import time
import threading
import psutil
import os
from shapely import Point, get_coordinates, points
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import h5py
import xarray as xr
from tqdm import tqdm
from concurrent.futures import as_completed
import stackstac
import planetary_computer as pc
from pystac_client import Client
from rasterio.transform import rowcol
import rasterio
from concurrent.futures import ThreadPoolExecutor

from const import MAX_HEIGHT_METERS, VSM_NODATA, FIGURE_SIZES, FONT_SIZES, set_plot_fonts, fewer_ticks

set_plot_fonts()

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

NON_VEGETATION_CLASSES = {50, 60, 70, 80}
REALISTIC_MAX=150 #m
REALISTIC_MIN=-150

class ProgressMonitor:
    """Background thread that prints speed and RAM stats."""

    def __init__(self, total_tiles, interval=5.0):
        self.total = total_tiles
        self.done = 0
        self.interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._start_time = time.time()
        self._process = psutil.Process(os.getpid())
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def tick(self):
        with self._lock:
            self.done += 1

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.wait(self.interval):
            self._print_status()
        self._print_status()

    def _print_status(self):
        with self._lock:
            done = self.done
        elapsed = time.time() - self._start_time
        mem = self._process.memory_info()
        rss_gb = mem.rss / (1024 ** 3)
        try:
            for child in self._process.children(recursive=True):
                rss_gb += child.memory_info().rss / (1024 ** 3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        rate = done / elapsed if elapsed > 0 else 0
        eta = (self.total - done) / rate if rate > 0 else float('inf')
        pct = 100 * done / self.total if self.total > 0 else 0
        print(
            f"  [{done}/{self.total}] {pct:5.1f}% | "
            f"{rate:.2f} tiles/s | "
            f"elapsed {elapsed:.0f}s | "
            f"ETA {eta:.0f}s | "
            f"RAM {rss_gb:.2f} GB (total)"
        )
        
def load_vsm_naturalness(h5_file: Path, **kwargs):
    '''
    Load the VSM patch statistics.
    Args:
        vsm_patch_stats_dir: path to the VSM patch statistics directory
    Returns:
        ddf: DataFrame of the VSM patch statistics
    '''
    with h5py.File(h5_file, 'r') as f:
        vsm_patches = f['vsm_median'][:]
        naturalness = f['land_use_id'][:]
        rowids = f['rowid'][:]
    return vsm_patches, naturalness, rowids

def prepare_rh_profile(rh_datacube):
    """
    Prepare the RH datacube for binning. This includes:
    - Converting nodata values to NaN
    - Converting from decimeters to meters
    - Masking out invalid profiles (all nodata, infinite, negative, or unrealistic values)
    NOTE: rh_datacube should have 4 dimensions (n, rows, cols, bands)
    """
    n_batch, n_bands, n_rows, n_cols = rh_datacube.shape
    dtype = rh_datacube.dtype
    # reshape
    rh_datacube = rh_datacube.transpose(0, 2, 3, 1).astype(np.float32)
    rh_datacube[rh_datacube == VSM_NODATA] = np.nan
    if dtype == np.int16: # rh_datacube is in decimeters, convert to meters
        rh_datacube = rh_datacube/10
    return rh_datacube, n_batch, n_bands, n_rows, n_cols

def mask_rh_profile(rh_datacube):
    """
    This masking is for calculating diversity indices, which only counts enegy returned above ground, and in range (0, max_height). 
    RH metrics that are all nodata, infinite and zeros, will be masked out as nodata
    There shouldn't be nodata or infinite appear in one of the 101 RH metrics, but I didn't check
    Negative values won't be counted later. 
    Values above max_height will be clipped to max_height, which is 150m by default.
    NOTE: rh_datacube should be in meters, nodata should be nan!!! rh_datacube should have 4 dimensions (n, rows, cols, bands)
    """
    valid = np.isfinite(rh_datacube) & (rh_datacube >= 0) # all 101 RH metrics <0 should be invalid
    invalid_mask = valid.sum(axis=-1) == 0
    realistic = (rh_datacube < REALISTIC_MIN) | (rh_datacube > REALISTIC_MAX)
    realistic_mask = realistic.any(axis=-1) # any value smaller than -150 or larger than 150, the whole RH profile will be masked out, 
    nodata_mask = invalid_mask | realistic_mask # (n, rows, cols) True for nodata pixels, False for valid pixels
    n, rows, cols, bands = rh_datacube.shape
    n_pixels = n * rows * cols
    pixel_valid = nodata_mask.reshape(n_pixels, 1) == False
    tile_flat = rh_datacube.reshape(n_pixels, bands) # (n_pixels, n_bands)
    tile_clean = np.where(pixel_valid, tile_flat, -1.0)
    band_valid_flat = valid.reshape(n_pixels, bands) & pixel_valid     # per-band AND per-pixel
    return tile_clean, nodata_mask, band_valid_flat, n_pixels

def batch_binning(tile, bin_width=5, max_height=MAX_HEIGHT_METERS):
    """
    Vectorized binning for a batch of spatial chunks using searchsorted.
    """
    tile, n_batch, n_bands, n_rows, n_cols = prepare_rh_profile(tile) # (n, rows, cols, bands) in meters, with nodata as nan
    tile_flat, nodata_mask, _, n_pixels = mask_rh_profile(tile) # (n, rows, cols, 1) nodata mask should work for all bands, and apply to the original data, therefore in original shape, not flattened shape.
    tile_clean = np.minimum(tile_flat, max_height) # clip to max height, (n_pixels, n_bands) in meters, with nodata as -1.0, which is smaller than 0 and won't be counted in the histogram later.
    
    # Sort along axis=1 (bands)
    profiles = np.sort(tile_clean, axis=1)
    profiles = np.ascontiguousarray(profiles)

    lower_edges = np.arange(0, max_height, bin_width)
    upper_edges = np.arange(bin_width, max_height + bin_width, bin_width)

    idx_low = np.stack(
        [np.searchsorted(profiles[i, :], lower_edges, side='left') for i in range(n_pixels)]
    ) # (n_pixels, n_bins)
    idx_high = np.stack(
        [np.searchsorted(profiles[i, :], upper_edges, side='left') for i in range(n_pixels)]
    )
    idx_high[:, -1] = np.array(
        [np.searchsorted(profiles[i, :], upper_edges[-1:], side='right')[0] for i in range(n_pixels)]
    )
    hist = (idx_high - idx_low).astype(np.float32) # (n_pixels, n_bins)
    n_bins = int(max_height / bin_width)
    hist = hist.reshape(n_batch, n_rows, n_cols, n_bins)

    return hist, nodata_mask


def batch_binning_add_at(tile, bin_width=5, max_height=MAX_HEIGHT_METERS):
    """
    Vectorized binning for a batch of spatial chunks.
    """
    tile, n_batch, n_bands, n_rows, n_cols = prepare_rh_profile(tile) # (n, rows, cols, bands) in meters, with nodata as nan
    tile_flat, nodata_mask, band_valid_flat, n_pixels = mask_rh_profile(tile) # nodata_mask: (n, rows, cols, 1) nodata mask should work for all bands, and apply to the original data, therefore in original shape, not flattened shape.
    tile_clean = np.minimum(tile_flat, max_height) # clip to max height, (n_pixels, n_bands) in meters, with nodata as -1.0, which is smaller than 0 and won't be counted in the histogram later.
    n_bins = int(max_height / bin_width)
    bin_idx = np.clip((tile_clean / bin_width).astype(np.int32), 0, n_bins - 1)
    hist = np.zeros((n_pixels, n_bins), dtype=np.float32)

    pixel_indices = np.broadcast_to(
        np.arange(n_pixels)[:, np.newaxis], (n_pixels, n_bands)
    )
    np.add.at(hist, (pixel_indices[band_valid_flat], bin_idx[band_valid_flat]), 1.0)
    hist = hist.reshape(n_batch, n_rows, n_cols, n_bins)
    return hist, nodata_mask

def loop_binning(tile, bin_width=5, max_height=MAX_HEIGHT_METERS):
    """
    Per-pixel np.histogram as reference. Same preprocessing/masking pipeline
    as batch_binning and batch_binning_add_at — just unvectorized histogramming.

    Parameters
    ----------
    tile : ndarray, shape (n_batch, n_bands, n_rows, n_cols)

    Returns
    -------
    hist : ndarray, shape (n_batch, n_rows, n_cols, n_bins)
    nodata_mask : ndarray, shape (n_batch, n_rows, n_cols, 1)
    """
    tile, n_batch, n_bands, n_rows, n_cols = prepare_rh_profile(tile)
    tile_flat, nodata_mask, band_valid_flat, n_pixels = mask_rh_profile(tile)
    tile_clean = np.minimum(tile_flat, max_height)   # (n_pixels, n_bands)

    n_bins = int(max_height / bin_width)
    bin_edges = np.arange(0, max_height + bin_width, bin_width, dtype=np.float64)

    hist = np.zeros((n_pixels, n_bins), dtype=np.float32)
    for i in range(n_pixels):
        valid = band_valid_flat[i]
        if not valid.any():
            continue
        h, _ = np.histogram(tile_clean[i, valid], bins=bin_edges)
        hist[i, :] = h

    hist = hist.reshape(n_batch, n_rows, n_cols, n_bins)
    return hist, nodata_mask

def verify_batch_binning():
    """
    Verify batch_binning (searchsorted) and batch_binning_add_at against
    loop_binning (per-pixel np.histogram reference). All three share the
    same prepare_rh_profile + mask_rh_profile pipeline, so results should
    be bit-identical.
    """
    np.random.seed(42)

    # Small tile for correctness verification: 1,210 pixels. loop_binning
    # runs one np.histogram per pixel, so keep this fast.
    # int16 decimeters mimics raw stored data; prepare_rh_profile divides by 10.
    n_batch, n_bands, n_rows, n_cols = 10, 101, 11, 11
    tile = np.random.randint(0, 1500, size=(n_batch, n_bands, n_rows, n_cols), dtype=np.int16)

    # Inject zeros and VSM_NODATA sentinels (no np.nan — int16 can't hold it)
    mask = np.random.random(tile.shape) < 0.3
    tile[mask] = 0
    mask2 = np.random.random(tile.shape[:-1]) < 0.05
    tile[mask2] = VSM_NODATA

    # Run all three
    hist_search, _ = batch_binning(tile.copy(), bin_width=5)
    hist_addat,  _ = batch_binning_add_at(tile.copy(), bin_width=5)
    hist_loop,   _ = loop_binning(tile.copy(), bin_width=5)

    print("=" * 60)
    print("SHAPE CHECK")
    print("=" * 60)
    print(f"searchsorted: {hist_search.shape}")
    print(f"add.at:       {hist_addat.shape}")
    print(f"loop:         {hist_loop.shape}")

    print()
    print("=" * 60)
    print("VALUE COMPARISON (vs loop reference)")
    print("=" * 60)
    for name, hist in [("searchsorted", hist_search), ("add.at", hist_addat)]:
        diff = np.abs(hist - hist_loop)
        max_diff, mean_diff = float(diff.max()), float(diff.mean())
        match = "YES" if max_diff < 1e-4 else "NO"
        print(f"{name:13s} | max {max_diff:.8f} | mean {mean_diff:.8f} | match {match}")

    # Show a few sample pixels
    print()
    print("=" * 60)
    print("SAMPLE PIXELS (first 3 non-empty)")
    print("=" * 60)
    count = 0
    for b in range(n_batch):
        for r in range(n_rows):
            for c in range(n_cols):
                if hist_loop[b, r, c].sum() > 0 and count < 3:
                    print(f"\nPixel (b={b}, r={r}, c={c}):")
                    print(f"  loop:         {hist_loop[b, r, c]}")
                    print(f"  searchsorted: {hist_search[b, r, c]}")
                    print(f"  add.at:       {hist_addat[b, r, c]}")
                    count += 1

    # Benchmark on larger tile
    print()
    print("=" * 60)
    print("BENCHMARK (10 x 101 x 64 x 64 tile)")
    print("=" * 60)
    big_tile = np.random.randint(0, 1500, size=(10, 101, 64, 64), dtype=np.int16)
    big_mask = np.random.random(big_tile.shape) < 0.3
    big_tile[big_mask] = 0

    timings = {}
    for name, fn in [
        ("searchsorted", batch_binning),
        ("add.at",       batch_binning_add_at),
        ("loop",         loop_binning),
    ]:
        t0 = time.time()
        fn(big_tile.copy(), bin_width=5)
        timings[name] = time.time() - t0
        print(f"{name:13s}: {timings[name]:.3f}s")

    print(f"\nspeedup vs loop:")
    print(f"  searchsorted: {timings['loop'] / timings['searchsorted']:.1f}x")
    print(f"  add.at:       {timings['loop'] / timings['add.at']:.1f}x")
    
    

def verify_diversity_indices_torch(n=256, bin_width=5, max_height=MAX_HEIGHT_METERS,
                                   rtol=1e-3, atol=1e-3, seed=42, verbose=True):
    """
    Cross-check datasets.h5_dataset.diversity_indices_torch (batched torch,
    the production path) against
    evaluation.on_diversity_indices.pixel_diversity_indices (per-pixel numpy
    reference). Both consume the SAME decimetre-quantised profile, so only the
    algorithm -- not quantisation -- is compared.

    Prints a per-index table and raises AssertionError if any index disagrees
    beyond (rtol, atol), so it works as both a quick script and a regression
    test:  python -c "from evaluation.utils import \
        verify_diversity_indices_torch as v; v()"

    NOTE: RH25/RH98 are forced valid here. CR is *defined* to differ between
    the two impls when RH25/RH98 are themselves invalid (reference indexes
    rhs[25]/rhs[98] with NaN; torch uses the raw value) -- a known, documented
    gap, out of scope for this equivalence check.
    """
    import torch
    from datasets.h5_dataset import diversity_indices_torch, DIVERSITY_NAMES
    from evaluation.on_diversity_indices import pixel_diversity_indices

    rng = np.random.default_rng(seed)
    n_bands = 101

    # Metres, sorted ascending (RH percentiles are non-decreasing), spanning a
    # bit past max_height to exercise the clipping path.
    prof_m = np.sort(
        rng.uniform(0, max_height * 1.2, size=(n, n_bands)).astype(np.float32), axis=1)

    zero_mask = rng.random((n, n_bands)) < 0.15
    nod_mask = rng.random((n, n_bands)) < 0.05
    prof_m[zero_mask] = 0.0
    # Keep RH25/RH98 valid (see NOTE) so CR is comparable.
    prof_m[:, 25] = rng.uniform(1, max_height, size=n).astype(np.float32)
    prof_m[:, 98] = rng.uniform(1, max_height, size=n).astype(np.float32)
    nod_mask[:, [25, 98]] = False

    # Decimetre-quantised raw (m*10 int16), nodata sentinel at nod positions.
    raw = np.clip(np.round(prof_m * 10.0), -32768, 32767).astype(np.int16)
    raw[nod_mask] = VSM_NODATA
    raw[0] = VSM_NODATA  # one fully-invalid profile -> both must be all-NaN

    # Reference gets the SAME quantised metres; invalid -> NaN so its
    # (finite & >0) filter drops exactly what torch drops via != nodata.
    prof_ref = raw.astype(np.float32) / 10.0
    prof_ref[raw == VSM_NODATA] = np.nan
    ref = np.array(
        [pixel_diversity_indices(prof_ref[i], bin_width=bin_width, max_height=max_height)
         for i in range(n)], dtype=np.float64)  # (n, 4)

    vsm = torch.from_numpy(np.ascontiguousarray(raw[:, :, None, None]))
    got = diversity_indices_torch(
        vsm, bin_width=bin_width, max_height=max_height
    ).numpy()[:, :, 0, 0].astype(np.float64)  # (n, 4)

    ok = True
    if verbose:
        print("=" * 64)
        print(f"diversity_indices_torch  vs  pixel_diversity_indices  (n={n})")
        print("=" * 64)
    for k, name in enumerate(DIVERSITY_NAMES):
        a, b = ref[:, k], got[:, k]
        one_nan = np.isnan(a) ^ np.isnan(b)
        both_nan = np.isnan(a) & np.isnan(b)
        finite = ~(np.isnan(a) | np.isnan(b))
        close = np.isclose(a[finite], b[finite], rtol=rtol, atol=atol)
        passed = (not one_nan.any()) and bool(close.all())
        ok &= passed
        if verbose:
            md = float(np.abs(a[finite] - b[finite]).max()) if finite.any() else 0.0
            print(f"{name:6s} | max|Δ| {md:.3e} | nan-mismatch {int(one_nan.sum()):3d}"
                  f" | both-nan {int(both_nan.sum()):3d} | {'PASS' if passed else 'FAIL'}")
    if verbose:
        print("-" * 64)
        print("OVERALL:", "PASS" if ok else "FAIL")
    assert ok, "diversity_indices_torch disagrees with pixel_diversity_indices"
    return ok


def plot_biome_samples(gdf_dissolved, points_gdf, save_path=None, figsize=FIGURE_SIZES['panel']):
    """Plot biome polygons with sampled points overlaid, no explicit loops."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot all biome polygons at once
    gdf_dissolved.plot(
        ax=ax, column="BIOME", cmap="tab20", alpha=0.5,
        edgecolor="black", linewidth=0.3, legend=True,
        # legend_kwds={"fontsize": 7, "loc": "lower left", "ncol": 2}
    )

    # Plot all points at once, colored by biome
    points_gdf.plot(
        ax=ax, column="BIOME", cmap="tab20", markersize=12,
        edgecolor="k", linewidth=0.3, zorder=5, alpha=0.8
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fewer_ticks(ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    

def plot_bars_frames(summary_file: str, per_class_file: str, all_cms_file: str,
                     save_dir: str, groups: tuple[str] = None, metric: str = 'F1',
                     avg: str = 'macro', baseline_name: str = 'rh98',
                     reveal_groups: list[list[str]] = None,
                     reveal_steps: list[int] = None, start_empty: bool = True,
                     show_improve: bool = True, improve_style: str = 'hatch',
                     hatch: str = '////', show_legend: bool = True,
                     dpi: int = 160, transparent: bool = True, **kwargs):
    '''
    Render `evaluation.on_naturalness.plot_bars` as a sequence of cumulative
    "reveal" frames for a Google-Slides / Keynote click-to-build animation:
    each frame draws one more model-group (one extra coloured bar per category)
    on top of a fixed geometry, so the PNGs overlay pixel-for-pixel and clicking
    advances frame_00 -> frame_01 -> ... .

    Mirrors plot_bars's data prep, ALS-first reordering, ALL/forest-type
    separator, baseline-lightening and legend exactly — the only difference is
    that un-revealed groups (and their legend swatches) are withheld/faded.
    Colours are pinned from the prop cycle up front so a group keeps the same
    colour across every frame even before it appears.

    Args:
        summary_file/per_class_file/all_cms_file: same artifacts plot_results
            reads (logistic_regression_*.csv / .npz).
        save_dir: frames are written to
            {save_dir}/barframes_{metric}_{avg}_baseline_{baseline_name}/frame_XX.png
        groups: model keys, in bar order (same as plot_bars/plot_results).
            Optional/ignored when reveal_groups is given (it sets the order).
        metric/avg/baseline_name: as in plot_bars.
        reveal_groups: list of model-key lists — one click reveals one inner
            list. This is the recommended, edit-friendly way to control the
            animation; it also fixes the bar order (its flattened concatenation
            becomes `groups`). Example reveal order rh98 -> the four diversity
            variants together -> key_rhs -> everything else:
                [[rh98],
                 [rh98_fhd, rh98_enl2d, rh98_cr, rh98_fhd_enl1d_enl2d_cr],
                 [key_rhs],
                 [rh98_s2, full_profile, full_profile_s2]]
        reveal_steps: lower-level alternative to reveal_groups — cumulative
            #groups shown per frame against `groups` order, e.g. [1, 5, 6, 9].
            Ignored if reveal_groups is set. Default (neither given) reveals one
            group per click: [0,1,..,N] (or [1,..,N] if start_empty=False).
        start_empty: include a frame_00 with only axes/legend scaffold.
        show_improve: split each bar into a baseline part and a gain-over-
            baseline part so the improvement stands out.
        improve_style: how to render the gain part when show_improve —
            'hatch' (default): solid baseline + the gain hatched on top;
            'lighten': full-colour gain on a lightened baseline (plot_bars look).
        hatch: hatch pattern used by improve_style='hatch' (e.g. '////', 'xx').
        transparent: save PNGs with a transparent background (for slide overlay).
    Returns:
        list[Path] of the frames written, in order.
    '''
    from matplotlib.patches import Patch
    # Reuse the canonical metadata/helpers — no duplication. Imported lazily
    # (not at module top) to avoid a circular import, since on_naturalness
    # imports this module.
    from evaluation.on_naturalness import (
        LAND_USE_NAMES, MODEL_NAMES, add_accuracy_from_cms, _lighten,
    )

    set_plot_fonts(label=16, annot=16, legend=14, ticks=14)
    assert avg in ['macro', 'weighted']
    assert improve_style in ['hatch', 'lighten'], improve_style
    save_dir = Path(save_dir).expanduser()
    out_dir = save_dir / f'barframes_{metric}_{avg}_baseline_{baseline_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cms = np.load(Path(all_cms_file).expanduser())
    summary_df = pd.read_csv(Path(summary_file).expanduser(), index_col=0)
    per_class_df = pd.read_csv(Path(per_class_file).expanduser(), index_col=0)
    labels = [v['short_name'] for v in LAND_USE_NAMES.values()]
    summary_df, per_class_df = add_accuracy_from_cms(summary_df, per_class_df, all_cms, labels)

    mask = summary_df.Metric == f'{avg} avg'
    summary_df = summary_df.loc[mask, [metric]]
    per_class_df = per_class_df.loc[:, ['Class', metric]]

    # reveal_groups (list of model-key lists) is the high-level control: it
    # fixes the bar order (flattened) AND the per-click chunks. Falls back to
    # reveal_steps (cumulative counts) or one-group-per-click.
    if reveal_groups is not None:
        reveal_groups = [list(chunk) for chunk in reveal_groups]
        groups = [g for chunk in reveal_groups for g in chunk]
        counts = np.cumsum([len(chunk) for chunk in reveal_groups]).tolist()
        reveal_steps = ([0] + counts) if start_empty else counts
    if groups is None:
        raise ValueError('pass either `groups` or `reveal_groups`')

    forest_types = [c['short_name'] for c in LAND_USE_NAMES.values()]
    n_models = len(groups)

    # Build values dict: model_name -> per-class values + ALL (same as plot_bars)
    values_dict = {}
    for model_name in groups:
        values_dict[model_name] = (
            per_class_df.loc[model_name, metric].values.tolist()
            + [summary_df.loc[model_name, metric]]
        )

    # ALL first, then forest types by Land_use_ID (identical to plot_bars)
    manual_order_ids = [11, 20, 53, 31, 32, 40, 0]
    land_use_ids = list(LAND_USE_NAMES.keys())
    manual_indices = [land_use_ids.index(_id) for _id in manual_order_ids]
    all_index = len(forest_types)
    reorder_indices = [all_index] + manual_indices
    forest_types = [forest_types[i] for i in manual_indices]
    for m in values_dict:
        values_dict[m] = [values_dict[m][i] for i in reorder_indices]

    class_names = ['ALL'] + forest_types
    baseline_vals = np.array(values_dict[baseline_name])

    # Fixed geometry (computed once -> every frame shares it)
    bar_width = 0.5
    group_spacing = 0.5
    all_gap = n_models * bar_width
    x_pos = np.arange(len(class_names)) * (n_models * bar_width + group_spacing)
    x_pos[1:] = x_pos[1:] + all_gap
    sep_x = (x_pos[0] + (n_models - 1) * bar_width + bar_width / 2 + x_pos[1] - bar_width / 2) / 2
    x_lim = (x_pos[0] - bar_width, x_pos[-1] + n_models * bar_width)

    # Pin a stable colour per group from the prop cycle, so a group keeps its
    # colour across frames even before it is revealed.
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    group_colors = [cycle[i % len(cycle)] for i in range(n_models)]

    if reveal_steps is None:
        reveal_steps = list(range(0 if start_empty else 1, n_models + 1))
    figsize = kwargs.get('figsize', FIGURE_SIZES['panel'])

    def _draw(reveal: int, frame_idx: int):
        fig, ax = plt.subplots(figsize=figsize)
        for i, model_name in enumerate(groups):
            if i >= reveal:
                continue
            values = np.array(values_dict[model_name])
            offsets = x_pos + i * bar_width
            color = group_colors[i]
            if show_improve and model_name != baseline_name:
                base_part = np.minimum(values, baseline_vals)
                if improve_style == 'hatch':
                    # Solid baseline, with the gain hatched on top (white lines
                    # over the model colour) — the requested slide look.
                    ax.bar(offsets, base_part, bar_width, color=color,
                           label=MODEL_NAMES[model_name]['name'])
                    ax.bar(offsets, values - base_part, bar_width, bottom=base_part,
                           color=color, hatch=hatch, edgecolor='white', linewidth=0.0)
                else:  # 'lighten'
                    ax.bar(offsets, values, bar_width, color=color,
                           label=MODEL_NAMES[model_name]['name'])
                    ax.bar(offsets, base_part, bar_width, color=_lighten(color))
            else:
                ax.bar(offsets, values, bar_width, color=color,
                       label=MODEL_NAMES[model_name]['name'])

        ax.axvline(sep_x, color='gray', linestyle='--', linewidth=1.5)
        ax.set_xticks(x_pos + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(class_names, fontsize=FONT_SIZES['ticks'], rotation=45, ha='center')
        ax.set_ylabel(f'{metric}', fontsize=FONT_SIZES['label'])
        ax.tick_params(axis='y', labelsize=FONT_SIZES['ticks'])
        ax.set_xlim(*x_lim)
        ax.set_ylim(0, 1.01)
        fewer_ticks(ax, axis='y')
        ax.grid(axis='y', alpha=0.3)

        leg = None
        if show_legend:
            # Always lay out all N entries (faded until revealed) so the figure
            # bbox — and thus the saved crop — is identical on every frame.
            faded = 0.15
            handles = [Patch(facecolor=group_colors[i], label=MODEL_NAMES[g]['name'],
                             alpha=1.0 if i < reveal else faded)
                       for i, g in enumerate(groups)]
            leg = ax.legend(handles=handles, loc='lower center',
                            bbox_to_anchor=(0.5, 1.02), ncol=3,
                            fontsize=FONT_SIZES['legend'], frameon=False)
            leg.set_in_layout(False)
            # Fade the label text too, so un-revealed entries dim with their swatch.
            for i, txt in enumerate(leg.get_texts()):
                txt.set_alpha(1.0 if i < reveal else faded)
        plt.tight_layout()
        out_path = out_dir / f'frame_{frame_idx:02d}.png'
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight', transparent=transparent,
                    bbox_extra_artists=None if leg is None else [leg])
        plt.close()
        return out_path

    frames = [_draw(reveal, k) for k, reveal in enumerate(reveal_steps)]
    print(f'wrote {len(frames)} frames -> {out_dir}')
    return frames


def _sample_tile(item, xs, ys, indices):
    """Read land cover values for points falling within one tile."""
    href = pc.sign(item.assets["map"].href)
    result = {}
    with rasterio.open(href) as src:
        tb = src.bounds
        in_tile = (
            (xs[indices] >= tb.left) & (xs[indices] <= tb.right) &
            (ys[indices] >= tb.bottom) & (ys[indices] <= tb.top)
        )
        idx = indices[in_tile]
        if len(idx) == 0:
            return result
        rows, cols = rowcol(src.transform, xs[idx], ys[idx])
        rows = np.clip(rows, 0, src.height - 1)
        cols = np.clip(cols, 0, src.width - 1)
        # Windowed read to avoid loading the full tile
        row_min, row_max = int(np.min(rows)), int(np.max(rows))
        col_min, col_max = int(np.min(cols)), int(np.max(cols))
        window = rasterio.windows.Window(
            col_min, row_min,
            col_max - col_min + 1, row_max - row_min + 1,
        )
        data = src.read(1, window=window)
        local_rows = np.array(rows) - row_min
        local_cols = np.array(cols) - col_min
        for i, r, c in zip(idx, local_rows, local_cols):
            result[i] = data[r, c]
    return result


def get_landcover_values(points_gdf, collection="esa-worldcover", year=2021, max_workers=16):
    """Query ESA WorldCover tile-by-tile in parallel via thread pool."""
    catalog = Client.open(STAC_URL, modifier=pc.sign_inplace)
    bbox = [float(b) for b in points_gdf.total_bounds]
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(search.items())
    print(f"  Found {len(items)} WorldCover tiles")

    xs = points_gdf.geometry.x.values
    ys = points_gdf.geometry.y.values
    lc_values = np.full(len(points_gdf), -1, dtype=np.int16)
    remaining = np.arange(len(points_gdf))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_sample_tile, item, xs, ys, remaining): item
            for item in items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Querying WorldCover tiles"):
            for i, val in future.result().items():
                lc_values[i] = val

    # Update remaining so already-resolved points aren't rechecked
    n_resolved = np.sum(lc_values != -1)
    print(f"  Resolved {n_resolved}/{len(points_gdf)} points")
    return lc_values

def sample_points_by_biome(biome_file: str, n_samples: int, seed: int = 42,
                           save_dir: str = None, plot_points: bool = True,
                           collection: str = "esa-worldcover", **kwargs):
    '''
    Sample vegetation points by biome from the GEDI ecoregions shapefile.
    It took ~2301s to sample 100000 points per biome.
    Parameters:
        biome_file: path to the GEDI ecoregions shapefile
        n_samples: number of samples per biome
        seed: random seed
        save_dir: path to save the sampled points
        plot_points: whether to plot the sampled points
        collection: name of the ESA WorldCover collection
        year: year of the ESA WorldCover data
    Returns:
        points_gdf: GeoDataFrame of the sampled points
    '''
    year = kwargs.get('year', 2021)
    from shapely import prepare, contains

    NON_VEGETATION_CLASSES = {50, 60, 70, 80}
    OVERSAMPLE_FACTOR = 2

    rng = np.random.default_rng(seed)
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    biome_file = Path(biome_file).expanduser()
    biome_df = gpd.read_file(biome_file)
    biome_df = biome_df.to_crs(epsg=4326)
    gdf_dissolved = biome_df.dissolve(by="BIOME").reset_index()
    gdf_dissolved = gdf_dissolved[~gdf_dissolved.BIOME.isin({98, 99})]
    for geom in gdf_dissolved.geometry:
        prepare(geom)

    # --- Stage 1: oversample 2x with rejection sampling on geometry only ---
    n_oversample = n_samples * OVERSAMPLE_FACTOR
    points_by_biome = {}
    for biome in gdf_dissolved.BIOME.unique():
        geom = gdf_dissolved[gdf_dissolved.BIOME == biome].geometry.iloc[0]
        minx, miny, maxx, maxy = geom.bounds
        points = []
        while len(points) < n_oversample:
            batch = max((n_oversample - len(points)) * 5, 100)
            xs = rng.uniform(minx, maxx, batch)
            ys = rng.uniform(miny, maxy, batch)
            pts = [Point(x, y) for x, y in zip(xs, ys)]
            mask = contains(geom, pts)
            points.extend(p for p, m in zip(pts, mask) if m)
        points_by_biome[biome] = points[:n_oversample]

    # Build full oversampled GeoDataFrame
    rows = [
        {"BIOME": biome, "geometry": pt}
        for biome, pts in points_by_biome.items()
        for pt in pts
    ]
    all_points = gpd.GeoDataFrame(rows, crs=gdf_dissolved.crs)
    
    lc_values = get_landcover_values(all_points, collection=collection, year=year)
    all_points["lc_class"] = lc_values
    vegetation = all_points[
        (~all_points.lc_class.isin(NON_VEGETATION_CLASSES)) & (all_points.lc_class != -1)
    ].copy()

    # --- Stage 3: take n_samples per biome from the filtered set ---
    points_gdf = (
        vegetation
        .groupby("BIOME", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_samples, len(g)), random_state=seed))
        # .drop(columns="lc_class")
        .reset_index(drop=True)
    )

    # Warn if any biome came up short
    counts = points_gdf.groupby("BIOME").size()
    short = counts[counts < n_samples]
    if not short.empty:
        for biome, count in short.items():
            print(f"  Warning: biome {biome} only got {count}/{n_samples} vegetation points")

    points_gdf.to_parquet(save_dir / f'random_sample_{n_samples}_points_per_biome.parquet')
    print(f"Saved {len(points_gdf)} points")

    if plot_points:
        plot_biome_samples(biome_df, points_gdf,
                           save_path=save_dir / f'random_sample_{n_samples}_points_per_biome.pdf')
        
               
def partition_points_by_tile(gdf_file: str, s2_tile_file: str, save_dir: str, **kwargs):
    '''
    Partition the points by tile.
    Args:
        gdf_file: path to the points GeoDataFrame
        s2_tile_file: path to the S2 tile file
        save_dir: path to save the partitioned points
    Returns:
    '''
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    gdf_file = Path(gdf_file).expanduser()
    s2_tile_file = Path(s2_tile_file).expanduser()
    points_gdf = gpd.read_parquet(gdf_file)
    s2_tile_df = gpd.read_parquet(s2_tile_file, columns=['Name', 'geometry', 'covered_by_gedi'])
    s2_tile_df = s2_tile_df.to_crs(epsg=4326)
    points_gdf = points_gdf.to_crs(epsg=4326)
    points_gdf = gpd.sjoin(points_gdf, s2_tile_df, how='left', predicate='intersects')
    points_gdf = points_gdf.drop_duplicates(subset='geometry', keep='first')
    points_gdf = points_gdf.drop(columns=['index_right'])
    for tile in points_gdf.Name.unique():
        points_gdf_tile = points_gdf[points_gdf.Name == tile]
        points_gdf_tile.to_parquet(save_dir / f'{tile}.parquet')
        print(f"Saved {len(points_gdf_tile)} points to {save_dir / f'{tile}.parquet'}")


def _short_count(n):
    if n >= 1_000_000:
        return f'{n/1e6:.1f}M'
    if n >= 1_000:
        return f'{n/1e3:.0f}K'
    return str(n)


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='utils',
    default_run='sample_points_by_biome',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()