"""
Protected Area Structural Integrity Analysis
=============================================
Compare FHD, ENL, RH98 between:
  - Primary forests within protected areas (reference baseline)
  - All forests within protected areas
  - All forests outside protected areas

Stratified by WWF biome.
"""

import gc
import shutil
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio.errors import RasterioIOError
from rasterio.features import rasterize
from rasterio.transform import rowcol, xy
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from scipy import stats

from const import BIOMES_BY_VALUE

warnings.filterwarnings("ignore")


# GDAL dtype names for the gdal_rasterize CLI. Cover the dtypes we'd realistically
# burn (binary masks, biome IDs, occasional float layers); anything else can be
# added on demand.
_GDAL_DTYPE = {
    "uint8": "Byte", "uint16": "UInt16", "int16": "Int16",
    "int32": "Int32", "uint32": "UInt32",
    "float32": "Float32", "float64": "Float64",
}


def _gdal_rasterize_to_cache(vector_path, profile, cache_file, attribute=None,
                             fill=0, dtype="uint8", layer=None):
    """Stream `vector_path` to a GeoTIFF at `cache_file` via the gdal_rasterize
    CLI. Used instead of geopandas + rasterio.features.rasterize because the
    latter materialises every feature as a shapely object in RAM — a 4.6 GB /
    268k-polygon gpkg blows past 32 GB before the rasterize call even starts.
    gdal_rasterize reads features in batches from OGR and writes blocks of the
    output as it goes, so peak RSS stays in the low GB regardless of input size.

    gdal_rasterize does NOT reproject vectors, so the source CRS must already
    match the target raster CRS. We check up front via fiona (metadata-only,
    no geometry read) and fail loudly with an `ogr2ogr` command the caller can
    run once if the CRSs disagree.
    """
    if not shutil.which("gdal_rasterize"):
        raise RuntimeError(
            "gdal_rasterize not on PATH. It ships with GDAL; install GDAL or "
            "pass `use_gdal_cli=False` to fall back to the in-memory path "
            "(which OOMs on multi-GB vectors).")

    with fiona.open(vector_path, layer=layer) as src:
        src_crs_dict = src.crs
    from pyproj import CRS as _PyCRS
    src_crs = _PyCRS.from_user_input(src_crs_dict) if src_crs_dict else None
    tgt_crs = profile["crs"]
    src_epsg = src_crs.to_epsg() if src_crs else None
    tgt_epsg = tgt_crs.to_epsg() if hasattr(tgt_crs, "to_epsg") else None

    # gdal_rasterize doesn't reproject. If the vector is in a different CRS we
    # pay a one-time ogr2ogr reprojection (also streamed; bounded memory) and
    # cache the reprojected gpkg next to the raster cache so subsequent cache
    # rebuilds reuse it instead of reprojecting again.
    cache_file = Path(cache_file).expanduser()
    if src_epsg and tgt_epsg and src_epsg != tgt_epsg:
        if not shutil.which("ogr2ogr"):
            raise RuntimeError("ogr2ogr not on PATH; install GDAL.")
        reproj_path = (cache_file.parent /
                       f"{Path(vector_path).stem}.reprojected_to_{tgt_epsg}.gpkg")
        if not reproj_path.exists():
            reproj_path.parent.mkdir(parents=True, exist_ok=True)
            ogr_cmd = [
                "ogr2ogr",
                "-t_srs", f"EPSG:{tgt_epsg}",
                "-f", "GPKG",
                str(reproj_path), str(vector_path),
            ]
            if layer:
                ogr_cmd += [layer]
            print(f"  reprojecting EPSG:{src_epsg} -> EPSG:{tgt_epsg}")
            print(f"  $ {' '.join(ogr_cmd)}")
            subprocess.run(ogr_cmd, check=True)
        else:
            print(f"  reusing reprojected vector: {reproj_path}")
        # ogr2ogr preserves the source layer name in the output gpkg, so the
        # caller's `layer` arg keeps working. For a single-layer shp input
        # `layer` was None and stays None — gdal_rasterize picks the only
        # layer in the reprojected gpkg automatically.
        vector_path = reproj_path

    t = profile["transform"]
    xres, yres = t.a, -t.e  # transform.e is negative for north-up rasters
    xmin, ymax = t.c, t.f
    xmax = xmin + t.a * profile["width"]
    ymin = ymax + t.e * profile["height"]

    gdal_dtype = _GDAL_DTYPE.get(str(dtype))
    if gdal_dtype is None:
        raise ValueError(f"unsupported dtype for gdal_rasterize: {dtype!r}")

    cache_file = Path(cache_file).expanduser()
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gdal_rasterize",
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-tr", str(xres), str(yres),
        "-init", str(fill),
        "-a_nodata", str(fill),
        "-ot", gdal_dtype,
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
    ]
    cmd += (["-a", attribute] if attribute else ["-burn", "1"])
    if layer:
        cmd += ["-l", layer]
    cmd += [str(vector_path), str(cache_file)]

    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_raster(path, band=1):
    """Load one band of a GeoTIFF, return array, profile, and transform.
    Nodata pixels are cast to float NaN so downstream masking is uniform."""
    with rasterio.open(path) as src:
        data = src.read(band)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data.astype(float))
    return data, profile, transform


def rasterize_vector(vector_path, profile, attribute=None, fill=0,
                     dtype="uint8", layer=None, cache_file=None,
                     use_gdal_cli=True):
    """Rasterize a vector file onto the raster grid defined by `profile`.

    If `attribute` is given, burn that attribute value; otherwise burn 1.
    `layer` picks a specific layer from multi-layer formats (e.g. GeoPackage).

    When `cache_file` is given:
      - if it exists, the rasterization is skipped and the array is read from
        the cached single-band GeoTIFF;
      - if it doesn't, the rasterization runs once and writes the cache so
        every subsequent run hits the fast path. The cache is grid-bound — its
        shape must match `profile`; delete it to rebuild on a new grid.

    For the build step, `use_gdal_cli=True` (default) shells out to the
    `gdal_rasterize` CLI, which streams features from disk and keeps RSS in
    the low GB. The in-memory `gpd.read_file` + `rasterio.features.rasterize`
    path is kept as a fallback for small vectors when no `cache_file` is given,
    but it OOMs on multi-GB / hundred-thousand-polygon inputs (peak ~3-5× the
    on-disk file size from shapely object overhead).
    """
    out_shape = (profile["height"], profile["width"])
    vector_path = Path(vector_path).expanduser()

    if cache_file is not None:
        cache_file = Path(cache_file).expanduser()
        if cache_file.exists():
            with rasterio.open(cache_file) as src:
                data = src.read(1)
            if data.shape != out_shape:
                raise ValueError(
                    f"cache {cache_file} has shape {data.shape}, target grid is "
                    f"{out_shape}; delete the cache to rebuild on the new grid."
                )
            print(f"  loaded rasterized cache: {cache_file}")
            return data

        if use_gdal_cli:
            _gdal_rasterize_to_cache(
                vector_path, profile, cache_file,
                attribute=attribute, fill=fill, dtype=dtype, layer=layer,
            )
            with rasterio.open(cache_file) as src:
                return src.read(1)

    # In-memory fallback: only safe for small vectors.
    print(f"  rasterizing {vector_path} in-memory...")
    gdf = gpd.read_file(vector_path, layer=layer) if layer else gpd.read_file(vector_path)
    if gdf.crs != profile["crs"]:
        gdf = gdf.to_crs(profile["crs"])
    if attribute:
        shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf[attribute])]
    else:
        shapes = [(geom, 1) for geom in gdf.geometry]
    out = rasterize(
        shapes, out_shape=out_shape, transform=profile["transform"],
        fill=fill, dtype=dtype,
    )
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_profile = {
            **profile, "count": 1, "dtype": dtype, "nodata": fill,
            "compress": "lzw", "tiled": True,
        }
        with rasterio.open(cache_file, "w", **cache_profile) as dst:
            dst.write(out, 1)
        print(f"  wrote rasterized cache: {cache_file}")
    return out


def sample_raster_at_points(raster_shape, transform, points_gdf):
    """Return (rows, cols, valid) of pixel coordinates for each point. `valid`
    masks points falling outside the raster footprint."""
    rows, cols = rowcol(transform, points_gdf.geometry.x, points_gdf.geometry.y)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    h, w = raster_shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return rows, cols, valid


# Shared per-metric colour palette: diversity metrics use the cool ramp,
# RH98 sits in the warm contrast so the "canopy height misses degradation"
# story reads at a glance.
_METRIC_COLORS = {
    "fhd": "#2166ac", "fhd_rel": "#1b7837",
    "enl": "#4393c3", "cr": "#92c5de", "rh98": "#d6604d",
    "auc": "#2e6b30",
}


def plot_biome_comparison(results_df, metrics, output_path):
    """Per-biome bar plot of Cohen's d (vs. primary forest) for each metric,
    split into protected vs. outside-protected panels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = [m.upper() for m in metrics]
    colors = [_METRIC_COLORS.get(m, "#888888") for m in metrics]

    for ax, group in zip(axes, ["in_pa", "outside_pa"]):
        group_data = results_df[results_df["group"] == group]
        pivot = group_data.pivot(
            index="biome", columns="metric", values="cohens_d"
        ).reindex(columns=metrics)
        if len(pivot) == 0:
            continue
        pivot.plot(kind="bar", ax=ax, width=0.8, color=colors)
        ax.set_ylabel("Cohen's d (vs. primary forest)")
        ax.set_xlabel("")
        title = "Protected areas" if group == "in_pa" else "Outside protected areas"
        ax.set_title(f"{title}: structural deviation from primary forest")
        ax.legend(labels)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_headline_bars(pcts: dict, output_path):
    """Bar chart of the % of PA forest pixels below biome-primary P10, one
    bar per metric. Takes an ordered dict so the column order matches
    `metrics` from the caller."""
    labels = [m.upper() for m in pcts]
    values = list(pcts.values())
    colors = [_METRIC_COLORS.get(m, "#888888") for m in pcts]

    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 1, 4))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylabel("% of PA forests below\nbiome primary forest P10")
    ax.set_title("Structural degradation in protected areas")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Headline plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main runnable
# ---------------------------------------------------------------------------
def analyze_protected_areas(
    diversity_file: str,
    rh98_file: str,
    wdpa_file: str,
    biomes_file: str,
    naturalness_file: str,
    save_dir: str,
    fhd_band: int = 1,
    enl_band: int = 3,
    cr_band: int = 4,
    bin_size: float = 5.0,
    biome_attribute: str = "BIOME_NUM",
    wdpa_layer: str = None,
    biomes_layer: str = None,
    wdpa_cache_file: str = None,
    biomes_cache_file: str = None,
    forest_biomes: list = None,
    forest_threshold: float = 5.0,
    naturalness_col: str = "naturalness",
    primary_forest_value: int = 1,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    naturalness_crs: str = "EPSG:4326",
    min_primary_samples: int = 100,
    **kwargs,
):
    """Compare FHD / ENL / RH98 between primary forest, protected forest, and
    unprotected forest, stratified by WWF biome.

    Parameters
    ----------
    diversity_file : Multi-band 1 km GeoTIFF holding the diversity metrics in
        band order [fhd, enl1d, enl2d, cr]. `fhd_band` and `enl_band` pick the
        two bands actually used by the analysis.
    rh98_file : 1 km single-band GeoTIFF of RH98, stored as int16 decimetres;
        converted to float metres in-memory. Must share the same grid as
        `diversity_file`.
    fhd_band, enl_band, cr_band : 1-indexed band numbers within
        `diversity_file`. Defaults: 1 (fhd), 3 (enl2d), 4 (cr). Use 2 for
        enl1d.
    bin_size : Height-bin size used when FHD was computed, in metres. Drives
        the `fhd_rel = FHD / ln(RH98 / bin_size)` normalisation, which puts
        FHD in [0, 1] by dividing out the height-dependent max entropy. Must
        match the `bin_width` used by `evaluation/diversity_maps.py`.
    wdpa_file : WDPA protected area boundaries (shapefile / geopackage).
    biomes_file : WWF biome polygons; `biome_attribute` is burned to the grid.
    naturalness_file : CSV of Lesiv et al. (2022) forest naturalness point
        annotations with `lon_col`, `lat_col`, and `naturalness_col` columns.
    save_dir : Output directory for `biome_results.csv` and the plots.
    biome_attribute : Vector attribute on `biomes_file` to burn as biome id.
    wdpa_layer, biomes_layer : Layer name to read from multi-layer GeoPackages
        (e.g. WDPA's `WDPA_poly_<release>`). Leave `None` for single-layer
        shapefiles.
    wdpa_cache_file, biomes_cache_file : Path to a single-band GeoTIFF cache of
        the rasterized layer. On first run the vector is reprojected and
        rasterized, then written to the cache; on subsequent runs the cache is
        loaded directly. Strongly recommended for big static layers (the WDPA
        gpkg is multi-GB / hundreds of thousands of polygons).
    forest_biomes : Biome ids to analyse. Defaults to the forested biomes in
        the WWF scheme (1, 2, 3, 4, 5, 6, 12, 14).
    forest_threshold : RH98 height (metres) above which a pixel counts as
        forest. Replaces a separate forest mask raster.
    naturalness_col : Column on `naturalness_file` carrying the naturalness
        class label.
    primary_forest_value : Value of `naturalness_col` flagging primary forest.
    lon_col, lat_col : Longitude / latitude columns of `naturalness_file`.
    naturalness_crs : CRS the lon/lat columns are in.
    min_primary_samples : Skip biomes with fewer primary-forest pixels.
    """
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    forest_biomes = forest_biomes or [1, 2, 3, 4, 5, 6, 12, 14]
    metrics = ["fhd", "fhd_rel", "enl", "cr", "rh98"]

    # --- 1. Load rasters -------------------------------------------------
    # FHD, ENL, CR come from the same 4-band diversity GeoTIFF
    # (band order: 1=fhd, 2=enl1d, 3=enl2d, 4=cr); RH98 is a separate file.
    print("Loading raster data...")
    diversity_file = Path(diversity_file).expanduser()
    rh98_file = Path(rh98_file).expanduser()
    fhd, profile, transform = load_raster(diversity_file, band=fhd_band)
    enl, _, _ = load_raster(diversity_file, band=enl_band)
    cr, _, _ = load_raster(diversity_file, band=cr_band)
    rh98, _, _ = load_raster(rh98_file)
    # rh98_file is stored as int16 in decimetres; convert to float metres so
    # the threshold and stats below are in real height units.
    rh98 = rh98.astype(float) / 10.0
    assert fhd.shape == enl.shape == cr.shape == rh98.shape, (
        f"Shape mismatch: FHD={fhd.shape}, ENL={enl.shape}, "
        f"CR={cr.shape}, RH98={rh98.shape}"
    )
    print(f"  Raster shape: {fhd.shape[0]} x {fhd.shape[1]}")

    # --- 2. Rasterize vector layers --------------------------------------
    print("Rasterizing WDPA protected areas...")
    protected_mask = rasterize_vector(
        wdpa_file, profile, layer=wdpa_layer, cache_file=wdpa_cache_file,
    )

    print("Rasterizing WWF biomes...")
    biome_raster = rasterize_vector(
        biomes_file, profile, attribute=biome_attribute, layer=biomes_layer,
        cache_file=biomes_cache_file,
    )

    print("Loading forest naturalness (point annotations)...")
    nat_df = pd.read_csv(naturalness_file)
    nat_gdf = gpd.GeoDataFrame(
        nat_df,
        geometry=gpd.points_from_xy(nat_df[lon_col], nat_df[lat_col]),
        crs=naturalness_crs,
    )
    if nat_gdf.crs != profile["crs"]:
        nat_gdf = nat_gdf.to_crs(profile["crs"])

    primary_pts = nat_gdf[nat_gdf[naturalness_col] == primary_forest_value].copy()
    print(f"  Total naturalness points: {len(nat_gdf):,}")
    print(f"  Primary forest points:    {len(primary_pts):,}")

    p_rows, p_cols, p_valid = sample_raster_at_points(fhd.shape, transform, primary_pts)
    primary_pixel_set = set(zip(p_rows[p_valid], p_cols[p_valid]))
    print(f"  Primary forest pixels (unique 1km cells): {len(primary_pixel_set):,}")

    primary_raster = np.zeros_like(fhd, dtype=bool)
    for r, c in primary_pixel_set:
        primary_raster[r, c] = True

    # --- 3. Build pixel-level dataframe ----------------------------------
    # Forest mask is derived from RH98 height instead of a separate raster:
    # pixels taller than `forest_threshold` metres count as forest.
    print("Building pixel dataframe...")
    is_forest = rh98 > forest_threshold
    valid = (is_forest & np.isfinite(fhd) & np.isfinite(enl)
             & np.isfinite(cr) & np.isfinite(rh98))
    rows, cols = np.where(valid)
    # Relative entropy: FHD / max possible FHD given canopy height. n_bins =
    # rh98 / bin_size; max entropy is ln(n_bins). Pixels with n_bins <= 1
    # have no diversity to measure (single bin) -> mark as NaN so the per-
    # biome nanmean/nanpercentile path skips them cleanly.
    n_bins = rh98[valid] / bin_size
    with np.errstate(divide="ignore", invalid="ignore"):
        fhd_rel_vals = np.where(n_bins > 1, fhd[valid] / np.log(n_bins), np.nan)
    df = pd.DataFrame({
        "row": rows,
        "col": cols,
        "fhd": fhd[valid],
        "fhd_rel": fhd_rel_vals,
        "enl": enl[valid],
        "cr": cr[valid],
        "rh98": rh98[valid],
        "biome": biome_raster[valid],
        "protected": protected_mask[valid].astype(bool),
        "is_primary": primary_raster[valid],
    })
    df["group"] = np.where(df["protected"], "in_pa", "outside_pa")

    print(f"  Total forest pixels:    {len(df):,}")
    print(f"  Protected:              {df['protected'].sum():,}")
    print(f"  Primary forest:         {df['is_primary'].sum():,}")
    print(f"  Primary + protected:    {(df['is_primary'] & df['protected']).sum():,}")

    # --- 4. Per-biome analysis ------------------------------------------
    print("\nPer-biome analysis:")
    print("=" * 80)
    results = []
    for biome_id in forest_biomes:
        biome_name = BIOMES_BY_VALUE.get(biome_id, {}).get("name", f"Biome {biome_id}")
        bdf = df[df["biome"] == biome_id]
        primary = bdf[bdf["is_primary"]]
        in_pa = bdf[bdf["group"] == "in_pa"]
        outside = bdf[bdf["group"] == "outside_pa"]

        if len(primary) < min_primary_samples:
            print(f"\n{biome_name}: skipped (only {len(primary)} primary samples)")
            continue

        print(f"\n{biome_name} (n_primary={len(primary):,}, "
              f"n_pa={len(in_pa):,}, n_outside={len(outside):,})")

        for metric in metrics:
            ref_values = primary[metric].values
            ref_mean = np.nanmean(ref_values)
            ref_std = np.nanstd(ref_values)
            ref_p10 = np.nanpercentile(ref_values, 10)

            for group_name, group_df in [("in_pa", in_pa), ("outside_pa", outside)]:
                grp_values = group_df[metric].values
                grp_mean = np.nanmean(grp_values)

                pooled_std = np.sqrt((ref_std**2 + np.nanstd(grp_values)**2) / 2)
                cohens_d = (ref_mean - grp_mean) / pooled_std if pooled_std > 0 else 0
                pct_below = np.nanmean(grp_values < ref_p10) * 100

                if len(grp_values) > 1 and len(ref_values) > 1:
                    t_stat, p_val = stats.ttest_ind(
                        ref_values, grp_values, equal_var=False, nan_policy="omit"
                    )
                else:
                    t_stat, p_val = np.nan, np.nan

                results.append({
                    "biome": biome_name,
                    "biome_id": biome_id,
                    "metric": metric,
                    "group": group_name,
                    "ref_mean": ref_mean,
                    "group_mean": grp_mean,
                    "diff": ref_mean - grp_mean,
                    "cohens_d": cohens_d,
                    "pct_below_p10": pct_below,
                    "p_value": p_val,
                    "n_ref": len(ref_values),
                    "n_group": len(grp_values),
                })

            print(f"  {metric:>5s}: primary={ref_mean:.2f}, "
                  f"PA={np.nanmean(in_pa[metric]):.2f}, "
                  f"outside={np.nanmean(outside[metric]):.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / "biome_results.csv", index=False)
    if results_df.empty:
        n_primary_total = int(df["is_primary"].sum())
        biomes_present = sorted(df["biome"].unique().tolist())
        raise RuntimeError(
            "No biome passed `min_primary_samples` — nothing to analyse.\n"
            f"  primary forest pixels (all biomes): {n_primary_total:,}\n"
            f"  biome ids present in pixel grid:    {biomes_present}\n"
            f"  forest_biomes filter:               {forest_biomes}\n"
            f"  min_primary_samples:                {min_primary_samples}\n"
            "Common causes:\n"
            "  - `primary_forest_value` does not match `naturalness_col` "
            "encoding (Lesiv et al. uses 11 for primary natural forest, not 1)\n"
            "  - `biome_attribute` burns a column whose values don't overlap "
            "`forest_biomes` (WWF uses BIOME 1..14; some derivatives renumber)\n"
            "  - `forest_threshold` too high so no pixels qualify as forest"
        )

    # --- 5. Global headline numbers -------------------------------------
    print("\n" + "=" * 80)
    print("GLOBAL HEADLINE NUMBERS")
    print("=" * 80)
    for m in metrics:
        df[f"below_p10_{m}"] = False
    for biome_id in forest_biomes:
        bdf = df[df["biome"] == biome_id]
        primary = bdf[bdf["is_primary"]]
        if len(primary) < min_primary_samples:
            continue
        for metric in metrics:
            ref_p10 = np.nanpercentile(primary[metric].values, 10)
            mask = (df["biome"] == biome_id) & df["protected"]
            df.loc[mask, f"below_p10_{metric}"] = df.loc[mask, metric] < ref_p10

    pa_pixels = df[df["protected"]]
    n_pa = len(pa_pixels)
    pcts = {}
    if n_pa > 0:
        for m in metrics:
            pcts[m] = pa_pixels[f"below_p10_{m}"].mean() * 100
        print(f"\nAmong protected area forests (n={n_pa:,}):")
        for m in metrics:
            print(f"  {pcts[m]:5.1f}% below biome primary forest P10 in {m.upper()}")
        if "rh98" in pcts:
            divs = [m for m in metrics if m != "rh98"]
            div_max = max(pcts[m] for m in divs) if divs else 0
            print(f"\n  -> If diversity metrics (max={div_max:.0f}%) >> "
                  f"RH98 ({pcts['rh98']:.0f}%),")
            print(f"     canopy height underestimates structural degradation.")

    # --- 6. Visualization -----------------------------------------------
    plot_biome_comparison(results_df, metrics, save_dir / "biome_comparison.png")
    if n_pa > 0:
        plot_headline_bars(pcts, save_dir / "headline.png")

    # --- 7. Summary table -----------------------------------------------
    print("\n\nSummary table (Cohen's d by biome and metric, in_pa):")
    print("-" * 60)
    summary = results_df[results_df["group"] == "in_pa"].pivot(
        index="biome", columns="metric", values="cohens_d"
    ).round(3)
    print(summary.to_string())
    print("\nDone. Results saved to:", save_dir)
    return results_df


# ---------------------------------------------------------------------------
# AUC profile analysis: helpers
# ---------------------------------------------------------------------------
def _resolve_tile_dir(stac_col_dir, tile_id, year):
    """Folder holding one tile's RH GeoTIFFs, resolved through the STAC
    collection. Tiles live in different folders, so the directory comes from the
    item's `RH98_Q1` asset href rather than a fixed root — mirrors
    `postprocessing.core.sample_vsm._resolve_vsm_path`. Returns None if the tile
    isn't in the collection."""
    import pystac
    item_json = Path(stac_col_dir).expanduser() / f"{tile_id}_{year}" / f"{tile_id}_{year}.json"
    if not item_json.exists():
        return None
    item = pystac.Item.from_file(str(item_json))
    return Path(item.assets["RH98_Q1"].href.replace("file://", "")).parent


def _sample_tile_profiles(args):
    """ProcessPoolExecutor worker: sample the requested RH levels for every
    point that falls in one S2 tile.

    The 10 m RH product is one folder per S2 tile (resolved via STAC; see
    `_resolve_tile_dir`), one single-band file per level
    (`RH{level}_Q{q_idx}.tif`), each in the tile's local UTM CRS. Points'
    lon/lat are reprojected into the tile CRS (read off the first band — robust
    to naming, and the file is opened anyway).

    Each band is read ONCE over the bounding window of all the tile's points and
    the values pulled out with numpy fancy-indexing. On this latency-bound
    networked filesystem the cost is dominated by the *number* of read ops, and
    a windowed read is a fixed one-op-per-band regardless of point count —
    whereas `src.sample()` is one op per point per band, which blows up (and
    stalls the pool) on tiles holding many points. Mirrors
    `sample_vsm._sample_tile_points`.

    Returns (point_indices, profiles, stats) where profiles has shape
    (k, len(rh_levels)) (missing-band / out-of-footprint samples stay NaN) and
    stats = (tile_name, t_open, t_read, t_total, n_files, n_points, used_window)
    times the open vs read cost so the caller can see where the time goes.
    """
    tile_dir, idxs, lons, lats, rh_levels, q_idx, fname_pattern = args
    tile_dir = Path(tile_dir)
    files = [tile_dir / fname_pattern.format(level=lvl, q_idx=q_idx)
             for lvl in rh_levels]
    out = np.full((len(idxs), len(rh_levels)), np.nan, dtype=float)

    # Cap the GDAL block cache: workers are reused across many tiles and we read
    # each block ~once, so an unbounded cache just climbs in RAM for no benefit
    # (matches diversity_maps._process_tile). GDAL_DISABLE_READDIR_ON_OPEN skips
    # the per-open sidecar scan — the dominant open cost on networked storage.
    gdal.SetCacheMax(128 * 1024 * 1024)  # 128 MB per worker
    setup_done = False
    t_open = t_read = 0.0
    n_files = n_valid = 0
    t0_tile = time.perf_counter()
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for j, f in enumerate(files):
            t0 = time.perf_counter()
            try:
                src = rasterio.open(f)
            except RasterioIOError:
                continue
            t_open += time.perf_counter() - t0
            n_files += 1
            with src:
                if not setup_done:
                    # All bands share the tile grid: compute pixel coords and
                    # the bounding window once.
                    xs, ys = warp_transform("EPSG:4326", src.crs,
                                            list(lons), list(lats))
                    rows, cols = rowcol(src.transform, xs, ys)
                    rows, cols = np.asarray(rows), np.asarray(cols)
                    H, W = src.height, src.width
                    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
                    n_valid = int(valid.sum())
                    if n_valid == 0:
                        return idxs, out, (tile_dir.name, t_open, t_read,
                                           time.perf_counter() - t0_tile,
                                           n_files, 0, True)
                    r0, r1 = rows[valid].min(), rows[valid].max()
                    c0, c1 = cols[valid].min(), cols[valid].max()
                    window = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
                    rows_loc, cols_loc = rows - r0, cols - c0
                    setup_done = True
                nodata = src.nodata
                tr0 = time.perf_counter()
                data = src.read(1, window=window)
                t_read += time.perf_counter() - tr0
                vals = np.full(len(idxs), np.nan, dtype=float)
                vals[valid] = data[rows_loc[valid], cols_loc[valid]]
            if nodata is not None:
                vals[vals == nodata] = np.nan
            out[:, j] = vals
    stats = (tile_dir.name, t_open, t_read, time.perf_counter() - t0_tile,
             n_files, n_valid, True)
    return idxs, out, stats


def _extract_profiles_tiled(samples_df, s2_grid, stac_col_dir, year, rh_levels,
                            q_idx, fname_pattern, max_workers):
    """Extract the RH profile at `rh_levels` for all sample points from the
    per-tile 10 m product, one worker per S2 tile.

    Points (in lon/lat) are spatially joined to the S2 grid to find each
    point's tile; each tile's RH folder is then resolved through the STAC
    collection (`_resolve_tile_dir`, since tiles live in different folders).
    Points are grouped by tile so each worker opens a tile's files exactly once.
    Uses the same bounded-inflight ProcessPoolExecutor pattern as
    `diversity_maps._write_diversity_gtiff` so finished results don't pile up.

    `samples_df` must carry `lon` / `lat` (EPSG:4326). Returns a float array of
    shape (len(samples_df), len(rh_levels)) aligned to `samples_df.iloc` order;
    points whose tile is absent from the grid or the STAC collection stay
    all-NaN.
    """
    from evaluation.diversity_maps import ProgressMonitor

    n = len(samples_df)
    profiles = np.full((n, len(rh_levels)), np.nan, dtype=float)

    # Assign each point to its containing tile via spatial join. Tiles overlap
    # slightly, so a point can match >1 tile — keep the first match per point.
    pts = gpd.GeoDataFrame(
        {"_i": np.arange(n)},
        geometry=gpd.points_from_xy(samples_df["lon"], samples_df["lat"]),
        crs="EPSG:4326",
    )
    grid = s2_grid[["Name", "geometry"]]
    if grid.crs is None:
        grid = grid.set_crs("EPSG:4326")
    elif grid.crs.to_epsg() != 4326:
        grid = grid.to_crs("EPSG:4326")
    matched = gpd.sjoin(pts, grid, predicate="within", how="left")
    matched = matched[~matched["_i"].duplicated(keep="first")]
    n_unmatched = int(matched["Name"].isna().sum())
    if n_unmatched:
        print(f"  {n_unmatched:,} / {n:,} points matched no S2 tile (left NaN)")

    # Build one work item per tile, resolving its RH folder through STAC.
    work_items = []
    n_not_in_stac = 0
    for tile_id, grp in matched.dropna(subset=["Name"]).groupby("Name"):
        tile_dir = _resolve_tile_dir(stac_col_dir, tile_id, year)
        if tile_dir is None or not tile_dir.is_dir():
            n_not_in_stac += 1
            continue
        idxs = grp["_i"].to_numpy()
        lons = samples_df["lon"].to_numpy()[idxs]
        lats = samples_df["lat"].to_numpy()[idxs]
        work_items.append((str(tile_dir), idxs, lons, lats,
                           rh_levels, q_idx, fname_pattern))
    if n_not_in_stac:
        print(f"  {n_not_in_stac} tiles not in STAC collection (points left NaN)")

    print(f"  Sampling {n:,} points across {len(work_items)} tiles "
          f"with {max_workers} processes")
    if not work_items:
        return profiles

    monitor = ProgressMonitor(total_tiles=len(work_items), interval=5.0)
    max_inflight = max_workers * 2
    # Summed across workers (concurrent, so these exceed wall time) — the
    # open:read ratio tells us whether time goes to file opens or to reads.
    tot_open = tot_read = 0.0
    n_slow_logged = 0
    monitor.start()
    try:
        # max_tasks_per_child recycles each worker after N tiles. Long-lived
        # GDAL/PROJ state (block cache, coordinate-transform contexts, dataset
        # handles) accumulates per process and makes a reused worker slow down
        # and grow in RAM over a run — recycling releases it. Costs a process
        # respawn every N tiles, negligible against ~100 file opens per tile.
        with ProcessPoolExecutor(max_workers=max_workers,
                                 max_tasks_per_child=50) as pool:
            work_iter = iter(work_items)
            inflight = set()
            for _ in range(max_inflight):
                try:
                    inflight.add(pool.submit(_sample_tile_profiles, next(work_iter)))
                except StopIteration:
                    break
            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    idxs, out, st = fut.result()
                    profiles[idxs] = out
                    name, t_open, t_read, t_total, n_files, n_pts, win = st
                    tot_open += t_open
                    tot_read += t_read
                    if t_total > 3.0 and n_slow_logged < 40:
                        n_slow_logged += 1
                        print(f"  slow tile {name}: {t_total:.1f}s "
                              f"(open {t_open:.1f}s + read {t_read:.1f}s), "
                              f"{n_files} files, {n_pts} pts, "
                              f"{'window' if win else 'sample'}")
                    monitor.tick()
                    try:
                        inflight.add(pool.submit(_sample_tile_profiles, next(work_iter)))
                    except StopIteration:
                        pass
    finally:
        monitor.stop()
    denom = max(tot_open + tot_read, 1e-9)
    print(f"  I/O totals (summed over workers): open {tot_open:.0f}s "
          f"({100 * tot_open / denom:.0f}%) + read {tot_read:.0f}s "
          f"({100 * tot_read / denom:.0f}%)")
    return profiles


def plot_mean_profiles(samples_df, forest_biomes, rh_levels, output_dir):
    """Per-biome plot of the mean normalized RH profile (RH_q / RH98) for
    primary / protected / outside-PA forests, over the sampled `rh_levels`. The
    gap between curves is the height-independent structural signal the AUC
    summarises."""
    norm_cols = [f"norm_rh{lvl}" for lvl in rh_levels]
    quantiles = np.asarray(rh_levels)
    top = quantiles.max()
    groups = [
        ("primary", lambda d: d[d["is_primary"]], "#1b5e20", "Primary forest"),
        ("in_pa", lambda d: d[d["group"] == "in_pa"], "#1565c0", "Protected area"),
        ("outside_pa", lambda d: d[d["group"] == "outside_pa"], "#c62828", "Outside PA"),
    ]
    for biome_id in forest_biomes:
        bdf = samples_df[samples_df["biome"] == biome_id]
        if not len(bdf[bdf["is_primary"]]):
            continue
        biome_name = BIOMES_BY_VALUE.get(biome_id, {}).get("name", f"Biome {biome_id}")
        fig, ax = plt.subplots(figsize=(6, 5))
        for _, sel, color, label in groups:
            gdf = sel(bdf)
            if not len(gdf):
                continue
            ax.plot(quantiles, gdf[norm_cols].mean().values, color=color,
                    linewidth=2, label=label)
        ax.plot(quantiles, quantiles / top, "k--", linewidth=0.8,
                alpha=0.4, label="Uniform")
        ax.set_xlabel("RH level (q)")
        ax.set_ylabel("RH$_q$ / RH98")
        ax.set_title(biome_name)
        ax.set_xlim(0, top)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = output_dir / f"profile_biome_{biome_id}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Profile plot saved: {out.name}")


def plot_auc_headline(pct_auc, pct_rh98, output_path):
    """Two-bar headline: AUC flags top-heavy structure (above primary P90),
    RH98 flags short canopy (below primary P10). AUC >> RH98 means structural
    simplification is more prevalent than height loss."""
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["AUC\n(profile shape)", "RH98\n(canopy height)"],
        [pct_auc, pct_rh98],
        color=[_METRIC_COLORS["auc"], _METRIC_COLORS["rh98"]], width=0.5,
    )
    ax.set_ylabel("% of PA forests flagged\nas structurally deviant")
    ax.set_title("Structural simplification vs. height loss\nin protected areas")
    for bar, val in zip(bars, [pct_auc, pct_rh98]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Headline plot saved to {output_path}")


# ---------------------------------------------------------------------------
# AUC profile analysis: sampling (the expensive, cacheable half)
# ---------------------------------------------------------------------------
def _build_sample_profiles(
    rng, rh98_1km_file, rh98_1km_in_decimetres,
    wdpa_file, wdpa_layer, wdpa_cache_file,
    biomes_file, biome_attribute, biomes_layer, biomes_cache_file,
    naturalness_file, lon_col, lat_col, naturalness_crs,
    naturalness_col, primary_forest_value,
    forest_biomes, forest_threshold, samples_per_biome,
    s2_grid_file, stac_col_dir, year,
    n_rh_bands, rh_level_step, q_idx, rh_10m_filename_pattern,
    max_workers, rh_10m_in_decimetres,
):
    """Sample points (primary annotations + stratified random forest) and
    extract their normalized RH profiles + AUC from the per-tile 10 m product.

    Factored out of `analyze_protected_areas_auc` so the result can be cached —
    this is the expensive half (point sampling + the multiprocessing per-tile
    extraction). Returns (samples_df, rh_levels); `samples_df` carries
    lon/lat/biome/protected/is_primary/group, `auc`, `rh98`, and one
    `norm_rh{level}` column per sampled level.
    """
    # --- 1. Load the 1 km sampling frame ---------------------------------
    print("Loading 1 km sampling frame (RH98)...")
    rh98_1km, profile, transform = load_raster(Path(rh98_1km_file).expanduser())
    if rh98_1km_in_decimetres:
        rh98_1km = rh98_1km / 10.0
    print(f"  Frame shape: {rh98_1km.shape[0]} x {rh98_1km.shape[1]}")

    # --- 2. Rasterize WDPA + biomes onto the frame -----------------------
    print("Rasterizing WDPA protected areas...")
    protected_mask = rasterize_vector(
        wdpa_file, profile, layer=wdpa_layer, cache_file=wdpa_cache_file,
    )
    print("Rasterizing WWF biomes...")
    biome_raster = rasterize_vector(
        biomes_file, profile, attribute=biome_attribute, layer=biomes_layer,
        cache_file=biomes_cache_file,
    )

    print("Loading forest naturalness (point annotations)...")
    nat_df = pd.read_csv(naturalness_file)
    nat_gdf = gpd.GeoDataFrame(
        nat_df,
        geometry=gpd.points_from_xy(nat_df[lon_col], nat_df[lat_col]),
        crs=naturalness_crs,
    )
    if nat_gdf.crs != profile["crs"]:
        nat_gdf = nat_gdf.to_crs(profile["crs"])
    primary_pts = nat_gdf[nat_gdf[naturalness_col] == primary_forest_value].copy()
    print(f"  Primary forest annotation points: {len(primary_pts):,}")

    # --- 3. Build sample points (two sources) ----------------------------
    # The primary reference and the PA/outside comparison are sampled
    # differently on purpose:
    #   primary    -> the naturalness primary points themselves, at their true
    #                 coordinates. Random forest sampling almost never lands on
    #                 the sparse annotation cells, so the baseline must be drawn
    #                 directly from them or it comes out empty.
    #   in_PA /    -> stratified random forest points per biome, so the headline
    #   outside_PA    reflects ALL protected/unprotected forest, not just the
    #                 annotation points.
    # Per-point attributes (biome, protected, frame RH98) are read off the 1 km
    # frame; lon/lat (EPSG:4326) drive the later point->tile join. Primary
    # points keep their true lon/lat; random points use the 1 km pixel centre.
    print("\nBuilding sample points:")
    is_forest = rh98_1km > forest_threshold
    lon_parts, lat_parts, biome_parts = [], [], []
    prot_parts, rh98f_parts, isprim_parts = [], [], []

    # (a) primary reference — naturalness primary points (true coords).
    prim_4326 = primary_pts.to_crs("EPSG:4326")
    pr_rows, pr_cols, pr_valid = sample_raster_at_points(rh98_1km.shape, transform, primary_pts)
    pr_rows, pr_cols = pr_rows[pr_valid], pr_cols[pr_valid]
    pr_lon = prim_4326.geometry.x.to_numpy()[pr_valid]
    pr_lat = prim_4326.geometry.y.to_numpy()[pr_valid]
    pr_biome = biome_raster[pr_rows, pr_cols]
    in_fb = np.isin(pr_biome, forest_biomes)
    pr_rows, pr_cols, pr_lon, pr_lat, pr_biome = (
        pr_rows[in_fb], pr_cols[in_fb], pr_lon[in_fb], pr_lat[in_fb], pr_biome[in_fb])
    # Cap per biome at samples_per_biome so no single biome dominates.
    keep = []
    for biome_id in forest_biomes:
        bi = np.where(pr_biome == biome_id)[0]
        if len(bi) > samples_per_biome:
            bi = rng.choice(bi, size=samples_per_biome, replace=False)
        keep.append(bi)
    keep = np.concatenate(keep) if keep else np.array([], dtype=int)
    pr_rows, pr_cols = pr_rows[keep], pr_cols[keep]
    lon_parts.append(pr_lon[keep]); lat_parts.append(pr_lat[keep])
    biome_parts.append(pr_biome[keep])
    prot_parts.append(protected_mask[pr_rows, pr_cols].astype(bool))
    rh98f_parts.append(rh98_1km[pr_rows, pr_cols])
    isprim_parts.append(np.ones(len(pr_rows), dtype=bool))
    print(f"  primary reference points (in forest biomes): {len(pr_rows):,}")

    # (b) in_PA / outside_PA — stratified random forest points per biome.
    print("  random forest points per biome:")
    for biome_id in forest_biomes:
        rows, cols = np.where(is_forest & (biome_raster == biome_id))
        if not len(rows):
            print(f"    Biome {biome_id}: no forest pixels, skipping")
            continue
        n_sample = min(samples_per_biome, len(rows))
        idx = rng.choice(len(rows), size=n_sample, replace=False)
        r, c = rows[idx], cols[idx]
        xs, ys = xy(transform, r, c)
        lon, lat = warp_transform(profile["crs"], "EPSG:4326",
                                  list(np.asarray(xs)), list(np.asarray(ys)))
        lon_parts.append(np.asarray(lon)); lat_parts.append(np.asarray(lat))
        biome_parts.append(np.full(n_sample, biome_id))
        prot_parts.append(protected_mask[r, c].astype(bool))
        rh98f_parts.append(rh98_1km[r, c])
        isprim_parts.append(np.zeros(n_sample, dtype=bool))
        biome_name = BIOMES_BY_VALUE.get(biome_id, {}).get("name", f"Biome {biome_id}")
        print(f"    {biome_name}: sampled {n_sample:,}")

    if sum(len(p) for p in lon_parts) == 0:
        raise RuntimeError("No sample points built; check the frame, biome "
                           "raster, naturalness points, and forest_threshold.")
    samples_df = pd.DataFrame({
        "lon": np.concatenate(lon_parts),
        "lat": np.concatenate(lat_parts),
        "biome": np.concatenate(biome_parts),
        "rh98_1km": np.concatenate(rh98f_parts),
        "protected": np.concatenate(prot_parts),
        "is_primary": np.concatenate(isprim_parts),
    })
    samples_df["group"] = np.where(samples_df["protected"], "in_pa", "outside_pa")
    print(f"\nTotal samples: {len(samples_df):,} "
          f"({int(samples_df['is_primary'].sum()):,} primary)")

    # Free the global 1 km arrays before forking the worker pool. They're only
    # needed for sampling (done above); everything downstream uses samples_df.
    # Without this, fork's copy-on-write hands every worker a view of these
    # multi-GB arrays — harmless physically, but it inflates summed RSS and
    # risks real copies under memory pressure.
    del rh98_1km, biome_raster, protected_mask, is_forest, nat_gdf, primary_pts
    gc.collect()

    # --- 4. Extract 10 m RH profiles + compute AUC -----------------------
    print("\nExtracting RH profiles at 10 m (per-tile, multiprocessing)...")
    s2_grid = gpd.read_parquet(Path(s2_grid_file).expanduser(),
                               columns=["Name", "geometry"])
    # Levels actually sampled: every `rh_level_step`-th, plus the top level
    # (RH98 denominator) which must always be present. Fewer levels => fewer
    # per-tile file opens => proportional speedup. RH0 is always negative
    # (ground return), so it clips to 0 and carries no information — skip
    # reading it entirely.
    rh_levels = sorted(set(range(0, n_rh_bands, rh_level_step)) | {n_rh_bands - 1})
    rh_levels = [lvl for lvl in rh_levels if lvl >= 1]
    print(f"  sampling {len(rh_levels)} RH levels (step={rh_level_step}, RH0 skipped)")
    profiles = _extract_profiles_tiled(
        samples_df, s2_grid, stac_col_dir, year, rh_levels, q_idx,
        rh_10m_filename_pattern, max_workers,
    )
    if rh_10m_in_decimetres:
        profiles = profiles / 10.0
    # Negative RH (sub-ground returns) is physically zero canopy height.
    # np.maximum keeps NaN nodata as NaN.
    profiles = np.maximum(profiles, 0.0)

    # AUC = mean(RH_q / RH98) over the sampled levels. The top level stands in
    # for RH98. Only valid where RH98 clears the forest threshold; the
    # normalised profile is scale-free so dm vs m input is irrelevant here.
    rh98_10m = profiles[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = profiles / rh98_10m[:, np.newaxis]
        auc = np.where(rh98_10m > forest_threshold,
                       np.nanmean(normalized, axis=1), np.nan)
    samples_df["auc"] = auc
    samples_df["rh98"] = rh98_10m
    for j, lvl in enumerate(rh_levels):
        samples_df[f"norm_rh{lvl}"] = np.where(
            (rh98_10m > forest_threshold), normalized[:, j], np.nan)
    n_valid = int(np.sum(~np.isnan(auc)))
    print(f"  Valid profiles: {n_valid:,} / {len(samples_df):,}")
    print(f"  AUC range {np.nanmin(auc):.4f}–{np.nanmax(auc):.4f}, "
          f"mean {np.nanmean(auc):.4f}")
    # Return a GeoDataFrame so the cached/saved points carry point geometry
    # (lon/lat, EPSG:4326) and round-trip via GeoParquet.
    samples_df = gpd.GeoDataFrame(
        samples_df,
        geometry=gpd.points_from_xy(samples_df["lon"], samples_df["lat"]),
        crs="EPSG:4326",
    )
    return samples_df, rh_levels


# ---------------------------------------------------------------------------
# AUC profile analysis: main runnable
# ---------------------------------------------------------------------------
def analyze_protected_areas_auc(
    rh98_1km_file: str,
    stac_col_dir: str,
    s2_grid_file: str,
    wdpa_file: str,
    biomes_file: str,
    naturalness_file: str,
    save_dir: str,
    samples_cache_file: str = None,
    year: int = 2020,
    q_idx: int = 1,
    rh_10m_filename_pattern: str = "RH{level}_Q{q_idx}.tif",
    max_workers: int = 8,
    n_rh_bands: int = 101,
    rh_level_step: int = 1,
    samples_per_biome: int = 10000,
    rh98_1km_in_decimetres: bool = True,
    rh_10m_in_decimetres: bool = True,
    biome_attribute: str = "BIOME",
    wdpa_layer: str = None,
    biomes_layer: str = None,
    wdpa_cache_file: str = None,
    biomes_cache_file: str = None,
    forest_biomes: list = None,
    forest_threshold: float = 5.0,
    naturalness_col: str = "naturalness",
    primary_forest_value: int = 1,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    naturalness_crs: str = "EPSG:4326",
    min_primary_samples: int = 100,
    random_seed: int = 42,
    **kwargs,
):
    """Structural analysis of protected forests via the AUC of the normalized
    RH profile — a height-independent shape metric — compared against RH98.

    The pipeline samples points from two sources: the primary-forest baseline
    is taken from the naturalness annotation points directly (at their true
    coordinates), while the protected / outside-PA comparison groups are
    stratified random forest points drawn on the 1 km frame — so the headline
    reflects all protected forest, not just the sparse annotations. It then
    extracts the full RH profile (`n_rh_bands` levels) at every point from the
    10 m product and computes AUC = mean(RH_q / RH98). Low AUC = understory-
    rich; high AUC = top-heavy / simplified. Protected and outside-PA forests
    are then compared against primary forest per biome, exactly as
    `analyze_protected_areas` does for the diversity metrics — the difference is
    only the metric and that it's sampled at 10 m rather than read off a 1 km
    grid.

    Parameters
    ----------
    rh98_1km_file : 1 km RH98 GeoTIFF used as the sampling frame (forest mask =
        RH98 > `forest_threshold`). Same grid as the WDPA/biome caches.
    stac_col_dir : STAC collection directory (`{tile_id}_{year}/...json` per
        tile). Each tile's RH folder is resolved from its `RH98_Q1` asset href,
        since tiles live in different folders (see `_resolve_tile_dir`). The
        profile for a point is read across that tile's `n_rh_bands` single-band
        files (local UTM); points are grouped by tile and sampled one worker per
        tile (see `_extract_profiles_tiled`).
    s2_grid_file : Parquet of the S2 tile grid with `Name` (tile id) and
        `geometry` columns, used to map each sample point to its tile.
    year : Product year; selects the STAC item `{tile_id}_{year}`.
    q_idx : Quantile index in the RH filenames (the `_Q{q_idx}` suffix).
    rh_10m_filename_pattern : `str.format` pattern for the per-level files
        inside a tile folder; `{level}` (0..n-1) and `{q_idx}` are substituted.
    max_workers : Processes for the per-tile profile sampling pool.
    n_rh_bands : Number of RH levels in the profile (default 101 → RH0..RH100;
        the last band is treated as RH98 for the normalisation).
    rh_level_step : Sample every `rh_level_step`-th RH level instead of all
        `n_rh_bands` (the top level is always kept as the RH98 denominator; RH0
        is always skipped since it's a negative ground return). AUC is the mean
        over a smooth, monotonic profile, so a coarser step (e.g. 4–5) barely
        changes it while cutting the per-tile file opens — the dominant cost on
        a networked filesystem — by the same factor. Default 1 = every level.
    samples_per_biome : Max stratified random forest points drawn per biome.
    rh98_1km_in_decimetres, rh_10m_in_decimetres : Convert int decimetres to
        float metres on load. The AUC itself is scale-invariant (ratio), but
        RH98-in-metres is needed for the forest threshold and the height stats.
    wdpa_file, biomes_file, naturalness_file, save_dir, biome_attribute,
    wdpa_layer, biomes_layer, wdpa_cache_file, biomes_cache_file, forest_biomes,
    forest_threshold, naturalness_col, primary_forest_value, lon_col, lat_col,
    naturalness_crs, min_primary_samples :
        Same meaning as in `analyze_protected_areas`.
    random_seed : Seed for the stratified point sampler.
    samples_cache_file : Parquet cache of the sampled per-point profiles
        (lon/lat, attributes, `auc`, `rh98`, `norm_rh*`). If it exists, sampling
        and the per-tile 10 m extraction are skipped and the cache is loaded —
        so re-running only the analysis/plots is instant. Delete it to
        re-sample. Defaults to `save_dir/samples_profiles.parquet`.
    """
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    forest_biomes = forest_biomes or [1, 2, 3, 4, 5, 6, 12, 14]
    rng = np.random.default_rng(random_seed)
    metrics = ["auc", "rh98"]

    # --- 1-4. Sample points + extract RH profiles (cached) ---------------
    # Sampling + per-tile 10 m extraction is the expensive part. Cache the
    # per-point profiles so re-runs (tweaking the analysis/plots) skip straight
    # to the stats. Delete the cache file to force a re-sample.
    samples_cache = (Path(samples_cache_file).expanduser() if samples_cache_file
                     else save_dir / "samples_profiles.parquet")
    samples_cache.parent.mkdir(parents=True, exist_ok=True)
    if samples_cache.exists():
        print(f"Loading cached sampled profiles: {samples_cache}")
        try:
            samples_df = gpd.read_parquet(samples_cache)
        except Exception:
            # Fall back for a legacy non-geo cache (no geometry metadata).
            samples_df = pd.read_parquet(samples_cache)
        rh_levels = sorted(int(c[len("norm_rh"):]) for c in samples_df.columns
                           if c.startswith("norm_rh"))
        print(f"  {len(samples_df):,} samples, {len(rh_levels)} RH levels")
    else:
        samples_df, rh_levels = _build_sample_profiles(
            rng, rh98_1km_file, rh98_1km_in_decimetres,
            wdpa_file, wdpa_layer, wdpa_cache_file,
            biomes_file, biome_attribute, biomes_layer, biomes_cache_file,
            naturalness_file, lon_col, lat_col, naturalness_crs,
            naturalness_col, primary_forest_value,
            forest_biomes, forest_threshold, samples_per_biome,
            s2_grid_file, stac_col_dir, year,
            n_rh_bands, rh_level_step, q_idx, rh_10m_filename_pattern,
            max_workers, rh_10m_in_decimetres,
        )
        samples_df.to_parquet(samples_cache)  # GeoParquet (samples_df is a GeoDataFrame)
        print(f"Cached sampled profiles -> {samples_cache}")

    # --- 5. Per-biome analysis (mirrors analyze_protected_areas) ---------
    # AUC degradation lives in the UPPER tail (top-heavy → above primary P90);
    # RH98 degradation lives in the LOWER tail (short → below primary P10).
    stat_df = samples_df.dropna(subset=["auc", "rh98"])
    print("\nPer-biome analysis:")
    print("=" * 80)
    results = []
    for biome_id in forest_biomes:
        biome_name = BIOMES_BY_VALUE.get(biome_id, {}).get("name", f"Biome {biome_id}")
        bdf = stat_df[stat_df["biome"] == biome_id]
        primary = bdf[bdf["is_primary"]]
        in_pa = bdf[bdf["group"] == "in_pa"]
        outside = bdf[bdf["group"] == "outside_pa"]
        if len(primary) < min_primary_samples:
            print(f"\n{biome_name}: skipped (only {len(primary)} primary samples)")
            continue
        print(f"\n{biome_name} (n_primary={len(primary):,}, "
              f"n_pa={len(in_pa):,}, n_outside={len(outside):,})")

        for metric in metrics:
            ref_values = primary[metric].values
            ref_mean = np.nanmean(ref_values)
            ref_std = np.nanstd(ref_values)
            ref_p10 = np.nanpercentile(ref_values, 10)
            ref_p90 = np.nanpercentile(ref_values, 90)
            for group_name, group_df in [("in_pa", in_pa), ("outside_pa", outside)]:
                grp_values = group_df[metric].values
                grp_mean = np.nanmean(grp_values)
                pooled_std = np.sqrt((ref_std**2 + np.nanstd(grp_values)**2) / 2)
                cohens_d = (ref_mean - grp_mean) / pooled_std if pooled_std > 0 else 0
                pct_below_p10 = np.nanmean(grp_values < ref_p10) * 100
                pct_above_p90 = (np.nanmean(grp_values > ref_p90) * 100
                                 if metric == "auc" else np.nan)
                if len(grp_values) > 1 and len(ref_values) > 1:
                    _, p_val = stats.ttest_ind(
                        ref_values, grp_values, equal_var=False, nan_policy="omit")
                else:
                    p_val = np.nan
                results.append({
                    "biome": biome_name, "biome_id": biome_id,
                    "metric": metric, "group": group_name,
                    "ref_mean": ref_mean, "group_mean": grp_mean,
                    "diff": ref_mean - grp_mean, "cohens_d": cohens_d,
                    "pct_below_p10": pct_below_p10, "pct_above_p90": pct_above_p90,
                    "p_value": p_val, "n_ref": len(ref_values),
                    "n_group": len(grp_values),
                })
            print(f"  {metric:>5s}: primary={ref_mean:.4f}, "
                  f"PA={np.nanmean(in_pa[metric]):.4f}, "
                  f"outside={np.nanmean(outside[metric]):.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / "biome_results_auc.csv", index=False)
    if results_df.empty:
        raise RuntimeError(
            "No biome passed `min_primary_samples` — nothing to analyse. "
            "Check `primary_forest_value` / `naturalness_col` encoding, the "
            "`biome_attribute`, and `forest_threshold`.")

    # --- 6. Global headline numbers --------------------------------------
    # AUC: fraction of PA forest above its biome primary P90 (simplified).
    # RH98: fraction below its biome primary P10 (short).
    print("\n" + "=" * 80)
    print("GLOBAL HEADLINE NUMBERS")
    print("=" * 80)
    stat_df = stat_df.copy()
    stat_df["auc_above_p90"] = False
    stat_df["rh98_below_p10"] = False
    for biome_id in forest_biomes:
        bdf = stat_df[stat_df["biome"] == biome_id]
        primary = bdf[bdf["is_primary"]]
        if len(primary) < min_primary_samples:
            continue
        auc_p90 = np.nanpercentile(primary["auc"].values, 90)
        rh98_p10 = np.nanpercentile(primary["rh98"].values, 10)
        mask = (stat_df["biome"] == biome_id) & stat_df["protected"]
        stat_df.loc[mask, "auc_above_p90"] = stat_df.loc[mask, "auc"] > auc_p90
        stat_df.loc[mask, "rh98_below_p10"] = stat_df.loc[mask, "rh98"] < rh98_p10

    pa = stat_df[stat_df["protected"]]
    pct_auc = pct_rh98 = None
    if len(pa) > 0:
        pct_auc = pa["auc_above_p90"].mean() * 100
        pct_rh98 = pa["rh98_below_p10"].mean() * 100
        print(f"\nAmong protected area forests (n={len(pa):,}):")
        print(f"  {pct_auc:5.1f}% have AUC above biome primary P90 "
              f"(structurally simplified / top-heavy)")
        print(f"  {pct_rh98:5.1f}% have RH98 below biome primary P10 "
              f"(shorter than primary)")
        print(f"\n  -> If AUC ({pct_auc:.0f}%) >> RH98 ({pct_rh98:.0f}%), structural "
              f"simplification is more prevalent than height loss.")

    # --- 7. Visualization ------------------------------------------------
    plot_biome_comparison(results_df, metrics, save_dir / "auc_vs_rh98_biome.png")
    plot_mean_profiles(samples_df, forest_biomes, rh_levels, save_dir)
    if pct_auc is not None:
        plot_auc_headline(pct_auc, pct_rh98, save_dir / "headline_auc.png")

    # --- 8. Save samples + summary ---------------------------------------
    # Lightweight points table (no per-level norm_rh columns) as GeoParquet so
    # it loads back as a GeoDataFrame. save_dir was created at the top.
    samples_df.drop(
        columns=[c for c in samples_df.columns if c.startswith("norm_rh")]
    ).to_parquet(save_dir / "samples_auc.parquet", index=False)
    print("\n\nSummary table (Cohen's d by biome and metric, in_pa):")
    print("-" * 60)
    summary = results_df[results_df["group"] == "in_pa"].pivot(
        index="biome", columns="metric", values="cohens_d"
    ).round(3)
    print(summary.to_string())
    print("\nDone. Results saved to:", save_dir)
    return results_df


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='protected_area_analysis',
    default_run='analyze_protected_areas',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()
