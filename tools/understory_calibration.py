"""Continuous understory calibration: VSM and GEDI vs LVIS, height removed.

For each footprint, remove canopy top height from RH25 to get an `understory
signal`. Put the LVIS understory signal (airborne lidar truth) on x, the VSM
and GEDI understory signals on y. The slope of VSM relative to GEDI is the
recovered fraction of the lidar-detectable understory contrast at matched
height — the continuous analogue of the grouped-model headline in
`tools.vsm_understory_matched`.

Pipeline (no banding, no terciles)
----------------------------------
1. Load the per-tile LVIS-GEDI-VSM pair parquets used by
   `vsm_understory_matched` and `gedi_penetration_bias`. Each row carries
   `lvis_RH25`/`lvis_RH98` (uppercase), `gedi_rh25`/`gedi_rh98` (lowercase),
   `vsm_RH25`/`vsm_RH98` (uppercase), `gedi_sensitivity`, geometry; the
   parquet filename stem is stamped as the S2 MGRS `tile` (bootstrap block).
2. Drop rows with non-finite values in any of the 8 RH columns or `tile`,
   then keep rows with `gedi_sensitivity > sensitivity_threshold` (default
   0.95 — same clean window the rest of the paper uses).
3. Residualize each instrument's RH25 on its OWN RH98 (OLS, one global fit).
   Each on its OWN — not LVIS RH98 — so each sensor's height channel is
   removed maximally and leftover agreement is genuinely beyond height.
   Print `corr(resid, rh98)` (~0 by construction) and `corr(resid, rh98^2)`;
   if any `|corr(resid, rh98^2)| > quadratic_tol`, switch ALL THREE to a
   natural cubic spline (df=4) and re-report.
4. Fix x-bin edges once from the FULL-sample `resid_lvis` quantiles (15
   equal-count bins). Per bin: x_center, n, mean(resid_vsm), mean(resid_gedi),
   mean(lvis_RH98) (the height-balance check).
5. Tile-block bootstrap (B=1000) for per-bin CIs and the recovery-ratio CI:
   each iteration resamples whole S2 tiles WITH replacement, re-runs step 3
   (residualisation, OLS or spline matching the full sample) and assigns
   resampled points to the FIXED full-sample bin edges, then stores per-bin
   means + `slope_vsm/slope_gedi`. Per bin: 2.5/97.5 percentile of the means.
6. Recovery = slope_vsm / slope_gedi on the full sample (OLS of each y
   residual on the LVIS x residual). Bootstrap percentile CI from step 5.
   Cross-reference against the grouped-model headline 25% [20, 29]%.

Outputs (under <save_dir>)
--------------------------
    understory_calibration_bins.csv     bin, x_center, n, mean_rh98_lvis,
                                        y_vsm, y_vsm_lo, y_vsm_hi,
                                        y_gedi, y_gedi_lo, y_gedi_hi
    figs/understory_calibration.png     headline figure (PNG)
    figs/understory_calibration.pdf     headline figure (PDF)

Robustness checks (printed to the log, no extra figures)
--------------------------------------------------------
    strict control: additionally partial out lvis_RH98 from resid_vsm /
        resid_gedi (expected to barely move VSM).
    n_bins = 12, 20: re-run the recovery (no re-bootstrap; full-sample slopes
        do not depend on the bin grid so the values match the headline but
        the bin-level picture is reported for the supplement).

  python -m tools.run run=understory_calibration
"""
from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf

from const import FIGURE_SIZES, FONT_SIZES

# Per-instrument (rh25_col, rh98_col) tuples on the raw pair parquets. Mirrors
# `tools.vsm_understory_matched._cols`: LVIS/VSM keep RH casing (uppercase),
# GEDI lowercases it. Order chosen so 'lvis' is the x-axis reference.
_INSTRUMENTS = ('lvis', 'vsm', 'gedi')
_INSTRUMENT_LABEL = {'lvis': 'LVIS', 'vsm': 'VSM', 'gedi': 'GEDI'}
# Paper scheme (matches plot_understory_main).
_INSTRUMENT_COLOR = {'lvis': '#2ca02c', 'vsm': '#d62728', 'gedi': '#1f77b4'}


def _raw_cols(instr: str, rh_under: str, rh_top: str) -> tuple:
    if instr == 'gedi':
        return f'gedi_{rh_under.lower()}', f'gedi_{rh_top.lower()}'
    return f'{instr}_{rh_under}', f'{instr}_{rh_top}'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_pairs(pairs_dir: Path, glob: str, rh_under: str, rh_top: str,
                sensitivity_col: str, say) -> pd.DataFrame:
    """Pool the per-tile triple-sensor pair parquets into a flat frame with
    `rh25_<s>`, `rh98_<s>`, `rh98_lvis_true` (= rh98_lvis, kept under the
    `_lvis_true` alias so the strict-control variant reads cleanly),
    `sensitivity`, `lat`, `lon`, and `tile` (filename stem; the S2 MGRS id
    the bootstrap blocks on). Fails loudly on missing required columns."""
    files = sorted(pairs_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f'No files matching {glob!r} under {pairs_dir}')
    raw_cols = []
    for s in _INSTRUMENTS:
        raw_cols += list(_raw_cols(s, rh_under, rh_top))
    needed = {'geometry', sensitivity_col, *raw_cols}
    available = set(pq.ParquetFile(files[0]).schema.names)
    missing = sorted(needed - available)
    if missing:
        raise KeyError(
            f'Pair parquet {files[0].name} is missing required columns '
            f'{missing}.\nAvailable columns:\n  {sorted(available)}\n'
            f'Align column names before rerunning (do not guess).')
    load = sorted(needed)
    parts = []
    for f in files:
        g = gpd.read_parquet(f, columns=load)
        if g.empty:
            continue
        g = g.to_crs('EPSG:4326')
        d = pd.DataFrame()
        for s in _INSTRUMENTS:
            rh25c, rh98c = _raw_cols(s, rh_under, rh_top)
            d[f'rh25_{s}'] = g[rh25c].to_numpy()
            d[f'rh98_{s}'] = g[rh98c].to_numpy()
        d['rh98_lvis_true'] = d['rh98_lvis'].to_numpy()
        d['sensitivity'] = g[sensitivity_col].to_numpy()
        d['lon'] = g.geometry.x.to_numpy()
        d['lat'] = g.geometry.y.to_numpy()
        d['tile'] = f.stem
        parts.append(d)
    if not parts:
        raise ValueError(f'All {len(files)} pair parquets under {pairs_dir} '
                         f'were empty.')
    df = pd.concat(parts, ignore_index=True)
    say(f'Loaded {len(df):,} footprints from {df["tile"].nunique()} tiles.')
    return df


def _apply_guards(df: pd.DataFrame, sensitivity_threshold: float,
                  say) -> pd.DataFrame:
    """Drop rows with non-finite values in any of the 8 RH columns or `tile`,
    then keep rows with `sensitivity > sensitivity_threshold`. Matches the
    `vsm_understory_matched` Step-5 clean window."""
    cols = []
    for s in _INSTRUMENTS:
        cols += [f'rh25_{s}', f'rh98_{s}']
    clean = df.replace([np.inf, -np.inf], np.nan)
    finite = clean[cols].notna().all(axis=1) & clean['tile'].notna()
    sens_ok = clean['sensitivity'].to_numpy() > sensitivity_threshold
    keep = finite.to_numpy() & sens_ok
    out = clean[keep].reset_index(drop=True)
    say(f'Guards: dropped {int((~finite).sum()):,} non-finite, '
        f'{int(finite.sum() - keep.sum()):,} '
        f'<= sensitivity {sensitivity_threshold:g} -> {len(out):,} kept '
        f'(from {len(df):,}).')
    return out


# ---------------------------------------------------------------------------
# Residualization (OLS or natural cubic spline df=4) + slopes
# ---------------------------------------------------------------------------
def _resid_on(rh25: np.ndarray, rh98: np.ndarray, kind: str,
              extra: np.ndarray = None) -> np.ndarray:
    """Residuals of `rh25 ~ f(rh98) (+ extra)` across the full vector.

    `kind` is 'ols' (linear in rh98) or 'spline' (natural cubic regression
    spline `cr(rh98, df=4)`). If `extra` is given, it is concatenated as an
    extra linear column — used by the strict-control variant to also partial
    out the TRUE top height (lvis_RH98) from VSM and GEDI residuals.
    """
    d = pd.DataFrame({'y': rh25, 'x': rh98})
    if extra is not None:
        d['z'] = extra
        rhs = ('x' if kind == 'ols' else 'cr(x, df=4)') + ' + z'
    else:
        rhs = 'x' if kind == 'ols' else 'cr(x, df=4)'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = smf.ols(f'y ~ {rhs}', data=d).fit()
    return res.resid.to_numpy()


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of `y ~ a + b * x` on aligned 1-d arrays."""
    A = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[1])


def _orthogonality_log(resid: dict, rh98: dict, tag: str, say) -> bool:
    """Print `corr(resid, rh98)` and `corr(resid, rh98^2)` per instrument
    and return True if the quadratic check fails for ANY instrument
    (|corr| > 0.10)."""
    quadratic_fail = False
    say(f'Orthogonality check ({tag}; corr(resid, rh98) ~ 0 by construction, '
        f'corr(resid, rh98^2) tests nonlinearity):')
    for s in _INSTRUMENTS:
        r1 = float(np.corrcoef(resid[s], rh98[s])[0, 1])
        r2 = float(np.corrcoef(resid[s], rh98[s] ** 2)[0, 1])
        flag = '  <-- nonlinear' if abs(r2) > 0.10 else ''
        if abs(r2) > 0.10:
            quadratic_fail = True
        say(f'  {_INSTRUMENT_LABEL[s]:<4}  corr(resid, rh98)={r1:+.4f}  '
            f'corr(resid, rh98^2)={r2:+.4f}{flag}')
    return quadratic_fail


# ---------------------------------------------------------------------------
# Binning (fixed full-sample edges) + tile-block bootstrap
# ---------------------------------------------------------------------------
def _bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count bin edges over a 1-d vector (NaN-free), preserving uniques
    if there are ties at the quantile boundaries."""
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, q))
    return edges


def _bin_means_fixed(x: np.ndarray, y_vsm: np.ndarray, y_gedi: np.ndarray,
                     edges: np.ndarray) -> tuple:
    """Per fixed bin: x_center (mean x), y_vsm_mean, y_gedi_mean, n. Returns
    1-d arrays aligned to `len(edges) - 1`. NaN where a bin is empty."""
    n_bins = len(edges) - 1
    labels = np.digitize(x, edges[1:-1], right=False)
    x_center = np.full(n_bins, np.nan)
    yv = np.full(n_bins, np.nan)
    yg = np.full(n_bins, np.nan)
    n = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = labels == b
        n[b] = int(m.sum())
        if n[b] > 0:
            x_center[b] = float(x[m].mean())
            yv[b] = float(y_vsm[m].mean())
            yg[b] = float(y_gedi[m].mean())
    return x_center, yv, yg, n


def _block_bootstrap(df: pd.DataFrame, edges: np.ndarray, fit_kind: str,
                     n_boot: int, rng: np.random.Generator, say,
                     strict: bool = False) -> tuple:
    """Tile-block bootstrap. Per iteration: pick N_tiles with replacement,
    re-residualize each instrument on the resampled rows (matching fit_kind
    and strict flag), bin to the FIXED full-sample edges. Stores per-bin
    means + the full-sample slopes/recovery from that iteration. Returns
    (boot_vsm, boot_gedi, boot_recovery) shaped (n_boot, n_bins) / (n_boot,)."""
    n_bins = len(edges) - 1
    unique_tiles, inv = np.unique(df['tile'].to_numpy(), return_inverse=True)
    tile_rows = [np.where(inv == k)[0] for k in range(len(unique_tiles))]
    n_tiles = len(unique_tiles)
    rh25 = {s: df[f'rh25_{s}'].to_numpy() for s in _INSTRUMENTS}
    rh98 = {s: df[f'rh98_{s}'].to_numpy() for s in _INSTRUMENTS}
    rh98_true = df['rh98_lvis_true'].to_numpy()

    boot_vsm = np.full((n_boot, n_bins), np.nan)
    boot_gedi = np.full((n_boot, n_bins), np.nan)
    boot_rec = np.full(n_boot, np.nan)
    say(f'Tile-block bootstrap: n_tiles={n_tiles}, B={n_boot}'
        + (', strict (+ lvis_RH98 control)' if strict else ''))
    progress = max(1, n_boot // 10)
    for i in range(n_boot):
        picks = rng.integers(0, n_tiles, n_tiles)
        idx = np.concatenate([tile_rows[k] for k in picks])
        resid = {}
        z_true = rh98_true[idx]
        for s in _INSTRUMENTS:
            extra = (z_true if (strict and s != 'lvis') else None)
            resid[s] = _resid_on(rh25[s][idx], rh98[s][idx], fit_kind, extra)
        b_vsm = _slope(resid['lvis'], resid['vsm'])
        b_gedi = _slope(resid['lvis'], resid['gedi'])
        boot_rec[i] = (b_vsm / b_gedi if abs(b_gedi) > 1e-12 else np.nan)
        _, yv, yg, _ = _bin_means_fixed(resid['lvis'], resid['vsm'],
                                        resid['gedi'], edges)
        boot_vsm[i] = yv
        boot_gedi[i] = yg
        if (i + 1) % progress == 0:
            say(f'  iter {i + 1}/{n_boot}')
    return boot_vsm, boot_gedi, boot_rec


def _percentile_ci(arr: np.ndarray, axis=None) -> tuple:
    lo = np.nanquantile(arr, 0.025, axis=axis)
    hi = np.nanquantile(arr, 0.975, axis=axis)
    return lo, hi


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _plot_calibration(bin_df: pd.DataFrame, recovery: float, rec_ci: tuple,
                      png_path: Path, pdf_path: Path, dpi: int) -> None:
    """LVIS resid on x; VSM (red, thick) and GEDI (blue) bin means joined by
    lines with bootstrap CI bands. No 1:1 line — VSM is a shrunk prediction
    and 1:1 is not a fair reference; the fair reference is GEDI."""
    x = bin_df['x_center'].to_numpy()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES.get('medium', (6.5, 5.0)))
    ax.axhline(0, color='0.6', lw=0.7, zorder=1)
    ax.axvline(0, color='0.6', lw=0.7, zorder=1)

    # GEDI first so VSM sits on top.
    ax.fill_between(x, bin_df['y_gedi_lo'], bin_df['y_gedi_hi'],
                    color=_INSTRUMENT_COLOR['gedi'], alpha=0.18,
                    linewidth=0, zorder=2)
    ax.plot(x, bin_df['y_gedi'], '-o', color=_INSTRUMENT_COLOR['gedi'],
            lw=1.6, markersize=4.5, zorder=3, label='GEDI')

    ax.fill_between(x, bin_df['y_vsm_lo'], bin_df['y_vsm_hi'],
                    color=_INSTRUMENT_COLOR['vsm'], alpha=0.22,
                    linewidth=0, zorder=2)
    ax.plot(x, bin_df['y_vsm'], '-o', color=_INSTRUMENT_COLOR['vsm'],
            lw=2.6, markersize=5, zorder=4, label='VSM')

    ax.set_xlabel('LVIS understory signal\n'
                  '(RH25 not explained by canopy height, m)',
                  fontsize=FONT_SIZES['label'])
    ax.set_ylabel('VSM / GEDI understory signal\n'
                  '(RH25 not explained by canopy height, m)',
                  fontsize=FONT_SIZES['label'])
    pct = recovery * 100.0
    lo, hi = rec_ci[0] * 100.0, rec_ci[1] * 100.0
    annot = (f'VSM understory response approximately {pct:.0f}% '
             f'[{lo:.0f}, {hi:.0f}] of GEDI')
    ax.text(0.02, 0.98, annot, transform=ax.transAxes,
            fontsize=FONT_SIZES['annot'], ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='0.6', linewidth=0.6, alpha=0.92))
    ax.legend(loc='lower right', fontsize=FONT_SIZES['legend'], frameon=True)
    ax.grid(True, linestyle='--', linewidth=0.5, color='0.85', zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Entrypoint (registered as run=understory_calibration)
# ===========================================================================
def understory_calibration_scatter(
        pairs_dir: str,
        save_dir: str,
        rh_understory: str = 'RH25',
        rh_top: str = 'RH98',
        pairs_glob: str = '*.parquet',
        sensitivity_col: str = 'gedi_sensitivity',
        sensitivity_threshold: float = 0.95,
        n_bins: int = 15,
        n_boot_blocks: int = 1000,
        quadratic_tol: float = 0.10,
        robustness_bins: tuple = (12, 20),
        fig_subdir: str = 'figs',
        dpi: int = 300,
        random_state: int = 0,
        **kwargs) -> None:
    """Continuous understory calibration plot (VSM, GEDI on y; LVIS on x; each
    instrument's RH25 residualized on its OWN RH98).

    Args:
        pairs_dir: dir of per-tile LVIS-GEDI-VSM pair parquets (same source as
            `vsm_understory_matched` / `gedi_penetration_bias`) carrying
            `lvis_RH<NN>` / `gedi_rh<NN>` / `vsm_RH<NN>` + `gedi_sensitivity`
            + geometry. The parquet filename stem is stamped as the S2 MGRS
            `tile`, used as the bootstrap block.
        save_dir: output dir for the bin CSV and the `figs/` subdir holding
            the headline PNG/PDF.
        rh_understory: understory RH percentile (numerator). Default RH25.
        rh_top: top-height RH percentile residualized OUT. Default RH98.
        pairs_glob: glob under `pairs_dir`. Default '*.parquet'.
        sensitivity_col: GEDI sensitivity column. Default 'gedi_sensitivity'.
        sensitivity_threshold: STRICT lower bound (`sensitivity > thresh`) —
            matches the spec wording 'GEDI sensitivity > 0.95'. Default 0.95.
        n_bins: equal-count x-bins built from the full-sample LVIS-residual
            quantiles. Default 15.
        n_boot_blocks: tile-block bootstrap iterations. Default 1000.
        quadratic_tol: switch all three residualizations to a natural cubic
            spline (df=4) when ANY instrument's `|corr(resid, rh98^2)|`
            exceeds this. Default 0.10.
        robustness_bins: extra bin counts to report in the log for the
            supplement (no extra figures). Default (12, 20).
        fig_subdir: subdir under `save_dir` for the PNG/PDF. Default 'figs'.
        dpi: PNG resolution. Default 300.
        random_state: RNG seed.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    fig_dir = save_dir / fig_subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    report: list = []

    def say(line: str = '') -> None:
        print(line)
        report.append(line)

    say('=' * 74)
    say('VSM understory continuous calibration vs LVIS '
        '(GEDI as comparable spaceborne reference)')
    say('=' * 74)
    say(f'Sensitivity guard: gedi_sensitivity > {sensitivity_threshold:g}; '
        f'n_bins={n_bins}; n_boot_blocks={n_boot_blocks}; seed={random_state}.')
    say('Each instrument residualized on its OWN RH98 (OLS, one global fit; '
        'spline fallback only if the quadratic check fails).')
    say('')

    # -- load + guards ---------------------------------------------------
    raw = _load_pairs(pairs_dir, pairs_glob, rh_understory, rh_top,
                      sensitivity_col, say)
    df = _apply_guards(raw, sensitivity_threshold, say)
    if len(df) < n_bins * 10:
        raise ValueError(
            f'Only {len(df)} footprints survived the guards — not enough for '
            f'{n_bins} bins. Lower `sensitivity_threshold` or reduce `n_bins`.')

    # -- residualize OLS, then check quadratic orthogonality -------------
    rh25 = {s: df[f'rh25_{s}'].to_numpy() for s in _INSTRUMENTS}
    rh98 = {s: df[f'rh98_{s}'].to_numpy() for s in _INSTRUMENTS}
    resid = {s: _resid_on(rh25[s], rh98[s], 'ols') for s in _INSTRUMENTS}
    fit_kind = 'ols'
    if _orthogonality_log(resid, rh98, 'OLS', say):
        say('  -> nonlinear in rh98 for at least one instrument; switching '
            'all three to natural cubic spline cr(rh98, df=4).')
        fit_kind = 'spline'
        resid = {s: _resid_on(rh25[s], rh98[s], 'spline')
                 for s in _INSTRUMENTS}
        _orthogonality_log(resid, rh98, 'spline (df=4)', say)
    for s in _INSTRUMENTS:
        df[f'resid_{s}'] = resid[s]
    say(f'  fit family used: {fit_kind}.')

    # -- fixed bin edges + full-sample slopes + recovery -----------------
    edges = _bin_edges(resid['lvis'], n_bins)
    n_eff_bins = len(edges) - 1
    if n_eff_bins < n_bins:
        say(f'  NOTE: bin edges collapsed from {n_bins} to {n_eff_bins} '
            f'(ties in resid_lvis quantiles).')
    x_center, yv, yg, n_per = _bin_means_fixed(
        resid['lvis'], resid['vsm'], resid['gedi'], edges)
    slope_vsm = _slope(resid['lvis'], resid['vsm'])
    slope_gedi = _slope(resid['lvis'], resid['gedi'])
    recovery = (slope_vsm / slope_gedi if abs(slope_gedi) > 1e-12 else np.nan)
    say('')
    say(f'Full-sample slopes (resid_y ~ resid_lvis): VSM={slope_vsm:+.4f}, '
        f'GEDI={slope_gedi:+.4f}; recovery = {recovery:.4f} '
        f'({recovery * 100:.1f}%).')

    # -- mean lvis_RH98 per fixed bin (height-balance check) -------------
    labels = np.digitize(resid['lvis'], edges[1:-1], right=False)
    mean_rh98_lvis = np.full(n_eff_bins, np.nan)
    for b in range(n_eff_bins):
        m = labels == b
        if m.any():
            mean_rh98_lvis[b] = float(df.loc[m, 'rh98_lvis'].mean())

    # -- tile-block bootstrap --------------------------------------------
    boot_vsm, boot_gedi, boot_rec = _block_bootstrap(
        df, edges, fit_kind, n_boot_blocks, rng, say, strict=False)
    yv_lo, yv_hi = _percentile_ci(boot_vsm, axis=0)
    yg_lo, yg_hi = _percentile_ci(boot_gedi, axis=0)
    rec_lo, rec_hi = _percentile_ci(boot_rec)
    rec_ci = (float(rec_lo), float(rec_hi))
    say(f'Recovery 95% CI (tile-block bootstrap): '
        f'[{rec_ci[0] * 100:.1f}%, {rec_ci[1] * 100:.1f}%]; '
        f'median={float(np.nanmedian(boot_rec)) * 100:.1f}%.')
    say('Grouped-model headline (from vsm_understory_matched): '
        '25% [20, 29]%; the continuous recovery should sit near it but does '
        'not have to match exactly (different estimators).')

    # -- write bin CSV ---------------------------------------------------
    bin_df = pd.DataFrame({
        'bin': np.arange(n_eff_bins, dtype=int),
        'x_center': x_center,
        'n': n_per,
        'mean_rh98_lvis': mean_rh98_lvis,
        'y_vsm': yv,
        'y_vsm_lo': yv_lo,
        'y_vsm_hi': yv_hi,
        'y_gedi': yg,
        'y_gedi_lo': yg_lo,
        'y_gedi_hi': yg_hi,
    })
    csv_path = save_dir / 'understory_calibration_bins.csv'
    bin_df.to_csv(csv_path, index=False)

    # -- figure ----------------------------------------------------------
    png_path = fig_dir / 'understory_calibration.png'
    pdf_path = fig_dir / 'understory_calibration.pdf'
    _plot_calibration(bin_df, recovery, rec_ci, png_path, pdf_path, dpi)

    # -- robustness: strict control (+ lvis_RH98 partialled out of VSM/GEDI)
    say('')
    say('Robustness — strict control (additionally partial out lvis_RH98 from '
        'resid_vsm and resid_gedi; LVIS unchanged):')
    rh98_true = df['rh98_lvis_true'].to_numpy()
    resid_strict = {'lvis': resid['lvis']}
    for s in ('vsm', 'gedi'):
        resid_strict[s] = _resid_on(rh25[s], rh98[s], fit_kind,
                                    extra=rh98_true)
    sv = _slope(resid_strict['lvis'], resid_strict['vsm'])
    sg = _slope(resid_strict['lvis'], resid_strict['gedi'])
    rec_strict = sv / sg if abs(sg) > 1e-12 else np.nan
    say(f'  slope VSM={sv:+.4f}, slope GEDI={sg:+.4f}, '
        f'recovery_strict={rec_strict * 100:.1f}% '
        f'(grouped strict moved +1.08 -> +1.05; VSM should barely shift).')
    _, _, boot_rec_strict = _block_bootstrap(
        df, edges, fit_kind, n_boot_blocks, rng, say, strict=True)
    rs_lo, rs_hi = _percentile_ci(boot_rec_strict)
    say(f'  strict recovery 95% CI: [{rs_lo * 100:.1f}%, {rs_hi * 100:.1f}%].')

    # -- robustness: bin count 12 and 20 ---------------------------------
    say('')
    say('Robustness — bin count (full-sample slopes are bin-independent; this '
        'is a visual robustness pass on the per-bin curve, no re-bootstrap):')
    for nb in robustness_bins:
        edges_alt = _bin_edges(resid['lvis'], nb)
        x_c, yvc, ygc, nc = _bin_means_fixed(
            resid['lvis'], resid['vsm'], resid['gedi'], edges_alt)
        bin_min, bin_max = int(np.min(nc)), int(np.max(nc))
        say(f'  n_bins={nb}: {len(edges_alt) - 1} effective bins, per-bin N '
            f'[{bin_min}, {bin_max}]; VSM mean across bins '
            f'{float(np.nanmean(yvc)):+.4f}, GEDI {float(np.nanmean(ygc)):+.4f}.')

    say('')
    say('=' * 74)
    say('SUMMARY')
    say('=' * 74)
    say(f'N footprints: {len(df):,}; tiles: {df["tile"].nunique()}; '
        f'fit family: {fit_kind}.')
    say(f'Headline recovery (VSM/GEDI slope ratio): {recovery * 100:.1f}% '
        f'[{rec_ci[0] * 100:.1f}%, {rec_ci[1] * 100:.1f}%].')
    say(f'-> {csv_path}')
    say(f'-> {png_path}')
    say(f'-> {pdf_path}')

    (save_dir / 'understory_calibration_log.md').write_text(
        '\n'.join(report) + '\n')
