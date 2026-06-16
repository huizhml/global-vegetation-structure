"""Falsifiable out-of-fold residual partial-correlation test: does the VSM
encode understory structure *beyond* canopy top height and geographic priors?

This answers the SAR-community critique "an optical model cannot see the
understory". The test is built to be able to FAIL: the control set is
deliberately aggressive (both reference and VSM top height + lat/lon), the
residuals are out-of-fold (so a flexible control model cannot manufacture
spurious residual structure), and the conclusion is only allowed to stand if a
positive residual correlation survives the *entire* control ladder
(linear -> spline -> gradient boosting). If the correlation collapses toward
zero as the control gets stronger, the script says so.

Method (per target metric Y and its VSM counterpart vsm_Y)
----------------------------------------------------------
Control set  C = [lvis_RH98, vsm_RH98, lat, lon]
  - both top heights -> removes the reference height structure AND the VSM's
    own height channel, so any leftover signal is genuinely beyond-height.
  - lat/lon -> removes a geographic / ecological prior (VSM "guessing"
    understory from location rather than seeing it).

For each control model on the ladder:
  1. 5-fold out-of-fold predictions of lvis_Y ~ C   -> residual e_L
  2. 5-fold out-of-fold predictions of vsm_Y  ~ C   -> residual e_V
  3. corr(e_V, e_L): Pearson AND Spearman, each with a 1000x bootstrap
     95% CI and a (two-sided, against zero) bootstrap p-value.

A raw single-control partial r (control = lvis_RH98 only, in-sample OLS with a
quadratic term) is reported alongside as the "before hardening" baseline, so
the figures show how much of the naive partial correlation survives the full
control ladder.

Disattenuation
--------------
Reliability correction needs a reference-noise estimate (e.g. GEDI-vs-LVIS
divergence on the same footprint as a sigma_ref proxy). The LVIS-VSM pair
population carries no second independent reference, so disattenuation is
SKIPPED and the reported correlations are a LOWER BOUND on the true
beyond-height association.

Data
----
Part A (RH profile): the LVIS-VSM RH pairs from
`evaluation.on_lvis.extract_lvis_vsm_pairs` (aggregate_within_pixel=mean),
`eval_lvis_<tile>.parquet`, columns `lvis_RH<NN>` / `vsm_RH<NN>` +
`tile_id/row/col` + geometry (pixel center in the per-tile UTM CRS; reprojected
to EPSG:4326 here for lat/lon).

Part B (FHD): the LVIS-VSM FHD pairs from
`evaluation.on_lvis.extract_lvis_vsm_fhd_pairs`,
`eval_lvis_fhd_<tile>.parquet`, columns `lvis_fhd` / `vsm_fhd` +
`tile_id/row/col`. Inner-joined to the RH pairs on `(tile_id,row,col)` to
attach the `lvis_RH98` / `vsm_RH98` / lat / lon controls. Both products
aggregate LVIS shots to the same VSM 10 m pixel grid, so the join is exact
(this is the same key `attach_lvis_fhd_to_rh_pairs` uses).

  python -m tools.run run=vsm_understory_residual
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from const import FIGURE_SIZES, FONT_SIZES

# Control columns. lat/lon are appended after they are derived from geometry.
_CONTROL_RH = ('lvis_RH{lvl}', 'vsm_RH{lvl}')   # both top heights at control_level
_GEO = ['lat', 'lon']


# ---------------------------------------------------------------------------
# Control-ladder model factories
# ---------------------------------------------------------------------------
def _make_model(kind: str, seed: int, spline_knots: int, spline_degree: int):
    """One regressor per control-ladder rung. Each maps control set C -> y.

    linear : ordinary least squares on the raw controls.
    spline : natural cubic B-splines on every control, then OLS — absorbs
             smooth nonlinear height/geographic dependence.
    gbm    : HistGradientBoostingRegressor — absorbs arbitrary interactions,
             the most aggressive control (hardest test to pass).
    """
    if kind == 'linear':
        return LinearRegression()
    if kind == 'spline':
        return make_pipeline(
            SplineTransformer(n_knots=spline_knots, degree=spline_degree,
                              include_bias=False),
            LinearRegression())
    if kind == 'gbm':
        return HistGradientBoostingRegressor(random_state=seed)
    raise ValueError(f'unknown ladder rung {kind!r}')


def _oof_residual(C: np.ndarray, y: np.ndarray, kind: str, n_splits: int,
                  seed: int, spline_knots: int, spline_degree: int) -> np.ndarray:
    """Out-of-fold residual of y ~ C under the given control model.

    Each fold's prediction comes from a model that never saw that fold, so a
    flexible control cannot overfit the training rows and fabricate residual
    structure that then correlates across the two sensors."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(C):
        m = _make_model(kind, seed, spline_knots, spline_degree)
        m.fit(C[tr], y[tr])
        oof[te] = m.predict(C[te])
    return y - oof


# ---------------------------------------------------------------------------
# Correlation of two residual vectors + bootstrap CI / p
# ---------------------------------------------------------------------------
def _corr_block(eV: np.ndarray, eL: np.ndarray, n_boot: int,
                rng: np.random.Generator, alpha: float = 0.05) -> dict:
    """Pearson & Spearman corr(eV, eL) with percentile bootstrap 95% CI and a
    two-sided bootstrap p-value against zero. Returns NaNs if either residual
    is (near-)constant — e.g. the control-anchor level where lvis_RH98 sits in
    the control set and its residual collapses to ~0."""
    n = len(eV)
    out = {'n': int(n)}
    degenerate = (n < 10 or np.nanstd(eV) < 1e-9 or np.nanstd(eL) < 1e-9)
    for name, fn in (('pearson', lambda a, b: stats.pearsonr(a, b)),
                     ('spearman', lambda a, b: stats.spearmanr(a, b))):
        if degenerate:
            out.update({f'{name}_r': np.nan, f'{name}_ci_lo': np.nan,
                        f'{name}_ci_hi': np.nan, f'{name}_p_boot': np.nan,
                        f'{name}_p_analytic': np.nan})
            continue
        r, p = fn(eV, eL)
        draws = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            a, b = eV[idx], eL[idx]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                draws[i] = np.nan
                continue
            draws[i] = fn(a, b)[0]
        draws = draws[np.isfinite(draws)]
        if draws.size:
            lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
            # Two-sided bootstrap p against r=0, floored at 1/n_boot.
            p_boot = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
            p_boot = max(p_boot, 1.0 / n_boot)
        else:
            lo = hi = p_boot = np.nan
        out.update({f'{name}_r': float(r), f'{name}_ci_lo': float(lo),
                    f'{name}_ci_hi': float(hi), f'{name}_p_boot': float(p_boot),
                    f'{name}_p_analytic': float(p)})
    return out


def _raw_partial_r(y_v: np.ndarray, y_l: np.ndarray, ctrl: np.ndarray) -> float:
    """'Before hardening' baseline: Pearson partial r between vsm_Y and lvis_Y
    controlling ONLY for the single reference top height (in-sample OLS on
    [1, ctrl, ctrl^2]). Matches the existing structure_partial_correlation
    estimator so the figures show the naive-vs-hardened contrast."""
    valid = np.isfinite(y_v) & np.isfinite(y_l) & np.isfinite(ctrl)
    if valid.sum() < 10:
        return np.nan
    y_v, y_l, c = y_v[valid], y_l[valid], ctrl[valid]
    Z = np.column_stack([np.ones(len(c)), c, c ** 2])
    rv = y_v - Z @ np.linalg.lstsq(Z, y_v, rcond=None)[0]
    rl = y_l - Z @ np.linalg.lstsq(Z, y_l, rcond=None)[0]
    if np.std(rv) < 1e-12 or np.std(rl) < 1e-12:
        return np.nan
    return float(stats.pearsonr(rv, rl)[0])


# ---------------------------------------------------------------------------
# Run the full ladder for one (lvis_Y, vsm_Y) target on a finite-row frame
# ---------------------------------------------------------------------------
def _ladder_for_target(df: pd.DataFrame, lvis_col: str, vsm_col: str,
                       control_cols: list, ladder: list, n_splits: int,
                       n_boot: int, seed: int, spline_knots: int,
                       spline_degree: int) -> tuple:
    """Residualize lvis_Y and vsm_Y on C at every ladder rung and correlate.
    Returns (rows, hard_residuals) where rows is one dict per rung and
    hard_residuals is the (e_V, e_L) pair from the hardest rung (last in the
    ladder) for the scatter — or (None, None) if that rung was skipped."""
    needed = list(dict.fromkeys([lvis_col, vsm_col] + control_cols))
    sub = df[needed].replace([np.inf, -np.inf], np.nan).dropna()
    # Control anchor: the target IS a control (e.g. RH98 with RH98 in C). Its
    # residual collapses to ~0 by construction, so the correlation is
    # undefined. Emit NaN blocks rather than fit a degenerate model.
    if lvis_col in control_cols or vsm_col in control_cols:
        rows = []
        for kind in ladder:
            b = {'ladder': kind, 'n': int(len(sub))}
            for name in ('pearson', 'spearman'):
                b.update({f'{name}_r': np.nan, f'{name}_ci_lo': np.nan,
                          f'{name}_ci_hi': np.nan, f'{name}_p_boot': np.nan,
                          f'{name}_p_analytic': np.nan})
            rows.append(b)
        return rows, (None, None)
    rng = np.random.default_rng(seed)
    C = sub[control_cols].to_numpy(dtype=float)
    yl = sub[lvis_col].to_numpy(dtype=float)
    yv = sub[vsm_col].to_numpy(dtype=float)
    hard_rung = ladder[-1] if ladder else None
    rows, hard_resid = [], (None, None)
    for kind in ladder:
        if len(sub) < max(n_splits * 2, 20):
            rows.append({'ladder': kind, 'n': int(len(sub))})
            continue
        eL = _oof_residual(C, yl, kind, n_splits, seed, spline_knots,
                           spline_degree)
        eV = _oof_residual(C, yv, kind, n_splits, seed, spline_knots,
                           spline_degree)
        m = np.isfinite(eL) & np.isfinite(eV)
        block = _corr_block(eV[m], eL[m], n_boot, rng)
        block['ladder'] = kind
        rows.append(block)
        if kind == hard_rung:
            hard_resid = (eV, eL)
    return rows, hard_resid


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _validate(file: Path, needed: set) -> set:
    available = set(pq.ParquetFile(file).schema.names)
    missing = sorted(needed - available)
    if missing:
        raise KeyError(
            f'{file.name} is missing required columns {missing}.\n'
            f'Available columns:\n  {sorted(available)}\n'
            f'Align column names before rerunning (do not guess).')
    return available


def _load_rh_pairs(pairs_dir: Path, glob: str, rh_cols: list) -> pd.DataFrame:
    """Load + pool the RH pairs, deriving lat/lon from each tile's geometry
    (reprojected to EPSG:4326 per file since tiles carry distinct UTM CRSs)."""
    files = sorted(pairs_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f'No files matching {glob!r} under {pairs_dir}')
    needed = {'tile_id', 'row', 'col', 'geometry', *rh_cols}
    _validate(files[0], needed)
    parts = []
    for f in files:
        g = gpd.read_parquet(f, columns=sorted(needed))
        if g.empty:
            continue
        g = g.to_crs('EPSG:4326')
        d = pd.DataFrame({c: g[c].to_numpy() for c in
                          ['tile_id', 'row', 'col', *rh_cols]})
        d['lon'] = g.geometry.x.to_numpy()
        d['lat'] = g.geometry.y.to_numpy()
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    return df


def _load_fhd_pairs(pairs_dir: Path, glob: str) -> pd.DataFrame:
    files = sorted(pairs_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f'No files matching {glob!r} under {pairs_dir}')
    needed = {'tile_id', 'row', 'col', 'lvis_fhd', 'vsm_fhd'}
    _validate(files[0], needed)
    return pd.concat(
        [pd.read_parquet(f, columns=sorted(needed)) for f in files],
        ignore_index=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
_LADDER_STYLE = {'linear': ('C0', 'o', 'linear C'),
                 'spline': ('C4', 's', 'spline C'),
                 'gbm':    ('C3', 'D', 'GBM C')}


def _plot_profile_main(res: pd.DataFrame, raw: pd.DataFrame, hard_rung: str,
                       low_rh_max: int, save_path: Path) -> None:
    """Main Part A figure: hardened (GBM) residual Pearson r vs RH level with a
    bootstrap CI band, the raw single-control partial r overlaid, y=0 line, and
    the low-RH positive lobe highlighted."""
    h = res[(res.ladder == hard_rung) & (res.corr_type == 'pearson')] \
        .sort_values('rh_level')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
    x = h['rh_level'].to_numpy()
    y = h['r'].to_numpy()
    ax.axvspan(x.min() - 1, low_rh_max + 0.5, color='gold', alpha=0.12,
               zorder=0, label=f'low-RH lobe (<={low_rh_max})')
    ax.axhline(0, color='gray', lw=1, zorder=1)
    ax.fill_between(x, h['ci_lo'], h['ci_hi'], color='C3', alpha=0.2, zorder=2)
    ax.plot(x, y, '-D', color='C3', zorder=4,
            label=f'hardened residual r ({hard_rung} C, OOF)')
    rr = raw.sort_values('rh_level')
    ax.plot(rr['rh_level'], rr['raw_partial_r'], '--o', color='gray',
            zorder=3, label='raw partial r (control = lvis_RH98 only)')
    for xi, yi, ni in zip(x, y, h['n']):
        if np.isfinite(yi):
            ax.annotate(f'N={ni}', (xi, yi), textcoords='offset points',
                        xytext=(0, 6), ha='center', fontsize=7, color='dimgray')
    ax.set_xlabel('RH level', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('corr(e_V, e_L)', fontsize=FONT_SIZES['label'])
    ax.set_title('VSM beyond-height understory signal (RH profile)',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'], loc='best')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_profile_ladder(res: pd.DataFrame, save_path: Path) -> None:
    """SAR-proofing figure: Pearson residual r vs RH level, one line per ladder
    rung. The beyond-height claim only holds if the positive lobe is stable as
    the control hardens (lines stay together and above zero at low RH)."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
    ax.axhline(0, color='gray', lw=1, zorder=1)
    for rung, (color, marker, label) in _LADDER_STYLE.items():
        h = res[(res.ladder == rung) & (res.corr_type == 'pearson')] \
            .sort_values('rh_level')
        if h.empty:
            continue
        ax.plot(h['rh_level'], h['r'], marker=marker, ls='-', color=color,
                label=label, zorder=3)
        ax.fill_between(h['rh_level'], h['ci_lo'], h['ci_hi'], color=color,
                        alpha=0.12, zorder=2)
    ax.set_xlabel('RH level', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('corr(e_V, e_L)  (Pearson)', fontsize=FONT_SIZES['label'])
    ax.set_title('Control ladder: residual correlation stability',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'], loc='best')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_residual_scatter(eV: np.ndarray, eL: np.ndarray, r: float,
                           label: str, save_path: Path,
                           rng: np.random.Generator) -> None:
    n = len(eV)
    if n > 20000:
        sel = rng.choice(n, 20000, replace=False)
        eV, eL = eV[sel], eL[sel]
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['square'])
    ax.scatter(eL, eV, s=4, alpha=0.15, color='C3', edgecolors='none')
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlabel('e_L  (LVIS residual)', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('e_V  (VSM residual)', fontsize=FONT_SIZES['label'])
    ax.set_title(f'{label}: Pearson r={r:.3f} (N={n:,})',
                 fontsize=FONT_SIZES['title'])
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_fhd_ladder(res: pd.DataFrame, raw_r: float, save_path: Path) -> None:
    """Part B figure: residual r for FHD at each ladder rung (Pearson +
    Spearman), with the raw single-control partial r as a reference line."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['small'])
    rungs = [r for r in ('linear', 'spline', 'gbm')
             if not res[res.ladder == r].empty]
    x = np.arange(len(rungs))
    w = 0.38
    for k, (ctype, off, color) in enumerate(
            [('pearson', -w / 2, 'C3'), ('spearman', w / 2, 'C0')]):
        rs, los, his = [], [], []
        for rung in rungs:
            row = res[(res.ladder == rung) & (res.corr_type == ctype)].iloc[0]
            rs.append(row['r']); los.append(row['r'] - row['ci_lo'])
            his.append(row['ci_hi'] - row['r'])
        ax.bar(x + off, rs, w, yerr=[los, his], capsize=3, color=color,
               alpha=0.85, label=ctype)
    ax.axhline(0, color='gray', lw=1)
    if np.isfinite(raw_r):
        ax.axhline(raw_r, ls='--', color='dimgray',
                   label=f'raw partial r={raw_r:.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels([_LADDER_STYLE[r][2] for r in rungs])
    ax.set_ylabel('corr(e_V, e_L)', fontsize=FONT_SIZES['label'])
    ax.set_title('FHD beyond-height signal (control ladder)',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'])
    ax.grid(True, axis='y', ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Result flattening
# ---------------------------------------------------------------------------
def _flatten(blocks: list, **stamp) -> list:
    """Expand a list of per-rung corr blocks into one tidy row per
    (rung, corr_type) for the CSV."""
    rows = []
    for b in blocks:
        for ctype in ('pearson', 'spearman'):
            if f'{ctype}_r' not in b:
                continue
            rows.append({**stamp, 'ladder': b['ladder'], 'corr_type': ctype,
                         'n': b['n'], 'r': b[f'{ctype}_r'],
                         'ci_lo': b[f'{ctype}_ci_lo'],
                         'ci_hi': b[f'{ctype}_ci_hi'],
                         'p_boot': b[f'{ctype}_p_boot'],
                         'p_analytic': b[f'{ctype}_p_analytic']})
    return rows


# ===========================================================================
# Main entrypoint  (registered as run=vsm_understory_residual)
# ===========================================================================
def understory_residual_partial_correlation(
        rh_pairs_dir: str,
        fhd_pairs_dir: str,
        save_dir: str,
        rh_test_levels: list = None,
        control_level: int = 98,
        rh_pairs_glob: str = '*.parquet',
        fhd_pairs_glob: str = 'eval_lvis_fhd_*.parquet',
        ladder: list = None,
        low_rh_max: int = 30,
        scatter_level: int = 25,
        n_splits: int = 5,
        n_boot: int = 1000,
        spline_knots: int = 5,
        spline_degree: int = 3,
        fhd_min: float = 0.0,
        fhd_max: float = None,
        run_part_a: bool = True,
        run_part_b: bool = True,
        random_state: int = 0,
        **kwargs) -> None:
    """Out-of-fold residual partial-correlation test of VSM beyond-height
    understory signal, on (A) the lower RH profile and (B) FHD.

    Args:
        rh_pairs_dir: dir of LVIS-VSM RH pair parquets (`eval_lvis_<tile>`)
            carrying `lvis_RH<NN>` / `vsm_RH<NN>` + `tile_id/row/col` +
            geometry. From `extract_lvis_vsm_pairs` (agg mean).
        fhd_pairs_dir: dir of LVIS-VSM FHD pair parquets
            (`eval_lvis_fhd_<tile>`) carrying `lvis_fhd` / `vsm_fhd` +
            `tile_id/row/col`. From `extract_lvis_vsm_fhd_pairs` (agg mean).
        save_dir: output dir for figures, CSVs, conclusion.md.
        rh_test_levels: RH percentiles to test. Default
            [10,15,20,25,30,50,75,98]. The control_level (98) is included as
            a near-zero control anchor — its residual collapses because
            lvis_RH98 is in the control set; this is expected.
        control_level: top-height RH level used in the control set (both
            sensors). Default 98.
        ladder: control-ladder rungs, weakest first. Default
            ['linear','spline','gbm']. The conclusion requires a positive,
            CI-excludes-zero residual correlation across the whole ladder.
        low_rh_max: RH level at/below which the understory "positive lobe" is
            highlighted and summarized. Default 30.
        scatter_level: RH level for the e_V-vs-e_L residual scatter. Default 25.
        n_splits: out-of-fold KFold splits. Default 5.
        n_boot: bootstrap resamples for the CI / p. Default 1000.
        spline_knots, spline_degree: SplineTransformer config for the spline
            ladder rung. Defaults 5 / 3.
        fhd_min, fhd_max: keep FHD pairs with fhd in (fhd_min, fhd_max] on
            BOTH sensors (drops sentinels / nonpositive). fhd_max=None = no
            upper bound.
        run_part_a, run_part_b: toggle each evidence line.
        random_state: RNG seed (folds + bootstrap).
    """
    rh_pairs_dir = Path(rh_pairs_dir).expanduser()
    fhd_pairs_dir = Path(fhd_pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    rh_test_levels = list(rh_test_levels or [10, 15, 20, 25, 30, 50, 75, 98])
    ladder = list(ladder or ['linear', 'spline', 'gbm'])
    hard_rung = ladder[-1]
    report: list = []

    def say(line: str = '') -> None:
        print(line)
        report.append(line)

    control_cols = [_CONTROL_RH[0].format(lvl=control_level),
                    _CONTROL_RH[1].format(lvl=control_level)] + _GEO
    say('=' * 74)
    say('VSM beyond-height understory test (OOF residual partial correlation)')
    say('=' * 74)
    say(f'Control set C = {control_cols}')
    say(f'Control ladder = {ladder} (hardest = {hard_rung})')
    say(f'n_splits={n_splits}  n_boot={n_boot}  seed={random_state}')
    say('Disattenuation: SKIPPED (LVIS-VSM pairs carry no second independent '
        'reference for a sigma_ref proxy) -> reported r is a LOWER BOUND.')

    # ======================================================================
    # PART A — RH profile
    # ======================================================================
    if run_part_a:
        say('')
        say('#' * 74)
        say(f'PART A — RH profile  (levels {rh_test_levels})')
        say('#' * 74)
        rh_cols = sorted({f'lvis_RH{control_level}', f'vsm_RH{control_level}'}
                         | {f'lvis_RH{l}' for l in rh_test_levels}
                         | {f'vsm_RH{l}' for l in rh_test_levels})
        df = _load_rh_pairs(rh_pairs_dir, rh_pairs_glob, rh_cols)
        say(f'Loaded {len(df):,} RH pairs from {rh_pairs_dir.name}')

        a_rows, raw_rows = [], []
        scatter_resid = None
        for lvl in rh_test_levels:
            lvis_col, vsm_col = f'lvis_RH{lvl}', f'vsm_RH{lvl}'
            blocks, resid = _ladder_for_target(
                df, lvis_col, vsm_col, control_cols, ladder, n_splits,
                n_boot, random_state, spline_knots, spline_degree)
            a_rows += _flatten(blocks, part='A', metric=f'RH{lvl}',
                               rh_level=lvl)
            # Raw single-control (lvis_RH98 only) partial r baseline. Undefined
            # at the control anchor (target is the control itself).
            ctrl0 = control_cols[0]
            if lvis_col == ctrl0 or vsm_col == ctrl0:
                raw_r = np.nan
            else:
                sub = df[[lvis_col, vsm_col, ctrl0]] \
                    .replace([np.inf, -np.inf], np.nan).dropna()
                raw_r = _raw_partial_r(sub[vsm_col].to_numpy(),
                                       sub[lvis_col].to_numpy(),
                                       sub[ctrl0].to_numpy())
            raw_rows.append({'rh_level': lvl, 'raw_partial_r': raw_r})
            hard = next((b for b in blocks if b.get('ladder') == hard_rung
                         and 'pearson_r' in b), None)
            hp = hard['pearson_r'] if hard else np.nan
            say(f'  RH{lvl:<3d} N={hard["n"] if hard else 0:>6} | '
                f'raw partial r={raw_r:+.3f} | {hard_rung} residual '
                f'r={hp:+.3f}' + (f' [{hard["pearson_ci_lo"]:+.3f},'
                f'{hard["pearson_ci_hi"]:+.3f}] p_boot={hard["pearson_p_boot"]:.3g}'
                if hard and np.isfinite(hp) else ' (degenerate/anchor)'))
            if lvl == scatter_level and resid[0] is not None:
                scatter_resid = resid

        a_res = pd.DataFrame(a_rows)
        raw_df = pd.DataFrame(raw_rows)
        a_csv = save_dir / 'partA_rh_residual_corr.csv'
        a_res.merge(raw_df, on='rh_level', how='left').to_csv(a_csv, index=False)
        say(f'-> {a_csv.name}')

        _plot_profile_main(a_res, raw_df, hard_rung, low_rh_max,
                           save_dir / 'partA_main_profile.png')
        _plot_profile_ladder(a_res, save_dir / 'partA_ladder.png')
        say('-> partA_main_profile.png, partA_ladder.png')
        if scatter_resid is not None:
            eV, eL = scatter_resid
            m = np.isfinite(eV) & np.isfinite(eL)
            r = stats.pearsonr(eV[m], eL[m])[0]
            _plot_residual_scatter(
                eV[m], eL[m], r, f'RH{scatter_level} ({hard_rung} C, OOF)',
                save_dir / f'partA_scatter_RH{scatter_level}.png',
                np.random.default_rng(random_state))
            say(f'-> partA_scatter_RH{scatter_level}.png')

    # ======================================================================
    # PART B — FHD (independent second evidence)
    # ======================================================================
    b_res = None
    fhd_raw_r = np.nan
    if run_part_b:
        say('')
        say('#' * 74)
        say('PART B — FHD')
        say('#' * 74)
        rh_ctrl_cols = [f'lvis_RH{control_level}', f'vsm_RH{control_level}']
        rh_for_join = _load_rh_pairs(rh_pairs_dir, rh_pairs_glob, rh_ctrl_cols)
        fhd = _load_fhd_pairs(fhd_pairs_dir, fhd_pairs_glob)
        say(f'RH pairs (for controls): {len(rh_for_join):,} | '
            f'FHD pairs: {len(fhd):,}')
        merged = fhd.merge(
            rh_for_join[['tile_id', 'row', 'col', *rh_ctrl_cols, 'lat', 'lon']],
            on=['tile_id', 'row', 'col'], how='inner')
        say(f'inner join on (tile_id,row,col) -> N={len(merged):,}')
        # Quality filter: finite + in-range FHD on both sensors.
        before = len(merged)
        merged = merged.replace([np.inf, -np.inf], np.nan)
        keep = (merged['lvis_fhd'] > fhd_min) & (merged['vsm_fhd'] > fhd_min)
        if fhd_max is not None:
            keep &= (merged['lvis_fhd'] <= fhd_max) & (merged['vsm_fhd'] <= fhd_max)
        merged = merged[keep].dropna(
            subset=['lvis_fhd', 'vsm_fhd', *control_cols])
        say(f'after FHD quality filter (fhd>{fhd_min}'
            f'{f", <={fhd_max}" if fhd_max else ""}, finite controls): '
            f'N={len(merged):,} (dropped {before - len(merged):,})')

        if len(merged) < max(n_splits * 2, 20):
            say('Too few FHD pairs after filtering — skipping Part B test.')
        else:
            blocks, resid = _ladder_for_target(
                merged, 'lvis_fhd', 'vsm_fhd', control_cols, ladder, n_splits,
                n_boot, random_state, spline_knots, spline_degree)
            b_res = pd.DataFrame(_flatten(blocks, part='B', metric='fhd',
                                          rh_level=np.nan))
            fhd_raw_r = _raw_partial_r(
                merged['vsm_fhd'].to_numpy(), merged['lvis_fhd'].to_numpy(),
                merged[control_cols[0]].to_numpy())
            b_csv = save_dir / 'partB_fhd_residual_corr.csv'
            b_out = b_res.copy()
            b_out['raw_partial_r'] = fhd_raw_r
            b_out.to_csv(b_csv, index=False)
            say(f'raw partial r (control=lvis_RH{control_level} only)='
                f'{fhd_raw_r:+.3f}')
            for rung in ladder:
                row = b_res[(b_res.ladder == rung) &
                            (b_res.corr_type == 'pearson')]
                if not row.empty:
                    r = row.iloc[0]
                    say(f'  {rung:<7} pearson r={r["r"]:+.3f} '
                        f'[{r["ci_lo"]:+.3f},{r["ci_hi"]:+.3f}] '
                        f'p_boot={r["p_boot"]:.3g}  N={int(r["n"])}')
            say(f'-> {b_csv.name}')
            _plot_fhd_ladder(b_res, fhd_raw_r, save_dir / 'partB_fhd_ladder.png')
            if resid[0] is not None:
                eV, eL = resid
                m = np.isfinite(eV) & np.isfinite(eL)
                r = stats.pearsonr(eV[m], eL[m])[0]
                _plot_residual_scatter(
                    eV[m], eL[m], r, f'FHD ({hard_rung} C, OOF)',
                    save_dir / 'partB_scatter_fhd.png',
                    np.random.default_rng(random_state))
            say('-> partB_fhd_ladder.png, partB_scatter_fhd.png')

    # ======================================================================
    # CONCLUSION (data-driven, falsifiable)
    # ======================================================================
    say('')
    say('=' * 74)
    say('CONCLUSION')
    say('=' * 74)

    def _ladder_stable_positive(res: pd.DataFrame, levels=None) -> tuple:
        """A signal is 'stable & positive' if, across EVERY ladder rung, the
        Pearson r is > 0 and its bootstrap 95% CI excludes zero. For Part A we
        require this at the low-RH levels. Returns (ok, summary_r)."""
        ok = True
        rs = []
        for rung in ladder:
            sel = res[(res.ladder == rung) & (res.corr_type == 'pearson')]
            if levels is not None:
                sel = sel[sel.rh_level.isin(levels)]
            sel = sel[np.isfinite(sel['r'])]
            if sel.empty:
                ok = False
                continue
            ok &= bool(((sel['r'] > 0) & (sel['ci_lo'] > 0)).all())
            rs.append(sel['r'].mean())
        return ok, (float(np.mean(rs)) if rs else np.nan)

    if run_part_a:
        low = [l for l in rh_test_levels if l <= low_rh_max]
        ok_a, r_a = _ladder_stable_positive(a_res, levels=low)
        gbm_low = a_res[(a_res.ladder == hard_rung) &
                        (a_res.corr_type == 'pearson') &
                        (a_res.rh_level.isin(low))]
        say(f'[A] low-RH ({low}) {hard_rung} mean Pearson r='
            f'{gbm_low["r"].mean():+.3f}; stable & positive across ladder: '
            f'{ok_a}.')

    if run_part_b and b_res is not None:
        ok_b, r_b = _ladder_stable_positive(b_res)
        gbm_b = b_res[(b_res.ladder == hard_rung) &
                      (b_res.corr_type == 'pearson')]
        rb = gbm_b['r'].iloc[0] if not gbm_b.empty else np.nan
        say(f'[B] FHD {hard_rung} Pearson r={rb:+.3f}; stable & positive '
            f'across ladder: {ok_b}.')
    else:
        ok_b = False

    a_pass = run_part_a and ok_a
    b_pass = run_part_b and (b_res is not None) and ok_b
    say('')
    if a_pass or b_pass:
        ev = []
        if a_pass:
            ev.append(f'the lower RH profile (mean low-RH r={r_a:+.3f})')
        if b_pass:
            ev.append(f'FHD (r={r_b:+.3f})')
        say('VERDICT: VSM captures a weak but significant component of '
            'understory structure beyond canopy height and geographic priors. '
            'The positive out-of-fold residual correlation survives the full '
            'control ladder (linear -> spline -> GBM) on: ' + '; '.join(ev) +
            '. Because disattenuation was not applied, these are LOWER BOUNDS '
            'on the true beyond-height association.')
    else:
        say('VERDICT: No robust beyond-height evidence. The residual '
            'correlation does not stay positive with a zero-excluding CI '
            'across the full control ladder (it weakens/collapses as the '
            'control hardens), so the data do not support a claim that VSM '
            'sees understory structure beyond canopy height and geographic '
            'priors.')

    (save_dir / 'conclusion.md').write_text('\n'.join(report) + '\n')
    print(f'\n-> {save_dir / "conclusion.md"}')
