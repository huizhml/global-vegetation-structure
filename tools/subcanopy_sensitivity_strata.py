"""Does the LVIS-defined lower-canopy residual separation strengthen as GEDI beam
sensitivity rises?

This repeats the Part-D residual-class separation test of
`subcanopy_matched_residual` but, instead of stratifying by canopy-top height,
stratifies by GEDI beam sensitivity. The class definition is held fixed (it comes
from LVIS, which is independent of GEDI sensitivity), so any change in the GEDI /
VSM separation across sensitivity bins reflects detectability, not a moving truth.

Method
    1. Pool the per-tile LVIS-GEDI-VSM pair parquets (+ `gedi_sensitivity`),
       clean exactly as the other sub-canopy tools (RH98 > rh98_min on all three
       sensors, finite RH), and keep footprints with a valid sensitivity.
    2. Fit f: RH98 -> RH<num> on ALL valid LVIS footprints (RandomForest or a
       spline "gam") — the LVIS height->lower-canopy reference relationship.
       Residual R_s = RH<num>_s - f(RH98_s) per sensor = more/less lower-canopy
       structure than expected for the canopy height.
    3. Define high / low classes ONCE from LVIS residual terciles (global; height
       already controlled by f). The discarded middle is excluded from Δ.
    4. Within each GEDI sensitivity bin (default 0.95-0.97, 0.97-0.98, 0.98-0.99,
       0.99-1.00) compute, for the footprints in that bin:
         ΔGEDI = mean(R_GEDI_high) - mean(R_GEDI_low)
         ΔVSM  = mean(R_VSM_high)  - mean(R_VSM_low)
         ΔVSM / ΔGEDI            (VSM's share of the GEDI-detectable separation)
       plus Cohen's d for each sensor and the residual correlations / R².
    5. Test whether the separation increases with sensitivity, two ways:
         (a) across-bin trend — OLS slope of Δ on the bin midpoint and Spearman
             rank correlation over the bins (monotonicity);
         (b) footprint-level interaction — OLS R ~ sens + group + sens:group on
             the classed footprints; the sens:group coefficient is how fast the
             high-low gap widens per unit sensitivity (with a p-value). This is
             the more powerful test since it uses every footprint, not 4 means.

Reads the same per-tile triple pairs as `subcanopy_matched_residual`
(`lvis_RH<NN>` / `vsm_RH<NN>` uppercase, `gedi_rh<NN>` lowercase) plus the
`gedi_sensitivity` column.

  python -m tools.run run=subcanopy_sensitivity_strata
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from const import FIGURE_SIZES, FONT_SIZES
from tools.subcanopy_classify import _SHAPE_LEVELS, _TOP, _clean, _rh_col
from tools.subcanopy_matched_residual import (_SENSOR_COLOR, _SENSORS,
                                              _fit_height_model,
                                              _retention_block, _safe_ratio)

_SENS_COL = 'gedi_sensitivity'


# ---------------------------------------------------------------------------
# Loading — mirror _load_pairs but also pull gedi_sensitivity
# ---------------------------------------------------------------------------
def _load_pairs_with_sensitivity(pairs_dir: Path, glob: str, num_level: int,
                                 say) -> pd.DataFrame:
    """Pool the per-tile pair parquets, reading every RH column the residual test
    needs plus `gedi_sensitivity`. Fails loudly if a required column is absent."""
    files = sorted(pairs_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f'No files matching {glob!r} under {pairs_dir}')

    sensors = _SENSORS[1:]  # gedi, vsm
    required = [_rh_col('lvis', lv) for lv in sorted({_TOP, num_level})]
    for s in sensors:
        required += [_rh_col(s, lv) for lv in (*_SHAPE_LEVELS, num_level, _TOP)]
    required = sorted(set(required))

    available = set(pq.ParquetFile(files[0]).schema.names)
    missing = sorted((set(required) | {_SENS_COL}) - available)
    if missing:
        raise KeyError(
            f'Pair parquet {files[0].name} is missing required columns '
            f'{missing}.\nAvailable columns:\n  {sorted(available)}\n'
            f'Align column names before rerunning (do not guess).')

    read_cols = sorted(set(required) | {_SENS_COL})
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=read_cols)
        if d.empty:
            continue
        d = d.copy()
        d['tile'] = f.stem
        parts.append(d)
    if not parts:
        raise ValueError(f'All {len(files)} pair parquets under {pairs_dir} '
                         f'were empty.')
    df = pd.concat(parts, ignore_index=True)
    say(f'Loaded {len(df):,} footprints from {len(files)} tile(s); '
        f'{df["tile"].nunique()} unique tiles.')
    return df


# ---------------------------------------------------------------------------
# Residualize + global LVIS classes (same recipe as Part D)
# ---------------------------------------------------------------------------
def _residualize_and_class(df: pd.DataFrame, num_level: int, q: float,
                           model: str, n_estimators: int,
                           rf_min_samples_leaf: int, spline_knots: int,
                           random_state: int, say) -> pd.DataFrame:
    """Fit f:RH98->RH<num> on all valid LVIS, residualize every sensor, and tag
    high / low from global LVIS residual terciles. Returns the working frame."""
    rh98_lvis = df[_rh_col('lvis', _TOP)].to_numpy(float)
    rhnum_lvis = df[_rh_col('lvis', num_level)].to_numpy(float)
    f = _fit_height_model(rh98_lvis, rhnum_lvis, model, n_estimators,
                          rf_min_samples_leaf, spline_knots, random_state)

    work = df.copy()
    for s in _SENSORS:
        top = df[_rh_col(s, _TOP)].to_numpy(float)
        num = df[_rh_col(s, num_level)].to_numpy(float)
        work[f'resid_{s}'] = num - f.predict(top.reshape(-1, 1))

    rl = work['resid_lvis'].to_numpy(float)
    qlo, qhi = np.nanquantile(rl, q), np.nanquantile(rl, 1 - q)
    work['understory_group'] = ''
    work.loc[work['resid_lvis'] >= qhi, 'understory_group'] = 'high'
    work.loc[work['resid_lvis'] <= qlo, 'understory_group'] = 'low'
    say(f'LVIS residual terciles (global): low <= {qlo:.3f} m, high >= {qhi:.3f} m '
        f'(high={int((work["understory_group"]=="high").sum())}, '
        f'low={int((work["understory_group"]=="low").sum())}).')
    return work


# ---------------------------------------------------------------------------
# Per-sensitivity-bin Δ table
# ---------------------------------------------------------------------------
def _sensitivity_bin_table(work: pd.DataFrame, sens_edges: tuple,
                           min_group_n: int, say) -> pd.DataFrame:
    """ΔGEDI, ΔVSM, ΔVSM/ΔGEDI (+ Cohen's d, R²) within each gedi_sensitivity
    bin. Last bin is closed on the right so sensitivity == 1.0 is kept."""
    sens = work[_SENS_COL].to_numpy(float)
    edges = list(sens_edges)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        mb = (sens >= lo) & (sens <= hi) if last else (sens >= lo) & (sens < hi)
        label = f'{lo:.2f}-{hi:.2f}'
        block = _retention_block(work[mb], min_group_n)
        rows.append({
            'sens_bin': label, 'sens_mid': float((lo + hi) / 2),
            'n_footprints': int(mb.sum()),
            'n_high': block['n_high'], 'n_low': block['n_low'],
            'delta_GEDI': block['delta_GEDI'], 'delta_VSM': block['delta_VSM'],
            'd_GEDI': block['d_GEDI'], 'd_VSM': block['d_VSM'],
            'retention_VSM_vs_GEDI': block['retention_VSM_vs_GEDI'],
            'r2_GEDI': block['r2_GEDI'], 'r2_VSM': block['r2_VSM'],
        })
        say(f'  [{label:>10}] N={int(mb.sum()):>6} '
            f'(high={block["n_high"]}, low={block["n_low"]}) | '
            f'ΔGEDI={block["delta_GEDI"]:+.3f} ΔVSM={block["delta_VSM"]:+.3f} '
            f'ΔVSM/ΔGEDI={block["retention_VSM_vs_GEDI"]:.2f}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trend tests — does separation increase with sensitivity?
# ---------------------------------------------------------------------------
def _across_bin_trend(table: pd.DataFrame, col: str) -> dict:
    """OLS slope of Δ on the bin midpoint + Spearman rank corr over the bins.
    Uses only bins where Δ is finite."""
    sub = table[np.isfinite(table[col])]
    x = sub['sens_mid'].to_numpy(float)
    y = sub[col].to_numpy(float)
    out = {'metric': col, 'n_bins': int(len(x)),
           'ols_slope': np.nan, 'ols_p': np.nan,
           'spearman_rho': np.nan, 'spearman_p': np.nan}
    if len(x) >= 3 and np.ptp(x) > 0:
        fit = smf.ols('y ~ x', data=pd.DataFrame({'x': x, 'y': y})).fit()
        out['ols_slope'] = float(fit.params['x'])
        out['ols_p'] = float(fit.pvalues['x'])
        rho, p = spearmanr(x, y)
        out['spearman_rho'] = float(rho)
        out['spearman_p'] = float(p)
    return out


def _footprint_interaction(work: pd.DataFrame, sensor: str) -> dict:
    """Footprint-level test: R_sensor ~ sens + group + sens:group on the classed
    footprints (group = 1 high / 0 low). The sens:group coefficient is how fast
    the high-low residual gap widens per unit GEDI sensitivity; its p-value is the
    significance of 'separation increases with sensitivity'."""
    g = work[work['understory_group'].isin(('high', 'low'))].copy()
    g['R'] = g[f'resid_{sensor}'].to_numpy(float)
    g['group'] = (g['understory_group'] == 'high').astype(float)
    g['sens'] = g[_SENS_COL].to_numpy(float)
    g = g[np.isfinite(g['R']) & np.isfinite(g['sens'])]
    out = {'sensor': sensor.upper(), 'n': int(len(g)),
           'interaction_coef': np.nan, 'interaction_p': np.nan}
    if len(g) >= 20 and g['group'].nunique() == 2 and g['sens'].std() > 0:
        # Centre sensitivity so main effects stay interpretable.
        g['sens_c'] = g['sens'] - g['sens'].mean()
        fit = smf.ols('R ~ sens_c * group', data=g).fit()
        key = 'sens_c:group'
        out['interaction_coef'] = float(fit.params[key])
        out['interaction_p'] = float(fit.pvalues[key])
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _plot_strata(table: pd.DataFrame, num_level: int, save_path: Path) -> None:
    """Two panels: (a) ΔGEDI & ΔVSM vs sensitivity bin, (b) ΔVSM/ΔGEDI."""
    if table.empty:
        return
    bands = list(table['sens_bin'])
    x = np.arange(len(bands))
    fig, (ax, axr) = plt.subplots(1, 2, figsize=FIGURE_SIZES['small'])

    for s, col in (('gedi', 'delta_GEDI'), ('vsm', 'delta_VSM')):
        ax.plot(x, table[col].to_numpy(float), '-o', color=_SENSOR_COLOR[s],
                ms=5, lw=2, label=f'Δ{s.upper()}')
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=FONT_SIZES['ticks'], rotation=30)
    ax.set_xlabel('GEDI sensitivity bin', fontsize=FONT_SIZES['label'])
    ax.set_ylabel(f'Δ = mean(R_high) - mean(R_low), RH{num_level} (m)',
                  fontsize=FONT_SIZES['ticks'])
    ax.set_title('Residual separation vs sensitivity', fontsize=FONT_SIZES['ticks'])
    ax.legend(fontsize=FONT_SIZES['legend'])
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)

    axr.plot(x, table['retention_VSM_vs_GEDI'].to_numpy(float), '-s',
             color=_SENSOR_COLOR['vsm'], ms=5, lw=2)
    axr.axhline(1, color='k', ls='--', lw=1, alpha=0.6)
    axr.set_xticks(x)
    axr.set_xticklabels(bands, fontsize=FONT_SIZES['ticks'], rotation=30)
    axr.set_xlabel('GEDI sensitivity bin', fontsize=FONT_SIZES['label'])
    axr.set_ylabel('ΔVSM / ΔGEDI', fontsize=FONT_SIZES['ticks'])
    axr.set_title('VSM retention vs sensitivity', fontsize=FONT_SIZES['ticks'])
    axr.grid(True, ls='--', alpha=0.4)
    axr.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ===========================================================================
# Main entrypoint (registered as run=subcanopy_sensitivity_strata)
# ===========================================================================
def subcanopy_sensitivity_strata_analysis(
        pairs_dir: str,
        save_dir: str,
        pairs_glob: str = '*.parquet',
        num_level: int = 25,
        sens_bin_edges: tuple = (0.95, 0.97, 0.98, 0.99, 1.00),
        tercile_q: float = 0.30,
        min_group_n: int = 30,
        rh98_min: float = 5.0,
        resid_model: str = 'random_forest',
        n_estimators: int = 300,
        rf_min_samples_leaf: int = 50,
        spline_knots: int = 5,
        random_state: int = 0,
        **kwargs) -> None:
    """Residual-class separation stratified by GEDI beam sensitivity: does the
    LVIS-defined lower-canopy separation that GEDI / VSM recover grow with
    sensitivity?

    Args:
        pairs_dir: dir of per-tile LVIS-GEDI-VSM pair parquets carrying
            `lvis_RH<NN>` / `vsm_RH<NN>` (uppercase), `gedi_rh<NN>` (lowercase)
            and `gedi_sensitivity`.
        save_dir: output dir for the figure, CSVs and conclusion.md.
        pairs_glob: glob under pairs_dir. Default '*.parquet'.
        num_level: lower-canopy RH numerator residualised against RH98. Default 25.
        sens_bin_edges: GEDI sensitivity bin edges; consecutive pairs form the
            bins, last bin closed on the right. Default
            (0.95,0.97,0.98,0.99,1.0) -> 0.95-0.97,0.97-0.98,0.98-0.99,0.99-1.00.
        tercile_q: high/low quantile split of the LVIS residual. Default 0.30
            (top 30% high, bottom 30% low, middle 40% discarded).
        min_group_n: min high (or low) footprints in a bin for its Δ to be
            reported. Default 30.
        rh98_min: keep footprints with ALL sensors' RH98 > this (m). Default 5.
        resid_model: height model — 'random_forest' or 'gam'. Default
            'random_forest'.
        n_estimators: trees for the random_forest height model. Default 300.
        rf_min_samples_leaf: min samples per leaf for the random_forest. Default 50.
        spline_knots: knots for the 'gam' spline basis. Default 5.
        random_state: RNG seed for the residual model.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    report: list = []

    def say(line: str = '') -> None:
        print(line)
        report.append(line)

    num_level = int(num_level)
    sens_bin_edges = tuple(float(e) for e in sens_bin_edges)

    say('=' * 74)
    say('Sub-canopy residual separation stratified by GEDI sensitivity')
    say('=' * 74)
    say(f'Residual: f(RH98)->RH{num_level} ({resid_model}); '
        f'classes = LVIS residual terciles (q={tercile_q:g}); '
        f'rh98_min={rh98_min:g} m; sensitivity bins from {sens_bin_edges}.')

    # ---- Load + clean -----------------------------------------------------
    df = _load_pairs_with_sensitivity(pairs_dir, pairs_glob, num_level, say)
    df = _clean(df, _SENSORS[1:], (num_level,), rh98_min, say)
    n_pre = len(df)
    df = df[np.isfinite(df[_SENS_COL].to_numpy(float))].reset_index(drop=True)
    lo0, hi0 = sens_bin_edges[0], sens_bin_edges[-1]
    in_range = (df[_SENS_COL] >= lo0) & (df[_SENS_COL] <= hi0)
    say(f'Sensitivity filter: dropped {n_pre - int(in_range.sum()):,} footprints '
        f'(non-finite or outside [{lo0:g}, {hi0:g}]) -> {int(in_range.sum()):,} kept.')
    df = df[in_range].reset_index(drop=True)

    # ---- Residualize + global LVIS classes --------------------------------
    work = _residualize_and_class(df, num_level, tercile_q, resid_model,
                                  n_estimators, rf_min_samples_leaf,
                                  spline_knots, random_state, say)

    # ---- Per-sensitivity-bin Δ table --------------------------------------
    say('')
    say('=' * 74)
    say('Δ by GEDI sensitivity bin')
    say('=' * 74)
    table = _sensitivity_bin_table(work, sens_bin_edges, min_group_n, say)
    table.to_csv(save_dir / 'sensitivity_strata_delta.csv', index=False)

    say('')
    say('| Sens bin | N | ΔGEDI | ΔVSM | ΔVSM/ΔGEDI | d_GEDI | d_VSM | R²_GEDI | R²_VSM |')
    say('|---|---|---|---|---|---|---|---|---|')
    for _, r in table.iterrows():
        say(f'| {r["sens_bin"]} | {r["n_footprints"]:,} | {r["delta_GEDI"]:+.3f} '
            f'| {r["delta_VSM"]:+.3f} | {r["retention_VSM_vs_GEDI"]:.2f} '
            f'| {r["d_GEDI"]:+.2f} | {r["d_VSM"]:+.2f} '
            f'| {r["r2_GEDI"]:.3f} | {r["r2_VSM"]:.3f} |')

    # ---- Trend tests ------------------------------------------------------
    say('')
    say('=' * 74)
    say('Does structural separation increase with sensitivity?')
    say('=' * 74)
    trend_rows = [_across_bin_trend(table, c)
                  for c in ('delta_GEDI', 'delta_VSM', 'retention_VSM_vs_GEDI')]
    trend = pd.DataFrame(trend_rows)
    trend.to_csv(save_dir / 'sensitivity_trend_acrossbins.csv', index=False)
    say('')
    say('Across-bin trend (Δ vs bin midpoint):')
    say('| Metric | n bins | OLS slope | OLS p | Spearman ρ | Spearman p |')
    say('|---|---|---|---|---|---|')
    for _, r in trend.iterrows():
        say(f'| {r["metric"]} | {r["n_bins"]} | {r["ols_slope"]:+.3f} '
            f'| {r["ols_p"]:.3f} | {r["spearman_rho"]:+.2f} | {r["spearman_p"]:.3f} |')

    inter_rows = [_footprint_interaction(work, s) for s in ('gedi', 'vsm')]
    inter = pd.DataFrame(inter_rows)
    inter.to_csv(save_dir / 'sensitivity_interaction_footprint.csv', index=False)
    say('')
    say('Footprint-level interaction (R ~ sens * group; sens:group coefficient):')
    say('| Sensor | N | coef (Δ per unit sens) | p |')
    say('|---|---|---|---|')
    for _, r in inter.iterrows():
        say(f'| {r["sensor"]} | {r["n"]:,} | {r["interaction_coef"]:+.3f} '
            f'| {r["interaction_p"]:.3g} |')

    _plot_strata(table, num_level, save_dir / 'fig_sensitivity_strata.png')

    # ---- Interpretation ---------------------------------------------------
    say('')
    say('=' * 74)
    say('INTERPRETATION')
    say('=' * 74)
    g_slope = trend.set_index('metric').loc['delta_GEDI']
    v_slope = trend.set_index('metric').loc['delta_VSM']
    g_int = inter.set_index('sensor').loc['GEDI']
    v_int = inter.set_index('sensor').loc['VSM']

    def verdict(slope_row, int_row, name):
        up_bins = (np.isfinite(slope_row['ols_slope'])
                   and slope_row['ols_slope'] > 0)
        sig = (np.isfinite(int_row['interaction_p'])
               and int_row['interaction_p'] < 0.05
               and int_row['interaction_coef'] > 0)
        if up_bins and sig:
            return (f'{name}: separation INCREASES with sensitivity '
                    f'(slope {slope_row["ols_slope"]:+.3f}/unit; footprint '
                    f'interaction p={int_row["interaction_p"]:.3g}).')
        if up_bins or sig:
            return (f'{name}: weak/mixed evidence of an increase '
                    f'(across-bin slope {slope_row["ols_slope"]:+.3f}, '
                    f'interaction p={int_row["interaction_p"]:.3g}).')
        return (f'{name}: no evidence separation grows with sensitivity '
                f'(slope {slope_row["ols_slope"]:+.3f}, '
                f'interaction p={int_row["interaction_p"]:.3g}).')

    say('  ' + verdict(g_slope, g_int, 'GEDI'))
    say('  ' + verdict(v_slope, v_int, 'VSM'))

    finite_ret = table['retention_VSM_vs_GEDI'].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if not finite_ret.empty:
        say(f'  ΔVSM/ΔGEDI ranges {finite_ret.min():.2f}-{finite_ret.max():.2f} '
            f'across bins (mean {finite_ret.mean():.2f}); a falling ratio with '
            f'sensitivity would mean GEDI gains more from sensitivity than VSM.')

    say('')
    say('Outputs: sensitivity_strata_delta.csv, '
        'sensitivity_trend_acrossbins.csv, '
        'sensitivity_interaction_footprint.csv, '
        'fig_sensitivity_strata.png, conclusion.md')
    (save_dir / 'conclusion.md').write_text('\n'.join(report) + '\n')
    print(f'\n-> {save_dir / "conclusion.md"}')
