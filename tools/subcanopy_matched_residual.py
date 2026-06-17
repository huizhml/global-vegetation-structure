"""Does the LVIS-defined lower-canopy structural contrast survive once canopy-top
height is controlled — and how much of it do GEDI and VSM preserve?

The headline metric is the understory ratio U = RH25 / RH98. The worry is that a
high-vs-low U separation, even inside broad RH98 bins, is still partly driven by
RH98 itself. This tool controls canopy-top height two complementary ways and, for
each, measures how much of the LVIS separation GEDI and VSM retain.

Part A  Are the U classes already height-matched?
    U_LVIS = lvis_RH<num> / lvis_RH98. Within 5 m-wide LVIS RH98 bins (5-10, 10-15,
    ..., 45-50, >50 m) label the top `tercile_q` of U high-understory and the
    bottom `tercile_q` low-understory (middle discarded). Report mean/median RH98
    of each group and dRH98. Small dRH98 (<~1 m) means the original figure was
    already acceptable.

Part B  Strict 1:1 height matching.
    Within each bin, greedy nearest-neighbour match every high footprint to a
    distinct low footprint on lvis_RH98 with a |dRH98| < `match_tol` caliper.
    Keep only matched pairs -> a dataset where canopy-top height is effectively
    identical across the two classes. Report #pairs, median & 95th pct |dRH98|.

Part C  Class separation on the matched pairs.
    For each sensor U = RH25/RH98. Per height band: mean/median U for high & low,
    dU = mean(U_high) - mean(U_low), Cohen's d. Preservation ratio
    R_sensor = dU_sensor / dU_LVIS for GEDI and VSM. Two-panel figure: the six
    sensor x class U lines across bands + a dU panel.

Part D  Residual robustness (the rigorous test) + signal-retention metrics.
    Fit f: RH98 -> RH25 on ALL valid LVIS footprints (RandomForest, or a spline
    "gam") = the LVIS reference relationship. Residual R_s = RH25_s - f(RH98_s)
    per sensor = more/less lower-canopy structure than expected for the canopy
    height. Classes are LVIS-residual terciles (top `tercile_q` high, bottom
    `tercile_q` low). Per height bin AND pooled overall, for each sensor:
      Δ = mean(R_high) - mean(R_low), Cohen's d, and footprint-level
      corr(R_s, R_LVIS) with R² = corr².
    Then the retention metrics:
      (1) preservation  Pres_GEDI = Δ_GEDI/Δ_LVIS, Pres_VSM = Δ_VSM/Δ_LVIS,
      (2) HEADLINE      Retention_VSM_vs_GEDI = Δ_VSM/Δ_GEDI  = fraction of
          GEDI-detectable height-independent RH25 variation preserved by VSM,
      (3) effect-size ratios d_GEDI/d_LVIS, d_VSM/d_LVIS, d_VSM/d_GEDI,
      (4) residual-correlation ratio R²_VSM/R²_GEDI.
    Reported as a by-bin table, a pooled overall table, and mean +/- std across
    bins. Figure: mean residual RH25 vs canopy-height band, six sensor x class
    lines.

Part E  LVIS residual vs LVIS waveform complexity.
    Correlate R_LVIS = RH25 - f(RH98) with the LVIS `COMPLEXITY` metric, first
    over all footprints, then split by the Part-D LVIS-residual classes
    (dense = high R_LVIS, sparse = low R_LVIS). Pearson r and Spearman ρ (each
    with a p-value) per group + a scatter with per-group trend lines. Tests
    whether "more lower-canopy structure than expected for the height" coincides
    with a more complex LVIS waveform.

Reads the same per-tile LVIS-GEDI-VSM triple pairs as `vsm_understory_matched`
(`lvis_RH<NN>` / `vsm_RH<NN>` uppercase, `gedi_rh<NN>` lowercase) plus the
`lvis_COMPLEXITY` column for Part E.

  python -m tools.run run=subcanopy_matched_residual
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from const import FIGURE_SIZES, FONT_SIZES
# Reuse the loading / cleaning / column conventions of the classifier tool so all
# sub-canopy analyses pool and clean footprints identically.
from tools.subcanopy_classify import _TOP, _clean, _load_pairs, _rh_col

# LVIS defines the classes and is the preservation reference for GEDI / VSM.
_SENSORS = ('lvis', 'gedi', 'vsm')
_SENSOR_COLOR = {'lvis': 'C2', 'gedi': 'C0', 'vsm': 'C3'}
_GROUP_STYLE = {'high': '-', 'low': '--'}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _u(df: pd.DataFrame, sensor: str, num_level: int) -> np.ndarray:
    """Understory ratio RH<num>/RH98 for one sensor; NaN where RH98 <= 0."""
    top = df[_rh_col(sensor, _TOP)].to_numpy(float)
    num = df[_rh_col(sensor, num_level)].to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(top > 0, num / top, np.nan)


def _cohen_d(high: np.ndarray, low: np.ndarray) -> float:
    """Cohen's d with pooled SD; NaN if either group is too small/degenerate."""
    n1, n0 = len(high), len(low)
    if n1 < 2 or n0 < 2:
        return np.nan
    s1, s0 = high.std(ddof=1), low.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return float((high.mean() - low.mean()) / pooled)


def _bin_label(lo: float, hi: float) -> str:
    return f'{lo:g}+' if not np.isfinite(hi) else f'{lo:g}-{hi:g}'


def _safe_ratio(a: float, b: float) -> float:
    """a / b, but NaN when b is 0 or either side is non-finite."""
    if np.isfinite(a) and np.isfinite(b) and b != 0:
        return float(a / b)
    return np.nan


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the finite-on-both footprints; NaN if degenerate."""
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return np.nan
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _preservation(rows: pd.DataFrame, col: str) -> pd.Series:
    """delta_sensor / delta_LVIS matched within each height band. LVIS row of the
    same band supplies the denominator (so LVIS preservation == 1)."""
    lvis = rows[rows['sensor'] == 'LVIS'].set_index('height_bin')[col]

    def one(r):
        denom = lvis.get(r['height_bin'], np.nan)
        return float(r[col] / denom) if np.isfinite(denom) and denom != 0 else np.nan

    return rows.apply(one, axis=1)


# ---------------------------------------------------------------------------
# Part A — understory grouping inside RH98 bins + height-match diagnostics
# ---------------------------------------------------------------------------
def _group_understory(df: pd.DataFrame, num_level: int, height_edges: tuple,
                      q: float, min_bin_n: int, say) -> pd.DataFrame:
    """Tag each footprint with `height_bin` and `understory_group` (high/low/'')
    from LVIS U terciles taken WITHIN each LVIS RH98 bin. Bins below min_bin_n are
    dropped. Returns only the high/low footprints (middle 40% discarded)."""
    out = df.copy()
    top = out[_rh_col('lvis', _TOP)].to_numpy(float)
    out['U_lvis'] = _u(out, 'lvis', num_level)
    out['height_bin'] = ''
    out['understory_group'] = ''

    edges = list(height_edges) + [np.inf]
    say(f'Understory grouping: U_LVIS = LVIS_RH{num_level}/LVIS_RH98; '
        f'high = top {q:.0%}, low = bottom {q:.0%} within each RH98 bin.')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (top >= lo) & (top < hi)
        label = _bin_label(lo, hi)
        nbin = int(m.sum())
        if nbin < min_bin_n:
            say(f'  [{label:>7}] N={nbin:>6} -> skipped (< min_bin_n={min_bin_n})')
            continue
        u = out.loc[m, 'U_lvis']
        qlo, qhi = u.quantile(q), u.quantile(1 - q)
        out.loc[m, 'height_bin'] = label
        out.loc[m & (out['U_lvis'] >= qhi), 'understory_group'] = 'high'
        out.loc[m & (out['U_lvis'] <= qlo), 'understory_group'] = 'low'
        say(f'  [{label:>7}] N={nbin:>6} | U q{q:.0%}={qlo:.3f} '
            f'q{1-q:.0%}={qhi:.3f} -> high={int((out.loc[m,"understory_group"]=="high").sum())} '
            f'low={int((out.loc[m,"understory_group"]=="low").sum())}')

    grouped = out[out['understory_group'].isin(('high', 'low'))].reset_index(drop=True)
    say(f'  Grouped footprints: {len(grouped):,} '
        f'({int((grouped["understory_group"]=="high").sum())} high / '
        f'{int((grouped["understory_group"]=="low").sum())} low) across '
        f'{grouped["height_bin"].nunique()} height bins.')
    return grouped


def _height_match_table(grouped: pd.DataFrame) -> pd.DataFrame:
    """Part A deliverable: per height bin, mean/median LVIS RH98 of the high and
    low understory groups and dRH98 = |mean_high - mean_low|."""
    rh98 = grouped[_rh_col('lvis', _TOP)].to_numpy(float)
    is_high = grouped['understory_group'].to_numpy() == 'high'
    rows = []
    for b in dict.fromkeys(grouped['height_bin']):
        mb = grouped['height_bin'].to_numpy() == b
        hi, lo = rh98[mb & is_high], rh98[mb & ~is_high]
        if len(hi) == 0 or len(lo) == 0:
            continue
        rows.append({
            'height_bin': b, 'n_high': len(hi), 'n_low': len(lo),
            'rh98_high_mean': float(hi.mean()), 'rh98_low_mean': float(lo.mean()),
            'rh98_high_median': float(np.median(hi)),
            'rh98_low_median': float(np.median(lo)),
            'delta_rh98_mean': float(abs(hi.mean() - lo.mean())),
            'delta_rh98_median': float(abs(np.median(hi) - np.median(lo))),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part B — strict 1:1 nearest-neighbour height matching within each bin
# ---------------------------------------------------------------------------
def _greedy_match(high_h: np.ndarray, low_h: np.ndarray, tol: float) -> tuple:
    """Greedy 1:1 nearest-neighbour matching on canopy-top height within a caliper
    `tol`. Highs are processed in ascending height; each takes its nearest still
    unused low if |dh| < tol. Returns (high_pos, low_pos) index arrays into the
    passed arrays (one entry per matched pair)."""
    order = np.argsort(high_h, kind='stable')
    low_sort = np.argsort(low_h, kind='stable')
    low_vals = low_h[low_sort]
    used = np.zeros(len(low_h), dtype=bool)
    hi_pos, lo_pos = [], []
    for hp in order:
        h = high_h[hp]
        # Nearest insertion point, then walk outward over unused lows.
        j = int(np.searchsorted(low_vals, h))
        best_k, best_d = -1, np.inf
        left, right = j - 1, j
        # Expand symmetrically; stop once both sides exceed the current best.
        while left >= 0 or right < len(low_vals):
            if right < len(low_vals):
                d = abs(low_vals[right] - h)
                if d < best_d and not used[low_sort[right]]:
                    best_d, best_k = d, right
                if d >= best_d and (best_k != -1):
                    right = len(low_vals)  # no closer unused low to the right
                else:
                    right += 1
            if left >= 0:
                d = abs(low_vals[left] - h)
                if d < best_d and not used[low_sort[left]]:
                    best_d, best_k = d, left
                if d >= best_d and (best_k != -1):
                    left = -1
                else:
                    left -= 1
        if best_k != -1 and best_d < tol:
            used[low_sort[best_k]] = True
            hi_pos.append(hp)
            lo_pos.append(low_sort[best_k])
    return np.asarray(hi_pos, int), np.asarray(lo_pos, int)


def _match_pairs(grouped: pd.DataFrame, tol: float, say) -> tuple:
    """Build the matched-pair dataset by greedy 1:1 RH98 matching within each
    height bin. Returns (matched_df, diagnostics_df). matched_df keeps every
    matched footprint with its `pair_id`, `understory_group` and `height_bin`."""
    rh98 = grouped[_rh_col('lvis', _TOP)].to_numpy(float)
    is_high = grouped['understory_group'].to_numpy() == 'high'
    parts, diag_rows, pair_id = [], [], 0
    say('')
    say(f'Strict 1:1 nearest-neighbour matching on LVIS RH98 (caliper {tol:g} m):')
    for b in dict.fromkeys(grouped['height_bin']):
        mb = grouped['height_bin'].to_numpy() == b
        hi_idx = np.where(mb & is_high)[0]
        lo_idx = np.where(mb & ~is_high)[0]
        if len(hi_idx) == 0 or len(lo_idx) == 0:
            continue
        hp, lp = _greedy_match(rh98[hi_idx], rh98[lo_idx], tol)
        if len(hp) == 0:
            say(f'  [{b:>7}] no pairs within caliper -> skipped')
            continue
        hrows = grouped.iloc[hi_idx[hp]].copy()
        lrows = grouped.iloc[lo_idx[lp]].copy()
        ids = np.arange(pair_id, pair_id + len(hp))
        pair_id += len(hp)
        hrows['pair_id'], lrows['pair_id'] = ids, ids
        adh = np.abs(rh98[hi_idx[hp]] - rh98[lo_idx[lp]])
        parts += [hrows, lrows]
        diag_rows.append({'height_bin': b, 'n_pairs': len(hp),
                          'median_abs_drh98': float(np.median(adh)),
                          'p95_abs_drh98': float(np.percentile(adh, 95))})
        say(f'  [{b:>7}] pairs={len(hp):>5} | median|dRH98|={np.median(adh):.3f} '
            f'p95|dRH98|={np.percentile(adh, 95):.3f} m')
    matched = (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
    diag = pd.DataFrame(diag_rows)
    if not matched.empty:
        adh_all = matched.groupby('pair_id').apply(
            lambda g: abs(g[_rh_col('lvis', _TOP)].iloc[0]
                          - g[_rh_col('lvis', _TOP)].iloc[1]), include_groups=False)
        say(f'  TOTAL pairs={int(len(matched)/2):,} | '
            f'median|dRH98|={adh_all.median():.3f} '
            f'p95|dRH98|={np.percentile(adh_all, 95):.3f} m')
    return matched, diag


# ---------------------------------------------------------------------------
# Part C — class separation on matched pairs
# ---------------------------------------------------------------------------
def _separation_table(matched: pd.DataFrame, num_level: int,
                      min_group_n: int, say) -> pd.DataFrame:
    """Per (height_bin, sensor) high/low mean & median of U=RH<num>/RH98, dU,
    Cohen's d and the preservation ratio dU_sensor/dU_LVIS."""
    is_high = matched['understory_group'].to_numpy() == 'high'
    bins = matched['height_bin'].to_numpy()
    metric = {s: _u(matched, s, num_level) for s in _SENSORS}
    rows = []
    for b in dict.fromkeys(bins):
        mb = bins == b
        for s in _SENSORS:
            hi = metric[s][mb & is_high]
            lo = metric[s][mb & ~is_high]
            hi, lo = hi[np.isfinite(hi)], lo[np.isfinite(lo)]
            if len(hi) < min_group_n or len(lo) < min_group_n:
                say(f'  [{s.upper():<4}] bin {b:>7}: high={len(hi)} low={len(lo)} '
                    f'-> skipped (< min_group_n={min_group_n})')
                continue
            rows.append({
                'sensor': s.upper(), 'height_bin': b,
                'n_high': len(hi), 'n_low': len(lo),
                'mean_high': float(hi.mean()), 'mean_low': float(lo.mean()),
                'median_high': float(np.median(hi)),
                'median_low': float(np.median(lo)),
                'delta_U': float(hi.mean() - lo.mean()),
                'delta_U_median': float(np.median(hi) - np.median(lo)),
                'cohens_d': _cohen_d(hi, lo),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out['preservation'] = _preservation(out, 'delta_U')
        out['preservation_median'] = _preservation(out, 'delta_U_median')
    return out


def _plot_matched(matched: pd.DataFrame, sep: pd.DataFrame, num_level: int,
                  save_path: Path) -> None:
    """Two-panel matched-pair figure: (a) six sensor x class U lines across bands,
    (b) dU per sensor across bands."""
    bands = list(dict.fromkeys(sep['height_bin']))
    x = np.arange(len(bands))
    fig, (ax, axd) = plt.subplots(1, 2, figsize=FIGURE_SIZES['small'])

    for s in _SENSORS:
        sub = sep[sep['sensor'] == s.upper()].set_index('height_bin')
        for grp, col in (('high', 'mean_high'), ('low', 'mean_low')):
            y = [sub.loc[b, col] if b in sub.index else np.nan for b in bands]
            ax.plot(x, y, _GROUP_STYLE[grp], color=_SENSOR_COLOR[s], marker='o',
                    ms=4, lw=1.8, label=f'{s.upper()} {grp}')
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=FONT_SIZES['ticks'])
    ax.set_xlabel('LVIS RH98 band (m)', fontsize=FONT_SIZES['label'])
    ax.set_ylabel(f'U = RH{num_level}/RH98', fontsize=FONT_SIZES['label'])
    ax.set_title('Matched-pair separation', fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'], ncol=3, loc='best')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)

    for s in _SENSORS:
        sub = sep[sep['sensor'] == s.upper()].set_index('height_bin')
        y = [sub.loc[b, 'delta_U'] if b in sub.index else np.nan for b in bands]
        axd.plot(x, y, '-', color=_SENSOR_COLOR[s], marker='s', ms=5, lw=2,
                 label=s.upper())
    axd.axhline(0, color='k', lw=1)
    axd.set_xticks(x)
    axd.set_xticklabels(bands, fontsize=FONT_SIZES['ticks'])
    axd.set_xlabel('LVIS RH98 band (m)', fontsize=FONT_SIZES['label'])
    axd.set_ylabel('dU = mean(U_high) - mean(U_low)', fontsize=FONT_SIZES['label'])
    axd.set_title('Preserved separation dU', fontsize=FONT_SIZES['title'])
    axd.legend(fontsize=FONT_SIZES['legend'])
    axd.grid(True, ls='--', alpha=0.4)
    axd.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part D — residual robustness
# ---------------------------------------------------------------------------
def _fit_height_model(rh98: np.ndarray, rh25: np.ndarray, model: str,
                      n_estimators: int, rf_min_samples_leaf: int,
                      spline_knots: int, random_state: int):
    """Fit f: RH98 -> RH25 on LVIS. 'random_forest' (nonlinear) or 'gam'
    (natural-ish cubic spline basis + linear regression)."""
    X = rh98.reshape(-1, 1)
    if model == 'gam':
        return Pipeline([
            ('spline', SplineTransformer(n_knots=spline_knots, degree=3,
                                         extrapolation='continue')),
            ('lin', LinearRegression()),
        ]).fit(X, rh25)
    if model == 'random_forest':
        return RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=rf_min_samples_leaf,
            random_state=random_state, n_jobs=-1).fit(X, rh25)
    raise ValueError(f'Unknown resid_model {model!r} (use "random_forest"/"gam").')


def _retention_block(work: pd.DataFrame, min_group_n: int) -> dict:
    """All retention metrics for ONE subset (a height bin, or the whole pool).

    Δ and Cohen's d come from the high vs low LVIS residual classes (the
    `understory_group` column; the discarded middle is excluded). The
    footprint-level residual correlations corr(R_sensor, R_LVIS) and R² come from
    ALL rows in the subset (not just the classed ones), since they measure shared
    height-independent variation, not class contrast."""
    grp = work['understory_group'].to_numpy()
    is_high, is_low = grp == 'high', grp == 'low'
    d, delta, mean_hi, mean_lo, n_hi, n_lo = {}, {}, {}, {}, {}, {}
    for s in _SENSORS:
        r = work[f'resid_{s}'].to_numpy(float)
        hi, lo = r[is_high], r[is_low]
        hi, lo = hi[np.isfinite(hi)], lo[np.isfinite(lo)]
        if len(hi) < min_group_n or len(lo) < min_group_n:
            delta[s] = d[s] = mean_hi[s] = mean_lo[s] = np.nan
        else:
            mean_hi[s], mean_lo[s] = float(hi.mean()), float(lo.mean())
            delta[s] = mean_hi[s] - mean_lo[s]
            d[s] = _cohen_d(hi, lo)
        n_hi[s], n_lo[s] = len(hi), len(lo)

    rl = work['resid_lvis'].to_numpy(float)
    corr = {s: _corr(work[f'resid_{s}'].to_numpy(float), rl) for s in _SENSORS}
    r2 = {s: (corr[s] ** 2 if np.isfinite(corr[s]) else np.nan) for s in _SENSORS}

    out = {'n_high': n_hi['lvis'], 'n_low': n_lo['lvis'], 'n_all': len(work)}
    for s in _SENSORS:
        S = s.upper()
        out[f'mean_high_{S}'] = mean_hi[s]
        out[f'mean_low_{S}'] = mean_lo[s]
        out[f'delta_{S}'] = delta[s]
        out[f'd_{S}'] = d[s]
        out[f'corr_{S}'] = corr[s]
        out[f'r2_{S}'] = r2[s]
    # (1) separation preservation relative to LVIS
    out['pres_GEDI'] = _safe_ratio(delta['gedi'], delta['lvis'])
    out['pres_VSM'] = _safe_ratio(delta['vsm'], delta['lvis'])
    # (2) HEADLINE — VSM signal relative to GEDI
    out['retention_VSM_vs_GEDI'] = _safe_ratio(delta['vsm'], delta['gedi'])
    # (3) Cohen's d ratios
    out['dratio_GEDI_LVIS'] = _safe_ratio(d['gedi'], d['lvis'])
    out['dratio_VSM_LVIS'] = _safe_ratio(d['vsm'], d['lvis'])
    out['dratio_VSM_GEDI'] = _safe_ratio(d['vsm'], d['gedi'])
    # (4) residual-correlation R² ratio
    out['r2ratio_VSM_GEDI'] = _safe_ratio(r2['vsm'], r2['gedi'])
    return out


# Ratio columns summarised as mean +/- std across bins, and shown in the tables.
_RATIO_COLS = ('pres_GEDI', 'pres_VSM', 'retention_VSM_vs_GEDI',
               'dratio_GEDI_LVIS', 'dratio_VSM_LVIS', 'dratio_VSM_GEDI',
               'r2ratio_VSM_GEDI')
_SUMMARY_COLS = (('delta_LVIS', 'delta_GEDI', 'delta_VSM',
                  'd_LVIS', 'd_GEDI', 'd_VSM',
                  'corr_GEDI', 'corr_VSM', 'r2_GEDI', 'r2_VSM') + _RATIO_COLS)


# ---------------------------------------------------------------------------
# Part E — LVIS residual vs LVIS waveform complexity
# ---------------------------------------------------------------------------
def _corr_with_p(a: np.ndarray, b: np.ndarray) -> dict:
    """Pearson r and Spearman ρ (each with a p-value) over the finite-on-both
    footprints. NaN everywhere if fewer than 3 usable points or either side is
    constant."""
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    out = {'n': n, 'pearson_r': np.nan, 'pearson_p': np.nan,
           'spearman_rho': np.nan, 'spearman_p': np.nan}
    if n >= 3 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
        pr, pp = pearsonr(a[m], b[m])
        sr, sp = spearmanr(a[m], b[m])
        out.update(pearson_r=float(pr), pearson_p=float(pp),
                   spearman_rho=float(sr), spearman_p=float(sp))
    return out


def _plot_complexity(R: np.ndarray, C: np.ndarray, grp: np.ndarray,
                     num_level: int, save_path: Path) -> None:
    """Scatter of R_LVIS vs LVIS complexity: all footprints (grey) plus the dense
    (high R_LVIS) and sparse (low R_LVIS) classes, each with an OLS trend line."""
    m = np.isfinite(R) & np.isfinite(C)
    if int(m.sum()) < 3:
        return
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
    ax.scatter(C[m], R[m], s=4, c='0.8', alpha=0.3, label='all', rasterized=True)
    for label, key, color in (('dense (high R_LVIS)', 'high', _SENSOR_COLOR['vsm']),
                              ('sparse (low R_LVIS)', 'low', _SENSOR_COLOR['gedi'])):
        gm = m & (grp == key)
        if int(gm.sum()) < 3:
            continue
        ax.scatter(C[gm], R[gm], s=5, color=color, alpha=0.4, label=label,
                   rasterized=True)
        b1, b0 = np.polyfit(C[gm], R[gm], 1)
        xs = np.linspace(C[gm].min(), C[gm].max(), 50)
        ax.plot(xs, b1 * xs + b0, color=color, lw=2)
    b1, b0 = np.polyfit(C[m], R[m], 1)
    xs = np.linspace(C[m].min(), C[m].max(), 50)
    ax.plot(xs, b1 * xs + b0, color='k', lw=2, ls='--', label='all trend')
    ax.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('LVIS complexity', fontsize=FONT_SIZES['label'])
    ax.set_ylabel(f'R_LVIS = RH{num_level} - f(RH98) (m)',
                  fontsize=FONT_SIZES['label'])
    ax.set_title('LVIS residual vs waveform complexity',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'])
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _complexity_analysis(work: pd.DataFrame, complexity_col: str, num_level: int,
                         save_dir: Path, say) -> pd.DataFrame:
    """Correlate the LVIS residual R_LVIS = RH<num> - f(RH98) with LVIS waveform
    complexity, overall and split by the LVIS-residual classes (dense = high,
    sparse = low). Skips cleanly (with a note) if the complexity column is
    absent."""
    say('')
    say('=' * 74)
    say(f'PART E — LVIS residual vs complexity (R_LVIS = RH{num_level} - f(RH98))')
    say('=' * 74)
    if complexity_col not in work.columns:
        say(f'Complexity column {complexity_col!r} not present -> Part E skipped. '
            f'Re-build pairs carrying it, or pass the correct complexity_col.')
        return pd.DataFrame()

    R = work['resid_lvis'].to_numpy(float)
    C = work[complexity_col].to_numpy(float)
    grp = work['understory_group'].to_numpy()

    specs = (('ALL', np.ones(len(work), bool)),
             ('dense (high R_LVIS)', grp == 'high'),
             ('sparse (low R_LVIS)', grp == 'low'))
    rows = []
    for name, mask in specs:
        stat = _corr_with_p(R[mask], C[mask])
        stat['group'] = name
        stat['mean_complexity'] = (float(np.nanmean(C[mask]))
                                   if mask.any() else np.nan)
        stat['mean_R_LVIS'] = (float(np.nanmean(R[mask]))
                               if mask.any() else np.nan)
        rows.append(stat)
    tab = pd.DataFrame(rows)[['group', 'n', 'pearson_r', 'pearson_p',
                              'spearman_rho', 'spearman_p',
                              'mean_complexity', 'mean_R_LVIS']]

    say('')
    say('| Group | N | Pearson r | p | Spearman ρ | p | mean complexity | mean R_LVIS |')
    say('|---|---|---|---|---|---|---|---|')
    for _, r in tab.iterrows():
        say(f'| {r["group"]} | {int(r["n"]):,} | {r["pearson_r"]:+.3f} '
            f'| {r["pearson_p"]:.2g} | {r["spearman_rho"]:+.3f} '
            f'| {r["spearman_p"]:.2g} | {r["mean_complexity"]:.3f} '
            f'| {r["mean_R_LVIS"]:+.3f} |')

    tab.to_csv(save_dir / 'complexity_correlation.csv', index=False)
    _plot_complexity(R, C, grp, num_level,
                     save_dir / f'fig_complexity_corr_RH{num_level}.png')

    ov = tab[tab['group'] == 'ALL'].iloc[0]
    say('')
    say(f'Overall corr(R_LVIS, complexity): Pearson {ov["pearson_r"]:+.3f} '
        f'(p={ov["pearson_p"]:.2g}), Spearman {ov["spearman_rho"]:+.3f}. '
        'A positive r means footprints with more lower-canopy structure than '
        'expected for their canopy height also have higher LVIS waveform '
        'complexity.')
    return tab


def _residual_analysis(df: pd.DataFrame, num_level: int, height_edges: tuple,
                       q: float, model: str, n_estimators: int,
                       rf_min_samples_leaf: int, spline_knots: int,
                       min_group_n: int, random_state: int, save_dir: Path,
                       say, complexity_col: str = 'lvis_COMPLEXITY') -> tuple:
    """Fit f:RH98->RH25 on all valid LVIS, residualize every sensor, class by LVIS
    residual terciles, and report the full retention metric suite (Δ, Cohen's d,
    preservation, VSM-vs-GEDI retention, residual correlations / R²) per height
    bin, pooled overall, and as mean +/- std across bins.

    Returns (overall_row, per_bin_df, across_bins_df)."""
    say('')
    say('=' * 74)
    say(f'PART D — residual robustness (f: RH98 -> RH{num_level}, model={model})')
    say('=' * 74)
    rh98_lvis = df[_rh_col('lvis', _TOP)].to_numpy(float)
    rh25_lvis = df[_rh_col('lvis', num_level)].to_numpy(float)
    f = _fit_height_model(rh98_lvis, rh25_lvis, model, n_estimators,
                          rf_min_samples_leaf, spline_knots, random_state)

    work = df.copy()
    for s in _SENSORS:
        top = df[_rh_col(s, _TOP)].to_numpy(float)
        num = df[_rh_col(s, num_level)].to_numpy(float)
        work[f'resid_{s}'] = num - f.predict(top.reshape(-1, 1))

    # Classes from LVIS residual terciles (global, height already controlled by f).
    rl = work['resid_lvis'].to_numpy(float)
    qlo, qhi = np.nanquantile(rl, q), np.nanquantile(rl, 1 - q)
    work['understory_group'] = ''
    work.loc[work['resid_lvis'] >= qhi, 'understory_group'] = 'high'
    work.loc[work['resid_lvis'] <= qlo, 'understory_group'] = 'low'
    say(f'LVIS residual terciles: low <= {qlo:.3f} m, high >= {qhi:.3f} m '
        f'(high={int((work["understory_group"]=="high").sum())}, '
        f'low={int((work["understory_group"]=="low").sum())}).')

    # ---- Overall (all bins pooled) ----------------------------------------
    # Δ/d use the classed rows; corr/R² use every footprint in the pool.
    overall = {'height_bin': 'ALL', **_retention_block(work, min_group_n)}

    # ---- Per height bin ---------------------------------------------------
    edges = list(height_edges) + [np.inf]
    top_all = work[_rh_col('lvis', _TOP)].to_numpy(float)
    bin_rows = []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        mb = (top_all >= lo_e) & (top_all < hi_e)
        if not mb.any():
            continue
        bin_rows.append({'height_bin': _bin_label(lo_e, hi_e),
                         **_retention_block(work[mb], min_group_n)})
    per_bin = pd.DataFrame(bin_rows)

    # ---- Mean +/- std across bins -----------------------------------------
    summ_rows = []
    for c in _SUMMARY_COLS:
        vals = per_bin[c].to_numpy(float) if c in per_bin else np.array([])
        vals = vals[np.isfinite(vals)]
        summ_rows.append({'metric': c, 'n_bins': int(vals.size),
                          'mean': float(vals.mean()) if vals.size else np.nan,
                          'std': float(vals.std(ddof=1)) if vals.size > 1 else np.nan})
    across_bins = pd.DataFrame(summ_rows)

    # ---- Console / report tables ------------------------------------------
    say('')
    say('Δ = mean(R_high) - mean(R_low); preservation = Δ_sensor / Δ_LVIS; '
        'retention = Δ_VSM / Δ_GEDI; R² = corr(R_sensor, R_LVIS)².')
    say('')
    say('PER HEIGHT BIN')
    say('| Bin | Δ_LVIS | Δ_GEDI | Δ_VSM | Pres_GEDI | Pres_VSM | '
        'Ret(VSM/GEDI) | d_GEDI/d_LVIS | d_VSM/d_LVIS | d_VSM/d_GEDI | '
        'R²_GEDI | R²_VSM | R²_VSM/R²_GEDI |')
    say('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for _, r in per_bin.iterrows():
        say(f'| {r["height_bin"]} | {r["delta_LVIS"]:+.3f} | {r["delta_GEDI"]:+.3f} '
            f'| {r["delta_VSM"]:+.3f} | {r["pres_GEDI"]:.2f} | {r["pres_VSM"]:.2f} '
            f'| {r["retention_VSM_vs_GEDI"]:.2f} | {r["dratio_GEDI_LVIS"]:.2f} '
            f'| {r["dratio_VSM_LVIS"]:.2f} | {r["dratio_VSM_GEDI"]:.2f} '
            f'| {r["r2_GEDI"]:.3f} | {r["r2_VSM"]:.3f} '
            f'| {r["r2ratio_VSM_GEDI"]:.2f} |')

    say('')
    say('OVERALL (all bins pooled)')
    say('| Δ_LVIS | Δ_GEDI | Δ_VSM | Pres_GEDI | Pres_VSM | Ret(VSM/GEDI) | '
        'd_GEDI/d_LVIS | d_VSM/d_LVIS | d_VSM/d_GEDI | R²_GEDI | R²_VSM | '
        'R²_VSM/R²_GEDI |')
    say('|---|---|---|---|---|---|---|---|---|---|---|---|')
    say(f'| {overall["delta_LVIS"]:+.3f} | {overall["delta_GEDI"]:+.3f} '
        f'| {overall["delta_VSM"]:+.3f} | {overall["pres_GEDI"]:.2f} '
        f'| {overall["pres_VSM"]:.2f} | {overall["retention_VSM_vs_GEDI"]:.2f} '
        f'| {overall["dratio_GEDI_LVIS"]:.2f} | {overall["dratio_VSM_LVIS"]:.2f} '
        f'| {overall["dratio_VSM_GEDI"]:.2f} | {overall["r2_GEDI"]:.3f} '
        f'| {overall["r2_VSM"]:.3f} | {overall["r2ratio_VSM_GEDI"]:.2f} |')

    say('')
    say('MEAN ± STD ACROSS BINS')
    say('| Metric | mean ± std (n bins) |')
    say('|---|---|')
    for _, r in across_bins.iterrows():
        if r['metric'] in _RATIO_COLS:
            say(f'| {r["metric"]} | {r["mean"]:.2f} ± {r["std"]:.2f} '
                f'({r["n_bins"]}) |')

    _plot_residual(per_bin, num_level, save_dir / f'fig_residual_separation_RH{num_level}.png')

    # ---- Part E — residual vs LVIS complexity -----------------------------
    _complexity_analysis(work, complexity_col, num_level, save_dir, say)
    return pd.DataFrame([overall]), per_bin, across_bins


def _plot_residual(per_bin: pd.DataFrame, num_level: int, save_path: Path) -> None:
    """Mean residual RH<num> vs canopy-height band; six sensor x class lines."""
    if per_bin.empty:
        return
    bands = list(per_bin['height_bin'])
    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
    for s in _SENSORS:
        S = s.upper()
        for grp, col in (('high', f'mean_high_{S}'), ('low', f'mean_low_{S}')):
            y = per_bin[col].to_numpy(float)
            ax.plot(x, y, _GROUP_STYLE[grp], color=_SENSOR_COLOR[s], marker='o',
                    ms=4, lw=1.8, label=f'{S} {grp}')
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=FONT_SIZES['ticks'])
    ax.set_xlabel('LVIS canopy-height band RH98 (m)', fontsize=FONT_SIZES['label'])
    ax.set_ylabel(f'mean residual RH{num_level} (m)', fontsize=FONT_SIZES['label'])
    ax.set_title('Residual class separation (height controlled)',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'], ncol=3)
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ===========================================================================
# Main entrypoint (registered as run=subcanopy_matched_residual)
# ===========================================================================
def subcanopy_matched_residual_analysis(
        pairs_dir: str,
        save_dir: str,
        pairs_glob: str = '*.parquet',
        num_level: int = 25,
        height_bin_edges: tuple = (15.0, 20.0, 25.0, 30.0, 35.0,
                                   40.0, 45.0, 50.0),
        tercile_q: float = 0.30,
        match_tol: float = 0.5,
        min_bin_n: int = 100,
        min_group_n: int = 30,
        rh98_min: float = 5.0,
        resid_model: str = 'random_forest',
        n_estimators: int = 300,
        rf_min_samples_leaf: int = 50,
        spline_knots: int = 5,
        random_state: int = 0,
        complexity_col: str = 'lvis_COMPLEXITY',
        **kwargs) -> None:
    """How much LVIS-defined lower-canopy structural separation survives canopy-top
    height control, and how much do GEDI and VSM preserve? Runs the height-match
    diagnostic (A), strict 1:1 RH98 matching (B), matched-pair U separation +
    preservation (C) and the residual robustness test (D).

    Args:
        pairs_dir: dir of per-tile LVIS-GEDI-VSM pair parquets carrying
            `lvis_RH<NN>` / `vsm_RH<NN>` (uppercase) and `gedi_rh<NN>` (lowercase).
        save_dir: output dir for figures, CSVs and conclusion.md.
        pairs_glob: glob under pairs_dir. Default '*.parquet'.
        num_level: understory numerator for U = RH<num>/RH98. Default 25.
        height_bin_edges: lower edges of the LVIS RH98 bins; last opens to +inf.
            Default 5 m steps (15,20,...,50) -> 15-20,20-25,...,45-50,50+.
        tercile_q: high/low quantile split of U (A-C) and LVIS residual (D).
            Default 0.30 (top 30% high, bottom 30% low, middle 40% discarded).
        match_tol: Part B caliper on |dRH98| (m) for a valid 1:1 match. Default 0.5.
        min_bin_n: drop an RH98 bin with fewer footprints before grouping.
            Default 100.
        min_group_n: min high (or low) footprints for a per-band/sensor stat.
            Default 30.
        rh98_min: keep footprints with ALL sensors' RH98 > this (m). Default 5.
        resid_model: Part D height model — 'random_forest' or 'gam' (cubic
            spline basis + linear regression). Default 'random_forest'.
        n_estimators: trees for the random_forest height model. Default 300.
        rf_min_samples_leaf: min samples per leaf for the random_forest. Default 50
            (a single RH98 feature overfits at leaf=1).
        spline_knots: knots for the 'gam' spline basis. Default 5.
        random_state: RNG seed for the residual model.
        complexity_col: pass-through LVIS waveform-complexity column for Part E
            (corr of R_LVIS vs complexity, overall + dense/sparse). Default
            'lvis_COMPLEXITY'; Part E is skipped with a note if it is absent.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    report: list = []

    def say(line: str = '') -> None:
        print(line)
        report.append(line)

    num_level = int(num_level)
    height_bin_edges = tuple(float(e) for e in height_bin_edges)
    def_tag = f'RH{num_level}/RH98'
    sensors_xtra = _SENSORS[1:]  # gedi, vsm for the loader's required-column set

    say('=' * 74)
    say('Sub-canopy matched + residual separation — LVIS vs GEDI vs VSM')
    say('=' * 74)
    say(f'U = {def_tag}; height bins from RH98={height_bin_edges}; '
        f'tercile_q={tercile_q:g}; match caliper={match_tol:g} m; '
        f'rh98_min={rh98_min:g} m.')

    df, _extra = _load_pairs(pairs_dir, pairs_glob, sensors_xtra, (num_level,), say,
                             extra_cols=(complexity_col,))
    df = _clean(df, sensors_xtra, (num_level,), rh98_min, say)

    # ---- Part A -----------------------------------------------------------
    say('')
    say('=' * 74)
    say('PART A — are the U classes already height-matched?')
    say('=' * 74)
    grouped = _group_understory(df, num_level, height_bin_edges, tercile_q,
                                min_bin_n, say)
    match_diag = _height_match_table(grouped)
    match_diag.to_csv(save_dir / 'height_match_diagnostics.csv', index=False)
    if not match_diag.empty:
        say('')
        say('| Height bin | RH98 high | RH98 low | dRH98 |')
        say('|---|---|---|---|')
        for _, r in match_diag.iterrows():
            say(f'| {r["height_bin"]} | {r["rh98_high_mean"]:.2f} | '
                f'{r["rh98_low_mean"]:.2f} | {r["delta_rh98_mean"]:.2f} |')
        worst = match_diag['delta_rh98_mean'].max()
        say(f'Max dRH98 across bins = {worst:.2f} m -> '
            + ('groups already well height-matched (<1 m); original figure '
               'acceptable.' if worst < 1.0 else
               'residual height imbalance present; strict matching (Part B) '
               'warranted.'))

    # ---- Part B -----------------------------------------------------------
    say('')
    say('=' * 74)
    say('PART B — strict 1:1 height matching')
    say('=' * 74)
    matched, match_pairs_diag = _match_pairs(grouped, match_tol, say)
    match_pairs_diag.to_csv(save_dir / 'matched_pair_diagnostics.csv', index=False)

    # ---- Part C -----------------------------------------------------------
    say('')
    say('=' * 74)
    say('PART C — class separation on matched pairs')
    say('=' * 74)
    if matched.empty:
        say('No matched pairs -> Part C skipped.')
        sep = pd.DataFrame()
    else:
        sep = _separation_table(matched, num_level, min_group_n, say)
        sep.to_csv(save_dir / 'matched_separation.csv', index=False)
        if not sep.empty:
            say('')
            say('| Height bin | Sensor | dU | Cohen d | preservation |')
            say('|---|---|---|---|---|')
            for _, r in sep.iterrows():
                say(f'| {r["height_bin"]} | {r["sensor"]} | {r["delta_U"]:+.4f} '
                    f'| {r["cohens_d"]:+.2f} | {r["preservation"]:.2f} |')
            say('')
            say('Mean preservation ratio over bands (matched pairs):')
            for s in ('GEDI', 'VSM'):
                sub = sep[sep['sensor'] == s]
                if not sub.empty:
                    say(f'  {s}: dU/dU_LVIS = {sub["preservation"].mean():.2f}  '
                        f'(mean Cohen d = {sub["cohens_d"].mean():+.2f})')
            _plot_matched(matched, sep, num_level,
                          save_dir / 'fig_matched_separation.png')

    # ---- Part D -----------------------------------------------------------
    overall, resid_by_bin, resid_across = _residual_analysis(
        df, num_level, height_bin_edges, tercile_q, resid_model, n_estimators,
        rf_min_samples_leaf, spline_knots, min_group_n, random_state, save_dir, say,
        complexity_col=complexity_col)
    overall.to_csv(save_dir / 'residual_overall.csv', index=False)
    resid_by_bin.to_csv(save_dir / 'residual_by_bin.csv', index=False)
    resid_across.to_csv(save_dir / 'residual_across_bins_summary.csv', index=False)

    # ---- Interpretation ---------------------------------------------------
    say('')
    say('=' * 74)
    say('INTERPRETATION')
    say('=' * 74)
    orow = overall.iloc[0]
    pres_c = (sep.groupby('sensor')['preservation'].mean()
              if not sep.empty else pd.Series(dtype=float))
    pres_d = {'GEDI': orow['pres_GEDI'], 'VSM': orow['pres_VSM']}
    for s in ('GEDI', 'VSM'):
        say(f'  {s}: matched-pair preservation = {pres_c.get(s, np.nan):.2f}, '
            f'residual preservation (Δ/Δ_LVIS) = {pres_d[s]:.2f}.')
    # Headline retention: fraction of GEDI-detectable height-independent RH25
    # variation preserved by VSM (Δ_VSM / Δ_GEDI), pooled + across-bins spread.
    ret = resid_across[resid_across['metric'] == 'retention_VSM_vs_GEDI']
    ret_m = float(ret['mean'].iloc[0]) if not ret.empty else np.nan
    ret_s = float(ret['std'].iloc[0]) if not ret.empty else np.nan
    say('')
    say(f'  HEADLINE Retention_VSM_vs_GEDI = Δ_VSM/Δ_GEDI '
        f'= {orow["retention_VSM_vs_GEDI"]:.2f} (pooled), '
        f'{ret_m:.2f} ± {ret_s:.2f} across bins.')
    say(f'  -> VSM preserves ~{orow["retention_VSM_vs_GEDI"]*100:.0f}% of the '
        f'GEDI-detectable height-independent RH{num_level} separation; '
        f'residual-correlation R²_VSM/R²_GEDI = {orow["r2ratio_VSM_GEDI"]:.2f}.')
    v_d = pres_d['VSM']
    if np.isfinite(v_d) and v_d > 0 and np.isfinite(pres_c.get('VSM', np.nan)) \
            and pres_c.get('VSM', np.nan) > 0:
        say('  => VSM retains positive lower-canopy separation under both tests '
            '(attenuated but not absent).')
    elif np.isfinite(v_d):
        say('  => VSM lower-canopy separation does not survive height control '
            'consistently.')
    if np.isfinite(orow['retention_VSM_vs_GEDI']) \
            and orow['retention_VSM_vs_GEDI'] < 0.5:
        say('  => GEDI retains substantially more lower-canopy information than VSM.')

    say('')
    say('Outputs: height_match_diagnostics.csv, matched_pair_diagnostics.csv, '
        'matched_separation.csv, residual_overall.csv, residual_by_bin.csv, '
        'residual_across_bins_summary.csv, complexity_correlation.csv, '
        'fig_matched_separation.png, fig_residual_separation.png, '
        'fig_complexity_corr.png, conclusion.md')
    (save_dir / 'conclusion.md').write_text('\n'.join(report) + '\n')
    print(f'\n-> {save_dir / "conclusion.md"}')
