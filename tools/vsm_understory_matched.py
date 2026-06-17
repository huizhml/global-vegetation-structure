"""Matched-grouping test: does the VSM reflect understory structure at *matched*
canopy top height? Headline test is on the understory NUMERATOR (rh25) with top
height controlled, not on the rh25:rh98 ratio (the ratio mixes a denominator
artifact into the contrast).

This answers the optical-remote-sensing critique "an optical model cannot see
the understory". It is deliberately built to be able to FAIL: the entire
footprint population is used (no example cherry-picking), the grouping variable
is the *independent reference* (LVIS), the comparison is anchored to a lidar
ceiling (GEDI), and every threshold is fixed up front so the VSM result cannot
feed back into the design.

Grouping (unchanged)
--------------------
Footprints are binned into narrow `band_width` m bands of LVIS top height
H_L = lvis_RH98, and inside each band split on the LVIS understory ratio
U_L = lvis_RH25/lvis_RH98 into terciles: hiU = top 1/3, loU = bottom 1/3, drop
the middle. NOTE the direction: a LARGE RH25/RH98 means energy sits high in the
profile = SPARSE understory (top-heavy); a SMALL ratio = DENSE understory
(bottom-heavy). So hiU = sparse-understory canopies, loU = dense-understory.
Grouping on the tested sensor would be circular, so it is always LVIS.

Headline test — numerator with top height controlled (change A)
---------------------------------------------------------------
Per sensor a mixed model for the understory numerator rh25, with the sensor's
OWN top height as a covariate so the contrast is at matched height:
    vsm_RH25  ~ grp + vsm_RH98  + <spatial> + (1|band)
    gedi_RH25 ~ grp + gedi_RH98 + <spatial> + (1|band)     (lidar anchor/ceiling)
    lvis_RH25 ~ grp + lvis_RH98 + <spatial> + (1|band)     (grouping-strength cap)
`grp` coefficient = how many METRES higher rh25 the high-understory group has at
the same top height. coef_VSM / coef_GEDI = the fraction of the
top-height-controlled, lidar-detectable understory contrast the optical model
recovers (the new headline; the ratio version is kept only for the artifact
decomposition).

Strict variant (change B) — the HEADLINE
-----------------------------------------
The headline model also controls the TRUE top height (lvis_RH98), for VSM and
GEDI, so neither the sensor's own height channel NOR the true height can leak
into the contrast:
    vsm_RH25  ~ grp + vsm_RH98  + lvis_RH98 + <spatial> + (1|band)
    gedi_RH25 ~ grp + gedi_RH98 + lvis_RH98 + <spatial> + (1|band)
(LVIS own RH98 == the true height, so LVIS strict == LVIS base.) The headline
coef_VSM/coef_GEDI is computed strict-vs-strict. The base (own-RH98-only) models
are still reported for the control ladder. Controlling the OWN RH98 — not a
common lvis_RH98 alone — is deliberate: it removes the sensor's height channel,
so a VSM that merely shifts its top-height estimate with understory cannot be
mistaken for genuine understory signal.

Ratio-artifact decomposition (change C)
---------------------------------------
Per band and overall, ΔU_V (the old ratio separation) is split into
    numerator term   =  ΔRH25 / mean(RH98)
    denominator term = -mean(U) * ΔRH98 / mean(RH98)
(first order; the two should sum to ~ΔU_V) so a reviewer can see directly how
much of the ratio separation is real numerator signal vs a denominator artifact.

Ecology control (change D)
--------------------------
<spatial> is biome when available, geography otherwise:
  - biome present (`biome_col` in the parquet): models use lat+lon, and a second
    `+ C(biome)` fixed-factor version is reported (~unchanged -> not an
    ecological prior). A within-biome refit is run for each biome with N>=300.
  - biome absent (fallback): lat+lon are REPLACED by a ~`spatial_cell_deg`-degree
    grid cell as a fixed categorical absorber `C(cell)`, soaking up regional /
    ecological structure nonlinearly.

Data: per-tile LVIS-GEDI-VSM pair parquets from
`evaluation/on_lvis.extract_vsm_on_pair_locations`
(`Gabon2016_vs_GEDI2020_stable_forest_with_vsm2020/<tile>.parquet`), columns
`lvis_RH<NN>` (uppercase) / `gedi_rh<NN>` (lowercase) / `vsm_RH<NN>` (uppercase)
+ optional `gedi_BIOME` + geometry (GEDI shot centre in per-tile UTM, reprojected
to EPSG:4326 here for lat/lon). Same source the GEDI penetration-bias diagnostic
uses.

  python -m tools.run run=vsm_understory_matched
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

# Three sensors keyed by the label used in figures/tables and the (numerator,
# denominator) column-name templates. GEDI uses lowercase dense `rh<NN>`; LVIS
# and VSM use uppercase sparse `RH<NN>` (see on_lvis._sensor_cols).
_SENSORS = ('LVIS', 'VSM', 'GEDI')
_SENSOR_COLOR = {'LVIS': 'C2', 'VSM': 'C3', 'GEDI': 'C0'}


def _cols(sensor: str, rh_under: str, rh_top: str) -> tuple:
    """(numerator_col, denominator_col) for one sensor. LVIS/VSM keep the RH
    casing as given; GEDI lowercases it."""
    if sensor == 'GEDI':
        return f'gedi_{rh_under.lower()}', f'gedi_{rh_top.lower()}'
    pre = sensor.lower()
    return f'{pre}_{rh_under}', f'{pre}_{rh_top}'


def _z(a: np.ndarray) -> np.ndarray:
    """z-score (covariate scaling for the optimiser; does not affect the grp
    coefficient, which is a 0/1 dummy reported in raw rh25 metres)."""
    a = np.asarray(a, dtype=float)
    s = a.std(ddof=0)
    return (a - a.mean()) / (s if s > 1e-12 else 1.0)


# ---------------------------------------------------------------------------
# Data loading (pool tiles, derive lat/lon from geometry, carry biome if any)
# ---------------------------------------------------------------------------
def _load_pairs(pairs_dir: Path, glob: str, rh_under: str, rh_top: str,
                biome_col: str, sensitivity_col: str) -> tuple:
    """Pool the per-tile triple-sensor pair parquets into a flat frame with the
    six RH columns + lat/lon (geometry reprojected to EPSG:4326 per file) and,
    if present, `biome` and `sensitivity` columns. Each row is also stamped
    with `tile` — the parquet's filename stem (a Sentinel-2 MGRS tile id) —
    which the cluster bootstrap uses as the resampling block. Returns
    (df, has_biome, has_sens). Fails loudly on any missing REQUIRED column;
    biome (change-D fallback) and sensitivity (the stratified step) are
    optional."""
    files = sorted(pairs_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f'No files matching {glob!r} under {pairs_dir}')
    rh_cols = []
    for s in _SENSORS:
        rh_cols += list(_cols(s, rh_under, rh_top))
    needed = {'geometry', *rh_cols}
    available = set(pq.ParquetFile(files[0]).schema.names)
    missing = sorted(needed - available)
    if missing:
        raise KeyError(
            f'Pair parquet {files[0].name} is missing required columns '
            f'{missing}.\nAvailable columns:\n  {sorted(available)}\n'
            f'Align column names before rerunning (do not guess).')
    has_biome = biome_col in available
    has_sens = sensitivity_col in available
    load = sorted(needed | ({biome_col} if has_biome else set())
                  | ({sensitivity_col} if has_sens else set()))
    parts = []
    for f in files:
        g = gpd.read_parquet(f, columns=load)
        if g.empty:
            continue
        g = g.to_crs('EPSG:4326')
        d = pd.DataFrame({c: g[c].to_numpy() for c in rh_cols})
        if has_biome:
            d['biome'] = g[biome_col].to_numpy()
        if has_sens:
            d['sensitivity'] = g[sensitivity_col].to_numpy()
        d['lon'] = g.geometry.x.to_numpy()
        d['lat'] = g.geometry.y.to_numpy()
        # S2 MGRS tile id, lifted from the filename so the cluster bootstrap
        # resamples whole tiles instead of iid footprints.
        d['tile'] = f.stem
        parts.append(d)
    if not parts:
        raise ValueError(f'All {len(files)} pair parquets under {pairs_dir} '
                         f'were empty.')
    return pd.concat(parts, ignore_index=True), has_biome, has_sens


# ---------------------------------------------------------------------------
# Step 1: clip + filter + understory ratio (numerator/denominator both carried)
# ---------------------------------------------------------------------------
def _build_indices(df: pd.DataFrame, rh_under: str, rh_top: str,
                   rh98_min: float, winsor: tuple, cell_deg: float,
                   has_biome: bool, has_sens: bool, say) -> pd.DataFrame:
    """Clip rh25<0->0, keep footprints with all three rh98>rh98_min, carry each
    sensor's numerator (rh25) and top height (rh98), and build the winsorized U
    ratio (for the artifact decomposition + legacy figure). H_L = LVIS top
    height; also assign a ~cell_deg-degree spatial grid cell + carry biome."""
    num = {s: _cols(s, rh_under, rh_top)[0] for s in _SENSORS}
    den = {s: _cols(s, rh_under, rh_top)[1] for s in _SENSORS}
    work = df.replace([np.inf, -np.inf], np.nan).copy()
    for s in _SENSORS:                       # clip negative understory RH to 0
        work[num[s]] = work[num[s]].clip(lower=0)

    n0 = len(work)
    finite = work[list(num.values()) + list(den.values())].notna().all(axis=1)
    keep = finite.copy()
    for s in _SENSORS:
        keep &= work[den[s]] > rh98_min
    out = work[keep].copy()
    say(f'Step 1: {n0:,} footprints loaded; dropped '
        f'{int((~keep).sum()):,} (non-finite or some {rh_top} <= {rh98_min:g} '
        f'm) -> {len(out):,} kept.')
    fin = work[finite]
    for s in _SENSORS:
        n_low = int((fin[den[s]] <= rh98_min).sum())
        say(f'    {s} {rh_top} <= {rh98_min:g} m: {n_low:,} '
            f'({n_low / max(len(fin), 1):.1%} of finite)')

    res = pd.DataFrame({'lat': out['lat'].to_numpy(),
                        'lon': out['lon'].to_numpy()})
    if 'tile' in out.columns:
        res['tile'] = out['tile'].to_numpy()
    res['H_L'] = out[den['LVIS']].to_numpy()
    lo_q, hi_q = winsor
    for s in _SENSORS:
        res[f'num_{s}'] = out[num[s]].to_numpy()
        res[f'top_{s}'] = out[den[s]].to_numpy()
        u = out[num[s]].to_numpy() / out[den[s]].to_numpy()
        ql, qh = np.quantile(u, [lo_q, hi_q])
        res[f'U_{s}'] = np.clip(u, ql, qh)
    # ~cell_deg-degree grid cell for the change-D spatial fallback.
    res['cell'] = (np.floor(res['lat'] / cell_deg).astype(int).astype(str)
                   + '_' + np.floor(res['lon'] / cell_deg).astype(int).astype(str))
    if has_biome:
        res['biome'] = out['biome'].to_numpy()
    if has_sens:
        res['sensitivity'] = out['sensitivity'].to_numpy()
    say(f'    winsorized each U at {lo_q:.0%}/{hi_q:.0%}; '
        f'U_LVIS range [{res.U_LVIS.min():.3f}, {res.U_LVIS.max():.3f}], '
        f'median {res.U_LVIS.median():.3f}.')
    say(f'    spatial grid ({cell_deg:g} deg): {res.cell.nunique()} cells'
        + (f'; biome: {res.biome.nunique()} classes.' if has_biome
           else '; no biome field -> spatial-cell fallback (change D).'))
    return res


# ---------------------------------------------------------------------------
# Step 2: fixed height bands with a min-N merge
# ---------------------------------------------------------------------------
def _height_bands(H: np.ndarray, width: float, start: float, top_merge: float,
                  min_n: int) -> tuple:
    """Fixed `width`-m bands from `start`, with everything >= `top_merge`
    pooled into a single open-ended band; bands below `min_n` are merged into a
    neighbour by dropping an interior edge. Returns (labels, edges)."""
    edges = list(np.arange(start, top_merge + 1e-9, width)) + [np.inf]
    while True:
        labels = np.digitize(H, edges[1:-1], right=False)
        counts = np.bincount(labels, minlength=len(edges) - 1)
        small = np.where(counts < min_n)[0]
        if small.size == 0 or len(edges) <= 2:
            break
        b = int(small[np.argmin(counts[small])])
        if b == 0:
            drop = 1
        elif b == len(edges) - 2:
            drop = len(edges) - 2
        else:
            drop = b if counts[b - 1] <= counts[b + 1] else b + 1
        edges.pop(drop)
    labels = np.digitize(H, edges[1:-1], right=False)
    return labels, np.array(edges)


def _band_label(lo: float, hi: float) -> str:
    return f'{lo:g}+' if not np.isfinite(hi) else f'{lo:g}-{hi:g}'


# ---------------------------------------------------------------------------
# Bootstrap diff-of-means CI + pooled Cohen's d + within-band top-height resid
# ---------------------------------------------------------------------------
def _boot_diff(hi: np.ndarray, lo: np.ndarray, n_boot: int,
               rng: np.random.Generator) -> tuple:
    """Percentile-bootstrap 95% CI for mean(hi) - mean(lo)."""
    if len(hi) < 2 or len(lo) < 2:
        return np.nan, np.nan
    d = np.empty(n_boot)
    for i in range(n_boot):
        a = hi[rng.integers(0, len(hi), len(hi))]
        b = lo[rng.integers(0, len(lo), len(lo))]
        d[i] = a.mean() - b.mean()
    lo_, hi_ = np.quantile(d, [0.025, 0.975])
    return float(lo_), float(hi_)


def _cohens_d(hi: np.ndarray, lo: np.ndarray) -> float:
    nh, nl = len(hi), len(lo)
    if nh < 2 or nl < 2:
        return np.nan
    sp = np.sqrt(((nh - 1) * hi.var(ddof=1) + (nl - 1) * lo.var(ddof=1))
                 / (nh + nl - 2))
    return float((hi.mean() - lo.mean()) / sp) if sp > 1e-12 else np.nan


def _topheight_residual(num_hi: np.ndarray, top_hi: np.ndarray,
                        num_lo: np.ndarray, top_lo: np.ndarray) -> tuple:
    """OLS-residualize rh25 on own rh98 across both groups in a band, then
    return (resid_high, resid_low). The high-vs-low gap in these residuals is
    the top-height-controlled understory separation drawn in the main figure."""
    num = np.concatenate([num_hi, num_lo])
    top = np.concatenate([top_hi, top_lo])
    A = np.column_stack([np.ones(len(top)), top])
    beta = np.linalg.lstsq(A, num, rcond=None)[0]
    resid = num - A @ beta
    return resid[:len(num_hi)], resid[len(num_hi):]


def _resid_on_top(num: np.ndarray, top: np.ndarray) -> np.ndarray:
    """OLS residual of rh25 on own rh98 (top-height channel removed) over a
    footprint set — the continuous analogue of _topheight_residual."""
    A = np.column_stack([np.ones(len(top)), top])
    beta = np.linalg.lstsq(A, num, rcond=None)[0]
    return num - A @ beta


def _corr_ci(a: np.ndarray, b: np.ndarray, n_boot: int,
             rng: np.random.Generator) -> tuple:
    """Pearson r between two residual vectors with a percentile bootstrap 95%
    CI. Returns (r, lo, hi)."""
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan, np.nan, np.nan
    r = float(np.corrcoef(a, b)[0, 1])
    n = len(a)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        aa, bb = a[idx], b[idx]
        draws[i] = (np.corrcoef(aa, bb)[0, 1]
                    if aa.std() > 1e-12 and bb.std() > 1e-12 else np.nan)
    draws = draws[np.isfinite(draws)]
    if not draws.size:
        return r, np.nan, np.nan
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return r, float(lo), float(hi)


# Sensor pairs for the sensitivity-stratified residual-correlation step.
_CORR_PAIRS = (('VSM', 'LVIS', 'C3'), ('VSM', 'GEDI', 'C4'),
               ('GEDI', 'LVIS', 'C7'))


def _sens_residual_corr(idx: pd.DataFrame, sens_range: tuple, n_bins: int,
                        n_boot: int, rng: np.random.Generator,
                        save_dir: Path, say) -> dict:
    """Stratify the (clean) [sens_lo, sens_hi] GEDI-sensitivity range into
    equal-count bins; in each bin residualize every sensor's rh25 on its OWN
    rh98 and report the pairwise residual correlations (VSM-LVIS, VSM-GEDI,
    GEDI-LVIS) with bootstrap CIs.

    Logic: VSM is optical, so GEDI sensitivity is just a footprint property —
    a genuine understory signal (VSM-LVIS) should be ~flat across sensitivity,
    while a penetration-driven artifact would trend with it (and track the
    GEDI-LVIS curve). Returns a summary dict for the conclusion.

    Restriction-of-range guard: a correlation inside a narrow bin is attenuated
    if the understory-signal variance is small there, so the per-bin residual SD
    (the amplitude actually being correlated) is printed alongside each r, with
    the pooled (whole clean-range) residual SD as the unrestricted reference and
    u = SD_bin / SD_pool as the range-restriction factor. If u is ~constant
    across bins, the per-bin correlations are comparable and a flat curve is
    real; if u varies, the correlations are not directly comparable."""
    lo, hi = sens_range
    s = idx['sensitivity'].to_numpy()
    keep = np.isfinite(s) & (s >= lo) & (s <= hi)
    sub = idx[keep]
    say(f'  sensitivity in [{lo:g}, {hi:g}]: {len(sub):,} of {len(idx):,} '
        f'footprints ({int((~keep).sum()):,} dropped: out of range or '
        f'non-finite).')
    if len(sub) < max(50 * n_bins, 200):
        say('  too few footprints for a stable stratification — step skipped.')
        return {}
    # Unrestricted reference: residual SD over the whole clean range.
    sd_pool = {sn: float(_resid_on_top(sub[f'num_{sn}'].to_numpy(),
                                       sub[f'top_{sn}'].to_numpy()).std())
               for sn in _SENSORS}
    say('  pooled RH25-residual SD over the clean range (unrestricted '
        'reference, m): ' + ', '.join(f'{sn} {sd_pool[sn]:.2f}'
                                       for sn in _SENSORS))
    sv = sub['sensitivity'].to_numpy()
    edges = np.unique(np.quantile(sv, np.linspace(0, 1, n_bins + 1)))
    labels = np.clip(np.digitize(sv, edges[1:-1], right=False),
                     0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = labels == b
        bsub = sub[m]
        if len(bsub) < 30:
            continue
        resid = {sn: _resid_on_top(bsub[f'num_{sn}'].to_numpy(),
                                   bsub[f'top_{sn}'].to_numpy())
                 for sn in _SENSORS}
        row = {'sens_bin': b, 'edge_lo': float(edges[b]),
               'edge_hi': float(edges[b + 1]),
               'sens_median': float(np.median(bsub['sensitivity'])),
               'n': int(len(bsub))}
        # per-bin residual variance/SD + range-restriction factor u vs pool.
        for sn in _SENSORS:
            v = float(resid[sn].var(ddof=1))
            row[f'var_resid_{sn}'] = v
            row[f'sd_resid_{sn}'] = float(np.sqrt(v))
            row[f'u_{sn}'] = (float(np.sqrt(v) / sd_pool[sn])
                              if sd_pool[sn] > 1e-12 else np.nan)
        for a, c, _ in _CORR_PAIRS:
            r, clo, chi = _corr_ci(resid[a], resid[c], n_boot, rng)
            key = f'{a}_{c}'
            row[f'r_{key}'] = r
            row[f'r_{key}_lo'] = clo
            row[f'r_{key}_hi'] = chi
        rows.append(row)
        say(f'    sens~{row["sens_median"]:.3f} N={row["n"]:>5} | '
            f'SD_resid m (V/G/L)={row["sd_resid_VSM"]:.2f}/'
            f'{row["sd_resid_GEDI"]:.2f}/{row["sd_resid_LVIS"]:.2f} '
            f'(u_L={row["u_LVIS"]:.2f}) | '
            f'r V-L={row["r_VSM_LVIS"]:+.3f}'
            f'[{row["r_VSM_LVIS_lo"]:+.3f},{row["r_VSM_LVIS_hi"]:+.3f}] '
            f'V-G={row["r_VSM_GEDI"]:+.3f} G-L={row["r_GEDI_LVIS"]:+.3f}')
    if len(rows) < 2:
        say('  fewer than 2 usable sensitivity bins — step skipped.')
        return {}
    df = pd.DataFrame(rows)
    csv = save_dir / 'sensitivity_residual_corr.csv'
    df.to_csv(csv, index=False)
    _plot_sens_corr(df, save_dir / 'sensitivity_residual_corr.png')
    say(f'  -> {csv.name}, sensitivity_residual_corr.png')
    # high-minus-low-sensitivity change in each pair's residual correlation.
    first, last = df.iloc[0], df.iloc[-1]
    u = df['u_LVIS'].to_numpy()
    return {'r_vl_lo_sens': float(first['r_VSM_LVIS']),
            'r_vl_hi_sens': float(last['r_VSM_LVIS']),
            'r_vl_mean': float(df['r_VSM_LVIS'].mean()),
            'delta_vl': float(last['r_VSM_LVIS'] - first['r_VSM_LVIS']),
            'delta_gl': float(last['r_GEDI_LVIS'] - first['r_GEDI_LVIS']),
            'u_lvis_min': float(np.nanmin(u)), 'u_lvis_max': float(np.nanmax(u)),
            'n_bins': len(df)}


# ---------------------------------------------------------------------------
# Numerator mixed model (change A/B/D)
# ---------------------------------------------------------------------------
def _fit_numerator(pooled: pd.DataFrame, sensor: str, spatial: str,
                   add_true98: bool = False, add_biome: bool = False,
                   mask: np.ndarray = None) -> dict:
    """rh25 ~ grp + own_rh98 (+ lvis_rh98) (+ spatial) (+ C(biome)) with a
    (1|band) random intercept when >= 2 bands, else OLS. `spatial` is 'latlon'
    or 'cell'. `mask` restricts to a subset (within-biome refit). Continuous
    covariates are z-scored (does not move the grp coef). Returns the grp effect
    in rh25 metres + meta."""
    d = pd.DataFrame({
        'y': pooled[f'num_{sensor}'].to_numpy(),
        'grp': pooled['grp'].to_numpy(float),
        'own98': _z(pooled[f'top_{sensor}'].to_numpy()),
        'band': pooled['band'].to_numpy()})
    rhs = ['grp', 'own98']
    if add_true98:
        d['true98'] = _z(pooled['top_LVIS'].to_numpy())
        rhs.append('true98')
    if spatial == 'latlon':
        d['lat'] = pooled['lat_z'].to_numpy()
        d['lon'] = pooled['lon_z'].to_numpy()
        rhs += ['lat', 'lon']
    elif spatial == 'cell':
        d['cell'] = pooled['cell'].to_numpy()
        rhs.append('C(cell)')
    if add_biome:
        d['biome'] = pooled['biome'].to_numpy()
        rhs.append('C(biome)')
    if mask is not None:
        d = d[mask].reset_index(drop=True)
    # Drop single-level categoricals (singular within a subset).
    rhs = [t for t in rhs if not (
        (t == 'C(cell)' and d['cell'].nunique() < 2) or
        (t == 'C(biome)' and d['biome'].nunique() < 2))]
    formula = 'y ~ ' + ' + '.join(rhs)
    n_bands = d['band'].nunique()
    out = {'sensor': sensor, 'formula': formula, 'n': int(len(d)),
           'n_bands': int(n_bands)}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            if n_bands >= 2:
                res = smf.mixedlm(formula, d, groups=d['band']).fit()
                out['kind'] = 'mixedlm'
                out['converged'] = bool(getattr(res, 'converged', True))
            else:
                res = smf.ols(formula, d).fit()
                out['kind'], out['converged'] = 'ols', True
            ci = res.conf_int()
            out.update(coef=float(res.params['grp']),
                       ci_lo=float(ci.loc['grp', 0]),
                       ci_hi=float(ci.loc['grp', 1]),
                       p=float(res.pvalues['grp']))
        except Exception as e:                       # singular / non-PD design
            out.update(kind='FAILED', converged=False, coef=np.nan,
                       ci_lo=np.nan, ci_hi=np.nan, p=np.nan, error=str(e))
    return out


def _sig_pos(m: dict) -> bool:
    return bool(np.isfinite(m.get('coef', np.nan)) and m['coef'] > 0
                and m.get('ci_lo', np.nan) > 0)


def _share(num: float, den: float) -> tuple:
    """(numerator_share, defined?) of the ratio decomposition. The share is
    only meaningful when the total separation is large relative to the
    cancelling terms — when ΔU ~ 0 (num and den cancel) the percentages blow
    up and are reported as undefined instead."""
    tot = num + den
    if abs(tot) > max(1e-6, 0.2 * (abs(num) + abs(den))):
        return num / tot, True
    return np.nan, False


# ---------------------------------------------------------------------------
# Block (cluster) bootstrap: one block = one S2 MGRS tile (the parquet stem).
# Replaces the footprint-level p (e-168) which treats spatially clustered
# footprints as iid and is anti-conservative.
# ---------------------------------------------------------------------------
# Eight statistics stored per iteration, in display order. Six grp coefficients
# (3 sensors x {base, strict}) plus the two ratios — formed INSIDE the
# iteration so numerator/denominator covariance propagates automatically
# instead of being lost to a hard CI division.
_BOOT_STATS = (
    'coef_VSM_base', 'coef_GEDI_base', 'coef_LVIS_base',
    'coef_VSM_strict', 'coef_GEDI_strict', 'coef_LVIS_strict',
    'ratio_V_G', 'ratio_V_L',
)


def _one_bootstrap_iter(sub_raw: pd.DataFrame, rh_under: str, rh_top: str,
                        rh98_min: float, winsor: tuple, cell_deg: float,
                        has_biome: bool, has_sens: bool,
                        band_width: float, band_start: float,
                        band_top_merge: float, min_band_n: int,
                        min_group_n: int, low_q: float, high_q: float,
                        min_effective_bands: int):
    """Run the full pipeline (drop RH98<=rh98_min, winsorize U, height bands,
    U_LVIS terciles, mixed-model fits) on one cluster-bootstrap resample and
    return the 8-vector of stored statistics. Returns None to signal the
    iteration was DROPPED (too few qualifying bands); a NaN inside the vector
    just means that particular sub-statistic could not be computed."""
    noop = lambda *a, **kw: None
    try:
        sub_idx = _build_indices(sub_raw, rh_under, rh_top, rh98_min, winsor,
                                 cell_deg, has_biome, has_sens, noop)
    except Exception:
        return None
    H = sub_idx['H_L'].to_numpy()
    if H.size == 0:
        return None
    labels, edges = _height_bands(H, band_width, band_start, band_top_merge,
                                  min_band_n)
    sub_idx = sub_idx.assign(_band=labels)

    pooled_parts = []
    for b in range(len(edges) - 1):
        sub = sub_idx[sub_idx['_band'] == b]
        if len(sub) < min_band_n:
            continue
        ul = sub['U_LVIS'].to_numpy()
        q_lo, q_hi = np.quantile(ul, [low_q, high_q])
        hi_mask = ul >= q_hi
        lo_mask = ul <= q_lo
        if (int(hi_mask.sum()) < min_group_n
                or int(lo_mask.sum()) < min_group_n):
            continue
        blab = _band_label(edges[b], edges[b + 1])
        for grp_val, mask in ((1, hi_mask), (0, lo_mask)):
            gdf = sub[mask]
            d = {'band': blab, 'grp': grp_val,
                 'lat': gdf['lat'].to_numpy(), 'lon': gdf['lon'].to_numpy(),
                 'cell': gdf['cell'].to_numpy()}
            for s in _SENSORS:
                d[f'num_{s}'] = gdf[f'num_{s}'].to_numpy()
                d[f'top_{s}'] = gdf[f'top_{s}'].to_numpy()
                d[f'U_{s}'] = gdf[f'U_{s}'].to_numpy()
            if has_biome:
                d['biome'] = gdf['biome'].to_numpy()
            pooled_parts.append(pd.DataFrame(d))

    n_eff_bands = len(pooled_parts) // 2          # one band -> two parts (hi, lo)
    if n_eff_bands < min_effective_bands:
        return None

    pooled = pd.concat(pooled_parts, ignore_index=True)
    pooled['lat_z'] = _z(pooled['lat'].to_numpy())
    pooled['lon_z'] = _z(pooled['lon'].to_numpy())
    spatial = 'latlon' if has_biome else 'cell'

    coef = {}
    for s in _SENSORS:
        m = _fit_numerator(pooled, s, spatial)
        coef[(s, 'base')] = m.get('coef', np.nan)
    for s in ('VSM', 'GEDI'):
        m = _fit_numerator(pooled, s, spatial, add_true98=True)
        coef[(s, 'strict')] = m.get('coef', np.nan)
    # LVIS strict == LVIS base (own RH98 IS the true height; collinear).
    coef[('LVIS', 'strict')] = coef[('LVIS', 'base')]

    cV_s = coef[('VSM', 'strict')]
    cG_s = coef[('GEDI', 'strict')]
    cL_b = coef[('LVIS', 'base')]
    rvg = (cV_s / cG_s if np.isfinite(cV_s) and np.isfinite(cG_s)
           and abs(cG_s) > 1e-12 else np.nan)
    rvl = (cV_s / cL_b if np.isfinite(cV_s) and np.isfinite(cL_b)
           and abs(cL_b) > 1e-12 else np.nan)
    return np.array([
        coef[('VSM',  'base')], coef[('GEDI', 'base')], coef[('LVIS', 'base')],
        coef[('VSM',  'strict')], coef[('GEDI', 'strict')], coef[('LVIS', 'strict')],
        rvg, rvl,
    ], dtype=float)


def _block_bootstrap_tile(raw: pd.DataFrame, n_boot: int, rh_under: str,
                          rh_top: str, rh98_min: float, winsor: tuple,
                          cell_deg: float, has_biome: bool, has_sens: bool,
                          band_width: float, band_start: float,
                          band_top_merge: float, min_band_n: int,
                          min_group_n: int, low_q: float, high_q: float,
                          min_effective_bands: int,
                          rng: np.random.Generator, say) -> tuple:
    """Cluster bootstrap on the S2 tile. Each iteration: pick N_block tiles
    with replacement (a tile drawn twice -> its rows enter twice), rerun the
    pipeline on the concatenated rows, store the 8-vector. Returns
    (draws_df, summary_df, per_block_n)."""
    if 'tile' not in raw.columns:
        raise KeyError('raw frame has no `tile` column — _load_pairs needs to '
                       'stamp the filename stem as tile id.')
    unique_blocks, inv = np.unique(raw['tile'].to_numpy(), return_inverse=True)
    block_rows = [np.where(inv == k)[0] for k in range(len(unique_blocks))]
    per_block_n = pd.DataFrame({
        'tile': unique_blocks,
        'n': [len(r) for r in block_rows],
    }).sort_values('n', ascending=False).reset_index(drop=True)
    n_blocks = len(unique_blocks)
    say(f'Block bootstrap unit: S2 tile (filename stem). n_blocks={n_blocks}, '
        f'total rows={len(raw):,}.')
    sizes = per_block_n['n'].to_numpy()
    say(f'  per-tile N: min={int(sizes.min())}, median='
        f'{int(np.median(sizes))}, max={int(sizes.max())}.')
    for _, r in per_block_n.iterrows():
        say(f'    {r["tile"]}: N={int(r["n"]):,}')

    if n_blocks < 2:
        say('  fewer than 2 tiles — cluster bootstrap not meaningful; '
            'step skipped.')
        return None, None, per_block_n

    draws = np.full((n_boot, len(_BOOT_STATS)), np.nan)
    n_dropped = 0
    progress = max(1, n_boot // 10)
    for i in range(n_boot):
        picks = rng.integers(0, n_blocks, n_blocks)
        row_idx = np.concatenate([block_rows[k] for k in picks])
        sub_raw = raw.iloc[row_idx].reset_index(drop=True)
        res = _one_bootstrap_iter(
            sub_raw, rh_under, rh_top, rh98_min, winsor, cell_deg, has_biome,
            has_sens, band_width, band_start, band_top_merge, min_band_n,
            min_group_n, low_q, high_q, min_effective_bands)
        if res is None:
            n_dropped += 1
        else:
            draws[i] = res
        if (i + 1) % progress == 0:
            say(f'  iter {i + 1}/{n_boot} (dropped so far: {n_dropped})')

    draws_df = pd.DataFrame(draws, columns=list(_BOOT_STATS))
    rows = []
    for k in _BOOT_STATS:
        v = draws_df[k].to_numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            rows.append({'stat': k, 'n_valid': 0, 'p2_5': np.nan,
                         'p50': np.nan, 'p97_5': np.nan, 'p_gt0': np.nan})
            continue
        lo, med, hi = np.quantile(v, [0.025, 0.5, 0.975])
        rows.append({'stat': k, 'n_valid': int(v.size),
                     'p2_5': float(lo), 'p50': float(med),
                     'p97_5': float(hi), 'p_gt0': float((v > 0).mean())})
    summary_df = pd.DataFrame(rows)
    say(f'  iterations: {n_boot} requested, kept {n_boot - n_dropped}, '
        f'dropped {n_dropped} (too few effective bands or build failure).')
    return draws_df, summary_df, per_block_n


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_sens_corr(df: pd.DataFrame, save_path: Path) -> None:
    """Residual correlation (rh25 | own rh98) vs GEDI sensitivity bin, one line
    per sensor pair with bootstrap CI bands. A flat VSM-LVIS curve = a genuine
    understory signal; a curve that trends with sensitivity and tracks GEDI-LVIS
    = a shared penetration artifact."""
    x = df['sens_median'].to_numpy()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
    ax.axhline(0, color='gray', lw=1, zorder=1)
    for a, c, color in _CORR_PAIRS:
        key = f'{a}_{c}'
        lw = 2.6 if key == 'VSM_LVIS' else 1.5
        ax.plot(x, df[f'r_{key}'], '-o', color=color, lw=lw, zorder=3,
                label=f'{a}-{c}')
        ax.fill_between(x, df[f'r_{key}_lo'], df[f'r_{key}_hi'], color=color,
                        alpha=0.18, zorder=2)
    ax.set_xlabel('GEDI sensitivity (bin median)',
                  fontsize=FONT_SIZES['label'])
    ax.set_ylabel('corr of RH25 residuals | own RH98',
                  fontsize=FONT_SIZES['label'])
    ax.set_title('Understory residual correlation vs GEDI sensitivity',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'], loc='best')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_numerator_main(per_band: pd.DataFrame, save_path: Path) -> None:
    """Change-E main figure: per band, the hiU- vs loU-group mean of rh25 AFTER
    residualizing on own rh98 (top height controlled). hiU = top tercile of
    U=RH25/RH98 = SPARSE understory (top-heavy); loU = bottom tercile = DENSE
    understory. Three sensors; VSM thickest. The vertical gap is the
    top-height-controlled understory numerator separation the headline model
    estimates. (The two lines per sensor mirror about 0 by construction: OLS
    residuals sum to zero over the equal-sized hiU+loU fit, so the gap, not the
    absolute level, is the signal.)"""
    x = np.arange(len(per_band))
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
    ax.axhline(0, color='gray', lw=1, zorder=1)
    for s in _SENSORS:
        c = _SENSOR_COLOR[s]
        lw = 2.6 if s == 'VSM' else 1.4
        ax.plot(x, per_band[f'resid_hiU_{s}'], '-o', color=c, lw=lw, zorder=3,
                label=f'{s} hiU (sparse)')
        ax.plot(x, per_band[f'resid_loU_{s}'], '--x', color=c, lw=lw, zorder=3,
                label=f'{s} loU (dense)', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(per_band['band'], rotation=45, ha='right',
                       fontsize=FONT_SIZES['ticks'])
    ax.set_xlabel('canopy top-height band H_L = lvis_RH98 (m)',
                  fontsize=FONT_SIZES['label'])
    ax.set_ylabel('mean RH25 residual | own RH98  (m)',
                  fontsize=FONT_SIZES['label'])
    ax.set_title('Top-height-controlled understory numerator '
                 '(hiU=sparse vs loU=dense)',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'], ncol=3, loc='best')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_delta_ratio(per_band: pd.DataFrame, save_path: Path) -> None:
    """Legacy figure: per-band ΔU = mean(U|high) - mean(U|low) ratio separation
    with bootstrap CIs. Kept for reference but explicitly labelled the
    UN-decomposed ratio version (mixes in the denominator artifact); the
    numerator figure is the one the headline refers to."""
    x = np.arange(len(per_band))
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
    ax.axhline(0, color='gray', lw=1, zorder=1)
    specs = [('VSM', '-o', 0.0, 2.6), ('GEDI', '--s', 0.12, 1.6),
             ('LVIS', ':D', -0.12, 1.6)]
    for s, style, dx, lw in specs:
        d = per_band[f'dU_{s}'].to_numpy()
        lo = d - per_band[f'dU_{s}_lo'].to_numpy()
        hi = per_band[f'dU_{s}_hi'].to_numpy() - d
        ax.errorbar(x + dx, d, yerr=[lo, hi], fmt=style,
                    color=_SENSOR_COLOR[s], lw=lw, capsize=3, zorder=3,
                    label=f'ΔU {s}')
    ax.set_xticks(x)
    ax.set_xticklabels(per_band['band'], rotation=45, ha='right',
                       fontsize=FONT_SIZES['ticks'])
    ax.set_xlabel('canopy top-height band H_L = lvis_RH98 (m)',
                  fontsize=FONT_SIZES['label'])
    ax.set_ylabel('ΔU = mean(U|high) - mean(U|low)  [RATIO, un-decomposed]',
                  fontsize=FONT_SIZES['label'])
    ax.set_title('Legacy ratio separation (NOT the headline; see numerator fig)',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'], loc='best')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ===========================================================================
# Main entrypoint (registered as run=vsm_understory_matched)
# ===========================================================================
def matched_understory_separation(
        pairs_dir: str,
        save_dir: str,
        rh_understory: str = 'RH25',
        rh_top: str = 'RH98',
        pairs_glob: str = '*.parquet',
        biome_col: str = 'gedi_BIOME',
        spatial_cell_deg: float = 1.5,
        sensitivity_col: str = 'gedi_sensitivity',
        sens_range: tuple = (0.95, 1.0),
        n_sens_bins: int = 6,
        rh98_min: float = 5.0,
        winsor: tuple = (0.01, 0.99),
        band_width: float = 5.0,
        band_start: float = 5.0,
        band_top_merge: float = 35.0,
        min_band_n: int = 200,
        min_group_n: int = 30,
        low_q: float = 1.0 / 3.0,
        high_q: float = 2.0 / 3.0,
        within_biome_min_n: int = 300,
        match_topheight_tol: float = 1.0,
        n_boot: int = 1000,
        anchor_required: bool = True,
        n_boot_blocks: int = 2000,
        min_effective_bands: int = 1,
        enable_block_bootstrap: bool = True,
        random_state: int = 0,
        **kwargs) -> None:
    """Matched-grouping test of whether VSM reflects understory structure at
    matched canopy top height. Headline = top-height-controlled NUMERATOR (rh25)
    separation, anchored to GEDI; the rh25:rh98 ratio is decomposed to expose
    any denominator artifact.

    Args:
        pairs_dir: dir of per-tile LVIS-GEDI-VSM pair parquets
            (`Gabon2016_vs_GEDI2020_stable_forest_with_vsm2020`) carrying
            `lvis_RH<NN>`/`gedi_rh<NN>`/`vsm_RH<NN>` (+ optional biome) +
            geometry.
        save_dir: output dir for figures, CSVs, conclusion.md.
        rh_understory: understory RH percentile (numerator). Default RH25.
        rh_top: top-height RH percentile (denominator + matched-on H_L + the
            numerator-model covariate). Default RH98.
        pairs_glob: glob under pairs_dir. Default '*.parquet'.
        biome_col: biome/ecoregion column to look for. Present -> models add a
            `+ C(biome)` version + within-biome refit; absent -> lat/lon are
            replaced by the spatial-cell fallback (change D). Default
            'gedi_BIOME'.
        spatial_cell_deg: grid-cell size (deg) for the no-biome spatial
            fallback. Default 1.5.
        sensitivity_col: GEDI sensitivity column for the stratified step. If
            present, Step 5 bins it within `sens_range` and recomputes the
            pairwise RH25-residual correlations; if absent the step is skipped.
            Default 'gedi_sensitivity'.
        sens_range: (lo, hi) GEDI-sensitivity window for Step 5 (the clean
            high-sensitivity range). Default (0.95, 1.0).
        n_sens_bins: number of equal-count sensitivity bins in Step 5.
            Default 6.
        rh98_min: keep footprints with ALL three rh_top > this. Default 5 m.
        winsor: (low, high) quantiles to winsorize each U ratio (ratio
            decomposition + legacy figure only). Default 1/99 %.
        band_width, band_start, band_top_merge: fixed top-height band edges (m);
            >= band_top_merge pooled into one open-ended band. Defaults 5/5/35.
        min_band_n: min footprints per band (else merge). Default 200.
        min_group_n: min footprints per high/low group (else drop band).
            Default 30.
        low_q, high_q: U_L tercile cuts. Default 1/3, 2/3.
        within_biome_min_n: min grouped footprints for a within-biome refit.
            Default 300.
        match_topheight_tol: |top-height group diff| (m) above which the
            matching is flagged as suspect (the numerator model controls top
            height regardless; this is a reporting guard). Default 1.0.
        n_boot: bootstrap resamples for per-band ΔU CIs. Default 1000.
        anchor_required: if True and the GEDI numerator grp coef is not
            significantly positive, declare the test void and stop. Default True.
        n_boot_blocks: cluster-bootstrap iterations (one block = one S2 MGRS
            tile, the parquet filename stem). Each iteration resamples tiles
            WITH replacement, reruns the whole pipeline on the concatenated
            rows, stores 6 grp coefs + 2 ratios; report 2.5/50/97.5 percentiles
            + P(>0) per statistic. Default 2000.
        min_effective_bands: drop an iteration when fewer qualifying bands
            survive its resample. Default 1 (OLS on a single band is still a
            usable draw).
        enable_block_bootstrap: turn the cluster bootstrap on/off. Default True.
        random_state: RNG seed.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    report: list = []

    def say(line: str = '') -> None:
        print(line)
        report.append(line)

    say('=' * 74)
    say('VSM understory matched-grouping test — NUMERATOR headline '
        '(top height controlled)')
    say('=' * 74)
    say(f'Group on LVIS U_L = lvis_{rh_understory}/lvis_{rh_top}; test '
        f'{rh_understory} with own {rh_top} as covariate; anchor = GEDI.')
    say(f'Bands: {band_width:g} m from {band_start:g} m, >= {band_top_merge:g} '
        f'm merged; min_band_n={min_band_n}, min_group_n={min_group_n}.')
    say(f'Terciles: low<=Q{low_q:.2f}, high>=Q{high_q:.2f} of U_LVIS; '
        f'n_boot={n_boot}; seed={random_state}.')
    say('Pre-registered: thresholds/covariates fixed before inspecting VSM.')
    say('')

    # --- load + Step 1 ----------------------------------------------------
    raw, has_biome, has_sens = _load_pairs(
        pairs_dir, pairs_glob, rh_understory, rh_top, biome_col,
        sensitivity_col)
    say(f'Biome field {biome_col!r}: '
        + ('FOUND -> lat/lon control + C(biome) version + within-biome refit.'
           if has_biome else
           f'NOT found -> fallback: lat/lon replaced by '
           f'{spatial_cell_deg:g}-deg C(cell) (change D).'))
    say(f'Sensitivity field {sensitivity_col!r}: '
        + ('FOUND -> Step 5 sensitivity-stratified residual correlation.'
           if has_sens else 'NOT found -> Step 5 skipped.'))
    spatial = 'latlon' if has_biome else 'cell'
    idx = _build_indices(raw, rh_understory, rh_top, rh98_min, winsor,
                         spatial_cell_deg, has_biome, has_sens, say)

    # --- Step 2 bands -----------------------------------------------------
    H = idx['H_L'].to_numpy()
    labels, edges = _height_bands(H, band_width, band_start, band_top_merge,
                                  min_band_n)
    idx = idx.assign(_band=labels)
    say('')
    say(f'Step 2: {len(edges) - 1} height bands after merge -> '
        f'{[_band_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]}')

    # --- Steps 3-4A per-band grouping, ΔU, ratio decomposition, residuals -
    say('')
    say('Step 3-4A: per-band tercile split (group on U_LVIS); ΔU ratio, '
        'numerator/denominator decomposition, top-height-controlled residuals.')
    per_rows = []
    pooled_parts = []
    cohend_acc = {s: [] for s in _SENSORS}        # (d, weight) per band
    for b in range(len(edges) - 1):
        sub = idx[idx['_band'] == b]
        blab = _band_label(edges[b], edges[b + 1])
        if len(sub) < min_band_n:
            say(f'  [{blab:>7}] N={len(sub)} < {min_band_n}; skipped.')
            continue
        ul = sub['U_LVIS'].to_numpy()
        q_lo, q_hi = np.quantile(ul, [low_q, high_q])
        hi_mask, lo_mask = ul >= q_hi, ul <= q_lo
        n_hi, n_lo = int(hi_mask.sum()), int(lo_mask.sum())
        if n_hi < min_group_n or n_lo < min_group_n:
            say(f'  [{blab:>7}] N={len(sub)} but groups too small '
                f'(high={n_hi}, low={n_lo} < {min_group_n}); band dropped.')
            continue
        hi_df, lo_df = sub[hi_mask], sub[lo_mask]
        row = {'band': blab, 'band_lo': float(edges[b]),
               'band_hi': float(edges[b + 1]), 'N_band': int(len(sub)),
               'N_high': n_hi, 'N_low': n_lo,
               'HL_diff': float(hi_df['H_L'].mean() - lo_df['H_L'].mean())}
        for s in _SENSORS:
            uh, ul_ = hi_df[f'U_{s}'].to_numpy(), lo_df[f'U_{s}'].to_numpy()
            nh, nl = hi_df[f'num_{s}'].to_numpy(), lo_df[f'num_{s}'].to_numpy()
            th, tl = hi_df[f'top_{s}'].to_numpy(), lo_df[f'top_{s}'].to_numpy()
            # legacy ratio separation
            d_u = float(uh.mean() - ul_.mean())
            clo, chi = _boot_diff(uh, ul_, n_boot, rng)
            row[f'dU_{s}'] = d_u
            row[f'dU_{s}_lo'], row[f'dU_{s}_hi'] = clo, chi
            # change C: numerator vs denominator decomposition of ΔU
            num_all = np.concatenate([nh, nl])
            top_all = np.concatenate([th, tl])
            mean_top = float(top_all.mean())
            mean_u = float((num_all / top_all).mean())
            d_num = float(nh.mean() - nl.mean())
            d_top = float(th.mean() - tl.mean())
            num_term = d_num / mean_top
            den_term = -mean_u * d_top / mean_top
            row[f'dRH25_{s}'] = d_num
            row[f'dRH98_{s}'] = d_top
            row[f'num_term_{s}'] = num_term
            row[f'den_term_{s}'] = den_term
            row[f'decomp_sum_{s}'] = num_term + den_term
            # change E: top-height-controlled residual means + Cohen's d (m).
            # hiU = top tercile of U=RH25/RH98 = SPARSE understory; loU = dense.
            r_hi, r_lo = _topheight_residual(nh, th, nl, tl)
            row[f'resid_hiU_{s}'] = float(r_hi.mean())
            row[f'resid_loU_{s}'] = float(r_lo.mean())
            row[f'resid_diff_{s}'] = float(r_hi.mean() - r_lo.mean())
            cd = _cohens_d(r_hi, r_lo)
            row[f'cohend_resid_{s}'] = cd
            if np.isfinite(cd):
                cohend_acc[s].append((cd, n_hi + n_lo))
            row[f'top_diff_{s}'] = d_top
        per_rows.append(row)
        say(f'  [{blab:>7}] N={len(sub):>5} hi={n_hi:>4} lo={n_lo:>4} | '
            f'VSM ΔRH25={row["dRH25_VSM"]:+.3f} (num {row["num_term_VSM"]:+.3f}'
            f' + den {row["den_term_VSM"]:+.3f} = {row["decomp_sum_VSM"]:+.3f}'
            f' ~ ΔU {row["dU_VSM"]:+.3f}) | H_L diff={row["HL_diff"]:+.2f} m')
        for grp_val, gdf in ((1, hi_df), (0, lo_df)):
            cols = {'band': blab, 'grp': grp_val,
                    'lat': gdf['lat'].to_numpy(), 'lon': gdf['lon'].to_numpy(),
                    'cell': gdf['cell'].to_numpy()}
            for s in _SENSORS:
                cols[f'num_{s}'] = gdf[f'num_{s}'].to_numpy()
                cols[f'top_{s}'] = gdf[f'top_{s}'].to_numpy()
                cols[f'U_{s}'] = gdf[f'U_{s}'].to_numpy()
            if has_biome:
                cols['biome'] = gdf['biome'].to_numpy()
            pooled_parts.append(pd.DataFrame(cols))

    if not per_rows:
        say('')
        say('No band qualified (need N>=%d with >=%d per group). Cannot run '
            'the test on this population.' % (min_band_n, min_group_n))
        (save_dir / 'conclusion.md').write_text('\n'.join(report) + '\n')
        print(f'\n-> {save_dir / "conclusion.md"}')
        return

    per_band = pd.DataFrame(per_rows)
    pooled = pd.concat(pooled_parts, ignore_index=True)
    pooled['lat_z'] = _z(pooled['lat'].to_numpy())
    pooled['lon_z'] = _z(pooled['lon'].to_numpy())
    w_band = (per_band['N_high'] + per_band['N_low']).to_numpy()

    # --- Step 3 match-quality guard --------------------------------------
    say('')
    say('Step 3: match-quality guard (high vs low, n-weighted over bands).')
    hl_pooled = float(np.average(per_band['HL_diff'].to_numpy(), weights=w_band))
    say(f'  H_L (matched top height) mean diff = {hl_pooled:+.3f} m.')
    for s in _SENSORS:
        td = float(np.average(per_band[f'top_diff_{s}'].to_numpy(),
                              weights=w_band))
        say(f'  {s} {rh_top} group diff = {td:+.3f} m '
            f'(numerator model controls it via own {rh_top}'
            + (' + true lvis_%s in the strict variant).' % rh_top
               if s == 'VSM' else ').'))
    if abs(hl_pooled) > match_topheight_tol:
        say(f'  NOTE: |H_L diff| {abs(hl_pooled):.2f} > {match_topheight_tol:g} '
            f'm — matching imperfect; the numerator models control top height '
            f'explicitly, but consider narrowing band_width.')

    # --- Step 4: numerator mixed models (A base, B strict, D +biome) -----
    say('')
    say('Step 4: numerator models  rh25 ~ grp + own_rh98 (+true98) '
        f'(+ {"lat+lon / C(biome)" if has_biome else "C(cell)"}) + (1|band).')
    model_rows = []

    def _log(m, tag):
        cd = float(np.average([c for c, _ in cohend_acc[m['sensor']]],
                              weights=[w for _, w in cohend_acc[m['sensor']]])) \
            if cohend_acc[m['sensor']] else np.nan
        m['variant'] = tag
        m['cohen_d'] = cd
        m['frac_bands_pos'] = float((per_band[f'resid_diff_{m["sensor"]}'] > 0)
                                    .mean())
        model_rows.append(m)
        say(f'  {m["sensor"]:<4} [{tag:<12}] grp={m["coef"]:+.4f} m '
            f'[{m.get("ci_lo", np.nan):+.4f},{m.get("ci_hi", np.nan):+.4f}] '
            f'p={m.get("p", np.nan):.3g} | d={cd:+.3f} | '
            f'resid_diff>0 in {m["frac_bands_pos"]:.0%} bands | '
            f'{m["kind"]}({m["n_bands"]}b'
            f'{"" if m.get("converged", True) else ",NO-CONV"})')

    base = {}
    for s in _SENSORS:                                   # change A: base
        base[s] = _fit_numerator(pooled, s, spatial)
        _log(base[s], 'base')
    # change B / headline: strict = own RH98 + TRUE lvis_RH98. Fit for VSM and
    # GEDI so the headline ratio is strict-vs-strict; LVIS own RH98 IS the true
    # height, so LVIS strict == LVIS base (adding it twice is collinear).
    vsm_strict = _fit_numerator(pooled, 'VSM', spatial, add_true98=True)
    _log(vsm_strict, 'strict')
    gedi_strict = _fit_numerator(pooled, 'GEDI', spatial, add_true98=True)
    _log(gedi_strict, 'strict')
    strict = {'VSM': vsm_strict, 'GEDI': gedi_strict, 'LVIS': base['LVIS']}
    biome_models = {}
    if has_biome:                                        # change D: +biome
        for s in _SENSORS:
            biome_models[s] = _fit_numerator(pooled, s, spatial, add_biome=True)
            _log(biome_models[s], 'base+biome')
        vsm_strict_biome = _fit_numerator(pooled, 'VSM', spatial,
                                          add_true98=True, add_biome=True)
        _log(vsm_strict_biome, 'strict+biome')

    # HEADLINE ratios use the STRICT coefs (own + true top height controlled).
    cV, cG, cL = (strict['VSM']['coef'], strict['GEDI']['coef'],
                  strict['LVIS']['coef'])
    ratio_vg = cV / cG if abs(cG) > 1e-12 else np.nan    # fraction of GEDI
    ratio_vl = cV / cL if abs(cL) > 1e-12 else np.nan    # fraction of ceiling

    # --- within-biome refit (change D) -----------------------------------
    wb_rows = []
    if has_biome:
        say('')
        say(f'Step 4D: within-biome VSM base refit (biomes with N>='
            f'{within_biome_min_n}).')
        for bval, cnt in pooled['biome'].value_counts().items():
            if cnt < within_biome_min_n:
                continue
            m = _fit_numerator(pooled, 'VSM', spatial,
                               mask=(pooled['biome'] == bval).to_numpy())
            m['biome'] = bval
            wb_rows.append({'biome': bval, 'n': m['n'], 'n_bands': m['n_bands'],
                            'coef': m.get('coef'), 'ci_lo': m.get('ci_lo'),
                            'ci_hi': m.get('ci_hi'), 'p': m.get('p'),
                            'kind': m['kind']})
            say(f'  biome {bval}: VSM grp={m.get("coef", np.nan):+.4f} m '
                f'[{m.get("ci_lo", np.nan):+.4f},{m.get("ci_hi", np.nan):+.4f}]'
                f' p={m.get("p", np.nan):.3g} (N={m["n"]}, '
                f'{"sig+" if _sig_pos(m) else "ns"})')

    # --- overall ratio decomposition (change C) --------------------------
    ov = {}
    for s in _SENSORS:
        ov[f'num_term_{s}'] = float(np.average(per_band[f'num_term_{s}'],
                                               weights=w_band))
        ov[f'den_term_{s}'] = float(np.average(per_band[f'den_term_{s}'],
                                               weights=w_band))
        ov[f'dU_{s}'] = float(np.average(per_band[f'dU_{s}'], weights=w_band))
    say('')
    say('Step 4C: overall ratio decomposition (n-weighted over bands).')
    for s in _SENSORS:
        fn, ok = _share(ov[f'num_term_{s}'], ov[f'den_term_{s}'])
        share = f'numerator share {fn:.0%}' if ok else 'ΔU~0, shares undefined'
        say(f'  {s:<4} ΔU={ov[f"dU_{s}"]:+.4f} = num {ov[f"num_term_{s}"]:+.4f} '
            f'+ den {ov[f"den_term_{s}"]:+.4f}  ({share})')

    # --- Step 4F: cluster (block) bootstrap on S2 tile -------------------
    # The footprint-level p in Step 4 treats spatially clustered footprints as
    # iid and is implausibly small (e.g. p~e-168). Resample WHOLE S2 tiles
    # (filename stem) with replacement, rerun the full pipeline inside every
    # iteration, and report 2.5/50/97.5 percentiles + P(>0) as the bootstrap
    # sign-consistency replacement for p. Bands and U_LVIS terciles are
    # recomputed every iteration so the threshold sampling uncertainty enters
    # the variance — fixing them would underestimate it.
    boot_summary_df = None
    if enable_block_bootstrap:
        say('')
        say(f'Step 4F: cluster bootstrap on S2 tile, n_boot_blocks='
            f'{n_boot_blocks}.')
        draws_df, boot_summary_df, per_block_n = _block_bootstrap_tile(
            raw, n_boot_blocks, rh_understory, rh_top, rh98_min, winsor,
            spatial_cell_deg, has_biome, has_sens, band_width, band_start,
            band_top_merge, min_band_n, min_group_n, low_q, high_q,
            min_effective_bands, rng, say)
        per_block_n.to_csv(save_dir / 'block_bootstrap_per_block_n.csv',
                           index=False)
        if draws_df is not None:
            draws_df.to_csv(save_dir / 'block_bootstrap_draws.csv',
                            index=False)
            boot_summary_df.to_csv(save_dir / 'block_bootstrap_summary.csv',
                                   index=False)
            say('  -> block_bootstrap_draws.csv, block_bootstrap_summary.csv, '
                'block_bootstrap_per_block_n.csv')
            say('  bootstrap (2.5 / 50 / 97.5 %; P(>0)):')
            for _, r in boot_summary_df.iterrows():
                say(f'    {r["stat"]:<18} '
                    f'[{r["p2_5"]:+.4f}, {r["p50"]:+.4f}, '
                    f'{r["p97_5"]:+.4f}]  P(>0)={r["p_gt0"]:.3f}  '
                    f'n_valid={int(r["n_valid"])}')
        else:
            say('  -> block_bootstrap_per_block_n.csv (bootstrap skipped: '
                'fewer than 2 tiles)')

    # --- write tables -----------------------------------------------------
    pb_csv = save_dir / 'per_band_separation.csv'
    per_band.to_csv(pb_csv, index=False)
    keep = ('sensor', 'variant', 'kind', 'n', 'n_bands', 'converged', 'coef',
            'ci_lo', 'ci_hi', 'p', 'cohen_d', 'frac_bands_pos', 'formula')
    model_df = pd.DataFrame([{k: m.get(k) for k in keep} for m in model_rows])
    # Ratios are strict-vs-strict; attach to the VSM strict row.
    model_df['ratio_VSM_over_GEDI'] = np.where(
        (model_df.sensor == 'VSM') & (model_df.variant == 'strict'),
        ratio_vg, np.nan)
    model_df['ratio_VSM_over_LVIS'] = np.where(
        (model_df.sensor == 'VSM') & (model_df.variant == 'strict'),
        ratio_vl, np.nan)
    md_csv = save_dir / 'numerator_model_coefficients.csv'
    model_df.to_csv(md_csv, index=False)
    decomp_df = pd.DataFrame([{'sensor': s, 'scope': 'overall',
                               'dU': ov[f'dU_{s}'],
                               'num_term': ov[f'num_term_{s}'],
                               'den_term': ov[f'den_term_{s}']}
                              for s in _SENSORS])
    dc_csv = save_dir / 'ratio_decomposition.csv'
    decomp_df.to_csv(dc_csv, index=False)
    out_files = [pb_csv.name, md_csv.name, dc_csv.name]
    if wb_rows:
        wb_csv = save_dir / 'within_biome_vsm.csv'
        pd.DataFrame(wb_rows).to_csv(wb_csv, index=False)
        out_files.append(wb_csv.name)

    # --- figures ----------------------------------------------------------
    _plot_numerator_main(per_band, save_dir / 'main_numerator_residual.png')
    _plot_delta_ratio(per_band, save_dir / 'legacy_delta_U_ratio.png')
    say('')
    say('-> ' + ', '.join(out_files))
    say('-> main_numerator_residual.png, legacy_delta_U_ratio.png')

    # --- Step 5: sensitivity-stratified residual correlation -------------
    sens_summary = {}
    if has_sens:
        say('')
        say(f'Step 5: GEDI-sensitivity-stratified RH25-residual correlations '
            f'({n_sens_bins} equal-count bins in [{sens_range[0]:g}, '
            f'{sens_range[1]:g}]).')
        sens_summary = _sens_residual_corr(idx, sens_range, n_sens_bins, n_boot,
                                           rng, save_dir, say)

    # --- conclusion -------------------------------------------------------
    say('')
    say('=' * 74)
    say('CONCLUSION')
    say('=' * 74)
    g = strict['GEDI']
    say(f'(1) GEDI anchor (strict numerator, precondition): grp={g["coef"]:+.4f}'
        f' m [{g["ci_lo"]:+.4f},{g["ci_hi"]:+.4f}] p={g["p"]:.3g} -> '
        f'{"significantly positive" if _sig_pos(g) else "NOT significantly positive"}.')

    if not _sig_pos(g) and anchor_required:
        say('')
        say('VERDICT (VOID): even GEDI cannot separate the LVIS-defined '
            'sparse/dense understory groups in the top-height-controlled '
            'numerator. '
            'Between two lidars the grouping signal is not detectable on this '
            'footprint population (LVIS-GEDI footprint mismatch likely drowns '
            'it), so the optical VSM cannot be fairly evaluated. Test stopped.')
        (save_dir / 'conclusion.md').write_text('\n'.join(report) + '\n')
        print(f'\n-> {save_dir / "conclusion.md"}')
        return

    vsm_variants = [('base', base['VSM']), ('strict', vsm_strict)]
    if has_biome:
        vsm_variants += [('base+biome', biome_models['VSM']),
                         ('strict+biome', vsm_strict_biome)]
    say('(2) VSM numerator grp coef (matched top height; grp>0 = sparse / '
        'high-RH25:RH98 group sits higher in RH25, the LVIS-defined ordering):')
    for tag, m in vsm_variants:
        say(f'    {tag:<12} {m["coef"]:+.4f} m '
            f'[{m.get("ci_lo", np.nan):+.4f},{m.get("ci_hi", np.nan):+.4f}] '
            f'p={m.get("p", np.nan):.3g} '
            f'-> {"sig+" if _sig_pos(m) else "NOT sig+"}')
    say(f'    effect size: Cohen\'s d={base["VSM"]["cohen_d"]:+.3f} (own-RH98 '
        f'residual); strict coef_VSM/coef_GEDI={ratio_vg:.2f}; '
        f'coef_VSM/coef_LVIS (ceiling)={ratio_vl:.2f}.')
    say('    (footprint-level p above treats spatially clustered footprints '
        'as iid and is anti-conservative; the S2-tile cluster bootstrap '
        'interval and P(>0) below are the honest inference.)')
    if boot_summary_df is not None:
        bsum = boot_summary_df.set_index('stat')
        for k, label in (
                ('coef_VSM_strict', 'VSM strict grp coef (m)'),
                ('ratio_V_G',       'ratio VSM_strict / GEDI_strict'),
                ('ratio_V_L',       'ratio VSM_strict / LVIS_base')):
            r = bsum.loc[k]
            say(f'    tile-block bootstrap {label}: median {r["p50"]:+.4f} '
                f'[{r["p2_5"]:+.4f}, {r["p97_5"]:+.4f}]  P(>0)={r["p_gt0"]:.3f}'
                f'  (n_valid={int(r["n_valid"])} / {n_boot_blocks})')
    num_share, share_ok = _share(ov['num_term_VSM'], ov['den_term_VSM'])
    if share_ok:
        say(f'(3) Ratio decomposition (VSM): of ΔU={ov["dU_VSM"]:+.4f}, '
            f'numerator term {ov["num_term_VSM"]:+.4f} ({num_share:.0%}), '
            f'denominator term {ov["den_term_VSM"]:+.4f} ({1 - num_share:.0%}).')
    else:
        say(f'(3) Ratio decomposition (VSM): ΔU={ov["dU_VSM"]:+.4f} is '
            f'~0 (numerator term {ov["num_term_VSM"]:+.4f} and denominator term '
            f'{ov["den_term_VSM"]:+.4f} cancel) — no net ratio separation to '
            f'attribute.')

    if sens_summary:
        dvl, dgl = sens_summary['delta_vl'], sens_summary['delta_gl']
        stable = abs(dvl) <= 0.05
        umin, umax = sens_summary['u_lvis_min'], sens_summary['u_lvis_max']
        # range-restriction guard: u = SD_bin/SD_pool roughly constant -> the
        # per-bin correlations are comparable, so a flat curve is meaningful.
        range_ok = np.isfinite(umin) and (umax - umin) <= 0.15
        say(f'(4) Sensitivity stratification ({sens_summary["n_bins"]} bins in '
            f'[{sens_range[0]:g},{sens_range[1]:g}]): VSM-LVIS residual r mean '
            f'{sens_summary["r_vl_mean"]:+.3f}, change high-vs-low sensitivity '
            f'{dvl:+.3f} (GEDI-LVIS change {dgl:+.3f}). '
            f'Range-restriction factor u_LVIS in [{umin:.2f}, {umax:.2f}] '
            + ('(roughly constant -> per-bin correlations comparable). '
               if range_ok else
               '(VARIES across bins -> understory range differs by bin, so '
               'the per-bin correlations are NOT directly comparable; '
               'range-restriction-correct before reading the trend). ')
            + ('Flat correlation -> consistent with a true understory signal, '
               'not a penetration artifact.' if stable and range_ok else
               'Interpret the trend with the range-restriction caveat above.'))

    # Prefer the S2-tile cluster-bootstrap CI for the verdict when available:
    # sig+ means the 2.5%ile excludes 0 for both VSM base and VSM strict;
    # nontrivial means the bootstrap median ratio VSM_strict/GEDI_strict>=0.1.
    if boot_summary_df is not None:
        bsum = boot_summary_df.set_index('stat')

        def _boot_sig(stat):
            r = bsum.loc[stat]
            return bool(np.isfinite(r['p2_5']) and r['p2_5'] > 0)

        all_pos = _boot_sig('coef_VSM_base') and _boot_sig('coef_VSM_strict')
        med_vg = float(bsum.loc['ratio_V_G', 'p50'])
        nontrivial = np.isfinite(med_vg) and med_vg >= 0.1
    else:
        all_pos = all(_sig_pos(m) for _, m in vsm_variants)
        nontrivial = np.isfinite(ratio_vg) and ratio_vg >= 0.1
    say('')
    if all_pos and nontrivial:
        bio_clause = ('and for biome' if has_biome
                      else 'and for regional structure (spatial cells)')
        if boot_summary_df is not None:
            rvg = bsum.loc['ratio_V_G']
            recov = (f'recovers {rvg["p50"]:.0%} of the lidar-detectable (GEDI) '
                     f'understory contrast (tile-block bootstrap 95% CI '
                     f'[{rvg["p2_5"]:.0%}, {rvg["p97_5"]:.0%}], '
                     f'P(>0)={rvg["p_gt0"]:.3f})')
        else:
            recov = (f'recovers ~{ratio_vg:.0%} of the lidar-detectable (GEDI) '
                     f'understory contrast')
        say('VERDICT: VSM separates sparse- vs dense-understory canopies '
            '(by LVIS RH25:RH98) at matched '
            "top height; the effect survives controlling for VSM's own and the "
            f'true top height, {bio_clause} — so it is not a denominator '
            f'artifact or an ecological prior, and {recov}. The numerator '
            f'separation is consistent across bands (>0 in '
            f'{base["VSM"]["frac_bands_pos"]:.0%} of them).')
    elif all_pos and not nontrivial:
        frac = (float(bsum.loc['ratio_V_G', 'p50'])
                if boot_summary_df is not None else ratio_vg)
        say('VERDICT (WEAK): the VSM numerator grp coefficient stays positive '
            'with a zero-excluding CI across variants, but its magnitude is a '
            f'negligible fraction ({frac:.0%}) of the GEDI-detectable '
            'contrast — statistically present, practically near-zero.')
    else:
        if boot_summary_df is not None:
            collapsed = [k for k in ('coef_VSM_base', 'coef_VSM_strict')
                         if not _boot_sig(k)]
        else:
            collapsed = [tag for tag, m in vsm_variants if not _sig_pos(m)]
        den_clause = (f'the ratio ΔU is {1 - num_share:.0%} denominator term, '
                      f'so the earlier ratio separation was largely a '
                      f'denominator artifact'
                      + (' / ecological prior' if has_biome else '')
                      if share_ok else
                      'the residual ratio ΔU is ~0 (numerator and denominator '
                      'terms cancel), i.e. no real top-height-controlled '
                      'understory signal')
        say('VERDICT (DOWNGRADED): once top height is controlled in the '
            'numerator, the VSM sparse/dense separation is NOT robustly positive '
            f'(fails at: {", ".join(collapsed)}); {den_clause}. We do not '
            'claim VSM resolves understory structure here.')

    (save_dir / 'conclusion.md').write_text('\n'.join(report) + '\n')
    print(f'\n-> {save_dir / "conclusion.md"}')
