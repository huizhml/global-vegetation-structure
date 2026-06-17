"""Does the VSM carry vertical *profile-shape* information beyond canopy-top
height? LVIS over Gabon is the independent reference; both LVIS and VSM RH
profiles are normalized by their own RH98 so the question is strictly about
shape, not height.

The analysis is a clean four-part battery (matching the study objective). It is
deliberately able to fail: the height-only baseline (Part 2) and the
height-regression residuals (Part 3) are trained on RH98 ALONE, so any
LVIS-VSM agreement that survives is shape information height cannot explain.

Part 1  Normalized-profile agreement
    P_i = RH_i / RH98 for each sensor.
    1.1  Pearson r, RMSE, MAE between LVIS and VSM for RH25/RH98, RH50/RH98,
         RH75/RH98 (+ scatter plots).
    1.2  Full-profile shape RMSE per footprint over the LVIS shape grid
         RH10..RH95 in 5-m steps (RMSE_shape = sqrt(mean_i (P_i^VSM-P_i^LVIS)^2));
         mean/median + hist.

Part 2  Shape-prediction baselines vs VSM (70/30 footprint split)
    Three baselines predict the normalized LVIS profile, each using NO S2, VSM,
    coords or neighbours, then compared on the test set to VSM-vs-LVIS shape RMSE:
      (0) mean profile  P_i = mean_train(P_i^LVIS)   — ignores RH98 (lowest bar);
      (1) linear height P_i = a_i + b_i * RH98^LVIS   — per-level linear trend;
      (2) RF height     RandomForest(RH98 -> profile) — nonlinear height model.
    For each, dRMSE = (RMSE_baseline - RMSE_VSM) / RMSE_baseline (> 0 -> VSM beats
    that baseline -> shape info beyond what height alone explains). Baseline (1)
    answers: can normalized shape be explained by a LINEAR trend in canopy height?

Part 3  Height-regression residuals
    For RH25/RH50/RH75 fit f: RH98 -> RH_x on LVIS (RandomForest, train split).
    Residuals on the test split: r^LVIS = RH_x^LVIS - f(RH98^LVIS),
    r^VSM = RH_x^VSM - f(RH98^VSM). Pearson + Spearman between r^LVIS and r^VSM
    (+ scatter). r ~ 0 -> VSM mostly reflects height; r > 0 -> VSM tracks
    shape variation beyond height.

Part 4  Height-bin comparison
    Group by lvis_RH98 (10-20, 20-30, 30-40, 40-50, 50+ m); within each bin
    compare RH25/RH98, RH50/RH98, RH75/RH98 between LVIS and VSM
    (boxplots, violin plots, per-bin correlation).

Data: the per-tile LVIS-GEDI-VSM pair parquets from
`evaluation/on_lvis.extract_vsm_on_pair_locations`
(`Gabon2016_vs_GEDI2020_stable_forest_with_vsm2020/<tile>.parquet`), columns
`lvis_RH<NN>` / `vsm_RH<NN>` (uppercase) + geometry. NOTE LVIS is recorded on a
SPARSE grid — RH10..RH95 every 5 m, plus RH96..RH100 — not every percentile, so
the shape grid defaults to those 5-m levels. Only the LVIS and VSM RH columns
are used here (GEDI is ignored).

  python -m tools.run run=vsm_profile_shape
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from const import FIGURE_SIZES, FONT_SIZES

_SENSORS = ('LVIS', 'VSM')
_KEY_LEVELS = (25, 50, 75)      # headline understory/mid-canopy metrics
_TOP = 98                       # canopy-top normalizer (RH98)


def _col(sensor: str, level: int) -> str:
    """Pair-parquet column for one sensor/RH level (LVIS & VSM are uppercase
    `RH<NN>`, no zero-padding)."""
    return f'{sensor.lower()}_RH{level}'


# ---------------------------------------------------------------------------
# Data loading: pool tiles, keep only the LVIS+VSM RH columns we need
# ---------------------------------------------------------------------------
def _load_profiles(pairs_dir: Path, glob: str, shape_levels: tuple,
                   say) -> pd.DataFrame:
    """Pool the per-tile pair parquets into a flat frame holding lvis_/vsm_ RH
    columns for every level in `shape_levels` plus RH98 (and the key levels,
    which are inside shape_levels). Fails loudly on any missing column. Stamps
    each row with its `tile` (filename stem) for reference."""
    files = sorted(pairs_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f'No files matching {glob!r} under {pairs_dir}')
    levels = sorted(set(shape_levels) | {_TOP} | set(_KEY_LEVELS))
    rh_cols = [_col(s, lv) for s in _SENSORS for lv in levels]
    available = set(pq.ParquetFile(files[0]).schema.names)
    missing = sorted(set(rh_cols) - available)
    if missing:
        raise KeyError(
            f'Pair parquet {files[0].name} is missing required columns '
            f'{missing[:8]}{" ..." if len(missing) > 8 else ""} '
            f'({len(missing)} total).\nAvailable columns:\n  {sorted(available)}'
            f'\nAlign column names before rerunning (do not guess).')
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=rh_cols)
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
        f'{len(levels)} RH levels x 2 sensors.')
    return df


def _clean_and_normalize(df: pd.DataFrame, shape_levels: tuple,
                         rh98_min: float, say) -> dict:
    """Clip negative RH to 0, keep footprints with BOTH sensors' RH98 > rh98_min
    and all needed RH finite, and build per-sensor normalized profiles
    P_i = RH_i / RH98. Returns a dict with the kept frame, the normalized
    matrices (n x len(shape_levels)) and per-key-level ratios."""
    levels = sorted(set(shape_levels) | {_TOP} | set(_KEY_LEVELS))
    rh_cols = [_col(s, lv) for s in _SENSORS for lv in levels]
    work = df.replace([np.inf, -np.inf], np.nan).copy()
    for c in rh_cols:                                # clip negative RH to 0
        work[c] = work[c].clip(lower=0)

    n0 = len(work)
    finite = work[rh_cols].notna().all(axis=1)
    keep = finite.copy()
    for s in _SENSORS:
        keep &= work[_col(s, _TOP)] > rh98_min
    out = work[keep].reset_index(drop=True)
    say(f'Cleaning: {n0:,} footprints; dropped {int((~keep).sum()):,} '
        f'(non-finite or RH98 <= {rh98_min:g} m on either sensor) -> '
        f'{len(out):,} kept.')

    shape_levels = tuple(shape_levels)
    res = {'df': out, 'shape_levels': shape_levels}
    for s in _SENSORS:
        top = out[_col(s, _TOP)].to_numpy()
        prof = np.column_stack([out[_col(s, lv)].to_numpy() for lv in shape_levels])
        res[f'P_{s}'] = prof / top[:, None]
        res[f'top_{s}'] = top
        for lv in _KEY_LEVELS:
            res[f'ratio_{s}_{lv}'] = out[_col(s, lv)].to_numpy() / top
            res[f'raw_{s}_{lv}'] = out[_col(s, lv)].to_numpy()
    return res


def _shape_rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-footprint RMSE between two normalized profile matrices (rows =
    footprints, cols = RH levels)."""
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str,
             title: str, save_path: Path, ann: str = '') -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['square'])
    ax.scatter(x, y, s=4, alpha=0.15, color='C0', edgecolors='none', zorder=2)
    lo = float(min(np.nanmin(x), np.nanmin(y)))
    hi = float(max(np.nanmax(x), np.nanmax(y)))
    ax.plot([lo, hi], [lo, hi], '--', color='k', lw=1, zorder=3, label='1:1')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES['label'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    ax.set_title(title, fontsize=FONT_SIZES['title'])
    if ann:
        ax.text(0.04, 0.96, ann, transform=ax.transAxes, va='top', ha='left',
                fontsize=FONT_SIZES['legend'],
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _hist_shape_rmse(rmse: np.ndarray, mean_v: float, median_v: float,
                     save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
    ax.hist(rmse, bins=60, color='C3', alpha=0.8, zorder=2)
    ax.axvline(mean_v, color='k', ls='-', lw=1.5, label=f'mean {mean_v:.3f}')
    ax.axvline(median_v, color='k', ls='--', lw=1.5,
               label=f'median {median_v:.3f}')
    ax.set_xlabel('per-footprint shape RMSE (RH1..RH95, normalized)',
                  fontsize=FONT_SIZES['label'])
    ax.set_ylabel('count', fontsize=FONT_SIZES['label'])
    ax.set_title('LVIS vs VSM full-profile shape RMSE',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'])
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _heightbin_plots(bin_data: list, level: int, save_dir: Path) -> None:
    """Side-by-side LVIS/VSM box + violin of RH<level>/RH98 across height bins."""
    labels = [b['label'] for b in bin_data]
    lvis = [b[f'ratio_LVIS_{level}'] for b in bin_data]
    vsm = [b[f'ratio_VSM_{level}'] for b in bin_data]
    x = np.arange(len(labels))
    for kind in ('box', 'violin'):
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
        if kind == 'box':
            ax.boxplot(lvis, positions=x - 0.18, widths=0.3, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor='C2', alpha=0.6),
                       medianprops=dict(color='k'))
            ax.boxplot(vsm, positions=x + 0.18, widths=0.3, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor='C3', alpha=0.6),
                       medianprops=dict(color='k'))
        else:
            vp_l = ax.violinplot(lvis, positions=x - 0.18, widths=0.3,
                                 showmedians=True)
            vp_v = ax.violinplot(vsm, positions=x + 0.18, widths=0.3,
                                 showmedians=True)
            for b in vp_l['bodies']:
                b.set_facecolor('C2'); b.set_alpha(0.6)
            for b in vp_v['bodies']:
                b.set_facecolor('C3'); b.set_alpha(0.6)
        ax.plot([], [], color='C2', lw=6, alpha=0.6, label='LVIS')
        ax.plot([], [], color='C3', lw=6, alpha=0.6, label='VSM')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right',
                           fontsize=FONT_SIZES['ticks'])
        ax.set_xlabel('LVIS canopy-top height bin RH98 (m)',
                      fontsize=FONT_SIZES['label'])
        ax.set_ylabel(f'RH{level} / RH98', fontsize=FONT_SIZES['label'])
        ax.set_title(f'RH{level}/RH98 by height bin ({kind})',
                     fontsize=FONT_SIZES['title'])
        ax.legend(fontsize=FONT_SIZES['legend'])
        ax.grid(True, ls='--', alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(save_dir / f'heightbin_{kind}_RH{level}.png',
                    bbox_inches='tight', dpi=150)
        plt.close(fig)


# ===========================================================================
# Main entrypoint (registered as run=vsm_profile_shape)
# ===========================================================================
def vsm_profile_shape_analysis(
        pairs_dir: str,
        save_dir: str,
        pairs_glob: str = '*.parquet',
        shape_levels: tuple = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
                               70, 75, 80, 85, 90, 95),
        rh98_min: float = 5.0,
        test_size: float = 0.3,
        group_split: bool = True,
        n_estimators: int = 100,
        rf_min_samples_leaf: int = 50,
        height_bin_edges: tuple = (10.0, 20.0, 30.0, 40.0, 50.0),
        random_state: int = 0,
        **kwargs) -> None:
    """Evaluate whether VSM captures vertical profile shape beyond canopy-top
    height, using LVIS over Gabon as the independent reference.

    Args:
        pairs_dir: dir of per-tile LVIS-GEDI-VSM pair parquets carrying
            `lvis_RH<NN>` / `vsm_RH<NN>` (NN = 0..100).
        save_dir: output dir for figures, CSVs, conclusion.md.
        pairs_glob: glob under pairs_dir. Default '*.parquet'.
        shape_levels: RH levels (excluding the RH98 normalizer) used for the
            full-profile shape RMSE and the height-only baseline target. Default
            is the LVIS 5-m grid RH10..RH95; every level must exist on BOTH
            sensors (the loader fails loudly otherwise). RH96..RH100 are left
            out on purpose (too close to the RH98 normalizer).
        rh98_min: keep footprints with BOTH sensors' RH98 > this (m). Default 5.
        test_size: held-out fraction for the baselines / residuals (fraction of
            tiles when group_split, else of footprints). Default 0.3.
        group_split: if True, split train/test by S2 tile (GroupShuffleSplit on
            the `tile` column) so no tile straddles the split — guards against
            spatial leakage. Falls back to a footprint-level random split if
            fewer than 2 tiles are present. Default True.
        n_estimators: trees in the RandomForest models. Default 300.
        rf_min_samples_leaf: min samples per leaf for ALL RandomForests (the
            Part 2 rf_height baseline and the Part 3 RH_x|RH98 fit). With a
            single feature (RH98) the default leaf=1 RF overfits and, under the
            grouped/spatial split, generalizes WORSE than the linear baseline;
            this regularizes it into a fair smooth nonlinear height model. The
            Part 2 log prints the rf_height train-vs-test shape RMSE so the
            overfit gap is visible. Default 50.
        height_bin_edges: lower edges of the RH98 height bins; the last opens to
            +inf. Default (10,20,30,40,50) -> 10-20,20-30,30-40,40-50,50+.
        random_state: RNG seed for the split and the forests.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    report: list = []

    def say(line: str = '') -> None:
        print(line)
        report.append(line)

    say('=' * 74)
    say('VSM profile-shape test — does VSM encode vertical shape beyond RH98?')
    say('=' * 74)
    shape_levels = tuple(int(lv) for lv in shape_levels)
    say(f'Reference = LVIS (Gabon). Normalize each profile by its own RH98.')
    say(f'Shape grid ({len(shape_levels)} levels): '
        f'RH{shape_levels[0]}..RH{shape_levels[-1]}; key levels {_KEY_LEVELS}; '
        f'rh98_min={rh98_min:g} m; test_size={test_size:g}; seed={random_state}.')
    say('')

    df = _load_profiles(pairs_dir, pairs_glob, shape_levels, say)
    data = _clean_and_normalize(df, shape_levels, rh98_min, say)
    n = len(data['df'])
    summary_rows: list = []

    # -- Part 1.1: normalized key-level agreement -------------------------
    say('')
    say('Part 1.1: normalized RH25/RH50/RH75 agreement (LVIS vs VSM)')
    for lv in _KEY_LEVELS:
        xl = data[f'ratio_LVIS_{lv}']
        yv = data[f'ratio_VSM_{lv}']
        r = float(stats.pearsonr(xl, yv)[0])
        rmse = float(np.sqrt(np.mean((yv - xl) ** 2)))
        mae = float(np.mean(np.abs(yv - xl)))
        say(f'  RH{lv}/RH98: Pearson r={r:+.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}')
        summary_rows.append({'part': '1.1', 'metric': f'RH{lv}/RH98',
                             'pearson_r': r, 'rmse': rmse, 'mae': mae, 'n': n})
        _scatter(xl, yv, f'LVIS RH{lv}/RH98', f'VSM RH{lv}/RH98',
                 f'Normalized RH{lv} (LVIS vs VSM)',
                 save_dir / f'scatter_normalized_RH{lv}.png',
                 ann=f'r={r:+.3f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}\nN={n:,}')

    # -- Part 1.2: full-profile shape RMSE --------------------------------
    say('')
    say(f'Part 1.2: full-profile shape RMSE over RH{shape_levels[0]}..'
        f'RH{shape_levels[-1]} ({len(shape_levels)} levels)')
    rmse_shape = _shape_rmse(data['P_VSM'], data['P_LVIS'])
    mean_sr, med_sr = float(np.mean(rmse_shape)), float(np.median(rmse_shape))
    say(f'  mean shape RMSE   = {mean_sr:.4f}')
    say(f'  median shape RMSE = {med_sr:.4f}')
    summary_rows.append({'part': '1.2', 'metric': 'shape_RMSE_all',
                         'mean': mean_sr, 'median': med_sr, 'n': n})
    _hist_shape_rmse(rmse_shape, mean_sr, med_sr,
                     save_dir / 'hist_shape_rmse.png')

    # -- Part 2: shape-prediction baselines vs VSM ------------------------
    say('')
    say('Part 2: shape-prediction baselines vs VSM (held-out test set)')
    idx = np.arange(n)
    tiles = data['df']['tile'].to_numpy()
    use_group = group_split and pd.unique(tiles).size >= 2
    if use_group:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=random_state)
        tr, te = next(gss.split(idx, groups=tiles))
        say(f'  grouped split by tile: {pd.unique(tiles[tr]).size} train / '
            f'{pd.unique(tiles[te]).size} test tiles -> {len(tr):,} train / '
            f'{len(te):,} test footprints (no tile straddles the split).')
    else:
        if group_split:
            say('  group_split requested but < 2 tiles present -> '
                'falling back to a footprint-level random split.')
        tr, te = train_test_split(idx, test_size=test_size,
                                  random_state=random_state)
        say(f'  footprint split: {len(tr):,} train / {len(te):,} test.')
    P_tr, P_te = data['P_LVIS'][tr], data['P_LVIS'][te]
    h_tr, h_te = data['top_LVIS'][tr][:, None], data['top_LVIS'][te][:, None]

    # (0) mean profile (ignores RH98); (1) per-level linear in RH98;
    # (2) RandomForest in RH98. Each predicts the normalized LVIS profile.
    pred_mean = np.broadcast_to(P_tr.mean(axis=0), P_te.shape)
    pred_lin = LinearRegression().fit(h_tr, P_tr).predict(h_te)
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               min_samples_leaf=rf_min_samples_leaf,
                               random_state=random_state,
                               n_jobs=-1).fit(h_tr, P_tr)
    pred_rf = rf.predict(h_te)
    rf_train_rmse = float(_shape_rmse(rf.predict(h_tr), P_tr).mean())
    v_mean = float(_shape_rmse(data['P_VSM'][te], P_te).mean())

    baselines = (
        ('mean_profile', pred_mean, 'mean profile (no RH98, lowest bar)'),
        ('linear_height', pred_lin, 'per-level linear in RH98'),
        ('rf_height', pred_rf, 'RandomForest in RH98'),
    )
    baseline_rmse = {}
    say(f'  VSM shape RMSE (test) = {v_mean:.4f}')
    for key, pred, desc in baselines:
        b_mean = float(_shape_rmse(pred, P_te).mean())
        d_rmse = (b_mean - v_mean) / b_mean if b_mean > 1e-12 else np.nan
        baseline_rmse[key] = (b_mean, d_rmse)
        say(f'  {desc:<34} RMSE={b_mean:.4f}  '
            f'dRMSE=(base-VSM)/base={d_rmse:+.3f}  '
            f'({"VSM better" if d_rmse > 0 else "VSM no better"})')
        summary_rows.append({'part': '2', 'metric': 'shape_RMSE_test',
                             'baseline_type': key, 'baseline': b_mean,
                             'vsm': v_mean, 'delta_rmse': d_rmse,
                             'n': int(len(te))})
    rf_test_rmse = baseline_rmse['rf_height'][0]
    say(f'  [diag] rf_height (min_samples_leaf={rf_min_samples_leaf}) train '
        f'shape RMSE={rf_train_rmse:.4f} vs test={rf_test_rmse:.4f}; a large '
        f'train<<test gap = RF overfitting RH98 -> raise rf_min_samples_leaf.')
    # Verdict references the strongest (hardest-to-beat) height baseline.
    d_rmse = baseline_rmse['rf_height'][1]

    # -- Part 3: height-regression residuals ------------------------------
    say('')
    say('Part 3: residuals after regressing RH_x on RH98 (f fit on LVIS train)')
    resid = {}
    for lv in _KEY_LEVELS:
        f_x = RandomForestRegressor(n_estimators=n_estimators,
                                    min_samples_leaf=rf_min_samples_leaf,
                                    random_state=random_state, n_jobs=-1)
        f_x.fit(data['top_LVIS'][tr][:, None], data[f'raw_LVIS_{lv}'][tr])
        r_lvis = (data[f'raw_LVIS_{lv}'][te]
                  - f_x.predict(data['top_LVIS'][te][:, None]))
        r_vsm = (data[f'raw_VSM_{lv}'][te]
                 - f_x.predict(data['top_VSM'][te][:, None]))
        pear = float(stats.pearsonr(r_lvis, r_vsm)[0])
        spear = float(stats.spearmanr(r_lvis, r_vsm)[0])
        resid[lv] = pear
        say(f'  RH{lv}: corr(r^LVIS, r^VSM)  Pearson={pear:+.3f}  '
            f'Spearman={spear:+.3f}')
        summary_rows.append({'part': '3', 'metric': f'resid_RH{lv}',
                             'pearson_r': pear, 'spearman_r': spear,
                             'n': int(len(te))})
        _scatter(r_lvis, r_vsm, f'LVIS RH{lv} residual (m)',
                 f'VSM RH{lv} residual (m)',
                 f'Height-regression residuals RH{lv}',
                 save_dir / f'scatter_residual_RH{lv}.png',
                 ann=f'Pearson={pear:+.3f}\nSpearman={spear:+.3f}\n'
                     f'N={len(te):,}')

    # -- Part 4: height-bin comparison ------------------------------------
    say('')
    say('Part 4: height-bin comparison of RH25/50/75 ratios (LVIS vs VSM)')
    edges = list(height_bin_edges) + [np.inf]
    h = data['top_LVIS']
    bin_data = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (h >= lo) & (h < hi)
        if int(m.sum()) < 30:
            continue
        label = f'{lo:g}+' if not np.isfinite(hi) else f'{lo:g}-{hi:g}'
        entry = {'label': label, 'n': int(m.sum())}
        line = f'  [{label:>7}] N={int(m.sum()):>6}'
        for lv in _KEY_LEVELS:
            rl = data[f'ratio_LVIS_{lv}'][m]
            rv = data[f'ratio_VSM_{lv}'][m]
            entry[f'ratio_LVIS_{lv}'] = rl
            entry[f'ratio_VSM_{lv}'] = rv
            r = float(stats.pearsonr(rl, rv)[0]) if m.sum() > 2 else np.nan
            rmse = float(np.sqrt(np.mean((rv - rl) ** 2)))
            line += f' | RH{lv}: r={r:+.3f} RMSE={rmse:.3f}'
            summary_rows.append({'part': '4', 'metric': f'RH{lv}/RH98',
                                 'height_bin': label, 'pearson_r': r,
                                 'rmse': rmse, 'n': int(m.sum())})
        say(line)
        bin_data.append(entry)
    for lv in _KEY_LEVELS:
        _heightbin_plots(bin_data, lv, save_dir)

    # -- Deliverable 1: summary table -------------------------------------
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(save_dir / 'summary_metrics.csv', index=False)

    # -- Deliverable 6: interpretation ------------------------------------
    say('')
    say('=' * 74)
    say('Interpretation: does VSM contain vertical profile-shape information '
        'beyond RH98?')
    say('=' * 74)
    pos_resid = [lv for lv in _KEY_LEVELS if resid[lv] > 0.1]
    beats_height = np.isfinite(d_rmse) and d_rmse > 0
    say(f'- Shape baselines (test shape RMSE; VSM={v_mean:.4f}): '
        + ', '.join(f'{k}={baseline_rmse[k][0]:.4f} '
                    f'(dRMSE={baseline_rmse[k][1]:+.3f})'
                    for k, _, _ in baselines) + '.')
    say('  ' + ('VSM beats the nonlinear height-only baseline -> it carries '
                'shape a height model cannot reproduce.'
                if beats_height else
                'VSM does not beat the nonlinear height-only baseline -> its '
                'shape is largely reproducible from height alone.'))
    say(f'- Residual correlations (shape variation after removing height): '
        + ', '.join(f'RH{lv} r={resid[lv]:+.3f}' for lv in _KEY_LEVELS) + '.')
    if pos_resid:
        say(f'  Positive residual correlation at RH{"/".join(map(str, pos_resid))}'
            f' -> VSM tracks lower/mid-canopy shape beyond canopy height.')
    else:
        say('  Residual correlations near zero -> VSM mostly reflects canopy '
            'height, with little independent shape signal.')
    verdict = ('YES — VSM encodes vertical profile shape (esp. RH25-RH75) '
               'beyond canopy-top height.'
               if (beats_height and pos_resid) else
               'PARTIAL — evidence is mixed; see per-metric numbers above.'
               if (beats_height or pos_resid) else
               'NO — VSM profile shape is largely explained by canopy height.')
    say(f'- Verdict: {verdict}')
    say('')
    say('Outputs: summary_metrics.csv, scatter_normalized_RH{25,50,75}.png, '
        'scatter_residual_RH{25,50,75}.png, hist_shape_rmse.png, '
        'heightbin_{box,violin}_RH{25,50,75}.png, conclusion.md')

    (save_dir / 'conclusion.md').write_text('\n'.join(report) + '\n')
    print(f'\n-> {save_dir / "conclusion.md"}')
