"""GEDI penetration-bias diagnostic.

Question: does the VSM optical model shed the systematic penetration bias
that raw GEDI carries on low-sensitivity (poor ground-detection) shots?

Context (for reading the output, not the code):
  - VSM is an optical model learned from GEDI, trained ONLY on shots with
    GEDI sensitivity > 0.95 ("clean-ground" labels).
  - The validation set is NOT filtered, so it still contains low-sensitivity
    shots where raw GEDI carries a penetration bias.
  - LVIS is an independent airborne reference that penetrates the canopy
    better, so we treat it as ground truth.
  - Hypothesis: because VSM only saw clean-ground labels, its error should
    stay flat across sensitivity, while raw GEDI's error sinks (underestimates
    height) in the low-sensitivity regime.

Data: per-tile LVIS-GEDI-VSM pair parquets from
`evaluation/on_lvis.extract_vsm_on_pair_locations`
(`Gabon2016_vs_GEDI2020_stable_forest_with_vsm2020/<tile>.parquet`). Column
naming (note the case split between sensors):
  - LVIS RH : `lvis_RH25`, `lvis_RH98`   (uppercase, sparse percentiles)
  - GEDI RH : `gedi_rh25`, `gedi_rh98`   (lowercase, dense rh0..rh100)
  - VSM RH  : `vsm_RH25`,  `vsm_RH98`    (uppercase, matches LVIS set)
  - sensitivity : `gedi_sensitivity`
  - height control / "LVIS top height" : `lvis_RH98` (no `lvis_top_h` column
    exists; RH98 is the codebase's `control_rh` convention).

Error sign convention: err = predicted - LVIS, so a negative ME = underestimate.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from const import FIGURE_SIZES, FONT_SIZES


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------
def _sensor_cols(rh: str) -> tuple:
    """Map a canonical RH name ('RH25') to the (lvis, gedi, vsm) column
    names. GEDI uses lowercase dense `rh<NN>`; LVIS and VSM use uppercase
    `RH<NN>`."""
    return f'lvis_{rh}', f'gedi_{rh.lower()}', f'vsm_{rh}'


# ---------------------------------------------------------------------------
# Bootstrap CIs (vectorized; shared resample indices)
# ---------------------------------------------------------------------------
def _boot_idx(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def _ci_mean(x: np.ndarray, n_boot: int, rng: np.random.Generator,
             alpha: float = 0.05) -> tuple:
    """Percentile bootstrap CI for the mean of `x`."""
    if len(x) < 2:
        return np.nan, np.nan
    idx = _boot_idx(len(x), n_boot, rng)
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    am, bm = a - a.mean(), b - b.mean()
    den = np.sqrt((am ** 2).sum() * (bm ** 2).sum())
    return float((am * bm).sum() / den) if den > 0 else np.nan


def _ci_corr(a: np.ndarray, b: np.ndarray, n_boot: int,
             rng: np.random.Generator, alpha: float = 0.05) -> tuple:
    """Percentile bootstrap CI for Pearson r between paired (a, b)."""
    if len(a) < 3:
        return np.nan, np.nan
    idx = _boot_idx(len(a), n_boot, rng)
    ax, bx = a[idx], b[idx]                       # (n_boot, n)
    am = ax - ax.mean(axis=1, keepdims=True)
    bm = bx - bx.mean(axis=1, keepdims=True)
    num = (am * bm).sum(axis=1)
    den = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        rs = np.where(den > 0, num / den, np.nan)
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return np.nan, np.nan
    lo, hi = np.quantile(rs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Quantile binning with a min-N merge safety net
# ---------------------------------------------------------------------------
def _make_bins(sens: np.ndarray, target_n_bins: int, min_bin_n: int) -> tuple:
    """Equal-sample quantile bins over `sens`, merging any bin below
    `min_bin_n` into a neighbor by dropping interior edges. Returns
    (labels, edges)."""
    qs = np.linspace(0, 1, target_n_bins + 1)
    edges = list(np.unique(np.quantile(sens, qs)))
    if len(edges) < 2:
        return np.zeros(len(sens), dtype=int), np.array([sens.min(), sens.max()])
    while True:
        labels = np.digitize(sens, edges[1:-1], right=False)
        counts = np.bincount(labels, minlength=len(edges) - 1)
        small = np.where(counts < min_bin_n)[0]
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
    return labels, np.array(edges)


# ---------------------------------------------------------------------------
# Per-group stats (one predictor)
# ---------------------------------------------------------------------------
def _group_stats(lvis: np.ndarray, pred: np.ndarray, sens: np.ndarray,
                 n_boot: int, rng: np.random.Generator, tag: str) -> dict:
    """ME, median error, Pearson r, and bootstrap 95% CIs for one
    predictor on one (already-finite) group of pairs."""
    err = pred - lvis
    me_lo, me_hi = _ci_mean(err, n_boot, rng)
    r = _pearson(pred, lvis)
    r_lo, r_hi = _ci_corr(pred, lvis, n_boot, rng)
    return {
        f'me_{tag}': float(err.mean()),
        f'me_{tag}_lo': me_lo,
        f'me_{tag}_hi': me_hi,
        f'medianerr_{tag}': float(np.median(err)),
        f'r_{tag}': r,
        f'r_{tag}_lo': r_lo,
        f'r_{tag}_hi': r_hi,
    }


def _both_predictor_stats(sub: pd.DataFrame, lvis_col: str, gedi_col: str,
                          vsm_col: str, sens_col: str, n_boot: int,
                          rng: np.random.Generator) -> dict:
    """VSM and GEDI stats on one group, plus sensitivity summary + N."""
    lvis = sub[lvis_col].to_numpy()
    row = {
        'n': int(len(sub)),
        'sens_median': float(np.median(sub[sens_col].to_numpy())),
        'sens_min': float(sub[sens_col].min()),
        'sens_max': float(sub[sens_col].max()),
    }
    row.update(_group_stats(lvis, sub[vsm_col].to_numpy(),
                            sub[sens_col].to_numpy(), n_boot, rng, 'vsm'))
    row.update(_group_stats(lvis, sub[gedi_col].to_numpy(),
                            sub[sens_col].to_numpy(), n_boot, rng, 'gedi'))
    return row


# ---------------------------------------------------------------------------
# Fork plot (ME on top, bin-wise r on bottom)
# ---------------------------------------------------------------------------
def _fork_plot(bins_df: pd.DataFrame, rh: str, threshold: float,
               save_path: Path, title_suffix: str = '') -> None:
    x = bins_df['sens_median'].to_numpy()
    fig, (ax_me, ax_r) = plt.subplots(
        2, 1, figsize=FIGURE_SIZES['medium'], sharex=True)

    for ax, metric, ylabel in [
        (ax_me, 'me', 'ME (pred - LVIS, m)'),
        (ax_r, 'r', 'bin-wise Pearson r'),
    ]:
        for tag, color, label in [('gedi', 'C2', 'GEDI'), ('vsm', 'C1', 'VSM')]:
            y = bins_df[f'{metric}_{tag}'].to_numpy()
            lo = bins_df[f'{metric}_{tag}_lo'].to_numpy()
            hi = bins_df[f'{metric}_{tag}_hi'].to_numpy()
            ax.plot(x, y, '-o', color=color, label=label, zorder=3)
            ax.fill_between(x, lo, hi, color=color, alpha=0.2, zorder=1)
        ax.axvline(threshold, ls='--', color='k', alpha=0.7, zorder=2)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
        ax.grid(True, ls='--', alpha=0.4)
        ax.set_axisbelow(True)
    ax_me.axhline(0, color='gray', lw=0.8, zorder=0)

    # Shade the OOD region (sensitivity < threshold = outside VSM training).
    ood = bins_df['sens_median'] < threshold
    if ood.any():
        for ax in (ax_me, ax_r):
            ax.axvspan(ax.get_xlim()[0], threshold, color='red', alpha=0.05,
                       zorder=0)
        ax_me.text(0.01, 0.97, 'OOD for VSM training\n(sensitivity < %.2f)'
                   % threshold, transform=ax_me.transAxes, va='top',
                   ha='left', fontsize=FONT_SIZES['legend'], color='firebrick')

    # Annotate N per bin along the top axis.
    y_top = ax_me.get_ylim()[1]
    for xi, ni in zip(x, bins_df['n'].to_numpy()):
        ax_me.annotate(f'N={ni}', (xi, y_top), textcoords='offset points',
                       xytext=(0, 2), ha='center', va='bottom',
                       fontsize=7, color='dimgray', rotation=45)

    ax_r.set_xlabel('GEDI sensitivity (bin median)',
                    fontsize=FONT_SIZES['label'])
    ax_me.legend(fontsize=FONT_SIZES['legend'], loc='lower right')
    ax_me.set_title(f'{rh} penetration-bias fork{title_suffix}',
                    fontsize=FONT_SIZES['title'])
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-RH binning driver (used for the full set and the height-band subset)
# ---------------------------------------------------------------------------
def _binned_table(df: pd.DataFrame, rh: str, sens_col: str, threshold: float,
                  target_n_bins: int, min_bin_n: int, n_boot: int,
                  rng: np.random.Generator) -> pd.DataFrame:
    lvis_col, gedi_col, vsm_col = _sensor_cols(rh)
    labels, edges = _make_bins(df[sens_col].to_numpy(), target_n_bins,
                               min_bin_n)
    rows = []
    for b in range(len(edges) - 1):
        sub = df[labels == b]
        if len(sub) < 3:
            continue
        row = _both_predictor_stats(sub, lvis_col, gedi_col, vsm_col,
                                    sens_col, n_boot, rng)
        row['bin'] = b
        row['edge_lo'] = float(edges[b])
        row['edge_hi'] = float(edges[b + 1])
        row['ood_for_vsm'] = bool(row['sens_median'] < threshold)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('sens_median').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5: mechanism decomposition (de-noise vs penetration de-bias)
# ---------------------------------------------------------------------------
def _advantage(df: pd.DataFrame, rh: str) -> dict:
    lvis_col, gedi_col, vsm_col = _sensor_cols(rh)
    lvis = df[lvis_col].to_numpy()
    vsm, gedi = df[vsm_col].to_numpy(), df[gedi_col].to_numpy()
    r_vsm, r_gedi = _pearson(vsm, lvis), _pearson(gedi, lvis)
    me_vsm, me_gedi = float((vsm - lvis).mean()), float((gedi - lvis).mean())
    return {
        'n': int(len(df)),
        'r_vsm': r_vsm, 'r_gedi': r_gedi, 'delta_r': r_vsm - r_gedi,
        'me_vsm': me_vsm, 'me_gedi': me_gedi,
        'delta_me': me_gedi - me_vsm,        # GEDI bias minus VSM bias
        'abs_me_vsm': abs(me_vsm), 'abs_me_gedi': abs(me_gedi),
        'delta_abs_me': abs(me_gedi) - abs(me_vsm),  # bias-magnitude reduction
    }


# ---------------------------------------------------------------------------
# Step 7: ME heatmap (sensitivity bin x RH metric)
# ---------------------------------------------------------------------------
def _me_heatmap(per_rh_bins: dict, threshold: float, save_path: Path) -> None:
    rhs = list(per_rh_bins.keys())
    # Use the bin order of the first RH for column labels; bins are aligned
    # because every RH is binned on the same pooled sensitivity distribution.
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES['small'],
                             constrained_layout=True)
    for ax, tag, name in [(axes[0], 'gedi', 'GEDI'), (axes[1], 'vsm', 'VSM')]:
        mat, col_labels = [], None
        for rh in rhs:
            bdf = per_rh_bins[rh]
            mat.append(bdf[f'me_{tag}'].to_numpy())
            col_labels = [f'{m:.2f}' for m in bdf['sens_median']]
        mat = np.array(mat)
        vmax = np.nanmax(np.abs(mat)) if mat.size else 1.0
        im = ax.imshow(mat, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(rhs)))
        ax.set_yticklabels(rhs, fontsize=9)
        ax.set_xlabel('sensitivity (bin median)', fontsize=FONT_SIZES['ticks'])
        ax.set_title(f'{name} ME', fontsize=FONT_SIZES['label'])
        fig.colorbar(im, ax=ax, fraction=0.046, label='ME (m)')
    fig.suptitle('Penetration signature: ME by sensitivity bin x RH',
                 fontsize=FONT_SIZES['title'])
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Main entrypoint (registered as run=gedi_penetration_bias)
# ===========================================================================
def diagnose_gedi_penetration_bias(
        pairs_dir: str,
        save_dir: str,
        rh_metrics: list = None,
        sensitivity_col: str = 'gedi_sensitivity',
        height_control_col: str = 'lvis_RH98',
        sensitivity_threshold: float = 0.95,
        target_n_bins: int = 8,
        min_bin_n: int = 200,
        n_boot: int = 1000,
        height_band: tuple = (20.0, 30.0),
        make_heatmap: bool = True,
        pairs_glob: str = '*.parquet',
        random_state: int = 0,
        **kwargs) -> None:
    """Diagnose whether VSM sheds raw GEDI's low-sensitivity penetration bias.

    Pools the per-tile LVIS-GEDI-VSM pair parquets, then for each requested
    RH metric: bins by GEDI sensitivity, computes VSM vs GEDI ME / median
    error / bin-wise Pearson r (with 1000x bootstrap 95% CIs), and emits a
    fork plot (ME on top, r on bottom). Also produces a sensitivity<0.95 vs
    >=0.95 binary split, a mechanism-decomposition table (de-noise vs
    penetration de-bias), an optional height-confound-controlled repeat, an
    optional ME heatmap, and a data-driven conclusion.

    Args:
        pairs_dir: dir of LVIS-GEDI-VSM pair parquets
            (`Gabon2016_vs_GEDI2020_stable_forest_with_vsm2020`).
        save_dir: where figures, CSVs, and the conclusion are written.
        rh_metrics: canonical RH names (uppercase, e.g. ['RH25', 'RH98']).
            Defaults to ['RH25', 'RH98'].
        sensitivity_col: GEDI sensitivity column. Default `gedi_sensitivity`.
        height_control_col: LVIS top-height column used for the step-6
            height-band confound control. Default `lvis_RH98` (no
            `lvis_top_h` column exists). Set to null to skip step 6.
        sensitivity_threshold: clean-ground / training cutoff. Default 0.95.
        target_n_bins: number of equal-sample quantile bins. Default 8.
        min_bin_n: minimum pairs per bin; smaller bins merge into a
            neighbor. Default 200.
        n_boot: bootstrap iterations for the 95% CIs. Default 1000.
        height_band: (min, max) LVIS `height_control_col` band for step 6.
        make_heatmap: emit the step-7 ME heatmap. Default True.
        pairs_glob: glob under `pairs_dir`. Default `*.parquet`.
        random_state: RNG seed for the bootstrap. Default 0.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    rh_metrics = list(rh_metrics) if rh_metrics else ['RH25', 'RH98']
    rng = np.random.default_rng(random_state)
    report: list = []                       # accumulates conclusion lines

    def say(line: str = '') -> None:
        print(line)
        report.append(line)

    # --- locate + schema-validate -----------------------------------------
    files = sorted(pairs_dir.glob(pairs_glob))
    if not files:
        raise FileNotFoundError(
            f'No files matching {pairs_glob!r} under {pairs_dir}')
    available = set(pq.ParquetFile(files[0]).schema.names)

    needed = {'tile_id', sensitivity_col}
    for rh in rh_metrics:
        needed.update(_sensor_cols(rh))
    use_height = height_control_col is not None
    if use_height and height_control_col in available:
        needed.add(height_control_col)
    elif use_height:
        print(f'WARNING: height_control_col {height_control_col!r} not in '
              f'parquet; skipping step 6 (height confound control).')
        use_height = False

    missing = sorted(needed - available)
    if missing:
        raise KeyError(
            f'Pair parquet {files[0].name} is missing required columns '
            f'{missing}. Available columns:\n  {sorted(available)}\n'
            f'Align column names before rerunning (do not guess).')

    df = pd.concat(
        [pd.read_parquet(f, columns=sorted(needed)) for f in files],
        ignore_index=True)
    say('=' * 72)
    say('GEDI penetration-bias diagnostic')
    say('=' * 72)
    say(f'Loaded {len(df):,} pairs from {len(files)} tiles in {pairs_dir.name}')
    say(f'RH metrics: {rh_metrics} | sensitivity: {sensitivity_col} | '
        f'threshold: {sensitivity_threshold}')

    # ======================================================================
    # Step 1: sensitivity health check
    # ======================================================================
    sens = df[sensitivity_col].to_numpy()
    finite = np.isfinite(sens)
    sens = sens[finite]
    n_total = len(sens)
    n_low = int((sens < sensitivity_threshold).sum())
    frac_low = n_low / n_total if n_total else float('nan')
    say('')
    say('--- Step 1: sensitivity health check ---')
    say(f'Total finite-sensitivity pairs: {n_total:,}')
    say(f'sensitivity < {sensitivity_threshold}: {n_low:,} '
        f'({frac_low:.1%})')
    say(f'sensitivity range: [{sens.min():.3f}, {sens.max():.3f}], '
        f'median {np.median(sens):.3f}')
    if n_low < 500:
        say(f'WARNING: only {n_low} low-sensitivity pairs (<500) — the '
            f'low-sensitivity bins may be statistically unstable.')

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['small'])
    ax.hist(sens, bins=60, color='C0', alpha=0.8)
    ax.axvline(sensitivity_threshold, ls='--', color='k',
               label=f'threshold {sensitivity_threshold}')
    ax.set_xlabel('GEDI sensitivity', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('count', fontsize=FONT_SIZES['label'])
    ax.set_title(f'GEDI sensitivity (N={n_total:,}, '
                 f'{frac_low:.1%} below {sensitivity_threshold})',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'])
    fig.tight_layout()
    hist_path = save_dir / 'step1_sensitivity_hist.png'
    fig.savefig(hist_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    say(f'-> {hist_path.name}')

    # ======================================================================
    # Steps 2-4: per-RH binned errors + fork plots
    # ======================================================================
    per_rh_bins: dict = {}
    binary_rows = []
    for rh in rh_metrics:
        lvis_col, gedi_col, vsm_col = _sensor_cols(rh)
        cols = [lvis_col, gedi_col, vsm_col, sensitivity_col]
        sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < max(min_bin_n, 10):
            say(f'\n[{rh}] only {len(sub)} finite pairs — skipping.')
            continue

        say(f'\n--- Steps 2-4: {rh} (N={len(sub):,} finite pairs) ---')
        bins_df = _binned_table(sub, rh, sensitivity_col,
                                sensitivity_threshold, target_n_bins,
                                min_bin_n, n_boot, rng)
        per_rh_bins[rh] = bins_df
        bin_csv = save_dir / f'bins_{rh}.csv'
        bins_df.to_csv(bin_csv, index=False)
        say(f'{len(bins_df)} sensitivity bins -> {bin_csv.name}')
        say(bins_df[['sens_median', 'n', 'me_gedi', 'me_vsm',
                     'r_gedi', 'r_vsm', 'ood_for_vsm']].to_string(index=False))

        fork_path = save_dir / f'fork_{rh}.png'
        _fork_plot(bins_df, rh, sensitivity_threshold, fork_path)
        say(f'-> {fork_path.name}')

        # Explicit binary split <thr vs >=thr.
        for label, grp in [('low (<%.2f)' % sensitivity_threshold,
                            sub[sub[sensitivity_col] < sensitivity_threshold]),
                           ('high (>=%.2f)' % sensitivity_threshold,
                            sub[sub[sensitivity_col] >= sensitivity_threshold])]:
            if len(grp) < 3:
                continue
            srow = _both_predictor_stats(grp, lvis_col, gedi_col, vsm_col,
                                         sensitivity_col, n_boot, rng)
            srow['rh'] = rh
            srow['split'] = label
            binary_rows.append(srow)

    if binary_rows:
        binary_df = pd.DataFrame(binary_rows)
        front = ['rh', 'split', 'n', 'me_gedi', 'me_vsm', 'r_gedi', 'r_vsm']
        binary_df = binary_df[front + [c for c in binary_df.columns
                                       if c not in front]]
        binary_csv = save_dir / 'binary_split.csv'
        binary_df.to_csv(binary_csv, index=False)
        say('\n--- Binary split (sensitivity < vs >= threshold) ---')
        say(binary_df[front].to_string(index=False))
        say(f'-> {binary_csv.name}')

    # ======================================================================
    # Step 5: mechanism decomposition (de-noise vs penetration de-bias)
    # ======================================================================
    say('\n--- Step 5: mechanism decomposition ---')
    decomp_rows = []
    for rh in per_rh_bins:
        lvis_col, gedi_col, vsm_col = _sensor_cols(rh)
        sub = df[[lvis_col, gedi_col, vsm_col, sensitivity_col]] \
            .replace([np.inf, -np.inf], np.nan).dropna()
        full = _advantage(sub, rh)
        clean = _advantage(
            sub[sub[sensitivity_col] >= sensitivity_threshold], rh)
        # (a)=full validation, (b)=clean subset, (a)-(b)=penetration de-bias.
        for tag, d in [('a_full', full), ('b_clean', clean)]:
            d = dict(d)
            d['rh'], d['subset'] = rh, tag
            decomp_rows.append(d)
        contrib = {
            'rh': rh, 'subset': 'penetration_contrib (a-b)',
            'n': full['n'] - clean['n'],
            'delta_r': full['delta_r'] - clean['delta_r'],
            'delta_me': full['delta_me'] - clean['delta_me'],
            'delta_abs_me': full['delta_abs_me'] - clean['delta_abs_me'],
        }
        decomp_rows.append(contrib)

    decomp_df = pd.DataFrame(decomp_rows)
    decomp_csv = save_dir / 'decomposition.csv'
    decomp_df.to_csv(decomp_csv, index=False)
    show_cols = ['rh', 'subset', 'n', 'delta_r', 'delta_me', 'delta_abs_me',
                 'r_vsm', 'r_gedi', 'me_vsm', 'me_gedi']
    say(decomp_df[[c for c in show_cols if c in decomp_df.columns]]
        .to_string(index=False))
    say(f'-> {decomp_csv.name}')
    say('Reading: (a) full validation = total VSM advantage; (b) clean '
        'subset = random de-noise / optical skill (GEDI is also clean here); '
        '(a)-(b) = penetration de-bias from training-label filtering.')

    # ======================================================================
    # Step 6: height-confound control (LVIS top-height band)
    # ======================================================================
    if use_height:
        lo, hi = float(height_band[0]), float(height_band[1])
        say(f'\n--- Step 6: height confound control '
            f'({height_control_col} in [{lo:g}, {hi:g}] m) ---')
        band = df[(df[height_control_col] >= lo) &
                  (df[height_control_col] <= hi)]
        say(f'{len(band):,} pairs in the height band')
        for rh in per_rh_bins:
            lvis_col, gedi_col, vsm_col = _sensor_cols(rh)
            sub = band[[lvis_col, gedi_col, vsm_col, sensitivity_col]] \
                .replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < max(min_bin_n, 10):
                say(f'[{rh}] only {len(sub)} pairs in band — skipping.')
                continue
            bins_df = _binned_table(sub, rh, sensitivity_col,
                                    sensitivity_threshold, target_n_bins,
                                    min_bin_n, n_boot, rng)
            bcsv = save_dir / f'bins_{rh}_heightband_{lo:g}_{hi:g}.csv'
            bins_df.to_csv(bcsv, index=False)
            fpath = save_dir / f'fork_{rh}_heightband_{lo:g}_{hi:g}.png'
            _fork_plot(bins_df, rh, sensitivity_threshold, fpath,
                       title_suffix=f' | {height_control_col} {lo:g}-{hi:g} m')
            say(f'[{rh}] {len(bins_df)} bins -> {fpath.name}, {bcsv.name}')

    # ======================================================================
    # Step 7: ME heatmap
    # ======================================================================
    if make_heatmap and per_rh_bins:
        hm_path = save_dir / 'step7_me_heatmap.png'
        _me_heatmap(per_rh_bins, sensitivity_threshold, hm_path)
        say(f'\n--- Step 7: ME heatmap -> {hm_path.name} ---')

    # ======================================================================
    # Conclusion (data-driven)
    # ======================================================================
    say('')
    say('=' * 72)
    say('CONCLUSION')
    say('=' * 72)
    for rh in per_rh_bins:
        bins_df = per_rh_bins[rh]
        low = bins_df[bins_df['sens_median'] < sensitivity_threshold]
        high = bins_df[bins_df['sens_median'] >= sensitivity_threshold]
        say(f'\n[{rh}]')
        if low.empty or high.empty:
            say('  Not enough bins on both sides of the threshold to judge '
                'the low- vs high-sensitivity contrast from binned data; '
                'see the binary-split table.')
        else:
            # (1) Does GEDI ME sink at low sensitivity?
            gedi_drop = high['me_gedi'].mean() - low['me_gedi'].mean()
            say(f'  1. GEDI ME: high-sens mean {high["me_gedi"].mean():+.2f} m '
                f'vs low-sens mean {low["me_gedi"].mean():+.2f} m '
                f'(drop {gedi_drop:+.2f} m at low sensitivity). '
                f'{"Penetration bias is visible." if gedi_drop > 0.5 else "No clear sink."}')
            # (2) Is VSM flatter AND does its bin-r hold?
            vsm_spread = bins_df['me_vsm'].max() - bins_df['me_vsm'].min()
            gedi_spread = bins_df['me_gedi'].max() - bins_df['me_gedi'].min()
            r_vsm_low = low['r_vsm'].mean()
            r_vsm_high = high['r_vsm'].mean()
            flatter = vsm_spread < gedi_spread
            r_holds = np.isfinite(r_vsm_low) and r_vsm_low >= 0.7 * r_vsm_high
            say(f'  2. VSM ME spread {vsm_spread:.2f} m vs GEDI {gedi_spread:.2f} m '
                f'({"flatter" if flatter else "NOT flatter"}); '
                f'VSM bin-r low {r_vsm_low:.2f} vs high {r_vsm_high:.2f} '
                f'({"holds -> true de-bias" if r_holds else "collapses -> flatline, not real de-bias"}).')
        # (3) Penetration de-bias share of total advantage.
        full = decomp_df[(decomp_df.rh == rh) & (decomp_df.subset == 'a_full')]
        contrib = decomp_df[(decomp_df.rh == rh) &
                            (decomp_df.subset == 'penetration_contrib (a-b)')]
        if not full.empty and not contrib.empty:
            tot_r = full['delta_r'].iloc[0]
            con_r = contrib['delta_r'].iloc[0]
            share = con_r / tot_r if abs(tot_r) > 1e-9 else float('nan')
            say(f'  3. Penetration de-bias contributes delta_r {con_r:+.3f} of '
                f'total VSM advantage delta_r {tot_r:+.3f} '
                f'({share:.0%} of the r-advantage); bias-magnitude reduction '
                f'attributable to penetration de-bias: '
                f'{contrib["delta_abs_me"].iloc[0]:+.2f} m.')

    (save_dir / 'conclusion.md').write_text('\n'.join(report) + '\n')
    print(f'\n-> {save_dir / "conclusion.md"}')
