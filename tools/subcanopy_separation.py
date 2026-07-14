"""Descriptive dense/sparse class-separation of the normalized sub-canopy metric
RH<num>/RH98, height-stratified and compared across sensors.

Sister analysis to `tools.subcanopy_classify` (which trains a classifier and
reports AUC). Here we do NOT classify — we directly measure, per LVIS RH98
height bin and per sensor, how far apart the LVIS-defined DENSE and SPARSE
classes sit on the SAME normalized metric RH<num>/RH98.

  1. Pool the per-tile LVIS-GEDI-VSM pair parquets; keep footprints with valid
     RH98 (> rh98_min) on all three sensors. Tile = filename stem (S2 MGRS).
  2. Bin by LVIS RH98 (10-20, 20-30, 30-40, 40-50, 50+ m).
  3. WITHIN each height bin, label the top `tercile_q` fraction of
     LVIS_RH<num>/LVIS_RH98 "dense" (1) and the bottom `tercile_q` "sparse" (0);
     discard the middle. Because the label is defined inside the RH98 bin it is
     independent of canopy-top height — the comparison is height-stratified and
     NOT driven by canopy-top height.
  4. For each sensor in (LVIS, GEDI, VSM) and each height bin, take that
     sensor's OWN RH<num>/RH98 on the dense and sparse footprints and report
       mean_dense, mean_sparse, median_dense, median_sparse,
       delta_mean   = mean_dense   - mean_sparse,
       delta_median = median_dense - median_sparse,
       Cohen's d    = delta_mean / pooled_sd,
       preservation = delta_sensor / delta_LVIS   (mean and median variants).
     LVIS is the reference, so its preservation ratio is 1 by construction; GEDI
     and VSM show how much of the LVIS-detectable separation they retain.
  5. Repeat the whole thing for each numerator in `sub_definitions`
     (default RH25/RH98, RH50/RH98, RH75/RH98).

Deliverables (per definition + pooled): per-height-bin grouped box plots with
the three LVIS classes (sparse/mid/dense) on the x axis and GEDI vs VSM as
grouped boxes, bar plots of delta across height bins for LVIS/GEDI/VSM, a tidy
summary CSV and a markdown summary table + conclusion.md.

Reads the same LVIS-GEDI-VSM triple pairs as `vsm_understory_matched`
(`lvis_RH<NN>` / `vsm_RH<NN>` uppercase, `gedi_rh<NN>` lowercase).

  python -m tools.run run=subcanopy_separation
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from const import FIGURE_SIZES, FONT_SIZES
# Reuse the loading / cleaning / labelling conventions of the classifier tool so
# both analyses bin and label footprints identically.
from tools.subcanopy_classify import (_TOP, _clean, _load_pairs, _make_labels,
                                       _rh_col)

# Sensors compared. LVIS first: it defines the labels and is the preservation
# reference for GEDI / VSM.
_SENSORS = ('lvis', 'gedi', 'vsm')
# Default sub-canopy numerators: RH25/RH98, RH50/RH98, RH75/RH98.
_DEFAULT_DEFS = (25, 50, 75)
_SENSOR_COLOR = {'lvis': 'C2', 'gedi': 'C0', 'vsm': 'C3'}
# Sensors shown as grouped boxes in the per-height-bin box plots.
_PLOT_SENSORS = ('gedi', 'vsm')
# LVIS classes kept on the x axis: bottom q sparse, middle band median, top q
# dense (so unlike the dense/sparse stats, the middle is NOT discarded here).
_TRI_CLASSES = ('sparse', 'mid', 'dense')


# ---------------------------------------------------------------------------
# Per-(definition, sensor, height-bin) separation statistics
# ---------------------------------------------------------------------------
def _cohen_d(dense: np.ndarray, sparse: np.ndarray) -> float:
    """Cohen's d with pooled SD; NaN if either group is too small/degenerate."""
    n1, n0 = len(dense), len(sparse)
    if n1 < 2 or n0 < 2:
        return np.nan
    s1, s0 = dense.std(ddof=1), sparse.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return float((dense.mean() - sparse.mean()) / pooled)


def _separation_rows(labelled: pd.DataFrame, num_level: int, def_tag: str,
                     min_group_n: int, say) -> pd.DataFrame:
    """One row per (height_bin, sensor): dense/sparse mean & median of the
    sensor's own RH<num>/RH98, the deltas and Cohen's d. Preservation ratios are
    filled in afterwards (need the LVIS row of the same bin)."""
    # Each sensor's own normalized sub-canopy metric.
    metric = {}
    for s in _SENSORS:
        top = labelled[_rh_col(s, _TOP)].to_numpy(float)
        num = labelled[_rh_col(s, num_level)].to_numpy(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            metric[s] = np.where(top > 0, num / top, np.nan)

    is_dense = labelled['label'].to_numpy() == 1
    bins = labelled['height_bin'].to_numpy()
    # Preserve the height-bin order as they first appear (already sorted by
    # _make_labels' edge iteration).
    bin_order = list(dict.fromkeys(bins))

    rows = []
    for b in bin_order:
        in_bin = bins == b
        for s in _SENSORS:
            m = metric[s]
            dense = m[in_bin & is_dense]
            sparse = m[in_bin & ~is_dense]
            dense = dense[np.isfinite(dense)]
            sparse = sparse[np.isfinite(sparse)]
            if len(dense) < min_group_n or len(sparse) < min_group_n:
                say(f'  [{def_tag}] {s.upper():<4} bin {b:>7}: '
                    f'dense={len(dense)} sparse={len(sparse)} '
                    f'-> skipped (< min_group_n={min_group_n})')
                continue
            mean_d, mean_s = float(dense.mean()), float(sparse.mean())
            med_d, med_s = float(np.median(dense)), float(np.median(sparse))
            rows.append({
                'definition': def_tag, 'num_level': num_level,
                'sensor': s.upper(), 'height_bin': b,
                'n_dense': len(dense), 'n_sparse': len(sparse),
                'mean_dense': mean_d, 'mean_sparse': mean_s,
                'median_dense': med_d, 'median_sparse': med_s,
                'delta_mean': mean_d - mean_s,
                'delta_median': med_d - med_s,
                'cohens_d': _cohen_d(dense, sparse),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Preservation ratio = delta_sensor / delta_LVIS, matched within each bin.
    lvis = (out[out['sensor'] == 'LVIS']
            .set_index('height_bin')[['delta_mean', 'delta_median']])

    def _pres(row, col):
        if row['height_bin'] not in lvis.index:
            return np.nan
        denom = lvis.loc[row['height_bin'], col]
        return float(row[col] / denom) if denom not in (0, np.nan) and np.isfinite(denom) else np.nan

    out['preservation_mean'] = out.apply(lambda r: _pres(r, 'delta_mean'), axis=1)
    out['preservation_median'] = out.apply(lambda r: _pres(r, 'delta_median'), axis=1)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _make_tri_labels(df: pd.DataFrame, num_level: int, height_edges: tuple,
                     q: float, min_bin_n: int) -> pd.DataFrame:
    """Like `_make_labels` but keeps the middle band as a third class instead of
    discarding it. Adds `height_bin` and `tri_class` in {'sparse','mid',
    'dense'}, taking the q / 1-q quantiles of LVIS_RH<num>/LVIS_RH98 WITHIN each
    LVIS RH98 height bin (so the split is independent of canopy-top height)."""
    out = df.copy()
    top = out[_rh_col('lvis', _TOP)].to_numpy()
    num = out[_rh_col('lvis', num_level)].to_numpy()
    out['lvis_sub'] = num / top

    edges = list(height_edges) + [np.inf]
    out['height_bin'] = ''
    out['tri_class'] = ''
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (top >= lo) & (top < hi)
        label = f'{lo:g}+' if not np.isfinite(hi) else f'{lo:g}-{hi:g}'
        if int(m.sum()) < min_bin_n:
            continue
        sub = out.loc[m, 'lvis_sub']
        qlo, qhi = sub.quantile(q), sub.quantile(1 - q)
        out.loc[m, 'height_bin'] = label
        out.loc[m & (out['lvis_sub'] <= qlo), 'tri_class'] = 'sparse'
        out.loc[m & (out['lvis_sub'] >= qhi), 'tri_class'] = 'dense'
        out.loc[m & (out['lvis_sub'] > qlo) & (out['lvis_sub'] < qhi),
                'tri_class'] = 'mid'
    return out[out['tri_class'] != ''].reset_index(drop=True)


def _safe_bin(label: str) -> str:
    """Filesystem-safe version of a height-bin label ('10-20'->'10_20',
    '50+'->'50plus')."""
    return label.replace('+', 'plus').replace('-', '_')


def _plot_boxes_per_bin(tri: pd.DataFrame, num_level: int, def_tag: str,
                        safe_tag: str, min_group_n: int, save_dir: Path,
                        say) -> None:
    """One box-plot figure per LVIS RH98 height bin: x axis = LVIS class
    (sparse / median / dense), with GEDI and VSM RH<num>/RH98 as grouped boxes."""
    bin_order = list(dict.fromkeys(tri['height_bin']))
    cls_arr = tri['tri_class'].to_numpy()
    bin_arr = tri['height_bin'].to_numpy()
    # Each sensor's own normalized sub-canopy metric.
    metric = {}
    for s in _PLOT_SENSORS:
        top = tri[_rh_col(s, _TOP)].to_numpy(float)
        num = tri[_rh_col(s, num_level)].to_numpy(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            metric[s] = np.where(top > 0, num / top, np.nan)

    n_sensors = len(_PLOT_SENSORS)
    width = 0.8 / n_sensors
    for b in bin_order:
        in_bin = bin_arr == b
        positions, data, colors = [], [], []
        for i, cls in enumerate(_TRI_CLASSES):
            for k, s in enumerate(_PLOT_SENSORS):
                vals = metric[s][in_bin & (cls_arr == cls)]
                vals = vals[np.isfinite(vals)]
                if len(vals) < min_group_n:
                    continue
                positions.append(i + (k - (n_sensors - 1) / 2) * width)
                data.append(vals)
                colors.append(_SENSOR_COLOR[s])
        if not data:
            say(f'  [{def_tag}] bin {b:>7}: no class/sensor group passed '
                f'min_group_n={min_group_n} -> no box plot.')
            continue

        fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='k', linewidth=1.2))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)
            patch.set_edgecolor('k')
        ax.set_xticks(range(len(_TRI_CLASSES)))
        ax.set_xticklabels(_TRI_CLASSES, fontsize=FONT_SIZES['ticks'])
        ax.set_xlim(-0.5, len(_TRI_CLASSES) - 0.5)
        ax.set_xlabel('LVIS class', fontsize=FONT_SIZES['label'])
        ax.set_ylabel(f'RH{num_level}/RH98', fontsize=FONT_SIZES['label'])
        ax.set_title(f'{def_tag} — LVIS RH98 {b} m', fontsize=FONT_SIZES['title'])
        handles = [plt.Line2D([0], [0], marker='s', ls='', markersize=10,
                              markerfacecolor=_SENSOR_COLOR[s],
                              markeredgecolor='k', label=s.upper())
                   for s in _PLOT_SENSORS]
        ax.legend(handles=handles, fontsize=FONT_SIZES['legend'],
                  loc='upper left')
        ax.grid(True, axis='y', ls='--', alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(save_dir / f'box_{safe_tag}_{_safe_bin(b)}.png',
                    bbox_inches='tight', dpi=150)
        plt.close(fig)


def _plot_delta_bars(sep: pd.DataFrame, def_tag: str, metric: str,
                     save_path: Path) -> None:
    """Grouped bar plot of delta (mean or median) across height bins, one bar
    group per sensor (LVIS, GEDI, VSM)."""
    col = f'delta_{metric}'
    bins = list(dict.fromkeys(sep['height_bin']))
    sensors = [s for s in ('LVIS', 'GEDI', 'VSM') if s in set(sep['sensor'])]
    x = np.arange(len(bins))
    w = 0.8 / max(len(sensors), 1)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])
    for k, s in enumerate(sensors):
        sub = sep[sep['sensor'] == s]
        vals = [float(sub[sub['height_bin'] == b][col].mean())
                if not sub[sub['height_bin'] == b].empty else np.nan
                for b in bins]
        ax.bar(x + (k - (len(sensors) - 1) / 2) * w, vals, w, label=s,
               color=_SENSOR_COLOR.get(s.lower(), None), alpha=0.85)
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=FONT_SIZES['ticks'])
    ax.set_xlabel('LVIS RH98 bin (m)', fontsize=FONT_SIZES['label'])
    ax.set_ylabel(f'delta_{metric}  (dense - sparse)',
                  fontsize=FONT_SIZES['label'])
    ax.set_title(f'Class separation by height bin — {def_tag}',
                 fontsize=FONT_SIZES['title'])
    ax.legend(fontsize=FONT_SIZES['legend'])
    ax.grid(True, axis='y', ls='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ===========================================================================
# Per-definition driver
# ===========================================================================
def _run_one_definition(df: pd.DataFrame, num_level: int, height_edges: tuple,
                        tercile_q: float, min_bin_n: int, min_group_n: int,
                        save_dir: Path, say) -> pd.DataFrame:
    def_tag = f'RH{num_level}/RH98'
    safe_tag = f'sub{num_level}'
    say('')
    say('=' * 74)
    say(f'DEFINITION: LVIS_sub = {def_tag}')
    say('=' * 74)

    labelled = _make_labels(df, num_level, height_edges, tercile_q, min_bin_n, say)
    if labelled.empty:
        say('  No labelled footprints -> definition skipped.')
        return pd.DataFrame()

    sep = _separation_rows(labelled, num_level, def_tag, min_group_n, say)
    if sep.empty:
        say('  No (sensor, bin) group passed min_group_n -> definition skipped.')
        return sep

    # Console readout of the headline numbers.
    say('')
    for s in ('LVIS', 'GEDI', 'VSM'):
        sub = sep[sep['sensor'] == s]
        if sub.empty:
            continue
        say(f'  {s}:')
        for _, r in sub.iterrows():
            say(f'    bin {r["height_bin"]:>7} | dense={r["mean_dense"]:.3f} '
                f'sparse={r["mean_sparse"]:.3f} | d_mean={r["delta_mean"]:+.3f} '
                f'd_med={r["delta_median"]:+.3f} | Cohen_d={r["cohens_d"]:+.2f} '
                f'| preserve(mean)={r["preservation_mean"]:.2f}')

    # Box plots: keep all three LVIS classes (sparse/median/dense) on the x axis
    # and group GEDI vs VSM, one figure per LVIS RH98 height bin.
    tri = _make_tri_labels(df, num_level, height_edges, tercile_q, min_bin_n)
    _plot_boxes_per_bin(tri, num_level, def_tag, safe_tag, min_group_n,
                        save_dir, say)
    _plot_delta_bars(sep, def_tag, 'mean', save_dir / f'delta_mean_{safe_tag}.png')
    _plot_delta_bars(sep, def_tag, 'median',
                     save_dir / f'delta_median_{safe_tag}.png')
    return sep


# ===========================================================================
# Main entrypoint (registered as run=subcanopy_separation)
# ===========================================================================
def subcanopy_separation_analysis(
        pairs_dir: str,
        save_dir: str,
        pairs_glob: str = '*.parquet',
        sub_definitions: tuple = _DEFAULT_DEFS,
        height_bin_edges: tuple = (10.0, 20.0, 30.0, 40.0, 50.0),
        tercile_q: float = 0.30,
        min_bin_n: int = 100,
        min_group_n: int = 30,
        rh98_min: float = 5.0,
        **kwargs) -> None:
    """Height-stratified dense/sparse class-separation of RH<num>/RH98 across
    sensors. Dense/sparse labels are defined WITHIN LVIS RH98 bins from LVIS
    RH<num>/RH98, so the separation is not driven by canopy-top height; GEDI and
    VSM are measured on their OWN RH<num>/RH98 and compared to LVIS via a
    preservation ratio.

    Args:
        pairs_dir: dir of per-tile LVIS-GEDI-VSM pair parquets carrying
            `lvis_RH<NN>` / `vsm_RH<NN>` (uppercase) and `gedi_rh<NN>`
            (lowercase).
        save_dir: output dir for figures, the summary CSV and conclusion.md.
        pairs_glob: glob under pairs_dir. Default '*.parquet'.
        sub_definitions: LVIS RH numerators for RH<num>/RH98; the analysis is
            repeated per entry. Default (25, 50, 75).
        height_bin_edges: lower edges of the LVIS RH98 height bins; last opens to
            +inf. Default (10,20,30,40,50) -> 10-20,20-30,30-40,40-50,50+.
        tercile_q: within-bin quantile for the dense/sparse split. Default 0.30
            (top 30% dense, bottom 30% sparse, middle 40% discarded).
        min_bin_n: drop a height bin with fewer than this many footprints before
            labelling. Default 100.
        min_group_n: min dense (or sparse) footprints for a sensor/bin stat to be
            reported. Default 30.
        rh98_min: keep footprints with ALL sensors' RH98 > this (m). Default 5.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    report: list = []

    def say(line: str = '') -> None:
        print(line)
        report.append(line)

    sub_definitions = tuple(int(d) for d in sub_definitions)
    height_bin_edges = tuple(float(e) for e in height_bin_edges)

    say('=' * 74)
    say('Sub-canopy class separation — LVIS dense/sparse on RH<num>/RH98')
    say('=' * 74)
    say(f'tercile_q={tercile_q:g}; rh98_min={rh98_min:g} m; '
        f'min_bin_n={min_bin_n}; min_group_n={min_group_n}.')
    say('Sub-canopy definitions: '
        + ', '.join(f'RH{d}/RH98' for d in sub_definitions) + '.')
    say('Labels are LVIS-defined WITHIN RH98 bins -> height-stratified, '
        'not driven by canopy-top height.')

    df, _extra = _load_pairs(pairs_dir, pairs_glob, _SENSORS[1:], sub_definitions,
                             say)
    df = _clean(df, _SENSORS[1:], sub_definitions, rh98_min, say)

    all_sep = []
    for num_level in sub_definitions:
        sep = _run_one_definition(df, num_level, height_bin_edges, tercile_q,
                                  min_bin_n, min_group_n, save_dir, say)
        if not sep.empty:
            all_sep.append(sep)

    separation = pd.concat(all_sep, ignore_index=True) if all_sep else pd.DataFrame()
    separation.to_csv(save_dir / 'separation_metrics.csv', index=False)

    # Deliverable: readable summary table in the report.
    if not separation.empty:
        say('')
        say('SUMMARY TABLE')
        say('| Definition | Sensor | RH98 bin | mean_dense | mean_sparse | '
            'median_dense | median_sparse | delta_mean | delta_median | '
            'Cohen d | preserve(mean) | preserve(median) |')
        say('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in separation.iterrows():
            say(f'| {r["definition"]} | {r["sensor"]} | {r["height_bin"]} '
                f'| {r["mean_dense"]:.3f} | {r["mean_sparse"]:.3f} '
                f'| {r["median_dense"]:.3f} | {r["median_sparse"]:.3f} '
                f'| {r["delta_mean"]:+.3f} | {r["delta_median"]:+.3f} '
                f'| {r["cohens_d"]:+.2f} | {r["preservation_mean"]:.2f} '
                f'| {r["preservation_median"]:.2f} |')

        # Headline: mean preservation ratio per sensor across all (def, bin).
        say('')
        say('Mean preservation ratio (delta_sensor / delta_LVIS) across all '
            'definitions and height bins:')
        for s in ('GEDI', 'VSM'):
            sub = separation[separation['sensor'] == s]
            if sub.empty:
                continue
            say(f'  {s}: preserve(mean)={sub["preservation_mean"].mean():.2f}  '
                f'preserve(median)={sub["preservation_median"].mean():.2f}  '
                f'mean|Cohen d|={sub["cohens_d"].abs().mean():.2f}')

    say('')
    say('Outputs: separation_metrics.csv, box_sub*_<bin>.png, '
        'delta_mean_sub*.png, delta_median_sub*.png, conclusion.md')
    (save_dir / 'conclusion.md').write_text('\n'.join(report) + '\n')
    print(f'\n-> {save_dir / "conclusion.md"}')
