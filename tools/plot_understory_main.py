"""Re-plot the VSM understory two-panel main figure from existing CSVs.

Pure re-plot: reads CSVs from a prior `vsm_understory_matched` run plus a
handful of constants (passed in via the config) and renders the publication
figure. Does NOT refit models, rerun bootstraps, or recompute any statistic.

Panel a (left): top-height-controlled understory contrast. Per height band,
mean RH25 residual conditional on each sensor's own RH98, split by the LVIS
U_L tercile (hiU = sparse / top-heavy, loU = dense / bottom-heavy). Six lines
(3 sensors x {hiU solid + circle, loU dashed + x}). An inset zoomed to
~[-1.2, 1.2] m shows the VSM band-by-band gap with the pooled tile-block
bootstrap contrast annotated.

Panel b (right): understory residual correlation vs GEDI sensitivity. Three
sensor pairs (VSM-LVIS red thick, VSM-GEDI purple, GEDI-LVIS gray) with their
bootstrap CI shaded; a secondary right axis shows the per-bin LVIS residual SD
(the unrestricted reference for the range-restriction factor u = SD/SD_pool).

Inputs (all CSVs are outputs of `tools.vsm_understory_matched`):
    per_band_separation.csv      -> panel a lines
    sensitivity_residual_corr.csv -> panel b lines (falls back to constants
                                     when the CSV is absent)
    block_bootstrap_summary.csv  -> panel a annotation (falls back to
                                     pooled_contrast_m + pooled_ci constants)

Outputs (under save_dir):
    fig_understory_main.{pdf,svg,png}   two-panel figure
    fig_understory_a.{pdf,svg,png}      panel a alone
    fig_understory_b.{pdf,svg,png}      panel b alone

  python -m tools.run run=plot_understory_main
"""
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_SENSOR_COLOR_A = {'VSM': '#d62728', 'GEDI': '#1f77b4', 'LVIS': '#2ca02c'}
_PAIR_COLOR_B = {'VSM_LVIS':  '#d62728',     # red, thick
                 'VSM_GEDI':  '#9467bd',     # purple
                 'GEDI_LVIS': '#7f7f7f'}     # gray
_PAIR_LABEL_B = {'VSM_LVIS':  'VSM–LVIS',
                 'VSM_GEDI':  'VSM–GEDI',
                 'GEDI_LVIS': 'GEDI–LVIS'}
_SENSORS = ('LVIS', 'VSM', 'GEDI')

_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}

# Fallback sensitivity table (matches the table the user pasted; used only when
# `sensitivity_residual_corr.csv` is absent from the results dir).
_FALLBACK_SENS = pd.DataFrame({
    'sens_median':     [0.961, 0.971, 0.977, 0.982, 0.987, 0.991],
    'r_VSM_LVIS':      [0.337, 0.363, 0.383, 0.328, 0.373, 0.357],
    'r_VSM_GEDI':      [0.380, 0.399, 0.342, 0.350, 0.374, 0.369],
    'r_GEDI_LVIS':     [0.296, 0.400, 0.420, 0.392, 0.351, 0.402],
    'sd_resid_LVIS':   [7.19, 6.92, 7.60, 6.97, 7.19, 7.01],
    'u_LVIS':          [1.01, 0.97, 1.06, 0.97, 1.01, 0.98],
})


# ---------------------------------------------------------------------------
# IO + fallback resolution
# ---------------------------------------------------------------------------
def _read_per_band(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'band_lo' in df.columns:
        df = df.sort_values('band_lo', kind='stable').reset_index(drop=True)
    return df


def _read_sens(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        if 'sens_median' in df.columns:
            df = df.sort_values('sens_median', kind='stable').reset_index(
                drop=True)
        return df
    return _FALLBACK_SENS.copy()


def _resolve_pooled_contrast(boot_path: Path, fallback_med: float,
                             fallback_ci: tuple) -> tuple:
    """Read coef_VSM_strict median + 95% CI from block_bootstrap_summary.csv if
    present, else fall back to the constants from the spec."""
    if boot_path.exists():
        df = pd.read_csv(boot_path).set_index('stat')
        if 'coef_VSM_strict' in df.index:
            row = df.loc['coef_VSM_strict']
            try:
                return (float(row['p50']),
                        (float(row['p2_5']), float(row['p97_5'])))
            except (KeyError, ValueError):
                pass
    return fallback_med, tuple(fallback_ci)


# ---------------------------------------------------------------------------
# Panel a — per-band residual hi vs lo, three sensors, inset on VSM only
# ---------------------------------------------------------------------------
def _draw_panel_a(ax, per_band: pd.DataFrame, contrast_med: float,
                  contrast_ci: tuple) -> None:
    """Six lines (3 sensors x {hiU solid+circle, loU dashed+x}) over the
    band axis, with an inset zoomed to ~[-1.2, 1.2] m for VSM only."""
    bands = per_band['band'].astype(str).tolist()
    x = np.arange(len(bands))
    ax.axhline(0, color='0.55', lw=0.8, zorder=1)

    # Draw the data lines WITHOUT individual labels — the legend is built
    # below from two factorized layers (color = sensor, linestyle = group)
    # rather than enumerating the 6 sensor x group combinations.
    for sensor in ('LVIS', 'GEDI', 'VSM'):     # LVIS first so VSM is on top
        c = _SENSOR_COLOR_A[sensor]
        lw = 2.4 if sensor == 'VSM' else 1.4
        y_hi = per_band[f'resid_hiU_{sensor}'].to_numpy()
        y_lo = per_band[f'resid_loU_{sensor}'].to_numpy()
        ax.plot(x, y_hi, '-o', color=c, lw=lw, markersize=5, zorder=3)
        ax.plot(x, y_lo, '--x', color=c, lw=lw, markersize=6, alpha=0.9,
                zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=30, ha='right')
    ax.set_xlabel('Canopy top height (LVIS RH98 [m])')
    ax.set_ylabel(r'RH25 independent of canopy top height [m] (RH25 $\perp$)')
    ax.grid(True, linestyle='--', linewidth=0.5, color='0.85', zorder=0)
    ax.set_axisbelow(True)

    # --- Two-layer legend ---------------------------------------------------
    # Layer 1: COLOR -> sensor (LVIS green, GEDI blue, VSM red), horizontal.
    # Layer 2: LINESTYLE+MARKER -> understory group (sparse vs dense), in
    # neutral gray so the linestyle/marker is the only differentiating
    # channel. Factorizing the 6 lines into colour x style cuts the cognitive
    # load of the original 6-item legend.
    sensor_proxies = [
        mlines.Line2D([], [], color=_SENSOR_COLOR_A['LVIS'], lw=2.6,
                      label='LVIS'),
        mlines.Line2D([], [], color=_SENSOR_COLOR_A['GEDI'], lw=2.6,
                      label='GEDI'),
        mlines.Line2D([], [], color=_SENSOR_COLOR_A['VSM'],  lw=2.6,
                      label='VSM'),
    ]
    group_proxies = [
        mlines.Line2D([], [], color='0.25', lw=1.5, ls='-', marker='o',
                      markersize=5,
                      label='sparse understory (high RH25/RH98)'),
        mlines.Line2D([], [], color='0.25', lw=1.5, ls='--', marker='x',
                      markersize=6,
                      label='dense understory (low RH25/RH98)'),
    ]
    leg_sensor = ax.legend(
        handles=sensor_proxies, loc='upper left',
        bbox_to_anchor=(0.01, 0.99), ncol=3, frameon=False, fontsize=8,
        handlelength=1.6, handletextpad=0.5, columnspacing=1.4)
    ax.add_artist(leg_sensor)            # so the second legend doesn't drop it
    ax.legend(
        handles=group_proxies, loc='upper left',
        bbox_to_anchor=(0.01, 0.92), ncol=1, frameon=False, fontsize=8,
        handlelength=2.2, handletextpad=0.5, labelspacing=0.3)

    # --- Pooled-contrast annotation: small label with a leader pointing at
    # the VSM red gap on the right-most band, where the hi-lo separation is
    # widest and the eye naturally lands. No inset/box — just a compact text
    # tag in the empty lower-right strip with a thin leader line. The text
    # rides on a thin white halo so it stays readable if it crosses a line.
    cV = _SENSOR_COLOR_A['VSM']
    y_hi_v = per_band['resid_hiU_VSM'].to_numpy()
    y_lo_v = per_band['resid_loU_VSM'].to_numpy()
    tgt_idx = len(x) - 1                              # rightmost band
    tgt_y = float((y_hi_v[tgt_idx] + y_lo_v[tgt_idx]) / 2)
    txt_x = x[tgt_idx] - 2.0
    txt_y = -5.0
    lo, hi = contrast_ci
    ax.annotate(
        f'VSM pooled hi$-$lo gap\n+{contrast_med:.2f} m '
        f'[{lo:.2f}, {hi:.2f}] (tile-block bootstrap)',
        xy=(x[tgt_idx], tgt_y), xytext=(txt_x, txt_y),
        fontsize=7.5, color=cV, ha='center', va='top',
        arrowprops=dict(arrowstyle='-', color=cV, lw=0.7,
                        connectionstyle='arc3,rad=-0.25'),
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=cV, linewidth=0.6, alpha=0.92),
        zorder=4)


# ---------------------------------------------------------------------------
# Panel b — residual correlation vs GEDI sensitivity, right-axis SD bars
# ---------------------------------------------------------------------------
def _draw_panel_b(ax, sens: pd.DataFrame, pooled_lvis_sd: float) -> tuple:
    """Three pair lines + CI shading on the left axis, light-gray LVIS resid
    SD bars on a twin right axis. Returns the twin axis so the caller can
    style/share the figure caption with the u-range derived from it."""
    x = sens['sens_median'].to_numpy()
    ax.axhline(0, color='0.55', lw=0.8, zorder=1)

    # ----- right axis first so the bars sit BELOW the correlation lines ----
    # Light-gray thin bars at each bin's LVIS RH25-residual SD. The right
    # y-axis is intentionally stretched well above the data (~6.9-7.6 m) so
    # the bars stay short (~half the panel height) and never compete with
    # the correlation lines on the left axis. A dotted reference at the
    # pooled SD (the denominator of u) anchors the per-bin variation.
    ax2 = ax.twinx()
    sd_vals = sens['sd_resid_LVIS'].to_numpy() if 'sd_resid_LVIS' in sens \
        else np.full(len(x), np.nan)
    if len(x) >= 2:
        spacing = float(np.median(np.diff(np.sort(x))))
    else:
        spacing = 0.005
    bar_w = max(spacing * 0.65, 1e-4)
    ax2.bar(x, sd_vals, width=bar_w, color='0.82', alpha=0.55,
            edgecolor='0.65', linewidth=0.5, zorder=0,
            label='LVIS resid SD')
    sd_finite = sd_vals[np.isfinite(sd_vals)]
    sd_max = float(sd_finite.max()) if sd_finite.size else 8.0
    # Headroom = ~2x the data max so the bars stop near the panel midline.
    hi_sd = max(15.0, np.ceil(sd_max * 2.0))
    ax2.set_ylim(0, hi_sd)
    ax2.axhline(pooled_lvis_sd, color='0.55', lw=0.7, ls=':', alpha=0.8,
                zorder=0)
    ax2.set_ylabel('LVIS residual SD (m)', color='0.45')
    ax2.tick_params(axis='y', colors='0.45')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('0.55')

    # ----- left axis: three pair lines with CI shading ---------------------
    line_handles = {}
    for key in ('GEDI_LVIS', 'VSM_GEDI', 'VSM_LVIS'):    # draw VSM-LVIS last
        c = _PAIR_COLOR_B[key]
        lw = 2.6 if key == 'VSM_LVIS' else 1.5
        y = sens[f'r_{key}'].to_numpy()
        h, = ax.plot(x, y, '-o', color=c, lw=lw, markersize=4.5, zorder=3,
                     label=_PAIR_LABEL_B[key])
        line_handles[key] = h
        lo_col, hi_col = f'r_{key}_lo', f'r_{key}_hi'
        if lo_col in sens.columns and hi_col in sens.columns:
            ylo = sens[lo_col].to_numpy()
            yhi = sens[hi_col].to_numpy()
            if np.isfinite(ylo).any() and np.isfinite(yhi).any():
                ax.fill_between(x, ylo, yhi, color=c, alpha=0.16, zorder=2,
                                linewidth=0)

    ax.set_xlabel('GEDI sensitivity')
    ax.set_ylabel('corr of RH25 residuals | own RH98')
    ax.set_ylim(0, 0.48)
    ax.set_xlim(min(x) - bar_w * 0.6, max(x) + bar_w * 0.6)
    ax.grid(True, linestyle='--', linewidth=0.5, color='0.85', zorder=0)
    ax.set_axisbelow(True)
    # Headline (VSM-LVIS) first. One-row horizontal legend INSIDE the panel
    # at upper-right with a white frame, so it floats above the correlation
    # lines (which peak at corr~0.42 = 87% panel) and stays clear of the SD
    # bars in the lower half.
    order = ('VSM_LVIS', 'VSM_GEDI', 'GEDI_LVIS')
    leg = ax.legend([line_handles[k] for k in order],
                    [_PAIR_LABEL_B[k] for k in order],
                    loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=3,
                    frameon=True, handlelength=2.0, handletextpad=0.5,
                    columnspacing=1.2)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('0.7')
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_alpha(0.92)

    # NOTE: the range-restriction factor u = SD_bin / SD_pool is now read
    # directly off the right axis (the dotted reference at the pooled SD).
    # Removed the redundant text annotation per user request.
    return ax2


# ---------------------------------------------------------------------------
# Panel-label helper + save-all helper
# ---------------------------------------------------------------------------
def _add_panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')


def _save_all(fig, stem: Path, dpi: int) -> list:
    out = []
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': dpi})):
        p = stem.with_suffix('.' + ext)
        fig.savefig(p, bbox_inches='tight', **kw)
        out.append(p)
    return out


# ===========================================================================
# Entrypoint (registered as run=plot_understory_main)
# ===========================================================================
def replot_understory_main(
        results_dir: str,
        save_dir: str = None,
        per_band_csv: str = 'per_band_separation.csv',
        sens_csv: str = 'sensitivity_residual_corr.csv',
        boot_summary_csv: str = 'block_bootstrap_summary.csv',
        pooled_contrast_m: float = 1.05,
        pooled_ci: tuple = (0.75, 1.27),
        pooled_lvis_resid_sd: float = 7.15,
        figsize_main: tuple = (12.0, 4.6),
        figsize_panel: tuple = (6.2, 4.6),
        dpi: int = 300,
        **kwargs) -> None:
    """Re-plot the VSM understory two-panel main figure.

    Args:
        results_dir: directory holding the prior `vsm_understory_matched`
            outputs (per_band_separation.csv, sensitivity_residual_corr.csv,
            block_bootstrap_summary.csv).
        save_dir: output directory for the figure files. Defaults to
            `results_dir` so plots land beside the CSVs.
        per_band_csv, sens_csv, boot_summary_csv: filenames under
            `results_dir`. Override when running on a renamed export.
        pooled_contrast_m: fallback pooled VSM hi-lo contrast (m) — used when
            `block_bootstrap_summary.csv` is missing or lacks the
            `coef_VSM_strict` row.
        pooled_ci: fallback (lo, hi) 95% CI for the pooled contrast.
        pooled_lvis_resid_sd: pooled LVIS RH25-residual SD (m) used as the
            unrestricted reference for u = SD_bin / SD_pool. Default 7.15
            (matches the spec).
        figsize_main: (w, h) inches for the combined two-panel figure.
        figsize_panel: (w, h) inches for each single-panel export.
        dpi: PNG resolution. PDF/SVG are vector.
    """
    results_dir = Path(results_dir).expanduser()
    save_dir = Path(save_dir).expanduser() if save_dir else results_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    per_band_path = results_dir / per_band_csv
    sens_path = results_dir / sens_csv
    boot_path = results_dir / boot_summary_csv

    if not per_band_path.exists():
        raise FileNotFoundError(
            f'{per_band_path} not found — panel a needs the per-band CSV; '
            f'pass a different `per_band_csv` or run vsm_understory_matched '
            f'first.')
    per_band = _read_per_band(per_band_path)
    sens = _read_sens(sens_path)
    contrast_med, contrast_ci = _resolve_pooled_contrast(
        boot_path, pooled_contrast_m, pooled_ci)

    print(f'Reading per-band lines from {per_band_path} '
          f'({len(per_band)} bands).')
    if sens_path.exists():
        print(f'Reading sensitivity table from {sens_path} '
              f'({len(sens)} bins).')
    else:
        print(f'Sensitivity CSV not found at {sens_path} — using the '
              f'fallback table from the spec ({len(sens)} bins).')
    if boot_path.exists():
        print(f'Reading pooled contrast from {boot_path}: '
              f'+{contrast_med:.2f} m '
              f'[{contrast_ci[0]:.2f}, {contrast_ci[1]:.2f}].')
    else:
        print(f'Bootstrap summary CSV not found at {boot_path} — using '
              f'constants: +{contrast_med:.2f} m '
              f'[{contrast_ci[0]:.2f}, {contrast_ci[1]:.2f}].')

    out_files = []
    with plt.rc_context(_RC):
        # -------- combined two-panel figure --------
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=figsize_main,
            gridspec_kw={'wspace': 0.30})
        _draw_panel_a(ax_a, per_band, contrast_med, contrast_ci)
        _draw_panel_b(ax_b, sens, pooled_lvis_resid_sd)
        _add_panel_label(ax_a, 'a')
        _add_panel_label(ax_b, 'b')
        fig.tight_layout()
        out_files += _save_all(fig, save_dir / 'fig_understory_main', dpi)
        plt.close(fig)

        # -------- panel a alone --------
        fig_a, ax_a2 = plt.subplots(figsize=figsize_panel)
        _draw_panel_a(ax_a2, per_band, contrast_med, contrast_ci)
        _add_panel_label(ax_a2, 'a')
        fig_a.tight_layout()
        out_files += _save_all(fig_a, save_dir / 'fig_understory_a', dpi)
        plt.close(fig_a)

        # -------- panel b alone --------
        fig_b, ax_b2 = plt.subplots(figsize=figsize_panel)
        _draw_panel_b(ax_b2, sens, pooled_lvis_resid_sd)
        _add_panel_label(ax_b2, 'b')
        fig_b.tight_layout()
        out_files += _save_all(fig_b, save_dir / 'fig_understory_b', dpi)
        plt.close(fig_b)

    for p in out_files:
        print(f'  -> {p}')
