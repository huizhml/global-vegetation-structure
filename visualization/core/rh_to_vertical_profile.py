"""
GEDI RH → Vertical Profile illustration.

Generates a three-panel figure:
  1. RH curve (cumulative energy vs height)
  2. Vertical profile (energy density per height bin + smoothed envelope)
  3. Forest structure cross-section

Usage:
    python gedi_figure.py
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
import numpy as np
from scipy.signal import savgol_filter

from const import FONT_SIZES, FIGURE_SIZES, set_plot_fonts

set_plot_fonts()


# ---------------------------------------------------------------------------
# Data & derived products
# ---------------------------------------------------------------------------

def derive_vertical_profile(rh, bin_size=0.4, h_min=-8, h_max=32,
                            savgol_window=5, savgol_poly=1):
    """Convert RH percentiles → binned energy density + smoothed envelope.

    Returns bin_centers, bin_amp_pct, smoothed_pct.
    """
    bin_centers = np.arange(h_min + bin_size / 2, h_max, bin_size)
    bin_amp = np.zeros(len(bin_centers))

    for b, hc in enumerate(bin_centers):
        for i in range(len(rh) - 1):
            if hc >= rh[i] and hc < rh[i + 1]:
                dh = rh[i + 1] - rh[i]
                bin_amp[b] = 1.0 / dh if dh > 0.01 else 0
                break

    bin_amp_pct = bin_amp / bin_amp.sum() * 100
    smoothed_pct = savgol_filter(bin_amp_pct, window_length=savgol_window,
                                 polyorder=savgol_poly)
    return bin_centers, bin_amp_pct, smoothed_pct


# ---------------------------------------------------------------------------
# Panel drawing functions
# ---------------------------------------------------------------------------

def plot_rh_curve(ax, percentiles, rh, rh_markers, cfg):
    """Panel 1: RH cumulative curve with labelled percentile dots."""
    ax.plot(percentiles, rh, color='#378ADD', linewidth=2, zorder=3)

    for p, color, label in rh_markers:
        ax.plot(p, rh[p], 'o', color=color, markersize=5, zorder=4)
        ax.axhline(y=rh[p], color=color, linewidth=0.5,
                   linestyle='--', alpha=0.4, zorder=1)
        # Place label left for dots near right edge, right otherwise
        if p >= 98:
            ax.annotate(label, xy=(p, rh[p]), xytext=(-40, -2),
                        textcoords='offset points', fontsize=cfg['ticks'],
                        color=color, va='center', zorder=5)
        else:
            ax.annotate(label, xy=(p, rh[p]), xytext=(6, -2),
                        textcoords='offset points', fontsize=cfg['ticks'],
                        color=color, va='center', zorder=5)

    ax.set_xlabel('RH0–100', fontsize=cfg['label'])
    ax.set_ylabel('Height (m)', fontsize=cfg['label'])
    ax.set_title('RH profile', fontsize=cfg['title'], pad=cfg['title_pad'])
    ax.set_xlim(0, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(np.arange(-5, 35, 5))
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=cfg['ticks'])


def plot_vertical_profile(ax, bin_centers, bin_amp_pct, smoothed_pct,
                          bin_size, rh, rh_markers, cfg):
    """Panel 2: horizontal bar chart + smoothed envelope."""
    ax.barh(bin_centers, bin_amp_pct, height=bin_size, color='#27500A',
            alpha=0.8, edgecolor='none', zorder=2)
    ax.plot(smoothed_pct, bin_centers, color='#D85A30', linewidth=2,
            alpha=0.9, zorder=3)

    for p, color, _ in rh_markers:
        ax.axhline(y=rh[p], color=color, linewidth=0.5,
                   linestyle='--', alpha=0.5, zorder=1)

    ax.set_xlabel('Energy (%)', fontsize=cfg['label'])
    ax.set_title('Vertical profile', fontsize=cfg['title'], pad=cfg['title_pad'])
    ax.set_yticks(np.arange(-5, 35, 5))
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=cfg['ticks'])

    # Legend
    legend_handles = [
        Patch(facecolor='#27500A', alpha=0.5, label='Binned'),
        Line2D([0], [0], color='#D85A30', linewidth=2, alpha=0.85,
               label='Smoothed'),
    ]
    ax.legend(handles=legend_handles, fontsize=cfg['legend'], loc='lower left',
              framealpha=0.85, edgecolor='#ccc', bbox_to_anchor=(0.6, 0.01))

    xmax = max(bin_amp_pct) * 1.15
    ax.set_xlim(0, xmax)
    return xmax


def add_rh_annotations(fig, ax, rh, label_configs, cfg):
    """Place RH height annotations in the gap right of panel 2."""
    fig.canvas.draw()

    for p, color, fmt, yoff in label_configs:
        h_val = 0 if p is None else rh[p]
        txt = fmt if p is None else fmt.format(rh[p])

        y_display = ax.transData.transform((0, h_val + yoff))[1]
        y_fig = fig.transFigure.inverted().transform((0, y_display))[1]
        x_fig = ax.get_position().x1 - 0.02

        fig.text(x_fig, y_fig, txt,
                 fontsize=cfg['annot'], color=color,
                 ha='left', va='center', zorder=6,
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                           edgecolor='none', alpha=0.75))


def draw_tree(ax, x, canopy_top, crown_rx=1.2, crown_ry=3.0,
              trunk_color='#3D2B1F'):
    """Draw a single stylised tree (trunk + elliptical crown layers)."""
    crown_cy = canopy_top - crown_ry
    crown_bottom = crown_cy - crown_ry

    lw = max(2.0, canopy_top / 10)
    ax.plot([x, x], [0, crown_bottom + crown_ry * 0.3],
            color=trunk_color, linewidth=lw, alpha=0.6, zorder=2,
            solid_capstyle='round')

    # Small branches
    bh = crown_bottom + crown_ry * 0.4
    ax.plot([x, x - 0.5], [bh - 1.5, bh],
            color=trunk_color, linewidth=0.9, alpha=0.35, zorder=2)
    ax.plot([x, x + 0.5], [bh - 0.8, bh + 0.5],
            color=trunk_color, linewidth=0.9, alpha=0.35, zorder=2)

    # Crown (layered ellipses)
    layers = [('#1a4d0a', 0.55, 0, 0, 1.0),
              ('#27500A', 0.42, -0.2, 0.3, 0.82),
              ('#3B6D11', 0.35, 0.25, -0.25, 0.65)]
    for cc, ca, ox, oy, sc in layers:
        e = Ellipse((x + ox, crown_cy + oy),
                    crown_rx * 2 * sc, crown_ry * 2 * sc,
                    facecolor=cc, edgecolor='none', alpha=ca, zorder=3)
        ax.add_patch(e)


def plot_forest(ax, rh, rh_markers, cfg):
    """Panel 3: stylised forest cross-section."""
    ax.set_xlim(0, 10)
    ax.set_title('Forest structure', fontsize=cfg['title'], pad=cfg['title_pad'])
    ax.set_xticks([])
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ground
    ax.axhline(y=0, color='#8B7355', linewidth=3, alpha=0.5, zorder=1)

    # RH dashed lines
    for p, color, _ in rh_markers:
        ax.axhline(y=rh[p], color=color, linewidth=0.4,
                   linestyle='--', alpha=0.3, zorder=1)

    # Trees (tallest reaches RH98)
    draw_tree(ax, 3.5, rh[98], crown_rx=1.6, crown_ry=4.0)
    draw_tree(ax, 7.0, 25.0,   crown_rx=1.3, crown_ry=3.2)
    draw_tree(ax, 1.5, 19.0,   crown_rx=1.1, crown_ry=2.3)
    draw_tree(ax, 5.5, 13.0,   crown_rx=0.9, crown_ry=1.8)
    draw_tree(ax, 8.5, 14.0,   crown_rx=0.8, crown_ry=1.6)

    # Understory shrubs
    for sx in [2.5, 4.5, 6.5, 8.0]:
        e = Ellipse((sx, 2.5), 1.0, 2.0, facecolor='#97C459',
                    edgecolor='none', alpha=0.22, zorder=2)
        ax.add_patch(e)


def add_ground_line(fig, ax_left, ax_right):
    """Draw a continuous dashed ground line (y=0) across all panels."""
    y_px = ax_left.transData.transform((0, 0))[1]
    y_fig = fig.transFigure.inverted().transform((0, y_px))[1]
    x0 = ax_left.get_position().x0
    x1 = ax_right.get_position().x1

    line = mlines.Line2D([x0, x1], [y_fig, y_fig],
                         transform=fig.transFigure, color='#8B7355',
                         linewidth=2, linestyle='--', alpha=0.6, zorder=5)
    fig.add_artist(line)
    return line


def _strip_labels(ax):
    """Remove title, axis labels, and tick labels from an axes."""
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_figure(rh, out_path, **kwargs):
    """Compose and save the three-panel GEDI illustration."""

    percentiles = np.arange(0, len(rh))

    # Style config
    cfg = dict(label=FONT_SIZES['label'], ticks=FONT_SIZES['ticks'],
               annot=FONT_SIZES['annot'], title=FONT_SIZES['title'],
               legend=FONT_SIZES['legend'], title_pad=16)

    # RH percentile markers to highlight
    rh_markers = [
        (25,  '#1D9E75', 'RH25'),
        (50,  '#378ADD', 'RH50'),
        (75,  '#7F77DD', 'RH75'),
        (98,  '#D85A30', 'RH98'),
        (100, '#E24B4A', 'RH100'),
    ]

    # RH annotation configs  (percentile | None for ground, color, format, y-offset)
    label_configs = [
        (None, '#8B7355', 'Ground',           -1.5),
        (25,   '#1D9E75', 'RH25 = {:.1f} m',   0.6),
        (50,   '#378ADD', 'RH50 = {:.1f} m',   0.6),
        (75,   '#7F77DD', 'RH75 = {:.1f} m',   0.6),
        (98,   '#D85A30', 'RH98 = {:.1f} m',  -1.2),
        (100,  '#E24B4A', 'RH100 = {:.1f} m',  0.6),
    ]

    # Derived data
    bin_size = 0.4
    h_min, h_max = -8, 32
    bin_centers, bin_amp_pct, smoothed_pct = derive_vertical_profile(
        rh, bin_size=bin_size, h_min=h_min, h_max=h_max)

    # Layout
    fig = plt.figure(figsize=kwargs.get('figsize', FIGURE_SIZES['wide']))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.5, 3.5, 1.6])
    gs.update(wspace=0.15)
    fig.subplots_adjust(right=0.78)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

    # Push panel 3 right to make room for annotations
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0 + 0.06, pos3.y0, pos3.width, pos3.height])

    for ax in (ax1, ax2, ax3):
        ax.set_ylim(h_min, h_max)

    # Draw panels
    plot_rh_curve(ax1, percentiles, rh, rh_markers, cfg)
    xmax = plot_vertical_profile(ax2, bin_centers, bin_amp_pct, smoothed_pct,
                                 bin_size, rh, rh_markers, cfg)
    plot_forest(ax3, rh, rh_markers, cfg)

    # Annotations & cross-panel ground line
    add_rh_annotations(fig, ax2, rh, label_configs, cfg)
    add_ground_line(fig, ax1, ax3)

    # Save
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix('.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(out_path.with_suffix('.pdf'), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved → {out_path.with_suffix('.png')}")
    print(f"Saved → {out_path.with_suffix('.pdf')}")


def make_separate_figures(rh, out_dir, annot_rhs: bool = False, **kwargs):
    """Export each panel as a separate figure with transparent background."""

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    percentiles = np.arange(0, len(rh))
    cfg = dict(label=FONT_SIZES['label'], ticks=FONT_SIZES['ticks'],
               annot=FONT_SIZES['annot'], title=FONT_SIZES['title'],
               legend=FONT_SIZES['legend'], title_pad=16)

    rh_markers = [
        (25,  '#1D9E75', 'RH25'),
        (50,  '#378ADD', 'RH50'),
        (75,  '#7F77DD', 'RH75'),
        (98,  '#D85A30', 'RH98'),
        (100, '#E24B4A', 'RH100'),
    ]

    label_configs = [
        (None, '#8B7355', 'Ground',           -1.5),
        (25,   '#1D9E75', 'RH25 = {:.1f} m',   0.6),
        (50,   '#378ADD', 'RH50 = {:.1f} m',   0.6),
        (75,   '#7F77DD', 'RH75 = {:.1f} m',   0.6),
        (98,   '#D85A30', 'RH98 = {:.1f} m',  -1.2),
        (100,  '#E24B4A', 'RH100 = {:.1f} m',  0.6),
    ]

    bin_size = 0.4
    h_min, h_max = -8, 32
    bin_centers, bin_amp_pct, smoothed_pct = derive_vertical_profile(
        rh, bin_size=bin_size, h_min=h_min, h_max=h_max)

    save_kw = dict(dpi=300, bbox_inches='tight', transparent=True)

    # --- Panel 1: RH curve ---
    fig1, ax1 = plt.subplots(figsize=kwargs.get('figsize', FIGURE_SIZES['mini']))
    ax1.set_ylim(h_min, h_max)
    plot_rh_curve(ax1, percentiles, rh, rh_markers, cfg)
    # ax1.axhline(y=0, color='#8B7355', linewidth=0.8, linestyle='--', alpha=0.5)
    _strip_labels(ax1)
    for ext in ('.png', '.pdf'):
        fig1.savefig(out_dir / f'rh_curve{ext}', **save_kw)
    plt.close(fig1)

    # --- Panel 2: Vertical profile ---
    fig2, ax2 = plt.subplots(figsize=kwargs.get('figsize', FIGURE_SIZES['mini']))
    ax2.set_ylim(h_min, h_max)
    xmax = plot_vertical_profile(ax2, bin_centers, bin_amp_pct, smoothed_pct,
                                 bin_size, rh, rh_markers, cfg)
    # ax2.axhline(y=0, color='#8B7355', linewidth=0.8, linestyle='--', alpha=0.5)
    if annot_rhs:
        add_rh_annotations(fig2, ax2, rh, label_configs, cfg)
    _strip_labels(ax2)
    for ext in ('.png', '.pdf'):
        fig2.savefig(out_dir / f'vertical_profile{ext}', **save_kw)
    plt.close(fig2)

    # --- Panel 3: Forest structure ---
    fig3, ax3 = plt.subplots(figsize=kwargs.get('figsize', FIGURE_SIZES['mini']))
    ax3.set_ylim(h_min, h_max)
    plot_forest(ax3, rh, rh_markers, cfg)
    # ax3.axhline(y=0, color='#8B7355', linewidth=0.8, linestyle='--', alpha=0.5)
    _strip_labels(ax3)
    for ext in ('.png', '.pdf'):
        fig3.savefig(out_dir / f'forest_structure{ext}', **save_kw)
    plt.close(fig3)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rh = np.array([
        -6.80,-5.41,-4.40,-3.28,-1.86,-0.78, 0.14, 1.15, 2.05, 2.69,
         3.17, 3.73, 4.40, 5.04, 5.64, 6.24, 6.83, 7.43, 8.03, 8.70,
         9.60,10.46,11.24,12.03,12.74,13.34,13.82,14.27,14.64,14.98,
        15.28,15.58,15.84,16.10,16.36,16.59,16.81,17.00,17.19,17.37,
        17.52,17.71,17.86,18.01,18.16,18.31,18.42,18.57,18.72,18.87,
        19.02,19.17,19.35,19.50,19.65,19.80,19.99,20.14,20.32,20.51,
        20.66,20.85,21.07,21.26,21.45,21.67,21.86,22.08,22.27,22.45,
        22.68,22.87,23.05,23.28,23.46,23.65,23.84,24.02,24.21,24.36,
        24.55,24.73,24.88,25.07,25.26,25.41,25.59,25.78,25.97,26.15,
        26.38,26.57,26.83,27.09,27.39,27.69,28.02,28.43,28.92,29.52,
        30.68
    ])

    make_figure(rh, '~/gvsm/results/illustrations/gedi_rh_vertical_profile')
    make_separate_figures(rh, '~/gvsm/results/illustrations/gedi_panels')