'''Shared plotting helpers for the evaluation module.

`hexbin_density_plot` is the common implementation behind the density scatter
plots in on_als / on_wsci / on_diversity_indices: a log-normed hexbin of two
variables with a reference line (1:1 identity or least-squares regression), an
optional sample-count colorbar, and a boxed stats annotation. Each caller keeps
its own stat formatting and passes the finished text in via `annotation`.

This module deliberately does NOT call set_plot_fonts; the importing module
owns the font configuration. FONT_SIZES is read at call time, so per-module
overrides (e.g. set_plot_fonts(annot=20)) are picked up.
'''
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress

from const import FONT_SIZES, FIGURE_SIZES, fewer_ticks


# corner name -> (x, y, ha, va) in axes-fraction coords for boxed annotations.
_ANNOT_CORNERS = {
    'upper left':  (0.05, 0.95, 'left',  'top'),
    'upper right': (0.95, 0.95, 'right', 'top'),
    'lower left':  (0.05, 0.05, 'left',  'bottom'),
    'lower right': (0.95, 0.05, 'right', 'bottom'),
}


def sci_notation(n) -> str:
    '''Mathtext scientific notation for use inside ``$...$``, e.g.
    1234567 -> "1.23 \\times 10^{6}". Renders as 1.23 x 10^6 with a proper
    superscript, matching the $R^2$ / $r$ mathtext annotations.'''
    mant, exp = f'{n:.2e}'.split('e')
    return f'{mant} \\times 10^{{{int(exp)}}}'


def annotate_stats(ax, text: str, corner: str = 'upper left', pos: tuple = None,
                   fontsize: float = None) -> None:
    '''Draw `text` in a rounded white box anchored at a corner of `ax`.

    `corner` is one of `_ANNOT_CORNERS`. `pos`, if given, overrides it with an
    explicit (x, y, ha, va) tuple in axes-fraction coordinates. `fontsize`
    overrides the default `FONT_SIZES['annot']`.
    '''
    x, y, ha, va = pos if pos is not None else _ANNOT_CORNERS[corner]
    ax.text(
        x, y, text,
        ha=ha, va=va, transform=ax.transAxes,
        fontsize=fontsize or FONT_SIZES['annot'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )


def hexbin_density_plot(
    x,
    y,
    *,
    ax=None,
    save_path: Path = None,
    figsize: tuple = None,
    gridsize: int = 50,
    cmap: str = 'Greens',
    vmin: float = 1,
    vmax: float = None,
    extent: tuple = None,
    refline: str = 'identity',
    equal_aspect: bool = False,
    annotation: str = None,
    annot_corner: str = 'upper left',
    annot_pos: tuple = None,
    annot_fontsize: float = None,
    show_colorbar: bool = True,
    reserve_colorbar_slot: bool = False,
    colorbar_label: str = 'Sample count',
    x_label: str = None,
    y_label: str = None,
    label_fontsize: float = None,
    title: str = None,
    title_fontsize: float = None,
    nbins: int = 3,
    dpi: int = None,
):
    '''Log-normed hexbin density scatter shared by the evaluation plots.

    Parameters
    ----------
    x, y : array-like
        The two variables to plot (x horizontal, y vertical).
    ax : matplotlib Axes, optional
        Draw into this axes instead of creating a new figure. Used to build
        multi-panel grids; pair with `show_colorbar=False` and add one shared
        colorbar at the figure level from the returned mappable.
    save_path : Path, optional
        If given, the figure is saved here (bbox_inches='tight') and closed.
    extent : (xmin, xmax, ymin, ymax), optional
        Hexbin extent and axis limits. Defaults to the data min/max.
    refline : {'identity', 'regression', None}
        '1:1' diagonal, least-squares fit line, or no reference line.
    equal_aspect : bool
        Force a 1:1 data aspect ratio (square plotting box).
    annotation : str, optional
        Boxed stats text; placement set by `annot_corner` / `annot_pos`.
    reserve_colorbar_slot : bool
        Always carve out the colorbar axes (kept invisible when
        `show_colorbar` is False) so the main axes width is identical with or
        without a colorbar. Matches the on_wsci layout.

    Returns
    -------
    (fig, ax, hb)
        `hb` is the hexbin mappable, for building a shared colorbar.
    '''
    x = np.asarray(x)
    y = np.asarray(y)

    if extent is None:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
    else:
        xmin, xmax, ymin, ymax = extent

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize or FIGURE_SIZES['square'])
    else:
        fig = ax.figure

    hb = ax.hexbin(
        x, y,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=1,
        edgecolors='none',
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=[xmin, xmax, ymin, ymax],
    )

    # ── Reference line ─────────────────────────────────────────────
    if refline == 'identity':
        lo, hi = min(xmin, ymin), max(xmax, ymax)
        ax.plot([lo, hi], [lo, hi], color='black', linestyle='--')
    elif refline == 'regression':
        slope, intercept, *_ = linregress(x, y)
        x_fit = np.linspace(xmin, xmax, 100)
        ax.plot(x_fit, slope * x_fit + intercept,
                color='black', linewidth=1.5, linestyle='--')
    elif refline is not None:
        raise ValueError(
            f"refline must be 'identity', 'regression' or None (got {refline!r})")

    # ── Colorbar ───────────────────────────────────────────────────
    # Drawn through a divider so the main axes keeps its size/aspect. When
    # reserve_colorbar_slot is set, the slot is carved out either way so the
    # plotting box has the same width with or without a visible colorbar.
    if show_colorbar or reserve_colorbar_slot:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        if show_colorbar:
            cb = fig.colorbar(hb, cax=cax)
            cb.set_label(colorbar_label, fontsize=FONT_SIZES['colorbar'])
            cb.ax.tick_params(labelsize=FONT_SIZES['ticks'])
        else:
            cax.set_visible(False)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=label_fontsize or FONT_SIZES['label'])
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=label_fontsize or FONT_SIZES['label'])
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize or FONT_SIZES['title'])

    fewer_ticks(ax, nbins=nbins)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['ticks'])
    if equal_aspect:
        ax.set_aspect('equal')

    if annotation is not None:
        annotate_stats(ax, annotation, corner=annot_corner, pos=annot_pos,
                       fontsize=annot_fontsize)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    return fig, ax, hb


def hexbin_density_grid(
    panels,
    *,
    save_path: Path = None,
    nrows: int = 3,
    ncols: int = 4,
    figsize: tuple = None,
    panel_size: float = 4.0,
    gridsize: int = 50,
    cmap: str = 'Greens',
    vmin: float = 1,
    vmax: float = None,
    extent: tuple = None,
    refline: str = 'identity',
    equal_aspect: bool = True,
    annot_corner: str = 'upper left',
    annot_fontsize: float = None,
    colorbar_label: str = 'Sample count',
    colorbar_width: float = 0.012,
    colorbar_pad: float = 0.02,
    x_label: str = None,
    y_label: str = None,
    panel_label_fontsize: float = None,
    panel_title_fontsize: float = None,
    nbins: int = 3,
    dpi: int = None,
):
    '''Grid of `hexbin_density_plot` panels with shared x/y axes and one colorbar.

    `panels` is a list of dicts, each with keys 'x', 'y' and optionally
    'annotation', 'title', 'x_label' and 'y_label'. Panels fill the grid
    row-major; any leftover axes are hidden. Every panel uses the same
    LogNorm(vmin, vmax), so a single figure-level colorbar (built from the last
    panel's mappable) applies to all.

    The figure-wide `x_label` / `y_label` name the axes once via
    `fig.supxlabel` / `fig.supylabel`. Per-panel 'x_label' / 'y_label' (e.g. an
    acquisition year) are drawn on each axes at `panel_label_fontsize`.

    Returns
    -------
    (fig, axes)
    '''
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize or (panel_size * ncols, panel_size * nrows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    if len(panels) > len(axes_flat):
        print(f'  hexbin_density_grid: {len(panels)} panels > {nrows}x{ncols} '
              f'cells; plotting the first {len(axes_flat)}')

    hb = None
    for ax, panel in zip(axes_flat, panels):
        # 'x'/'y' may be zero-arg callables so a caller with large per-panel
        # data can load one panel at a time instead of holding every panel's
        # arrays at once; hexbin only keeps the aggregated counts.
        px, py = panel['x'], panel['y']
        _, _, hb = hexbin_density_plot(
            px() if callable(px) else px, py() if callable(py) else py,
            ax=ax,
            gridsize=gridsize, cmap=cmap, vmin=vmin, vmax=vmax,
            extent=extent,
            refline=refline, equal_aspect=equal_aspect,
            annotation=panel.get('annotation'),
            annot_corner=annot_corner, annot_fontsize=annot_fontsize,
            title=panel.get('title'), title_fontsize=panel_title_fontsize,
            x_label=panel.get('x_label'), y_label=panel.get('y_label'),
            label_fontsize=panel_label_fontsize,
            show_colorbar=False,
            nbins=nbins,
        )

    # Hide any unused cells; re-expose x tick labels on the lowest visible row
    # of each column so a partially filled grid still shows its x axis.
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)
    for col in range(ncols):
        col_axes = [axes_flat[row * ncols + col] for row in range(nrows)]
        visible = [a for a in col_axes if a.get_visible()]
        if visible:
            visible[-1].tick_params(axis='x', labelbottom=True)

    if x_label is not None:
        fig.supxlabel(x_label, fontsize=FONT_SIZES['label'])
    if y_label is not None:
        fig.supylabel(y_label, fontsize=FONT_SIZES['label'])

    # Single colorbar pinned to the drawn panel stack. The panels are
    # equal-aspect, so constrained_layout leaves vertical slack; a colorbar that
    # stole space from the axes group would span that slack and read taller than
    # the panels. Instead, let constrained_layout finalize the panel positions,
    # freeze the layout, then add a slim colorbar axes spanning exactly from the
    # lowest panel bottom to the highest panel top, just right of the panels.
    if hb is not None:
        fig.canvas.draw()  # finalize constrained_layout (and aspect squeeze)
        try:
            fig.set_layout_engine('none')  # freeze positions for manual cax
        except (AttributeError, ValueError):
            fig.set_constrained_layout(False)
        boxes = [a.get_position() for a in axes_flat if a.get_visible()]
        x1 = max(b.x1 for b in boxes)
        y0 = min(b.y0 for b in boxes)
        y1 = max(b.y1 for b in boxes)
        cax = fig.add_axes([x1 + colorbar_pad, y0, colorbar_width, y1 - y0])
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label(colorbar_label, fontsize=FONT_SIZES['colorbar'])
        cb.ax.tick_params(labelsize=FONT_SIZES['ticks'])

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    return fig, axes
