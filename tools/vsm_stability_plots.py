"""Figures for the 2017-2026 VSM stability report (docs/reports/vsm_stability_2017_2026.md).

Inputs are the summary tables in vsm_stability_state/2017_2026 written by tools/vsm_stability.py
(summarize_stability_years, count_stability_images, summarize_stability_clouds,
summarize_stability_reflectance); the schematic reads the sampled values in vsm_stability/2017_2026.

    python -m tools.run run=plot_stability_report     # fig1-fig4
    python -m tools.run run=plot_stability_inputs     # fig5-fig6
    python -m tools.run run=plot_stability_per_image  # fig7
    python -m tools.run run=plot_stability_per_image_controls  # fig7b
    python -m tools.run run=plot_stability_inputs_2024  # fig8-fig10
    python -m tools.run run=plot_stability_anomaly_test  # fig11
    python -m tools.run run=plot_stability_schematic  # fig0
    python -m tools.run run=plot_stability_methods  # m01-m10, toy-data sketches of the test methods
    python -m tools.run run=plot_stability_sample_points  # m00, the 20 000 sample points of a tile
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Polygon

BIOMES = {1: 'Trop. moist broadleaf', 2: 'Trop. dry broadleaf', 3: 'Trop. coniferous',
          4: 'Temp. broadleaf & mixed', 5: 'Temp. coniferous', 6: 'Boreal', 7: 'Trop. savanna',
          8: 'Temp. grassland', 9: 'Flooded grassland', 10: 'Montane grassland', 11: 'Tundra',
          12: 'Mediterranean', 13: 'Desert', 14: 'Mangrove'}
INK, INK2, GRID, SURF, HL_RING = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#00c2ff'
BLUE, ORANGE, MUTED = '#2a78d6', '#eb6834', '#a9a8a4'
# diverging blue <-> red with a neutral grey midpoint
DIV = LinearSegmentedColormap.from_list('div', ['#104281', '#3987e5', '#cde2fb', '#f0efec',
                                                '#f7cfc9', '#e34948', '#8f1d1c'])
HL_YEAR = 2024


def _style(titlesize: int) -> dict:
    return {'font.family': 'sans-serif', 'font.size': 11, 'axes.edgecolor': GRID,
            'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
            'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
            'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.titlecolor': INK,
            'axes.titlesize': titlesize, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'}


def _save(fig, path: Path):
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig1-fig4: year offsets, per-pixel SD, GEDI accuracy
# ---------------------------------------------------------------------------

def _offset_heatmap(years_b: pd.DataFrame, rh: int, path: Path):
    """Year offset, biome x year (vegetated pixels, median over tiles)."""
    m = years_b[years_b.rh == rh].pivot(index='biome', columns='year', values='offset_vegetated')
    m = m.reindex(index=sorted(m.index))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    im = ax.imshow(m.values, cmap=DIV, norm=TwoSlopeNorm(0, -1.2, 1.2), aspect='auto')
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(v) > 0.8 else INK)
    ax.set_xticks(range(m.shape[1]), m.columns)
    ax.set_yticks(range(m.shape[0]), [f'{b} {BIOMES[b]}' for b in m.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, m.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, m.shape[0]), minor=True)
    ax.grid(which='minor', color=SURF, linewidth=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('m', color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f'RH{rh} Q1 year offset (m), vegetated pixels, median over tiles')
    fig.tight_layout()
    _save(fig, path)


def _tile_offsets(years_t: pd.DataFrame, path: Path):
    """Per-tile RH98 year offsets by year (box + points)."""
    years = sorted(years_t.year.unique())
    v = years_t[years_t.rh == 98]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    data_y = [v[v.year == y].offset_vegetated.dropna().values for y in years]
    ax.axhline(0, color=INK2, linewidth=1)
    ax.boxplot(data_y, positions=range(len(years)), widths=0.5, showfliers=False,
               medianprops=dict(color=BLUE, linewidth=2), boxprops=dict(color=INK2),
               whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    rng = np.random.default_rng(0)
    for i, d in enumerate(data_y):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=12, color=BLUE, alpha=0.45, linewidths=0)
    ax.set_xticks(range(len(years)), years)
    ax.set_ylabel('year offset (m)')
    ax.set_ylim(-4, 4)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_title('RH98 Q1 year offset per tile (vegetated pixels; 60 tiles with >= 100 such pixels)')
    fig.tight_layout()
    _save(fig, path)


def _pixel_sd(pix: pd.DataFrame, path: Path):
    """Per-pixel inter-annual SD by biome (RH98, vegetated pixels)."""
    p = pix[(pix.rh == 98) & (pix.pixels == 'vegetated')].sort_values('biome')
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(p))[::-1]
    ax.barh(y, p.sd_median, height=0.6, color=BLUE, label='per-pixel SD, median over pixels')
    ax.scatter(p.sd_p90, y, s=40, color=ORANGE, zorder=3, edgecolors=SURF, linewidths=2,
               label='per-pixel SD, 90th percentile')
    ax.scatter(p.tile_mean_sd, y, s=40, marker='D', color=INK2, zorder=3, edgecolors=SURF, linewidths=2,
               label='SD of the tile mean')
    ax.set_yticks(y, [f'{b} {BIOMES[b]}' for b in p.biome])
    ax.set_xlabel('SD over 10 years (m)')
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=9)
    ax.set_title('Inter-annual SD of RH98 Q1, vegetated pixels')
    fig.tight_layout()
    _save(fig, path)


def _gedi_per_year(gedi: pd.DataFrame, path: Path):
    """GEDI coverage and bias per year (two panels, all tiles)."""
    g = gedi[gedi.biome == 0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, metric, title, ref in ((axes[0], 'coverage', '[Q2, Q0] coverage of GEDI', 0.9),
                                   (axes[1], 'bias', 'Bias median(Q1 - GEDI) (m)', 0.0)):
        ax.axhline(ref, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        for rh, col in ((50, BLUE), (98, ORANGE)):
            s = g[g.rh == rh].sort_values('year')
            ax.plot(s.year, s[metric], color=col, linewidth=2, marker='o', markersize=7,
                    markeredgecolor=SURF, markeredgewidth=2)
            ax.plot([], [], color=col, linewidth=2, marker='o', label=f'RH{rh}')
        ax.set_xticks([2019, 2020, 2021, 2022, 2024])
        ax.set_xlim(2018.6, 2024.9)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.suptitle('Accuracy at GEDI shots per year (sensitivity >= 0.95, all 64 tiles with GEDI)',
                 x=0.01, ha='left', fontweight='bold', color=INK)
    fig.tight_layout()
    _save(fig, path)


def plot_stability_report(tables_dir: str, out_dir: str, **kwargs):
    """fig1/fig1b (year offset heatmaps RH98/RH50), fig2 (per-tile offsets), fig3 (per-pixel SD),
    fig4 (GEDI per year) from the summarize_stability_years tables."""
    data, out = Path(tables_dir).expanduser(), Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    years_t = pd.read_csv(data / 'stability_years_tiles.tsv', sep='\t')
    years_t = years_t[years_t.status == 'ok']
    years_b = pd.read_csv(data / 'stability_years_biome.tsv', sep='\t')
    pix = pd.read_csv(data / 'stability_pixels_biome.tsv', sep='\t')
    gedi = pd.read_csv(data / 'stability_gedi.tsv', sep='\t')
    with plt.rc_context(_style(12)):
        _offset_heatmap(years_b, 98, out / 'fig1_year_offset_rh98.png')
        _offset_heatmap(years_b, 50, out / 'fig1b_year_offset_rh50.png')
        _tile_offsets(years_t, out / 'fig2_tile_offsets_rh98.png')
        _pixel_sd(pix, out / 'fig3_pixel_sd_rh98.png')
        _gedi_per_year(gedi, out / 'fig4_gedi_per_year.png')
    print('figures ->', out)


# ---------------------------------------------------------------------------
# fig5-fig6: inputs per year, reflectance vs year offset
# ---------------------------------------------------------------------------

INPUT_YEARS = list(range(2017, 2026))   # complete years
REFLECTANCE_COLS = ['B02', 'B03', 'B04', 'B05', 'B08', 'B11', 'B12', 'NDVI']


def _circ_mean(doy):
    t = 2 * np.pi * np.asarray(doy) / 365.25
    return (np.angle(np.mean(np.exp(1j * t))) % (2 * np.pi)) * 365.25 / (2 * np.pi)


def _date_shift(selected_file: Path) -> pd.DataFrame:
    """Mean acquisition date of a tile-year minus the tile's usual date (2018-2025), days."""
    d = pd.read_parquet(selected_file)
    d['doy'] = pd.to_datetime(d.datetime, format='ISO8601').dt.dayofyear
    s = d.groupby(['tile', 'year']).doy.apply(_circ_mean).rename('doy').reset_index()
    ref = s[s.year.between(2018, 2025)].groupby('tile').doy.apply(_circ_mean).rename('ref')
    s = s.join(ref, on='tile')
    s['shift'] = ((s.doy - s.ref + 182.6) % 365.25) - 182.6
    return s


def _year_ticks(ax, pos):
    """Two-digit year labels; 2024 in orange."""
    ax.set_xticks(list(pos), [f"'{str(y)[2:]}" for y in INPUT_YEARS])
    for y, lab in zip(INPUT_YEARS, ax.get_xticklabels()):
        if y == HL_YEAR:
            lab.set_color(ORANGE)
            lab.set_fontweight('bold')


def _year_boxes(ax, df, col, title, ylabel, ylim=None):
    vals = [df[df.year == y][col].dropna().values for y in INPUT_YEARS]
    b = ax.boxplot(vals, positions=range(len(INPUT_YEARS)), widths=0.55, showfliers=False, patch_artist=True,
                   medianprops=dict(color=INK, linewidth=1.6), whiskerprops=dict(color=INK2),
                   capprops=dict(color=INK2), boxprops=dict(edgecolor=INK2))
    for y, patch in zip(INPUT_YEARS, b['boxes']):
        patch.set_facecolor(ORANGE if y == HL_YEAR else '#dcdbd7')
        patch.set_edgecolor(ORANGE if y == HL_YEAR else INK2)
    _year_ticks(ax, range(len(INPUT_YEARS)))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _inputs_per_year(data: Path, path: Path):
    """Image cloud cover, clear observations, residual cloud and date shift per year."""
    img = pd.read_csv(data / 'stability_images.tsv', sep='\t')
    clouds = pd.read_csv(data / 'stability_clouds_tiles.tsv', sep='\t')
    clouds['leak_pct'] = clouds.leak_share_mean * 100
    dates = _date_shift(data / 'stability_selected_images.parquet')
    dates['abs_shift'] = dates['shift'].abs()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    _year_boxes(axes[0], img, 'cloud_selected_mean', 'Scene cloud cover', '% (selected images)', (0, 60))
    _year_boxes(axes[1], clouds, 'n_used_median', 'Clear observations', 'per pixel (SCL)', (0, 22))
    _year_boxes(axes[2], clouds, 'leak_pct', 'Residual cloud', '% of clear obs. (Cloud Score+)', (0, 40))
    _year_boxes(axes[3], dates, 'abs_shift', 'Acquisition date shift', '|days| from the tile\'s usual date',
                (0, 60))
    fig.suptitle('Inputs per tile and year (box = tiles): 2024 (orange) is ordinary', x=0.01, ha='left',
                 fontweight='bold', color=INK)
    fig.tight_layout()
    _save(fig, path)


def _leave_year_out(r: pd.DataFrame) -> pd.DataFrame:
    """Median RH98 year offset per year, observed vs predicted from reflectance by a linear fit on the
    other years."""
    rows = []
    for y in INPUT_YEARS:
        train, test = r[r.year != y], r[r.year == y]
        a = np.c_[np.ones(len(train)), train[REFLECTANCE_COLS].values / 100]
        c, *_ = np.linalg.lstsq(a, train.rh98_offset.values, rcond=None)
        p = np.c_[np.ones(len(test)), test[REFLECTANCE_COLS].values / 100] @ c
        rows.append(dict(year=y, observed=test.rh98_offset.median(), from_reflectance=np.median(p)))
    return pd.DataFrame(rows)


def _reflectance_offset(data: Path, path: Path) -> pd.DataFrame:
    """NDVI / red offset vs RH98 year offset, and the share of the year offset reflectance explains."""
    r = pd.read_csv(data / 'stability_reflectance_tiles.tsv', sep='\t')
    r = r[(r['mask'] == 'scl_cs') & r.year.isin(INPUT_YEARS)].dropna(subset=REFLECTANCE_COLS)
    lyo = _leave_year_out(r)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    for ax, col, scale, xlabel in ((axes[0], 'NDVI', 1e-4, 'NDVI offset'),
                                   (axes[1], 'B04', 1e-4, 'Red (B04) reflectance offset')):
        o = r[r.year != HL_YEAR]
        h = r[r.year == HL_YEAR]
        ax.axhline(0, color=GRID, linewidth=1)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.scatter(o[col] * scale, o.rh98_offset, s=14, color=MUTED, alpha=0.6, linewidths=0, label='2017–2025')
        ax.scatter(h[col] * scale, h.rh98_offset, s=26, color=ORANGE, edgecolors=SURF, linewidths=0.8,
                   label='2024')
        rho = r[col].corr(r.rh98_offset, method='spearman')
        ax.text(0.97, 0.04, f'Spearman {rho:+.2f}'.replace('-', '−'), transform=ax.transAxes, ha='right',
                va='bottom', color=INK2, fontsize=10, bbox=dict(facecolor=SURF, edgecolor='none', pad=2))
        lo, hi = np.nanpercentile(r[col] * scale, [1, 99])
        pad = (hi - lo) * 0.1
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-2.5, 3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('RH98 year offset (m)')
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_title('Greener inputs → taller')
    axes[1].set_title('Darker red → taller')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    ax = axes[2]
    x = np.arange(len(INPUT_YEARS))
    ax.axhline(0, color=INK2, linewidth=1)
    ax.bar(x - 0.18, lyo.observed, width=0.34, color=[ORANGE if y == HL_YEAR else BLUE for y in INPUT_YEARS],
           label='observed')
    ax.bar(x + 0.18, lyo.from_reflectance, width=0.34, color='#dcdbd7', edgecolor=INK2, linewidth=0.6,
           label='expected from reflectance')
    _year_ticks(ax, x)
    ax.set_ylabel('median RH98 year offset (m)')
    ax.set_title('Reflectance explains ~⅓ of 2024')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.suptitle('Tile-year offsets of RH98 vs the inputs (vegetated pixels; clear = SCL and Cloud Score+)',
                 x=0.01, ha='left', fontweight='bold', color=INK)
    fig.tight_layout()
    _save(fig, path)
    return lyo


def plot_stability_inputs(tables_dir: str, out_dir: str, **kwargs):
    """fig5 (inputs per year) and fig6 (reflectance vs year offset). Tables: stability_images.tsv,
    stability_selected_images.parquet, stability_clouds_tiles.tsv, stability_reflectance_tiles.tsv."""
    data, out = Path(tables_dir).expanduser(), Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_style(11)):
        _inputs_per_year(data, out / 'fig5_inputs_per_year.png')
        lyo = _reflectance_offset(data, out / 'fig6_reflectance_offset.png')
    print(lyo.round(2).to_string(index=False))
    print('figures ->', out)


# ---------------------------------------------------------------------------
# fig7: per-image predictions on the strongest 2024 tiles
# ---------------------------------------------------------------------------

SEASON_WINDOW = 20   # days on either side for the "same season in the other years" reference


def _same_season(p: pd.DataFrame) -> pd.DataFrame:
    """Per tile-year: median over its images of (image offset - median offset of the other years'
    images within +-SEASON_WINDOW days of the same day of year)."""
    rows = []
    for (tile, year), cur in p.groupby(['tile', 'year']):
        other = p[(p.tile == tile) & (p.year != year)]
        d = [o - other[(other.doy - doy).abs() <= SEASON_WINDOW].offset.median()
             for doy, o in zip(cur.doy, cur.offset)]
        rows.append(dict(tile=tile, year=year, same_season=np.nanmedian(d)))
    return pd.DataFrame(rows)


def _running_median(doy, val, grid, half=SEASON_WINDOW, min_n=3):
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        w = val[np.abs(doy - g) <= half]
        if np.isfinite(w).sum() >= min_n:
            out[i] = np.nanmedian(w)
    return out


def plot_stability_per_image(tables_dir: str, out_file: str, tiles=None, **kwargs):
    """fig7: RH98 Q1 offset of every single image (vs the pixel's multi-year median) over the season,
    2024 against the other years, plus each tile-year's offset against the same season of the other
    years. Table: stability_per_image.tsv (summarize_stability_per_image).
    tiles: the five tiles to show (the table may hold more, e.g. the control tiles)."""
    data, out = Path(tables_dir).expanduser(), Path(out_file).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(data / 'stability_per_image.tsv', sep='\t', parse_dates=['date'])
    if tiles is not None:
        p = p[p.tile.isin(list(tiles))]
    p['doy'] = p.date.dt.dayofyear
    ss = _same_season(p)
    tiles = list(p.groupby('tile').biome.first().sort_values().index)
    month_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    grid = np.arange(1, 366)
    with plt.rc_context(_style(11)):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.0))
        for ax, tile in zip(axes.flat, tiles):
            g = p[p.tile == tile]
            o, h = g[g.year != HL_YEAR], g[g.year == HL_YEAR]
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.scatter(o.doy, o.offset, s=14, color=MUTED, alpha=0.7, linewidths=0, label='2022, 2023, 2025')
            ax.scatter(h.doy, h.offset, s=28, color=ORANGE, edgecolors=SURF, linewidths=0.8, label='2024')
            for d, col in ((o, MUTED), (h, ORANGE)):
                ax.plot(grid, _running_median(d.doy.values, d.offset.values, grid), color=col, linewidth=2)
            v = ss[(ss.tile == tile) & (ss.year == HL_YEAR)].same_season.iloc[0]
            ax.text(0.03, 0.96, f'2024 vs same season: {v:+.1f} m'.replace('-', '−'), transform=ax.transAxes,
                    va='top', fontsize=9.5, color=INK, bbox=dict(facecolor=SURF, edgecolor='none', pad=2))
            used = g.doy
            ax.set_xlim(max(used.min() - 15, 1), min(used.max() + 15, 365))
            ax.set_xticks(month_doy, ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax.set_ylim(-8, 7)
            ax.grid(axis='y', color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_title(f'{tile}  {BIOMES[int(g.biome.iloc[0])]}', fontsize=10.5)
        for ax in axes[:, 0]:
            ax.set_ylabel('image offset (m)')
        axes.flat[0].legend(frameon=False, fontsize=8.5, loc='lower right')

        ax = axes.flat[5]
        years = sorted(ss.year.unique())
        ax.axhline(0, color=INK2, linewidth=0.8)
        for i, y in enumerate(years):
            s = ss[ss.year == y]
            col = ORANGE if y == HL_YEAR else MUTED
            ax.scatter(np.full(len(s), i) + np.linspace(-0.15, 0.15, len(s)), s.same_season, s=40, color=col,
                       edgecolors=SURF, linewidths=1.2, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [s.same_season.median()] * 2, color=INK, linewidth=2)
        ax.set_xticks(range(len(years)), years)
        ax.get_xticklabels()[years.index(HL_YEAR)].set_color(ORANGE)
        ax.get_xticklabels()[years.index(HL_YEAR)].set_fontweight('bold')
        ax.set_ylabel('offset vs same season (m)')
        ax.set_ylim(-2, 2)
        ax.grid(axis='y', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title('Each year vs the same season of the others', fontsize=10.5)
        fig.suptitle('RH98 Q1 of single images (vegetated pixels, offset from the multi-year median): '
                     '2024 is higher at the same time of year', x=0.01, ha='left', fontweight='bold', color=INK)
        fig.tight_layout()
        _save(fig, out)
    print(ss.pivot(index='tile', columns='year', values='same_season').round(2).to_string())
    print('figure ->', out)


# ---------------------------------------------------------------------------
# fig0: metric schematic on real data
# ---------------------------------------------------------------------------

def plot_stability_schematic(values_dir: str, tile: str, biome: int, out_file: str,
                             years=tuple(range(2017, 2027)), n_show: int = 2500, **kwargs):
    """Metric schematic on real data: the sample points of one tile (x, y) stacked over the years.

    values_dir holds sample_pixels/ and pixel_values/. Same computation as
    tools.vsm_stability.summarize_stability_years (RH98 Q1, own-biome pixels valid in every year,
    vegetated = multi-year median >= 5 m)."""
    src = Path(values_dir).expanduser()
    out = Path(out_file).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    years = list(years)

    # ---- data, as in summarize_stability_years ----
    pts = pd.read_parquet(src / 'sample_pixels' / f'{tile}.parquet').set_index('point_id')
    q1 = np.stack([pd.read_parquet(src / 'pixel_values' / str(y) / f'{tile}.parquet')
                   .set_index('point_id').reindex(pts.index).RH98_Q1.to_numpy() for y in years])
    keep = (pts.BIOME.to_numpy() == biome) & np.isfinite(q1).all(0)
    q1, pts = q1[:, keep], pts[keep]
    med = np.median(q1, 0)
    veg = med >= 5
    dev = q1 - med
    offset = np.median(dev[:, veg], 1)
    sd = q1.std(0, ddof=1)
    print(tile, 'pixels', keep.sum(), 'vegetated', veg.sum(), 'offsets', offset.round(2),
          'median SD veg', round(float(np.median(sd[veg])), 2))

    u = (pts.lon - pts.lon.min()) / np.ptp(pts.lon)
    v = (pts.lat - pts.lat.min()) / np.ptp(pts.lat)
    rng = np.random.default_rng(0)
    show = rng.choice(len(pts), n_show, replace=False)  # plotted subset; statistics use all points
    vi = np.flatnonzero(veg)
    # a vegetated pixel that looks like plain noise: SD within 0.1 m of the median, no trend or step
    # (|correlation with the year| < 0.1, negative lag-1 autocorrelation), then the smallest largest deviation
    near = vi[np.abs(sd[vi] - np.median(sd[vi])) < 0.1]
    z = (q1[:, near] - q1[:, near].mean(0)) / q1[:, near].std(0)
    trend = np.abs(np.corrcoef(np.vstack([np.arange(len(years)), z.T]))[0, 1:])
    ok = (trend < 0.1) & ((z[1:] * z[:-1]).mean(0) < -0.1)
    pick = near[ok][np.argmin(np.abs(z[:, ok]).max(0))]
    k_hl = years.index(HL_YEAR)

    W, SX, SY, GAP = 10, 0.55, 0.32, 1.25

    def xy(x0, y0, level, uu, vv):
        return x0 + uu * W + SX * vv * W, y0 + SY * vv * W + level * GAP

    def slab(ax, vals, x0, y0, cmap, vmin, vmax, level=0, s=2.2):
        c = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        ax.add_patch(Polygon(np.column_stack(xy(x0, y0, level, c[:, 0], c[:, 1])), closed=True,
                             fc=SURF, ec=INK2, lw=0.8, zorder=level * 3))
        X, Y = xy(x0, y0, level, u.to_numpy()[show], v.to_numpy()[show])
        ax.scatter(X, Y, c=vals[show], cmap=cmap, vmin=vmin, vmax=vmax, s=s, linewidths=0, zorder=level * 3 + 1)
        px, py = xy(x0, y0, level, u.iloc[pick], v.iloc[pick])
        ax.scatter(px, py, s=36, facecolors='none', edgecolors=HL_RING, linewidths=1.8, zorder=level * 3 + 2)

    def stack(ax, data, x0, y0, cmap, vmin, vmax):
        for k in range(len(years)):
            slab(ax, data[k], x0, y0, cmap, vmin, vmax, level=k)
            if k in (0, k_hl, len(years) - 1):
                ax.text(x0 + W + SX * W + 0.6, y0 + SY * W / 2 + k * GAP, str(years[k]), fontsize=8,
                        color=ORANGE if k == k_hl else INK2, va='center',
                        fontweight='bold' if k == k_hl else None)

    def arrow(ax, a, b, text):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14, color=INK2, lw=1.2))
        ax.text((a[0] + b[0]) / 2, a[1] + 0.8, text, ha='center', va='bottom', fontsize=8.5, color=INK2)

    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 10, 'figure.facecolor': SURF}):
        fig = plt.figure(figsize=(13, 7.1))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-3, 97)
        ax.set_ylim(-6.2, 40.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # ---- A: year offset ----
        yA, x1, x2, x3 = 23, 26, 48, 80
        ax.text(-2.5, 39.3, f'A  Year offset: is one year shifted as a whole?   (tile {tile}, {BIOMES[biome]})',
                fontsize=12, fontweight='bold', color=INK)
        stack(ax, q1, 0, yA, 'inferno', 0, 50)
        ax.annotate('', xy=(-1.2, yA + 9 * GAP + 3), xytext=(-1.2, yA),
                    arrowprops=dict(arrowstyle='-|>', color=INK2))
        ax.text(-2.2, yA + 6, 'year', rotation=90, fontsize=8.5, color=INK2, ha='center')
        ax.text(W / 2, yA - 1.8, 'x', fontsize=8.5, color=INK2)
        ax.text(W + SX * W / 2 + 1.2, yA + SY * W / 2 - 1.2, 'y', fontsize=8.5, color=INK2)
        ax.text(7.5, yA - 4.4, f'Q1 RH98 at the fixed sample pixels\n'
                               f'({keep.sum():,} pixels × {len(years)} years; 0–50 m)',
                ha='center', fontsize=8.5, color=INK)

        arrow(ax, (19.5, yA + 7), (x1 - 0.5, yA + 7), 'median over years\n(each pixel)')
        slab(ax, med, x1, yA + 5.5, 'inferno', 0, 50)
        ax.text(x1 + 7.5, yA + 3.5, 'per-pixel baseline m', ha='center', fontsize=8.5, color=INK)

        arrow(ax, (x1 + 16, yA + 7), (x2 - 0.5, yA + 7), 'Q1 − m\n(every year)')
        stack(ax, dev, x2, yA, 'RdBu_r', -6, 6)
        ax.text(x2 + 7.5, yA - 4.4, 'anomaly cube (−6 … +6 m)\nbaseline removed', ha='center', fontsize=8.5,
                color=INK)

        arrow(ax, (x2 + 19.8, yA + 7), (x3 - 4.6, yA + 7), 'median over\nvegetated pixels,\nper year slice')
        bx = ax.inset_axes([x3, yA + 0.5, 16.5, 12], transform=ax.transData)
        bx.axhline(0, color=INK2, lw=0.8)
        bx.bar(range(len(years)), offset, color=[ORANGE if y == HL_YEAR else BLUE for y in years], width=0.65)
        bx.text(k_hl, offset[k_hl] + 0.05, f'+{offset[k_hl]:.2f}', ha='center', va='bottom', fontsize=8,
                color=INK)
        bx.set_ylim(offset.min() - 0.15, offset.max() + 0.4)
        bx.set_xticks(range(len(years)), [str(y)[2:] for y in years], fontsize=7.5)
        bx.tick_params(axis='y', labelsize=7.5)
        bx.set_ylabel('offset (m)', fontsize=8, color=INK2)
        for s in ('top', 'right'):
            bx.spines[s].set_visible(False)
        bx.set_facecolor(SURF)
        bx.set_title('one offset per year for this tile', fontsize=8.5, color=INK, loc='left')
        ax.text(x3 + 8, yA - 4.4, 'then: median over the tiles of a biome', ha='center', fontsize=8.5,
                color=INK2, style='italic')

        # ---- B: per-pixel SD ----
        yB = -1
        ax.text(-2.5, 16.2, 'B  Per-pixel SD: how noisy is a single pixel from year to year?', fontsize=12,
                fontweight='bold', color=INK)
        col = q1[:, pick]
        for k, val in enumerate(col):
            ax.add_patch(plt.Rectangle((4, yB + k * 1.15), 3, 1.0, color=plt.get_cmap('inferno')(val / 50)))
            ax.text(7.6, yB + k * 1.15 + 0.5, f'{years[k]}: {val:4.1f} m', fontsize=7, color=INK2, va='center')
        ax.add_patch(plt.Rectangle((4, yB), 3, 11.5, fill=False, ec=HL_RING, lw=2.2))
        ax.text(5.5, yB - 3.2, f'time column of the circled\npixel in A (SD {sd[pick]:.1f} m)', ha='center',
                fontsize=8.5, color=INK)

        arrow(ax, (16, yB + 6), (x1 - 0.5, yB + 6), 'SD over years\n(each pixel)')
        slab(ax, sd, x1, yB + 2, 'Greys', 0, 6)
        ax.text(x1 + 7.5, yB - 1.0, 'SD map s (0–6 m)', ha='center', fontsize=8.5, color=INK)

        arrow(ax, (x1 + 16, yB + 6), (x2 - 0.5, yB + 6), 'median over\nvegetated pixels')
        hx = ax.inset_axes([x2 + 3, yB + 0.5, 22, 10.5], transform=ax.transData)
        hx.hist(sd[veg], bins=np.arange(0, 8.01, 0.25), color=BLUE, edgecolor=SURF, linewidth=0.6)
        m = float(np.median(sd[veg]))
        hx.axvline(m, color=ORANGE, lw=1.6)
        hx.text(m + 0.15, hx.get_ylim()[1] * 0.92, f'median {m:.1f} m', color=INK, fontsize=8.5, va='top')
        hx.set_xlabel(f'per-pixel SD of RH98 over {len(years)} years (m)', fontsize=8, color=INK2)
        hx.tick_params(labelsize=7.5)
        hx.set_yticks([])
        for s in ('top', 'right', 'left'):
            hx.spines[s].set_visible(False)
        hx.set_facecolor(SURF)
        # report-wide numbers from stability_pixels_biome.tsv (all 69 tiles)
        ax.text(x2 + 29, yB + 6, 'all 69 tiles:\nmedian 1.7 m\n(biomes 0.9–2.3 m)', fontsize=10, color=INK,
                va='center')
        n_show_txt = f'{n_show:,}'.replace(',', ' ')
        ax.text(96.5, -5.9, f'real data: {tile}, {veg.sum():,} vegetated of {keep.sum():,} pixels; '
                            f'{n_show_txt} drawn per slice',
                ha='right', fontsize=8, color=INK2, style='italic')
        fig.savefig(out, dpi=200)
        plt.close(fig)
    print('figure ->', out)


# ---------------------------------------------------------------------------
# fig8-fig10: what is different about the 2024 inputs
# ---------------------------------------------------------------------------

STRONG_TILES = ('49UFS', '49UDT', '33PZL', '19TFN')
IMAGE_BANDS = ['veg_B02', 'veg_B04', 'veg_B05', 'veg_B08', 'veg_B11', 'veg_B12']


def _spread_labels(ax, items, x, min_gap=0.075):
    """Right-end labels (y, text, colour, weight), pushed apart to at least min_gap of the axis height."""
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    items = sorted((min(max(y, lo), hi), *rest) for y, *rest in items)
    ys = [it[0] for it in items]
    for _ in range(200):          # push overlapping neighbours apart symmetrically
        moved = False
        for i in range(1, len(ys)):
            d = ys[i] - ys[i - 1]
            if d < gap:
                ys[i - 1] -= (gap - d) / 2
                ys[i] += (gap - d) / 2
                moved = True
        if not moved:
            break
    for (_, text, color, weight), y in zip(items, ys):
        ax.text(x, y, text, fontsize=8, color=color, va='center', fontweight=weight)


def _year_lines(ax, ext, col, title, strong, label_right=True, ylim=None):
    """Same-season offset per year: one grey line per strong tile (2024 point orange), median over all tiles in ink."""
    years = sorted(ext.year.unique())
    ax.axhline(0, color=INK2, linewidth=0.8)
    labels = []
    for t in strong:
        g = ext[ext.tile == t].set_index('year')[col].reindex(years)
        ax.plot(years, g.values, color=MUTED, linewidth=1.5, marker='o', markersize=4, zorder=2)
        if HL_YEAR in g.index and np.isfinite(g[HL_YEAR]):
            ax.scatter([HL_YEAR], [g[HL_YEAR]], s=60, color=ORANGE, edgecolors=SURF, linewidths=1.5, zorder=4)
        if np.isfinite(g.iloc[-1]):
            labels.append((g.iloc[-1], t, INK2, 'normal'))
    med = ext.groupby('year')[col].median().reindex(years)
    ax.plot(years, med.values, color=INK, linewidth=2.2, zorder=3)
    labels.append((med.iloc[-1], 'all tiles', INK, 'bold'))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if label_right:
        _spread_labels(ax, labels, years[-1] + 0.15)
    ax.set_xticks(years, [str(y)[2:] for y in years])
    ax.get_xticklabels()[years.index(HL_YEAR)].set_color(ORANGE)
    ax.set_xlim(years[0] - 0.4, years[-1] + 1.2)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10.5)


def _fig8_s2_inputs(ext, strong, path):
    with plt.rc_context(_style(11)):
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
        for ax, (col, title) in zip(axes, [('s2_veg_B08', 'Vegetated pixels: NIR (B08)'),
                                           ('s2_veg_B12', 'Vegetated pixels: SWIR2 (B12)'),
                                           ('s2_water_B04', 'Water pixels: red (B04)'),
                                           ('s2_p2_B04', 'Darkest 2% of pixels: red (B04)')]):
            _year_lines(ax, ext, col, title, strong, ylim=(-0.025, 0.025))
        axes[0].set_ylabel('same-season offset (reflectance)')
        fig.suptitle('S2 inputs vs the same time of year in other years: in 2024 the strong tiles are darker in SWIR2,\n'
                     'brighter in NIR on 3 of 4; over all tiles, water and the darkest pixels stay at 0',
                     x=0.01, ha='left', fontweight='bold', color=INK, fontsize=11.5)
        fig.tight_layout()
        _save(fig, path)


def _fig9_external(ext, strong, path):
    ok = ext.dropna(subset=['modis_nir', 's2_veg_B08'])
    with plt.rc_context(_style(11)):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
        # top row: does MODIS see what S2 sees?
        for ax, (xc, yc, title, lim) in zip(axes[0], [
                ('modis_nir', 's2_veg_B08', 'NIR: S2 vs MODIS (tile-years)', 0.03),
                ('modis_swir2', 's2_veg_B12', 'SWIR2: S2 vs MODIS (tile-years)', 0.03),
                ('modis_ndvi', 'rh98_offset', 'RH98 offset vs MODIS NDVI anomaly', None)]):
            d = ext.dropna(subset=[xc, yc])
            o, h = d[d.year != HL_YEAR], d[d.year == HL_YEAR]
            ax.axhline(0, color=GRID, linewidth=0.8)
            ax.axvline(0, color=GRID, linewidth=0.8)
            ax.scatter(o[xc], o[yc], s=14, color=MUTED, alpha=0.7, linewidths=0, label='other years')
            ax.scatter(h[xc], h[yc], s=30, color=ORANGE, edgecolors=SURF, linewidths=0.8, label='2024')
            for t in strong:
                s = h[h.tile == t]
                if len(s):
                    ax.annotate(t, (s[xc].iloc[0], s[yc].iloc[0]), xytext=(4, 4), textcoords='offset points',
                                fontsize=8, color=INK2)
            rho = d[[xc, yc]].corr(method='spearman').iloc[0, 1]
            ax.text(0.03, 0.96, f'Spearman {rho:.2f}, n = {len(d)}', transform=ax.transAxes, va='top', fontsize=9,
                    color=INK)
            if lim:
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.plot([-lim, lim], [-lim, lim], color=INK2, linewidth=0.8, linestyle=':')
                ax.set_xlabel(f'MODIS {xc.split("_")[1].upper()} anomaly')
                ax.set_ylabel(f'S2 {xc.split("_")[1].upper()} same-season offset')
            else:
                ax.set_xlim(-0.06, 0.06)
                ax.set_xticks([-0.05, -0.025, 0, 0.025, 0.05])
                ax.set_xlabel('MODIS NDVI anomaly (growing months)')
                ax.set_ylabel('RH98 year offset (m)')
            ax.set_title(title, fontsize=10.5)
            ax.grid(color=GRID, linewidth=0.4)
            ax.set_axisbelow(True)
        axes[0, 0].legend(frameon=False, fontsize=8.5, loc='lower right')
        # bottom row: the strong tiles over the years, MODIS NDVI and ERA5
        for ax, (col, title, ylab) in zip(axes[1], [
                ('modis_ndvi', 'MODIS NDVI anomaly, growing months', 'NDVI'),
                ('era5_t2m', 'ERA5-Land air temperature (2 m)', 'K vs 2018-2025 mean'),
                ('era5_swvl1', 'ERA5-Land soil moisture (0-7 cm)', 'm³/m³ vs 2018-2025 mean')]):
            _year_lines(ax, ext, col, title, strong)
            ax.set_ylabel(ylab)
        fig.suptitle('Independent data (MODIS Terra, ERA5-Land) for the same tiles and years', x=0.01, ha='left',
                     fontweight='bold', color=INK)
        fig.tight_layout()
        _save(fig, path)


def _fig10_image_fit(refl, per, tiles, path):
    """Per tile: image RH98 offset predicted from the image's band offsets, fitted on the other years
    (2022/2023/2025: leave one year out; 2024: fitted on all three)."""
    m = per.merge(refl[['tile', 'item_id'] + IMAGE_BANDS], on=['tile', 'item_id']).dropna(subset=['offset'] + IMAGE_BANDS)
    rows = []
    for t in tiles:
        g = m[m.tile == t]
        for y in sorted(g.year.unique()):
            tr = g[(g.year != y) & (g.year != HL_YEAR)]
            te = g[g.year == y]
            coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[IMAGE_BANDS]], tr.offset, rcond=None)
            rows.append(te.assign(pred=np.c_[np.ones(len(te)), te[IMAGE_BANDS]] @ coef))
    f = pd.concat(rows)
    with plt.rc_context(_style(11)):
        fig, axes = plt.subplots(1, len(tiles), figsize=(3.2 * len(tiles), 3.9), sharex=True, sharey=True)
        for ax, t in zip(axes, tiles):
            g = f[f.tile == t]
            o, h = g[g.year != HL_YEAR], g[g.year == HL_YEAR]
            ax.plot([-9, 9], [-9, 9], color=INK2, linewidth=0.8, linestyle=':')
            ax.scatter(o.pred, o.offset, s=14, color=MUTED, alpha=0.8, linewidths=0, label='2022, 2023, 2025')
            ax.scatter(h.pred, h.offset, s=26, color=ORANGE, edgecolors=SURF, linewidths=0.8, label='2024')
            ax.text(0.04, 0.96, f'2024 median\nobserved {h.offset.median():+.1f} m\npredicted {h.pred.median():+.1f} m'
                    .replace('-', '−'), transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
            ax.set_title(t, fontsize=10.5)
            ax.set_xlabel('predicted from reflectance (m)')
            ax.grid(color=GRID, linewidth=0.4)
            ax.set_axisbelow(True)
        axes[0].set_ylabel('image RH98 offset (m)')
        axes[0].set_xlim(-8, 8)
        axes[0].set_ylim(-8, 8)
        axes[-1].legend(frameon=False, fontsize=8.5, loc='lower right')
        fig.suptitle('Single images: band offsets (B02, B04, B05, B08, B11, B12) predict the 2024 RH98 offset, '
                     'fitted on the other years', x=0.01, ha='left', fontweight='bold', color=INK, fontsize=11.5)
        fig.tight_layout()
        _save(fig, path)
    print(f.groupby(['tile', f.year == HL_YEAR]).apply(lambda d: pd.Series(
        dict(obs=d.offset.median(), pred=d.pred.median(), resid=(d.offset - d.pred).median()))).round(2).to_string())


def plot_stability_inputs_2024(tables_dir: str, out_dir: str, strong_tiles=STRONG_TILES,
                               image_tiles=('49UFS', '49UDT', '33PZL', '19TFN', '48QYK'), **kwargs):
    """fig8: same-season S2 offsets per year (vegetated NIR/SWIR2 vs water and dark pixels);
    fig9: S2 against MODIS, and MODIS / ERA5-Land anomalies of the strong tiles;
    fig10: image RH98 offsets predicted from the image band offsets.
    Tables: stability_external_tiles.tsv (summarize_stability_external),
    stability_reflectance_per_image.tsv, stability_per_image.tsv."""
    data, out = Path(tables_dir).expanduser(), Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ext = pd.read_csv(data / 'stability_external_tiles.tsv', sep='\t')
    strong = list(strong_tiles)
    _fig8_s2_inputs(ext, strong, out / 'fig8_s2_inputs_2024.png')
    _fig9_external(ext, strong, out / 'fig9_modis_era5_2024.png')
    refl = pd.read_csv(data / 'stability_reflectance_per_image.tsv', sep='\t')
    per = pd.read_csv(data / 'stability_per_image.tsv', sep='\t')
    _fig10_image_fit(refl, per, list(image_tiles), out / 'fig10_image_fit_2024.png')
    print('figures ->', out)


# ---------------------------------------------------------------------------
# fig11: does the 2024 RH98 change follow the 2024 MODIS greening, over many tiles?
# ---------------------------------------------------------------------------

ANOMALY_GROUPS = {'green': ('MODIS greener in 2024', BLUE), 'neutral': ('MODIS ~ unchanged', INK2),
                  'brown': ('MODIS browner in 2024', ORANGE), 'stability': ('69-tile stability test', MUTED)}


BOREAL = 6  # RESOLVE biome id of Boreal Forests/Taiga


def plot_stability_anomaly_test(report_file: str, out_file: str, **kwargs):
    """fig11: RH98 Q1 change of 2024 (minus the mean of 2023 and 2025) against the MODIS NDVI anomaly
    of the same definition, per tile; tiles picked by the anomaly plus the 69-tile stability test.
    Table: anomaly_test_tiles.tsv (summarize_anomaly_test)."""
    d = pd.read_csv(Path(report_file).expanduser(), sep='\t').dropna(subset=['rh98_change', 'ndvi_anomaly'])
    out = Path(out_file).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    order = [g for g in ('stability', 'brown', 'neutral', 'green') if g in set(d.group)]
    with plt.rc_context(_style(11)):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
        a1.axhline(0, color=GRID, linewidth=0.8)
        a1.axvline(0, color=GRID, linewidth=0.8)
        for g in order:
            s = d[d.group == g]
            label, col = ANOMALY_GROUPS[g]
            a1.scatter(s.ndvi_anomaly, s.rh98_change, s=22 if g == 'stability' else 34, color=col,
                       edgecolors=SURF, linewidths=0.8, alpha=0.85, label=f'{label} (n={len(s)})', zorder=3)
        boreal = d[d.biome == BOREAL]
        a1.scatter(boreal.ndvi_anomaly, boreal.rh98_change, s=95, facecolors='none', edgecolors=INK,
                   linewidths=1.1, label=f'Boreal forest (n={len(boreal)})', zorder=4)
        new = d[d.group != 'stability']
        rho = new[['ndvi_anomaly', 'rh98_change']].corr(method='spearman').iloc[0, 1]
        a1.text(0.03, 0.96, f'Spearman {rho:.2f} on the {len(new)} picked tiles', transform=a1.transAxes, va='top',
                fontsize=9.5, color=INK)
        a1.set_xlabel('MODIS NDVI anomaly 2024 (vs mean of 2023, 2025; tree pixels, growing months)')
        a1.set_ylabel('RH98 Q1 change 2024 vs mean of 2023, 2025 (m)')
        a1.grid(color=GRID, linewidth=0.4)
        a1.set_axisbelow(True)
        a1.legend(frameon=False, fontsize=8.5, loc='lower right')
        a1.set_title('Per tile', fontsize=10.5)

        groups = [g for g in ('brown', 'neutral', 'green') if g in set(d.group)]
        a2.axhline(0, color=INK2, linewidth=0.8)
        rng = np.random.default_rng(0)
        for i, g in enumerate(groups):
            s = d[d.group == g].rh98_change
            label, col = ANOMALY_GROUPS[g]
            x = i + rng.uniform(-0.18, 0.18, len(s))
            a2.scatter(x, s, s=26, color=col, edgecolors=SURF, linewidths=0.8, alpha=0.85, zorder=3)
            b = (d[d.group == g].biome == BOREAL).values
            a2.scatter(x[b], s[b], s=80, facecolors='none', edgecolors=INK, linewidths=1.1, zorder=4)
            a2.plot([i - 0.3, i + 0.3], [s.median()] * 2, color=INK, linewidth=2.2, zorder=4)
            a2.text(i + 0.33, s.median(), f'{s.median():+.2f} m'.replace('-', '−'), va='center', fontsize=9, color=INK)
        a2.set_xticks(range(len(groups)), [ANOMALY_GROUPS[g][0].replace(' in 2024', '') for g in groups], fontsize=9)
        a2.set_xlim(-0.5, len(groups) - 0.2)
        a2.set_ylabel('RH98 Q1 change (m)')
        a2.grid(axis='y', color=GRID, linewidth=0.6)
        a2.set_axisbelow(True)
        a2.set_title('Tiles picked by the MODIS anomaly (line = median)', fontsize=10.5)
        fig.suptitle('2024 RH98 is higher on most picked tiles (median +0.3 m), but greener tiles do not rise more'
                     ' than unchanged ones\nBoreal tiles (ringed) are all above 0, most by 0.7 m or more',
                     x=0.01, ha='left', fontweight='bold', color=INK)
        fig.tight_layout()
        _save(fig, out)
    print('figure ->', out)


# ---------------------------------------------------------------------------
# fig7b: single images on the strong tiles and on control tiles (2024 offset near 0)
# ---------------------------------------------------------------------------

def plot_stability_per_image_controls(tables_dir: str, out_file: str, strong_tiles, **kwargs):
    """fig7b: per tile, each year's single-image RH98 offset against the same season of the other
    years (2022-2025), strong tiles vs control tiles, and the 2024 value against the same-season
    S2 NDVI offset. Tables: stability_per_image.tsv, stability_external_tiles.tsv."""
    data, out = Path(tables_dir).expanduser(), Path(out_file).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(data / 'stability_per_image.tsv', sep='\t', parse_dates=['date'])
    p['doy'] = p.date.dt.dayofyear
    ss = _same_season(p)
    ext = pd.read_csv(data / 'stability_external_tiles.tsv', sep='\t').set_index(['tile', 'year'])
    biome = p.groupby('tile').biome.first()
    strong = list(strong_tiles)
    tiles = [t for t in strong if t in set(ss.tile)] + sorted(set(ss.tile) - set(strong), key=lambda t: biome[t])
    with plt.rc_context(_style(11)):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
        a1.axhline(0, color=INK2, linewidth=0.8)
        for i, t in enumerate(tiles):
            s = ss[ss.tile == t]
            o, h = s[s.year != HL_YEAR], s[s.year == HL_YEAR]
            a1.scatter(np.full(len(o), i), o.same_season, s=22, color=MUTED, linewidths=0, zorder=2)
            a1.scatter(np.full(len(h), i), h.same_season, s=60, color=ORANGE, edgecolors=SURF, linewidths=1.2, zorder=3)
        a1.axvline(len([t for t in tiles if t in strong]) - 0.5, color=GRID, linewidth=1.2)
        a1.set_xticks(range(len(tiles)), [f'{t}\n{BIOMES[int(biome[t])].split()[0]}' for t in tiles], fontsize=8)
        a1.text(0.01, 0.97, 'tiles with the largest 2024 offsets', transform=a1.transAxes, fontsize=9, color=INK2, va='top')
        a1.text(0.99, 0.97, 'control tiles (2024 year offset near 0)', transform=a1.transAxes, fontsize=9, color=INK2,
                va='top', ha='right')
        a1.set_ylabel('offset vs same season of other years (m)')
        a1.set_ylim(-2, 2)
        a1.grid(axis='y', color=GRID, linewidth=0.6)
        a1.set_axisbelow(True)
        a1.scatter([], [], s=60, color=ORANGE, label='2024')
        a1.scatter([], [], s=22, color=MUTED, label='2022, 2023, 2025')
        a1.legend(frameon=False, fontsize=8.5, loc='lower left')
        a1.set_title('Single images, RH98 Q1: each year vs the same season of the others', fontsize=10.5)

        h = ss[ss.year == HL_YEAR].set_index('tile')
        x = ext.xs(HL_YEAR, level='year').s2_veg_NDVI.reindex(h.index)
        a2.axhline(0, color=GRID, linewidth=0.8)
        a2.axvline(0, color=GRID, linewidth=0.8)
        for t in h.index:
            col = ORANGE if t in strong else BLUE
            a2.scatter(x[t], h.same_season[t], s=46, color=col, edgecolors=SURF, linewidths=1, zorder=3)
            a2.annotate(t, (x[t], h.same_season[t]), xytext=(4, 3), textcoords='offset points', fontsize=8, color=INK2)
        a2.scatter([], [], s=46, color=ORANGE, label='largest 2024 offsets')
        a2.scatter([], [], s=46, color=BLUE, label='controls')
        a2.legend(frameon=False, fontsize=8.5, loc='lower right')
        a2.set_xlabel('S2 NDVI, same-season offset 2024')
        a2.set_ylabel('RH98 single images, 2024 vs same season (m)')
        a2.grid(color=GRID, linewidth=0.4)
        a2.set_axisbelow(True)
        a2.set_title('2024: RH98 vs S2 NDVI, both against the same season', fontsize=10.5)
        fig.suptitle('Control tiles: the same-season 2024 rise is not general; of 8 controls only Boreal 20ULC shows it',
                     x=0.01, ha='left', fontweight='bold', color=INK)
        fig.tight_layout()
        _save(fig, out)
    print(ss.pivot(index='tile', columns='year', values='same_season').reindex(tiles).round(2).to_string())
    print('figure ->', out)


# ---------------------------------------------------------------------------
# methods: one toy-data figure per test method (for the methods deck in Trilium)
# ---------------------------------------------------------------------------

def _toy_tag(ax):
    ax.text(0.99, 0.02, 'toy data', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=MUTED)


def _lower_median(v):
    v = np.sort(np.asarray(v)[np.isfinite(v)])
    return v[(len(v) - 1) // 2]


def _season(doy, peak=200, width=55, lo=8., hi=16.):
    return lo + (hi - lo) * np.exp(-0.5 * ((doy - peak) / width) ** 2)


def plot_stability_methods(out_dir: str, seed: int = 0, **kwargs):
    """Toy-data sketches of the test methods of the VSM stability test, one PNG each:
    m01 yearly composite at fixed pixels, m02 year offset, m03 year-to-year change, m04 same-season
    single images, m05 image bootstrap, m06 disturbance mask, m07 input checks, m08 MODIS anomaly tile
    selection, m09 references tried, m10 GEDI bias."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    years = list(range(2017, 2026))
    ycol = lambda y: ORANGE if y == HL_YEAR else MUTED
    st = _style(11)

    # m01: fixed pixels, yearly value = lower median of ~20 single-image predictions
    with plt.rc_context(st):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1, 2.2]))
        p = rng.uniform(0, 1, (300, 2))
        a1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK2))
        a1.scatter(p[:, 0], p[:, 1], s=4, color=BLUE)
        a1.scatter([0.62], [0.41], s=60, facecolors='none', edgecolors=ORANGE, linewidths=1.6, zorder=3)
        a1.set_xlim(-0.05, 1.05), a1.set_ylim(-0.05, 1.05), a1.set_aspect('equal'), a1.axis('off')
        a1.set_title('S2 tile: 20 000 random pixels,\nthe same in every year', fontsize=10)
        for k, (y, shift) in enumerate([(2022, 0), (2023, 0.3), (2024, 1.6)]):
            doy = np.sort(rng.uniform(120, 290, 20))
            v = _season(doy) + shift + rng.normal(0, 1.2, 20)
            col = ORANGE if y == HL_YEAR else (MUTED if k == 0 else INK2)
            a2.scatter(doy, v, s=18, color=col, label=f'{y}: single images')
            m = _lower_median(v)
            a2.hlines(m, 115, 295, color=col, linewidth=2)
            a2.text(298, m, f'{y} median {m:.1f} m', va='center', fontsize=9, color=col)
        a2.set_xlim(110, 345)
        a2.set_xlabel('day of year')
        a2.set_ylabel('RH98 Q1 at one pixel (m)')
        a2.set_title('Yearly value = median of ~20 single-image predictions (lower middle for 20)', fontsize=10)
        _toy_tag(a2)
        fig.tight_layout()
        _save(fig, out / 'm01_composite.png')

    # m02: year offset vs the pixel's multi-year median
    with plt.rc_context(st):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        v = 18 + rng.normal(0, 0.8, len(years))
        v[years.index(2024)] += 1.6
        med = np.median(v)
        a1.axhline(med, color=INK2, linestyle='--', linewidth=1)
        a1.text(2025.4, med - 0.12, 'pixel median\n2017–2025', fontsize=8.5, color=INK2, ha='right', va='top')
        for y, x in zip(years, v):
            a1.annotate('', (y, x), (y, med), arrowprops=dict(arrowstyle='->', color=ycol(y), linewidth=1.4))
            a1.scatter(y, x, s=30, color=ycol(y), zorder=3)
        a1.set_xticks(years, [str(y)[2:] for y in years])
        a1.set_ylabel('RH98 Q1 (m)')
        a1.set_title('One pixel: offset = year value − pixel median', fontsize=10)
        offs = {y: rng.normal(0.05 + (0.35 if y == 2024 else 0), 1.5, 400) for y in years}
        for y in years:
            a2.scatter(np.full(60, y) + rng.uniform(-0.25, 0.25, 60), offs[y][:60], s=4, color=MUTED, alpha=0.6)
            a2.hlines(np.median(offs[y]), y - 0.35, y + 0.35, color=ycol(y), linewidth=2.5)
        a2.axhline(0, color=INK2, linewidth=0.8)
        a2.set_xticks(years, [str(y)[2:] for y in years])
        a2.set_ylabel('pixel offsets (m)')
        a2.set_title('Tile: year offset = median over vegetated pixels', fontsize=10)
        _toy_tag(a2)
        fig.tight_layout()
        _save(fig, out / 'm02_year_offset.png')

    # m03: year-to-year change, 2023 -> 2024 against the other one-year changes
    with plt.rc_context(st):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        v = np.array([15.2, 15.6, 15.4, 16.0, 15.7, 15.9, 15.6, 16.4, 16.0])
        a1.plot(years, v, color=MUTED, linewidth=1, zorder=1)
        for i in range(1, len(years)):
            y = years[i]
            a1.annotate('', (y, v[i]), (y - 1, v[i - 1]),
                        arrowprops=dict(arrowstyle='->', color=ycol(y), linewidth=2 if y == 2024 else 1.2))
        a1.scatter(years, v, s=26, color=INK2, zorder=3)
        a1.text(2022.9, 16.25, '2023→2024', color=ORANGE, fontsize=9, ha='right')
        a1.set_xticks(years, [str(y)[2:] for y in years])
        a1.set_ylabel('tile RH98 Q1 (m)')
        a1.set_title('One tile: change = H(t) − H(t−1), undisturbed pixels', fontsize=10)
        a2.axhline(0, color=INK2, linewidth=0.8)
        for y in years[1:]:
            c = rng.normal(0.55, 0.45, 24) if y == 2024 else rng.normal(0., 0.6, 24)
            a2.scatter(np.full(24, y) + rng.uniform(-0.22, 0.22, 24), c, s=10, color=ycol(y), alpha=0.8)
            a2.hlines(np.median(c), y - 0.35, y + 0.35, color=INK, linewidth=2)
            a2.text(y, 2.05, f'{(c > 0).mean():.0%}', ha='center', fontsize=8.5, color=ycol(y) if y == 2024 else INK2)
        a2.text(2017.6, 2.05, 'up:', fontsize=8.5, color=INK2, ha='right')
        a2.set_ylim(-2, 2.3)
        a2.set_xticks(years[1:], [str(y)[2:] for y in years[1:]])
        a2.set_ylabel('one-year change (m)')
        a2.set_title('Many tiles: is 2023→2024 outside the other years?', fontsize=10)
        _toy_tag(a2)
        fig.tight_layout()
        _save(fig, out / 'm03_year_to_year.png')

    # m04: same-season single images
    with plt.rc_context(st):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        for y, shift in [(2022, 0), (2023, -0.3), (2024, 1.4), (2025, 0.2)]:
            doy = np.sort(rng.uniform(130, 280, 20))
            v = _season(doy, lo=-5, hi=4) + shift + rng.normal(0, 0.8, 20)
            ax.scatter(doy, v, s=20, color=ycol(y), zorder=3 if y == 2024 else 2,
                       label='2024 images' if y == 2024 else ('other years\' images' if y == 2022 else None))
        d0 = 205
        ax.axvspan(d0 - 20, d0 + 20, color=BLUE, alpha=0.08)
        ax.axvline(d0, color=BLUE, linewidth=1)
        ax.text(d0, -6.3, '2024 image ±20 days:\ncompare with the other years in the window', ha='center',
                fontsize=9, color=BLUE)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_ylim(-7.5, 7)
        ax.set_xlabel('day of year')
        ax.set_ylabel('single-image offset (m)')
        ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper left')
        ax.set_title('Same season: each 2024 image vs the other years\' images at the same time of year',
                     fontsize=10)
        _toy_tag(ax)
        fig.tight_layout()
        _save(fig, out / 'm04_same_season.png')

    # m05: image bootstrap
    with plt.rc_context(st):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        v = 16 + rng.normal(0, 1.5, 20)
        a1.scatter(np.zeros(20) + rng.uniform(-0.1, 0.1, 20), v, s=18, color=INK2)
        a1.hlines(_lower_median(v), -0.25, 0.25, color=INK, linewidth=2)
        for k in range(1, 6):
            s = v[rng.integers(0, 20, 20)]
            a1.scatter(np.full(20, k) + rng.uniform(-0.1, 0.1, 20), s, s=12, color=MUTED)
            a1.hlines(_lower_median(s), k - 0.25, k + 0.25, color=BLUE, linewidth=2)
        a1.set_xticks(range(6), ['selected\n20 images'] + [f'resample\n{k}' for k in range(1, 6)], fontsize=8.5)
        a1.set_ylabel('RH98 Q1 at one pixel (m)')
        a1.set_title('Redraw the 20 images with replacement', fontsize=10)
        for c, sd, col, lab in [(3.7, 1.1, ORANGE, 'tile A: interval excludes 0'),
                                (0.7, 1.0, MUTED, 'tile B: interval includes 0')]:
            b = rng.normal(c, sd, 500)
            a2.hist(b, bins=30, color=col, alpha=0.6, label=lab)
            lo, hi = np.percentile(b, [2.5, 97.5])
            a2.hlines(-4, lo, hi, color=col, linewidth=3)
        a2.axvline(0, color=INK2, linewidth=0.8)
        a2.set_xlabel('2023→2024 change of the tile (m), 500 resamples')
        a2.set_ylabel('count')
        a2.legend(frameon=False, fontsize=8.5)
        a2.set_title('Does the change survive another image selection?', fontsize=10)
        _toy_tag(a2)
        fig.tight_layout()
        _save(fig, out / 'm05_image_bootstrap.png')

    # m06: disturbance mask
    with plt.rc_context(st):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        stable = 20 + 0.1 * (np.array(years) - 2017) + rng.normal(0, 0.6, len(years))
        dist = np.array([22, 22.3, 22.1, 22.4, 3, 4.2, 5.5, 6.8, 8])
        ax.plot(years, stable, '-o', color=BLUE, markersize=5, label='kept: no mapped disturbance')
        ax.plot(years, dist, '-o', color=MUTED, markersize=5, label='removed: Hansen loss in 2021')
        ax.annotate('clear-cut', (2021, 3), (2019.6, 8), arrowprops=dict(arrowstyle='->', color=INK2), fontsize=9,
                    color=INK2)
        ax.set_xticks(years)
        ax.set_ylabel('RH98 Q1 (m)')
        ax.legend(frameon=False, fontsize=9, loc='center left', bbox_to_anchor=(0.0, 0.55))
        ax.text(2025.2, 14, 'removed if any of:\nHansen loss 2001–2025 (+30 m buffer) or gain\n'
                'MODIS burned area 2016–2025\nJRC TMF degradation / deforestation', ha='right', va='center',
                fontsize=9, color=INK2)
        ax.set_title('Disturbance mask: keep only pixels without mapped disturbance', fontsize=10)
        _toy_tag(ax)
        fig.tight_layout()
        _save(fig, out / 'm06_disturbance_mask.png')

    # m07: input checks (dark targets and vegetation, same season)
    with plt.rc_context(st):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        yy = years[1:]
        for ax, title, base in [(a1, 'Dark targets (water, darkest 2 %): red', 0.0),
                                (a2, 'Vegetation: NIR (up = greener)', 0.0)]:
            vals = rng.normal(base, 0.002, len(yy))
            if ax is a2:
                vals[yy.index(2024)] = 0.014
            ax.bar(yy, vals, color=[ycol(y) for y in yy], width=0.7)
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.set_xticks(yy, [str(y)[2:] for y in yy])
            ax.set_title(title, fontsize=10)
            ax.set_ylim(-0.004, 0.019)
        a1.set_ylabel('same-season reflectance offset')
        a1.text(0.03, 0.95, 'a radiometric shift would move these too', transform=a1.transAxes, fontsize=9,
                color=INK2, va='top')
        a2.text(0.03, 0.95, 'a real vegetation anomaly moves only these', transform=a2.transAxes, fontsize=9,
                color=INK2, va='top')
        _toy_tag(a2)
        fig.tight_layout()
        _save(fig, out / 'm07_input_checks.png')

    # m08: MODIS anomaly tile selection
    with plt.rc_context(st):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        a = rng.normal(0, 0.023, 18000)
        lo, hi = np.quantile(a, [0.1, 0.9])
        bins = np.linspace(-0.09, 0.09, 91)
        ax.hist(a, bins=bins, color=MUTED)
        ax.hist(a[a <= lo], bins=bins, color=ORANGE, label='browner: lowest 10 %')
        ax.hist(a[a >= hi], bins=bins, color=BLUE, label='greener: highest 10 %')
        ax.hist(a[np.abs(a) <= 0.001], bins=bins, color=INK, label='unchanged: |anomaly| ≤ 0.001')
        ax.set_xlabel('MODIS NDVI anomaly 2024 = 2024 − mean(2023, 2025), tree pixels, growing months')
        ax.set_ylabel('tiles')
        ax.legend(frameon=False, fontsize=9)
        ax.set_title('All tiles screened; 20 tiles per group, spread over biomes, predicted 2023–2025', fontsize=10)
        _toy_tag(ax)
        fig.tight_layout()
        _save(fig, out / 'm08_anomaly_selection.png')

    # m09: references tried for "is 2024 unusual?"
    with plt.rc_context(st):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        v = np.array([15.0, 15.3, 15.2, 15.7, 15.4, 15.6, 15.3, 16.2, 15.6])
        ax.plot(years, v, '-o', color=INK2, markersize=5, zorder=3)
        ax.scatter([2024], [v[years.index(2024)]], s=70, color=ORANGE, zorder=4)
        refs = [('multi-year median (year offset)', np.median(v), BLUE, '-'),
                ('mean of 2023 and 2025 (dropped)', (v[6] + v[8]) / 2, MUTED, '--'),
                ('mean of 2021–2023 (dropped)', v[4:7].mean(), MUTED, ':'),
                ('2023 = H(t−1) (year-to-year change, used)', v[6], ORANGE, '-')]
        for lab, r, col, ls in refs:
            ax.hlines(r, 2016.6, 2025.4, color=col, linestyle=ls, linewidth=1.4, label=lab)
        k = np.polyfit(years[:7], v[:7], 1)
        ax.plot([2017, 2024], np.polyval(k, [2017, 2024]), color=MUTED, linestyle='-.', linewidth=1.2,
                label='linear trend 2017–2023 extrapolated (dropped)')
        ax.set_xticks(years)
        ax.set_ylabel('tile RH98 Q1 (m)')
        ax.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
        ax.set_ylim(14.6, 16.9)
        ax.set_title('Which reference for 2024? Each one judged against the same measure in other years',
                     fontsize=10)
        _toy_tag(ax)
        fig.tight_layout()
        _save(fig, out / 'm09_references.png')

    # m10: GEDI bias (against independent observations)
    with plt.rc_context(st):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        g = rng.gamma(4, 5, 300)
        pr = g + rng.normal(0.15, 3, 300)
        a1.scatter(g, pr, s=6, color=ORANGE, alpha=0.6)
        a1.plot([0, 50], [0, 50], color=INK2, linewidth=0.8)
        a1.set_xlabel('GEDI RH98 (m), shots of that year')
        a1.set_ylabel('VSM RH98 Q1 (m)')
        a1.set_title('Prediction vs GEDI at the shots of each year', fontsize=10)
        gy = [2019, 2020, 2021, 2022, 2024]
        bias = list(rng.normal(-0.08, 0.06, 4)) + [0.18]
        a2.bar(range(len(gy)), bias, color=[ycol(y) for y in gy], width=0.6)
        a2.axhline(0, color=INK2, linewidth=0.8)
        a2.set_xticks(range(len(gy)), gy)
        a2.set_ylabel('bias = mean(VSM − GEDI) (m)')
        a2.set_title('Bias per year (shots differ every year)', fontsize=10)
        _toy_tag(a2)
        fig.tight_layout()
        _save(fig, out / 'm10_gedi_bias.png')
    print('figures ->', out)


def plot_stability_sample_points(tile: str, tiles_file: str, pixels_dir: str, values_dir: str, disturbance_dir: str,
                                 out_file: str, years=range(2017, 2026), vegetated_rh98: float = 5., **kwargs):
    """The 20 000 sample points of one tile and which of them the year comparisons use (vegetated,
    undisturbed; any biome); also prints the median counts over all tiles of the test."""
    from tools.vsm_stability import _undisturbed
    tiles = pd.read_csv(Path(tiles_file).expanduser(), sep='\t').set_index('tile')
    pxd, vd = Path(pixels_dir).expanduser(), Path(values_dir).expanduser()

    def classes(t, biome):
        px = pd.read_parquet(pxd / f'{t}.parquet').set_index('point_id')
        q = np.stack([pd.read_parquet(vd / 'pixel_values' / str(y) / f'{t}.parquet').set_index('point_id')
                      .reindex(px.index)['RH98_Q1'].to_numpy(float) for y in years])
        veg = np.isfinite(q).all(0) & (np.median(q, 0) >= vegetated_rh98)
        used = veg & _undisturbed(t, px.index, disturbance_dir)
        return px, veg, used

    counts = []
    for t, b in tiles.biome.items():
        if all((vd / 'pixel_values' / str(y) / f'{t}.parquet').exists() for y in years):
            _, veg, used = classes(t, int(b))
            counts.append((len(veg), veg.sum(), used.sum()))
    med = np.median(np.array(counts), 0)
    px, veg, used = classes(tile, int(tiles.loc[tile, 'biome']))
    out = Path(out_file).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = [(used, BLUE, f'used ({used.sum():,})'),
              (veg & ~used, '#8f8e8a', f'excluded: mapped disturbance ({(veg & ~used).sum():,})'),
              (~veg, '#d4d3cf', f'excluded: RH98 < 5 m or no prediction ({(~veg).sum():,})')]
    with plt.rc_context(_style(11)):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for m, col, lab in groups[::-1]:
            ax.scatter(px.lon[m], px.lat[m], s=0.7, color=col, linewidths=0, rasterized=True, label=lab)
        ax.set_aspect(1 / np.cos(np.deg2rad(px.lat.mean())))
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[::-1], l[::-1], frameon=False, fontsize=10, markerscale=14, loc='center left',
                  bbox_to_anchor=(1.03, 0.5))
        ax.set_title(f'{tile}: 20 000 random points, the same in every year; blue = used in the statistics',
                     fontsize=10)
        fig.subplots_adjust(left=0.01, right=0.5, top=0.9, bottom=0.03)
        _save(fig, out)
    print(f'{tile}: {len(veg)} sampled, {veg.sum()} vegetated, {used.sum()} used; '
          f'median over {len(counts)} tiles {med.round().astype(int).tolist()}; share used/vegetated '
          f"{np.median([c[2] / c[1] for c in counts if c[1]]):.2f}")
    print('figure ->', out)
