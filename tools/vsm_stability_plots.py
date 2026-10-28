"""Figures for the 2017-2026 VSM stability report (docs/reports/vsm_stability_2017_2026.md).

Inputs are the summary tables in vsm_stability_state/2017_2026 written by tools/vsm_stability.py
(summarize_stability_years, count_stability_images, summarize_stability_clouds,
summarize_stability_reflectance); the schematic reads the sampled values in vsm_stability/2017_2026.

    python -m tools.run run=plot_stability_report     # fig1-fig4
    python -m tools.run run=plot_stability_inputs     # fig5-fig6
    python -m tools.run run=plot_stability_schematic  # fig0
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
