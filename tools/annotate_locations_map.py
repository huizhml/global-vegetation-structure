from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd


def annotate_locations_on_map(
    csv_file: str,
    out_file: str,
    extent: tuple = None,
    title: str = None,
    marker_color: str = '#d62728',
    marker_size: int = 60,
    label_fontsize: int = 9,
    pad_deg: float = 2.0,
    dpi: int = 200,
    **kwargs,
):
    '''
    Plot a CSV of locations as annotated points on a cartopy world map.

    Args:
        csv_file: CSV with columns `name`, `lat`, `lon` (case-insensitive;
            extra columns are ignored).
        out_file: output PNG path. Parent dirs are created.
        extent: (lon_min, lon_max, lat_min, lat_max). If None, auto-fits to
            the points with `pad_deg` padding.
        title: optional figure title.
        marker_color: scatter marker face color.
        marker_size: scatter marker size in points^2.
        label_fontsize: text label font size.
        pad_deg: padding around auto-fit extent, in degrees.
        dpi: output figure DPI.
    '''
    csv_file = Path(csv_file).expanduser()
    out_file = Path(out_file).expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in ('name', 'lat', 'lon') if c not in cols]
    if missing:
        raise ValueError(
            f'CSV {csv_file} is missing required columns: {missing}. '
            f'Found: {list(df.columns)}'
        )
    df = df.rename(columns={cols['name']: 'name', cols['lat']: 'lat', cols['lon']: 'lon'})

    if extent is None:
        lon_min = max(df['lon'].min() - pad_deg, -180)
        lon_max = min(df['lon'].max() + pad_deg, 180)
        lat_min = max(df['lat'].min() - pad_deg, -90)
        lat_max = min(df['lat'].max() + pad_deg, 90)
        extent = (lon_min, lon_max, lat_min, lat_max)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': proj})
    ax.set_extent(list(extent), crs=proj)

    ax.add_feature(cfeature.LAND, facecolor='#f5f1e8')
    ax.add_feature(cfeature.OCEAN, facecolor='#dfeaf4')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)

    ax.scatter(
        df['lon'], df['lat'],
        s=marker_size, color=marker_color,
        edgecolor='black', linewidth=0.6,
        zorder=5, transform=proj,
    )
    for _, row in df.iterrows():
        ax.annotate(
            str(row['name']),
            xy=(row['lon'], row['lat']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=label_fontsize,
            color='black',
            zorder=6,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7),
        )

    if title:
        ax.set_title(title)

    fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved map of {len(df)} locations to {out_file}')
    return out_file


if __name__ == '__main__':
    annotate_locations_on_map(
        csv_file='~/data/gvs/locations/sites.csv',
        out_file='~/data/gvs/results/tools/annotate_locations_map/sites_map.png',
    )
