"""Audit the per-S2-tile growing months that download_all_valid filters orbits by.

`s2_tiles_with_growing_months.parquet` comes from
download/products/growing_season.py (VNP22Q2 2020, cycle 1, aggregated per
tile in GEE): a month counts when more than half of the tile's phenology pixels
are in season; with fewer than 5 such months the 5 most frequent months are
taken; a tile without phenology pixels gets a fixed default (north 5-9,
south 12-5). This tool tells, per tile, which of those rules produced its
months, and why a tile ended up "no orbit in growing season" in the download:

    no_gedi_orbit             no GEDI orbit of the year crosses the tile at all
                              (e.g. beyond GEDI's ~51.6 deg coverage)
    orbits_outside_months     orbits exist, but none in the tile's growing months
    has_orbits                at least one orbit in the growing months

Writes <save_dir>/growing_months_audit.tsv (one row per tile) and
<save_dir>/growing_months_audit.png (tile centroids by rule and orbit class),
and prints crosstabs by rule, latitude band and biome.

    python -m tools.run run=growing_season_audit run.year=2020
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from tools.task_state import TaskState

_DEFAULT_NORTH = [5, 6, 7, 8, 9]
_DEFAULT_SOUTH = [12, 1, 2, 3, 4, 5]


def _months(v) -> list:
    """growing_months as stored (list, array or '[5, 6]' string) -> sorted ints."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, str):
        v = v.strip('[]').replace(',', ' ').split()
    return sorted(int(m) for m in v)


def _rule(row) -> str:
    """Which branch of GrowingSeason.get_growing_months produced this tile's months."""
    if 'mask' not in row.index or pd.isna(row['mask']) or row['mask'] == 0:
        return 'no_phenology_default'
    freq = [row.get(f'freq_month{i}', row.get(f'month{i}', np.nan) / row['mask']) for i in range(1, 13)]
    if sum(f > 0.5 for f in freq) >= 5:
        return 'freq_over_half'
    return 'top5_fallback'


def _wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # the orbit index (and maybe the grid) is saved without a CRS; it is lon/lat
    return gdf.set_crs('EPSG:4326') if gdf.crs is None else gdf.to_crs('EPSG:4326')


def audit_growing_months(s2_grid_file: str, gedi_table_index_file: str, save_dir: str,
                         state_dir: str = None, biome_file: str = None, **kwargs) -> pd.DataFrame:
    '''
    Args:
        * s2_grid_file: S2 tiles with Name, geometry, growing_months (+ month*/freq_month*/mask)
        * gedi_table_index_file: GEDI L2A orbit index of the year (table_id, time_start, geometry)
        * save_dir: where the tsv and png go
        * state_dir: download_all_valid TaskState of the year, for each tile's outcome (optional)
        * biome_file: WWF ecoregions shapefile with BIOME, to break results down by biome (optional)
    '''
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    grid = _wgs84(gpd.read_parquet(Path(s2_grid_file).expanduser()))
    print(f'grid: {len(grid)} rows, {grid["Name"].nunique()} tiles; columns: {list(grid.columns)}')

    # antimeridian tiles are two rows with one Name: same months, union of orbits
    tiles = grid.drop_duplicates('Name').set_index('Name')
    cen = grid.dissolve('Name').geometry.representative_point()
    out = pd.DataFrame(index=tiles.index)
    out['lat'], out['lon'] = cen.y.round(3), cen.x.round(3)
    out['months'] = tiles['growing_months'].map(_months)
    out['n_months'] = out['months'].map(len)
    out['rule'] = tiles.apply(_rule, axis=1)
    out['is_default_months'] = out['months'].map(lambda m: m in (_DEFAULT_NORTH, sorted(_DEFAULT_SOUTH)))

    from download.products.gedi import tile_orbit_pairs    # same join as the download
    orbits = _wgs84(gpd.read_parquet(Path(gedi_table_index_file).expanduser()))
    pairs = tile_orbit_pairs(grid, orbits).drop_duplicates(['Name', 'table_id'])
    pairs['in_months'] = pairs['in_growing']
    g = pairs.groupby('Name')
    out['orbits_any'] = g.size().reindex(out.index, fill_value=0)
    out['orbits_in_months'] = g['in_months'].sum().reindex(out.index, fill_value=0).astype(int)
    out['orbit_months'] = g['month'].apply(lambda s: sorted(set(s))).reindex(out.index)
    out['orbit_class'] = np.select(
        [out['orbits_any'] == 0, out['orbits_in_months'] == 0],
        ['no_gedi_orbit', 'orbits_outside_months'], 'has_orbits')

    if state_dir:
        st = TaskState(Path(state_dir).expanduser(), create=False).load()
        out['download'] = [f'{st[n][0]}: {st[n][2]}' if n in st else 'not in state' for n in out.index]

    if biome_file:
        eco = _wgs84(gpd.read_file(Path(biome_file).expanduser())[['BIOME', 'geometry']])
        pts = gpd.GeoDataFrame(geometry=cen.reindex(out.index), crs='EPSG:4326')
        b = gpd.sjoin(pts, eco, predicate='within', how='left')
        out['biome'] = b[~b.index.duplicated()]['BIOME'].reindex(out.index)

    out['lat_band'] = pd.cut(out['lat'], [-90, -51.6, -23.5, 0, 23.5, 51.6, 90],
                             labels=['<-51.6', '-51.6..-23.5', '-23.5..0', '0..23.5', '23.5..51.6', '>51.6'])
    tsv = out.assign(months=out['months'].map(lambda m: ','.join(map(str, m))),
                     orbit_months=out['orbit_months'].map(
                         lambda m: ','.join(map(str, m)) if isinstance(m, list) else ''))
    tsv.reset_index().rename(columns={'Name': 'tile'}).to_csv(save_dir / 'growing_months_audit.tsv',
                                                             sep='\t', index=False)

    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.max_rows', 100):
        print('\nrule x orbit_class')
        print(pd.crosstab(out['rule'], out['orbit_class'], margins=True))
        print('\nlat_band x orbit_class')
        print(pd.crosstab(out['lat_band'], out['orbit_class'], margins=True))
        print('\nlat_band x rule')
        print(pd.crosstab(out['lat_band'], out['rule'], margins=True))
        print('\nn_months by rule')
        print(out.groupby('rule')['n_months'].describe())
        if 'biome' in out:
            print('\nbiome x orbit_class (BIOME codes: WWF, 1 = tropical moist broadleaf ...)')
            print(pd.crosstab(out['biome'], out['orbit_class'], margins=True))
            print('\nbiome x rule')
            print(pd.crosstab(out['biome'], out['rule'], margins=True))
        if 'download' in out:
            print('\ndownload outcome x orbit_class')
            print(pd.crosstab(out['download'].str.replace(r'\d+ shots', 'N shots', regex=True),
                              out['orbit_class'], margins=True))

    _plot(out, save_dir / 'growing_months_audit.png')
    print(f'\n-> {save_dir / "growing_months_audit.tsv"}, {save_dir / "growing_months_audit.png"}')
    return out


def _plot(out: pd.DataFrame, fp: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 13))
    colors = {'freq_over_half': '#4c78a8', 'top5_fallback': '#f58518', 'no_phenology_default': '#e45756',
              'has_orbits': '#bab0ac', 'orbits_outside_months': '#e45756', 'no_gedi_orbit': '#72b7b2'}
    for ax, col in zip(axes, ['rule', 'orbit_class']):
        for k, d in out.groupby(col):
            ax.scatter(d['lon'], d['lat'], s=2, c=colors.get(k, 'k'), label=f'{k} ({len(d)})')
        for y in (-51.6, 51.6):
            ax.axhline(y, color='k', lw=0.5, ls='--')
        ax.set_title(f'S2 tiles by {col}')
        ax.legend(markerscale=5, loc='lower left')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
    fig.tight_layout()
    fig.savefig(fp, dpi=150)
    plt.close(fig)
