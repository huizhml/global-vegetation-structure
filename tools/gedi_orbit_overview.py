"""Per S2 tile and year: how many GEDI L2A orbit tables cross it, and when.

Reads the orbit index of each year (l2a_orbit_table_<year>.parquet, one row
per GEE table = orbit granule, with its footprint) and the S2 grid with
growing_months, intersects them the way download_all_valid does
(download.products.gedi.tile_orbit_pairs) and writes to <save_dir>:

    tile_orbits_<year>.parquet   one row per (tile, orbit table): tile, table_id,
                                 orbit, month, in_growing (month in the tile's
                                 growing_months)
    tile_orbit_counts.tsv        one row per tile: lat, lon, growing_months, and
                                 per year orbits_<y>, orbits_growing_<y>,
                                 few_orbits_<y> (fewer than 40% of the median of
                                 tiles in the same 8 deg latitude band: suspect
                                 orbit index, see README), and from the download
                                 state, if there: shots_<y>, download_<y>
    orbit_summary.tsv            one row per year

Years without an index file are skipped (the index comes from GEE, see
download_gedi_table_index).

    python -m tools.run run=gedi_orbit_overview
"""
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from tools.task_state import TaskState

_ORBIT = re.compile(r'_O(\d{5})_')


def _wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # the orbit index is saved without a CRS; it is lon/lat
    return gdf.set_crs('EPSG:4326') if gdf.crs is None else gdf.to_crs('EPSG:4326')


def gedi_orbit_overview(s2_grid_file: str, index_pattern: str, save_dir: str, years: list,
                        state_pattern: str = None, few_ratio: float = 0.4, **kwargs) -> pd.DataFrame:
    '''
    Args:
        * s2_grid_file: S2 tiles with Name, geometry, growing_months
        * index_pattern: orbit index path with {year}
        * save_dir: output directory
        * years: years to cover
        * state_pattern: download state dir with {year} (optional), for shots per tile
        * few_ratio: few_orbits_<y> when orbits < few_ratio x latitude-band median
    '''
    from download.products.gedi import tile_orbit_pairs

    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    grid = _wgs84(gpd.read_parquet(Path(s2_grid_file).expanduser())).reset_index(drop=True)
    cen = grid.dissolve('Name').geometry.representative_point()
    tiles = grid.drop_duplicates('Name').set_index('Name')
    out = pd.DataFrame(index=tiles.index.rename('tile'))
    out['lat'], out['lon'] = cen.y.round(3), cen.x.round(3)
    out['growing_months'] = tiles['growing_months'].map(lambda m: ','.join(str(int(x)) for x in sorted(m)))
    out['n_growing_months'] = tiles['growing_months'].map(len)
    band = pd.cut(out['lat'], np.arange(-56, 92, 8))

    summary = []
    for year in [int(y) for y in years]:
        fp = Path(str(index_pattern).format(year=year)).expanduser()
        if not fp.exists():
            print(f'{year}: no orbit index at {fp}, skipped')
            continue
        idx = _wgs84(gpd.read_parquet(fp))
        pairs = tile_orbit_pairs(grid, idx).drop(columns='row').rename(columns={'Name': 'tile'})
        pairs = pairs.drop_duplicates(['tile', 'table_id'])     # antimeridian tiles are two rows
        pairs['orbit'] = pairs['table_id'].str.extract(_ORBIT, expand=False).astype('Int64')
        pairs.to_parquet(save_dir / f'tile_orbits_{year}.parquet', index=False)

        g = pairs.groupby('tile')
        n = g.size().reindex(out.index, fill_value=0)
        out[f'orbits_{year}'] = n
        out[f'orbits_growing_{year}'] = g['in_growing'].sum().reindex(out.index, fill_value=0).astype(int)
        ref = n[n > 0].groupby(band[n > 0], observed=True).median()
        out[f'few_orbits_{year}'] = (n > 0) & (n < few_ratio * band.map(ref).astype(float))

        if state_pattern:
            sp = Path(str(state_pattern).format(year=year)).expanduser()
            if sp.exists():
                st = TaskState(sp, create=False).load()
                info = pd.Series({k: f'{s}: {i}' for k, (s, _, i) in st.items()})
                out[f'download_{year}'] = info.reindex(out.index).fillna('not in state')
                out[f'shots_{year}'] = (out[f'download_{year}'].str.extract(r'(\d+) shots', expand=False)
                                        .astype(float).astype('Int64'))

        covered = n > 0
        summary.append({
            'year': year, 'orbit_tables': idx['table_id'].nunique(),
            'tiles': len(out), 'tiles_with_orbits': int(covered.sum()),
            'tiles_without_orbits': int((~covered).sum()),
            'median_orbits_per_tile': float(n[covered].median()),
            'share_of_orbits_in_growing_months': round(float(out.loc[covered, f'orbits_growing_{year}'].sum() / n.sum()), 3),
            'tiles_no_growing_orbit': int((covered & (out[f'orbits_growing_{year}'] == 0)).sum()),
            'tiles_few_orbits': int(out[f'few_orbits_{year}'].sum()),
        })
        print(f'{year}: {summary[-1]}')

    out.to_csv(save_dir / 'tile_orbit_counts.tsv', sep='\t')
    summary = pd.DataFrame(summary)
    summary.to_csv(save_dir / 'orbit_summary.tsv', sep='\t', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(summary.to_string(index=False))
    print(f'-> {save_dir}')
    return out
