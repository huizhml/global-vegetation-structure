"""Split inference tiles over SLURM array jobs, prioritizing some countries.

Nothing is written to disk: every job recomputes the same order from the
zone-grouped items store (download/core/items_store.py), the S2 grid and the
prioritized countries, then takes its own contiguous block (see job_tiles). The
split therefore depends only on n_jobs, which may change between submissions.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

from download.core.items_store import list_tiles, split_block

COUNTRIES_FILE = '~/data/00_raw_data/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'


def prioritized_tiles(prioritized_countries, s2_grid_file: str, countries_file: str = COUNTRIES_FILE):
    """Names of the S2 grid tiles intersecting any of `prioritized_countries`."""
    countries = gpd.read_file(Path(countries_file).expanduser())
    regions = countries[countries['ADMIN'].isin(list(prioritized_countries))]
    unknown = set(prioritized_countries) - set(regions['ADMIN'])
    if unknown:
        raise ValueError(f'Countries not found in {countries_file} (ADMIN column): {sorted(unknown)}')
    s2_grid = gpd.read_parquet(Path(s2_grid_file).expanduser(), columns=['Name', 'geometry'])
    return set(s2_grid.loc[s2_grid.intersects(regions.union_all()), 'Name'])


def prioritized_tile_order(items_file, prioritized_countries=None, s2_grid_file: str = None,
                           countries_file: str = COUNTRIES_FILE):
    """Tiles with items in the store; those intersecting `prioritized_countries` first.

    Both groups are sorted by tile name, so the order is deterministic and tiles of a
    zone stay adjacent (a job then reads few row groups).
    """
    tiles = list_tiles(items_file)
    if not prioritized_countries:
        return tiles
    prior = prioritized_tiles(prioritized_countries, s2_grid_file, countries_file)
    return [t for t in tiles if t in prior] + [t for t in tiles if t not in prior]


def job_tiles(items_file, job_id: int, n_jobs: int, prioritized_countries=None, s2_grid_file: str = None,
              countries_file: str = COUNTRIES_FILE):
    """Tiles of block `job_id` out of `n_jobs` over prioritized_tile_order()."""
    order = prioritized_tile_order(items_file, prioritized_countries, s2_grid_file, countries_file)
    return split_block(order, job_id, n_jobs)


def preview_job_allocation(items_file: str, n_jobs: int, prioritized_countries=None, s2_grid_file: str = None,
                           countries_file: str = COUNTRIES_FILE, n_show: int = 5, **kwargs):
    """Print how tiles would be split over `n_jobs` jobs, without running anything."""
    items_file = Path(items_file).expanduser()
    order = prioritized_tile_order(items_file, prioritized_countries, s2_grid_file, countries_file)
    prior = prioritized_tiles(prioritized_countries, s2_grid_file, countries_file) if prioritized_countries else set()
    if s2_grid_file:
        grid_tiles = set(pd.read_parquet(Path(s2_grid_file).expanduser(), columns=['Name'])['Name'])
        print(f'{len(grid_tiles)} tiles in the S2 grid, {len(grid_tiles - set(order))} without items in {items_file.name}')
    rows = []
    for job_id in range(n_jobs):
        tiles = split_block(order, job_id, n_jobs)
        rows.append(dict(job_id=job_id, n_tiles=len(tiles), n_prioritized=len(prior.intersection(tiles)),
                         n_zones=len({t[:3] for t in tiles}),
                         first=tiles[0] if tiles else '', last=tiles[-1] if tiles else ''))
    df = pd.DataFrame(rows)
    print(f'{len(order)} tiles with items over {n_jobs} jobs; {len(prior.intersection(order))} prioritized '
          f'({list(prioritized_countries or [])})')
    print(df.head(n_show).to_string(index=False))
    if len(df) > 2 * n_show:
        print('  ...')
        print(df.tail(n_show).to_string(index=False, header=False))
    print(f'tiles/job min={df.n_tiles.min()} max={df.n_tiles.max()} | '
          f'zones/job median={int(df.n_zones.median())} max={df.n_zones.max()}')
    return df
