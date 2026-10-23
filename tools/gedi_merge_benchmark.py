'''
GEDI download with orbits merged per GEE request: how much faster, and where
is GEE's size limit for one table download?

On the S2 tiles with the most shots (download state of `year`), fetches the
same shots several ways, all with download.products.gedi.fetch_orbits:

    per_orbit          one request per orbit (the download before 2026-10)
    adaptive           orbits merged up to max_shots per request, sized by
                       the shots per orbit of prior_year, then of the
                       requests so far (what download_all_valid does)
    adaptive_no_prior  the same without prior_year
    probe_full         up to probe_max_shots per request, all columns
    probe_reduced      the same with REDUCED_COLUMNS (~40% of the columns)

The probes bracket the size limit (largest request that worked, smallest that
failed). If the limit moves up with fewer columns, it is bytes, not shots.
Only result tables are written. One tile per array task, then the summary:

    sbatch --array=0-3 scripts/lumi/gedi_merge_benchmark.sh
    python -m tools.run run=gedi_merge_benchmark_summary
'''
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from download.core.utils import ensure_authenticated, shapely_to_geojson
from download.products.gedi import (download_gedi_table_index, fetch_orbits, shot_filter_expr,
                                    shots_per_orbit_prior, state_shots, target_orbits,
                                    tile_orbit_pairs)

METHODS = ('per_orbit', 'adaptive', 'adaptive_no_prior', 'probe_full', 'probe_reduced')
REDUCED_COLUMNS = ['shot_number', 'delta_time', 'beam', 'sensitivity', 'pft_class'] + \
                  [f'rh{i}' for i in range(0, 101, 2)]


def tile_orbits(grid: gpd.GeoDataFrame, index: gpd.GeoDataFrame, orbit_months: str = 'growing') -> pd.DataFrame:
    '''Name, geometry, orbit_ids: the orbits download_all_valid takes per tile.'''
    pairs = tile_orbit_pairs(grid, index)
    mask, _ = target_orbits(pairs, orbit_months)
    orbits = pairs[mask].groupby('Name')['table_id'].apply(list).rename('orbit_ids')
    return grid[['Name', 'geometry']].merge(orbits, left_on='Name', right_index=True)


def method_kwargs(method: str, shots_per_orbit: float, n_orbits: int, max_shots: int,
                  max_orbits: int, probe_max_shots: int) -> dict:
    '''fetch_orbits arguments of each benchmark method.'''
    if method == 'per_orbit':
        return dict(max_orbits=1)
    if method == 'adaptive':
        return dict(shots_per_orbit=shots_per_orbit, max_shots=max_shots, max_orbits=max_orbits)
    if method == 'adaptive_no_prior':
        return dict(max_shots=max_shots, max_orbits=max_orbits)
    probe = dict(shots_per_orbit=shots_per_orbit, max_shots=probe_max_shots, max_orbits=n_orbits)
    if method == 'probe_full':
        return probe
    if method == 'probe_reduced':
        return dict(probe, columns=REDUCED_COLUMNS)
    raise ValueError(method)


def benchmark_orbit_merge(year: int, s2_grid_file: str, gedi_table_index_file: str, state_dir: str,
                          results_dir: str, prior_year: int = None, prior_state_dir: str = None,
                          prior_index_file: str = None, methods: list = METHODS, tiles: list = None,
                          n_tiles: int = 4, tile_rank: int = None, max_shots: int = 20000,
                          max_orbits: int = 50, probe_max_shots: int = 80000,
                          sensitivity_threshold: float = None, max_water_persistence: float = 50,
                          drop_no_dem: bool = True, key_file: str = None, **kwargs):
    '''
    Args:
        * tiles: S2 tiles to test; None = the n_tiles with the most shots in
            state_dir (the download state of `year`)
        * tile_rank: only the tile at this position of tiles (array task id)
        * prior_year, prior_state_dir, prior_index_file: shots per orbit from
            that year's download, for adaptive and the probes
        * max_shots, max_orbits: per request, as in download_all_valid
        * probe_max_shots: per request in the probes, above the limit
        * sensitivity_threshold, max_water_persistence, drop_no_dem: shot
            filter, as in download_all_valid_gedi
        * results_dir: summary_<tile>.tsv (one row per method) and
            requests_<tile>.tsv (one row per request)
    '''
    ensure_authenticated(key_file)
    shot_filter = shot_filter_expr(sensitivity_threshold, max_water_persistence, drop_no_dem)
    if tiles is None:
        counts = pd.Series(state_shots(state_dir)).sort_values(ascending=False)
        tiles = list(counts.index[:n_tiles])
        print(f'{n_tiles} tiles with the most shots in {year}: {counts.iloc[:n_tiles].to_dict()}')
    tiles = [str(t) for t in tiles]
    if tile_rank is not None:
        tiles = [tiles[tile_rank]]
    grid = gpd.read_parquet(Path(s2_grid_file).expanduser())
    grid = grid[grid['Name'].isin(tiles)].reset_index(drop=True)
    assert not grid['Name'].duplicated().any(), 'tiles split at the antimeridian are not supported'
    index = download_gedi_table_index(year=year, out_file=Path(gedi_table_index_file).expanduser())
    chosen = tile_orbits(grid, index.set_crs(epsg=4326, allow_override=True))
    prior = {} if prior_year is None else shots_per_orbit_prior(
        grid, prior_state_dir, prior_index_file, prior_year)
    results_dir = Path(results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = list(methods)
    for k, tile in enumerate(chosen.itertuples()):
        geom = shapely_to_geojson(tile.geometry)
        n_orbits = len(tile.orbit_ids)
        spo = prior.get(tile.Name)
        print(f'{tile.Name}: {n_orbits} orbits, prior {spo and round(spo)} shots per orbit ({prior_year})')
        # rotate the order per tile, so GEE caching from an earlier method
        # does not always favour the same later one
        r = ((tile_rank or 0) + k) % len(methods)
        rows, request_rows, shot_ids = [], [], {}
        for method in methods[r:] + methods[:r]:
            start = time.time()
            try:
                shots, log = fetch_orbits(tile.orbit_ids, geom, shot_filter, **method_kwargs(
                    method, spo, n_orbits, max_shots, max_orbits, probe_max_shots))
                error = ''
            except Exception as e:
                shots, log, error = None, [], repr(e)[:300]
            if not error:
                shot_ids[method] = (np.array([], dtype=str) if shots is None
                                    else np.sort(shots['shot_number'].astype(str).to_numpy()))
            row = dict(tile=tile.Name, method=method, n_orbits=n_orbits,
                       prior_shots_per_orbit=spo and round(spo),
                       requests=len(log), failed=sum(not r['ok'] for r in log),
                       seconds=round(time.time() - start, 1),
                       shots=len(shot_ids[method]) if not error else None,
                       columns=None if shots is None else shots.shape[1], error=error)
            print(row, flush=True)
            rows.append(row)
            request_rows += [dict(tile=tile.Name, method=method, **r) for r in log]
            del shots
        ref = shot_ids.get('per_orbit')
        for row in rows:
            if ref is not None and row['method'] in shot_ids:
                row['same_shots_as_per_orbit'] = np.array_equal(shot_ids[row['method']], ref)
        pd.DataFrame(rows).to_csv(results_dir / f'summary_{tile.Name}.tsv', sep='\t', index=False)
        pd.DataFrame(request_rows).to_csv(results_dir / f'requests_{tile.Name}.tsv', sep='\t', index=False)
        print(f'saved {results_dir}/summary_{tile.Name}.tsv, requests_{tile.Name}.tsv')


def summarize_orbit_merge(results_dir: str, **kwargs):
    '''Methods compared over all tiles, the size limit from the requests that
    worked and failed, and the time of one request (fixed + per shot).'''
    results_dir = Path(results_dir).expanduser()
    summary = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(results_dir.glob('summary_*.tsv'))])
    reqs = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(results_dir.glob('requests_*.tsv'))])
    pd.set_option('display.width', 200)
    print(summary.drop(columns='error').to_string(index=False))
    if summary['error'].notna().any():
        print(summary.loc[summary['error'].notna(), ['tile', 'method', 'error']].to_string(index=False))

    total = summary[summary['error'].isna()].groupby('method')[['requests', 'failed', 'seconds', 'shots']].sum()
    if 'per_orbit' in total.index:
        total['speedup'] = (total.loc['per_orbit', 'seconds'] / total['seconds']).round(2)
    print(f'\nall tiles:\n{total.to_string()}')

    reqs['columns'] = np.where(reqs['method'] == 'probe_reduced', 'reduced', 'all')
    limit = reqs.groupby('columns').apply(lambda g: pd.Series({
        'largest_ok': g.loc[g['ok'], 'shots'].max(),
        'smallest_failed': g.loc[~g['ok'], 'shots'].min(),
        'n_failed': int((~g['ok']).sum())}))
    print(f'\nsize limit (shots per request):\n{limit.to_string()}')

    ok = reqs[reqs['ok'] & (reqs['columns'] == 'all') & (reqs['shots'] > 0)]
    if len(ok) > 2:
        per_shot, fixed = np.polyfit(ok['shots'], ok['seconds'], 1)
        print(f'\none request (all columns): {fixed:.1f} s + {per_shot * 1000:.2f} s per 1000 shots '
              f'(fit on {len(ok)} requests)')
    summary.to_csv(results_dir / 'all_tiles_summary.tsv', sep='\t', index=False)
    print(f'saved {results_dir}/all_tiles_summary.tsv')
