'''
Fill orbit_number, minor_frame_number and shot_number_within_beam where merged
GEE downloads (2026-10-21/22, before GEDI_COLUMNS was passed as selectors)
dropped them: a CSV takes its columns from the first feature, so a batch led
by a table without these columns lost them for every orbit in it.

They are parts of shot_number (checked on 9.8M rows, 100% equal):
    orbit_number            = shot_number // 1e13
    minor_frame_number      = shot_number // 1e8 % 10
    shot_number_within_beam = shot_number % 1e8
orbit_number and minor_frame_number are filled in every empty row: they are
exact even from a rounded id (rounding changes the last digits only). That
includes rows of tables that do not have these columns in GEE: the tables
storing shot_number as a float (e.g. GEDI02_A_2020297211823_O10565_02_...).
shot_number_within_beam (the last 8 digits) only where the id is exact: its
orbit has no float table, or the id is no multiple of the float spacing (16,
8 below 2**56); an exact id that is such a multiple stays empty.
local_beam_azimuth/elevation cannot be derived and stay empty.

(The tables of a row cannot be told apart by time: the orbit index gives the
whole orbit's time range to each of its sub-orbit granules.)

    python -m tools.run run=gedi_fill_shot_fields run.year=2020
'''
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from download.core.utils import ensure_authenticated

FIELDS = ['orbit_number', 'minor_frame_number', 'shot_number_within_beam']


def null_orbit_number(path: Path) -> int:
    '''Rows without orbit_number, from the parquet statistics (all rows if the
    column is missing).'''
    md = pq.ParquetFile(path).metadata
    names = [md.schema.column(i).name for i in range(md.num_columns)]
    if 'orbit_number' not in names:
        return md.num_rows
    j = names.index('orbit_number')
    stats = [md.row_group(r).column(j).statistics for r in range(md.num_row_groups)]
    if all(s is not None and s.has_null_count for s in stats):
        return sum(s.null_count for s in stats)
    # some older files have no column statistics
    return int(pd.read_parquet(path, columns=['orbit_number'])['orbit_number'].isna().sum())


def table_fields(table_ids: list, chunk: int = 100) -> pd.DataFrame:
    '''Per GEE table: has orbit_number, and the type of shot_number.'''
    rows = []
    for i in range(0, len(table_ids), chunk):
        ids = table_ids[i:i + chunk]
        # built client side: a table id computed on the server cannot be loaded
        info = ee.List([ee.List([
            ee.Feature(ee.FeatureCollection(t).first()).propertyNames().contains('orbit_number'),
            ee.Algorithms.ObjectType(ee.FeatureCollection(t).first().get('shot_number'))]) for t in ids]).getInfo()
        rows += [dict(table_id=t, has_fields=h, shot_number_type=s) for t, (h, s) in zip(ids, info)]
    return pd.DataFrame(rows)


def fill_one(path: Path, float_orbits: set) -> dict:
    '''Fill one tile parquet in place (temp file + rename).'''
    df = gpd.read_parquet(path)
    for c in FIELDS + ['local_beam_azimuth', 'local_beam_elevation']:
        if c not in df:
            df[c] = np.nan
    rows = np.flatnonzero(df['orbit_number'].isna().to_numpy())
    sn = df['shot_number'].to_numpy('int64')[rows]
    orbit = sn // 10**13
    spacing = np.where(sn >= 2**56, 16, 8)
    exact = ~np.isin(orbit, list(float_orbits)) | (sn % spacing != 0)
    # by position: files written before 2026-10 keep a (not unique) pandas index
    col = df.columns.get_loc
    df.iloc[rows, col('orbit_number')] = orbit
    df.iloc[rows, col('minor_frame_number')] = sn // 10**8 % 10
    df.iloc[rows[exact], col('shot_number_within_beam')] = sn[exact] % 10**8
    if len(rows):
        tmp = path.with_name(f'.{path.name}.tmp')
        df.to_parquet(tmp)
        tmp.replace(path)
    return dict(tile=path.stem, rows=len(df), filled=len(rows), within_beam_filled=int(exact.sum()),
                within_beam_left_empty=int((~exact).sum()))


def fill_shot_fields(year: int, save_dir: str, gedi_table_index_file: str, key_file: str = None,
                     workers: int = 16, **kwargs):
    '''
    Args:
        * save_dir: the year's tile parquets (filled in place)
        * gedi_table_index_file: orbit index of the year (table_id, time_start, time_end)
    Writes <save_dir>_state/shot_fields_fill_<UTC time>.tsv (per tile) and
    <save_dir>_state/table_fields.tsv (per GEE table queried).
    '''
    save_dir = Path(save_dir).expanduser()
    files = sorted(save_dir.glob('*.parquet'))
    with ProcessPoolExecutor(workers) as ex:
        n_null = list(ex.map(null_orbit_number, files, chunksize=50))
    todo = [f for f, n in zip(files, n_null) if n]
    print(f'{year}: {len(todo)} of {len(files)} tiles have rows without orbit_number ({sum(n_null):,} rows)', flush=True)
    if not todo:
        return
    # the orbits of those rows, and their tables
    orbits = set()
    for f in todo:
        d = pd.read_parquet(f, columns=['shot_number'] + (['orbit_number'] if 'orbit_number' in pq.read_schema(f).names else []))
        sel = d['orbit_number'].isna() if 'orbit_number' in d else slice(None)
        orbits |= set((d.loc[sel, 'shot_number'].astype('int64') // 10**13).unique())
    index = gpd.read_parquet(Path(gedi_table_index_file).expanduser())
    index['orbit'] = index['table_id'].str.extract(r'_O(\d{5})_')[0].astype(int)
    index = index[index['orbit'].isin(orbits)]
    ensure_authenticated(key_file)
    print(f'querying {len(index)} GEE tables of {len(orbits)} orbits', flush=True)
    fields = table_fields(list(index['table_id']))
    state_dir = Path(f'{save_dir}_state')
    fields.to_csv(state_dir / 'table_fields.tsv', sep='\t', index=False)
    print(f'tables without orbit_number: {int((~fields["has_fields"]).sum())}, '
          f'float shot_number: {int((fields["shot_number_type"] != "String").sum())}', flush=True)
    fields = fields.merge(index[['table_id', 'orbit']], on='table_id')
    float_orbits = set(fields.loc[fields['shot_number_type'] != 'String', 'orbit'])
    with ProcessPoolExecutor(workers) as ex:
        report = pd.DataFrame(list(ex.map(fill_one, todo, [float_orbits] * len(todo))))
    out = state_dir / f'shot_fields_fill_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.tsv'
    report.to_csv(out, sep='\t', index=False)
    print(report[['filled', 'within_beam_filled', 'within_beam_left_empty']].sum().to_string())
    print(f'saved {out}')
