from typing import List
import os
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
from pathlib import Path
from omegaconf import OmegaConf
import omegaconf.listconfig
import pyarrow.parquet as pq
import dask
from dask.diagnostics import ProgressBar

def is_geoparquet(path) -> bool:
    schema = pq.read_schema(path)
    meta = schema.metadata or {}
    return b'geo' in meta


def make_parq_subcolumns(parq_dir: str, subcolumns: list[str], save_fp: str = None, **kwargs):
    '''
    Make subcolumns from a large geodataframe into a smaller one.
    For analysis on local machine.
    The large geodataframe is saved in multiple parquet files.
    Args:
        parq_dir: directory of the parquet file
        subcolumns: list of columns to make subcolumns of
        save_fp: path to save the smaller dataframe
    Returns:
        df: dataframe with subcolumns
    '''
    parq_dir = Path(parq_dir).expanduser()
    save_fp = Path(save_fp).expanduser()
    save_fp.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(subcolumns, omegaconf.listconfig.ListConfig):
        subcolumns = OmegaConf.to_container(subcolumns, resolve=True)
    df = dd.read_parquet(parq_dir / '*.parquet', columns=subcolumns)
    df.compute().to_parquet(save_fp)
    return df


def merge_columns_from_files(parq_dir: str, filename_pattern: str = '*.parquet',
                      save_fp: str = None, **kwargs) -> pd.DataFrame:
    '''
    Merge multiple parquet files in a directory by joining new columns.
    Merge per-data-type patch stats parquets into a single table, joined on rowid.
    Shared columns, are taken from the first file.
    '''
    parq_dir = Path(parq_dir).expanduser()
    files = sorted(parq_dir.glob(filename_pattern))
    if not files:
        raise ValueError(f'No parquet files matching {filename_pattern} in {parq_dir}')

    df = pd.read_parquet(files[0])
    for file in files[1:]:
        other = pd.read_parquet(file)
        # Drop columns already present to avoid collisions; keep only new feature columns
        new_cols = other.columns.difference(df.columns)
        df = df.join(other[new_cols], how='outer', validate='one_to_one')

    if save_fp is not None:
        save_fp = Path(save_fp).expanduser()
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_fp)
    return df

def check_missing_files(target_files, source_files):
    target_names = {f.name for f in target_files}
    source_names = {f.name for f in source_files}
    missing = target_names - source_names
    assert not missing, f'Files in target but not in source: {missing}'

def _already_processed(target_file: Path, source_file: Path) -> bool:
    target_schema = pq.read_schema(target_file)
    source_schema = pq.read_schema(target_file)
    new_cols = set(source_schema.names) - set(target_schema.names)
    return len(new_cols) == 0


def add_columns_from_dir(target_dir: str, source_dir: str, validate_cols: List[str], **kwargs):
    '''
    Add cols from source_dir into target_dir and write back to target_dir.
    The parquet filenames should match for the two directories.
    '''
    target_dir = Path(target_dir).expanduser()
    source_dir = Path(source_dir).expanduser()
    target_files = sorted(target_dir.glob('*.parquet'))
    source_files = sorted(source_dir.glob('*.parquet'))
    # check_missing_files(target_files, source_files)
    # remove tmp files
    for tmp in target_dir.glob("*.tmp"):
        tmp.unlink()
    

    is_geo = is_geoparquet(target_files[0])
    read_fn = gpd.read_parquet if is_geo else pd.read_parquet
    filenames = [f.name for f in target_files]
    filenames = [fn for fn in filenames if not _already_processed(target_dir /fn, source_dir/fn)]
    print(f"{len(filenames)} files to process, rest already done")

    def _update(filename: str):
        target_path = target_dir / filename
        source_path = source_dir / filename
        if not source_path.exists():
            df1 = read_fn(target_path)
            return 
        # assert source_path.exists(), f'{filename} not found in {source_dir}'

        df1 = read_fn(target_path)
        df2 = read_fn(source_path)
        df1 = df1.to_crs('EPSG:4326')
        df2 = df2.to_crs('EPSG:4326')
        assert len(df1) == len(df2), f'Row count mismatch for {filename}'

        for col in validate_cols:
            if col == 'geometry':
                # assert (df1.geometry.x.values == df2.geometry.x.values).all(), \
                #     f'geometry.x not aligned for {filename}'
                assert df1.geometry.geom_equals_exact(df2.geometry, tolerance=1e-10).all(), f'geometry not aligned for {filename}'
            else:
                assert (df1[col].values == df2[col].values).all(), \
                    f'{col} not aligned for {filename}'

        new_cols = df2.columns.difference(df1.columns)
        df = pd.concat([df1, df2[new_cols]], axis=1)

        if is_geo:
            df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            df.to_parquet(str(target_path) + '.tmp')
        else:
            df.to_parquet(str(target_path) + '.tmp')
        os.replace(str(target_path) + '.tmp', target_path)

    # filenames = [f for f in filenames if '43RBN.parquet' in f]
    # _update(filenames[0])
    tasks = [dask.delayed(_update)(fn) for fn in filenames]
    with ProgressBar():
        dask.compute(*tasks)


def merge_columns_from_dirs(target_dir: str, source_dir: str, save_fp: str,
                            validate_cols: List[str] = ['geometry', 'shot_number'],
                            row_group_size: int = 100_000,
                            **kwargs) -> pd.DataFrame:
    '''
    Merge new columns from ``source_dir`` into the matching files of
    ``target_dir`` and write everything as ONE combined parquet file.

    Same per-file alignment contract as :func:`add_columns_from_dir` (rows are
    concatenated side-by-side after asserting the tables line up on
    ``validate_cols``), but instead of writing each merged table back into
    ``target_dir`` in place, all tiles are stacked row-wise and saved to a
    single parquet at ``save_fp``. A ``tile_id`` column (filename stem) is
    added so downstream code can recover which tile each row came from.

    Parquet filenames must match across the two directories. Shared columns are
    kept from the target; only columns unique to the source are added. A target
    file with no source counterpart is kept as-is (no extra columns).

    Args:
        * target_dir: directory of per-tile parquet files (base table)
        * source_dir: directory of per-tile parquet files (extra columns)
        * save_fp: path of the single combined parquet file to write
        * validate_cols: columns asserted to be aligned row-for-row before the
          merge (``geometry`` is compared with ``geom_equals_exact``)
        * row_group_size: rows per parquet row group in the output file.
          Lowers peak RAM during write and enables parallel/column-projected
          reads later. Forwarded to ``to_parquet``.
    Returns:
        * the combined (Geo)DataFrame
    '''
    target_dir = Path(target_dir).expanduser()
    source_dir = Path(source_dir).expanduser()
    save_fp = Path(save_fp).expanduser()
    save_fp.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(validate_cols, omegaconf.listconfig.ListConfig):
        validate_cols = OmegaConf.to_container(validate_cols, resolve=True)

    target_files = sorted(target_dir.glob('*.parquet'))
    if not target_files:
        raise ValueError(f'No parquet files in {target_dir}')

    is_geo = is_geoparquet(target_files[0])
    _read = gpd.read_parquet if is_geo else pd.read_parquet

    def read_fn(path):
        # partitioning=None: don't let pyarrow parse a `key=value` path segment
        # (e.g. ".../version=masked/...") into a synthetic Hive-partition
        # column. Forwarded through to pyarrow.parquet.read_table by both
        # gpd.read_parquet and pd.read_parquet. Without this the combined
        # output gains a `version` column absent from every input file.
        return _read(path, partitioning=None)

    @dask.delayed
    def _merge(filename: str):
        target_path = target_dir / filename
        source_path = source_dir / filename
        tile_id = Path(filename).stem
        df1 = read_fn(target_path)
        df1['tile_id'] = tile_id
        if not source_path.exists():
            print(f'{filename}: no source counterpart, kept without extra columns')
            return df1

        df2 = read_fn(source_path)
        if is_geo:
            df1 = df1.to_crs('EPSG:4326')
            df2 = df2.to_crs('EPSG:4326')
        assert len(df1) == len(df2), f'Row count mismatch for {filename}'

        for col in validate_cols:
            if col not in df1.columns or col not in df2.columns:
                continue
            if col == 'geometry':
                assert df1.geometry.geom_equals_exact(df2.geometry, tolerance=1e-10).all(), \
                    f'geometry not aligned for {filename}'
            else:
                assert (df1[col].values == df2[col].values).all(), \
                    f'{col} not aligned for {filename}'

        new_cols = df2.columns.difference(df1.columns)
        return pd.concat([df1, df2[new_cols]], axis=1)

    tasks = [_merge(f.name) for f in target_files]
    with ProgressBar():
        parts = list(dask.compute(*tasks)) if tasks else []

    combined = pd.concat(parts, ignore_index=True)
    if is_geo:
        combined = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')

    combined.to_parquet(str(save_fp) + '.tmp', row_group_size=row_group_size)
    os.replace(str(save_fp) + '.tmp', save_fp)
    print(f'Merged {len(parts)} files, {len(combined)} rows -> {save_fp}')
    return combined
    


