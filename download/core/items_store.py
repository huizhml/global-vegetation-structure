"""Single-file store for inference S2 STAC items: one parquet row group per MGRS zone.

The zone (GZD, e.g. '32T') is the first 3 chars of the tile name, so a reader can
locate a tile's row group from the tile name alone and read only that row group.
This replaces the per-job copies written by deploy/schedule_tasks.py.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import geopandas as gpd
from geopandas.io.arrow import _arrow_to_geopandas

TILE_COL = 's2:mgrs_tile'
ZONE_COL = 'mgrs_zone'


def _expand_files(src):
    if isinstance(src, (list, tuple)):
        files = [Path(f).expanduser() for f in src]
    else:
        src = Path(src).expanduser()
        files = sorted(src.parent.glob(src.name))
    if not files:
        raise FileNotFoundError(f'No parquet files match {src}')
    return files


def _utc_timestamps(schema):
    """Mark tz-naive timestamps as UTC (values unchanged).

    WorldS2.get_top_10 strips the tz (tz_localize(None)) only for tiles with >20
    items, so the same UTC column is naive in some files and tz=UTC in others.
    """
    for i, f in enumerate(schema):
        if pa.types.is_timestamp(f.type) and f.type.tz is None:
            schema = schema.set(i, f.with_type(pa.timestamp(f.type.unit, tz='UTC')))
    return schema


def consolidate_items_by_zone(src, out_file, compression: str = 'zstd'):
    """Merge STAC item parquet files into one file with exactly one row group per zone.

    Args:
        src: glob (e.g. '~/data/gvs/deploy/deploy_s2_items_2024_part*.parquet') or list of files.
        out_file: output parquet path.

    Streams file by file: rows are buffered per zone and a zone is written out (and
    freed) as soon as every file containing it has been read, so peak memory is about
    one decoded part file plus the zones still pending, not the whole dataset.
    """
    files = _expand_files(src)

    # pass 1: footer + tile column only -> which files hold which zone, and the output schema
    pending = {}  # zone -> files not yet read that contain it
    for f in files:
        for z in set(pc.utf8_slice_codeunits(pq.read_table(f, columns=[TILE_COL])[TILE_COL], 0, 3).to_pylist()):
            pending.setdefault(z, set()).add(f)
    schemas = [pq.read_schema(f) for f in files]
    geo_meta = schemas[0].metadata.get(b'geo')
    schema = pa.unify_schemas([_utc_timestamps(s.remove_metadata()) for s in schemas], promote_options='permissive')
    if ZONE_COL in schema.names:
        schema = schema.remove(schema.get_field_index(ZONE_COL))
    # keep only geo metadata (drop pandas index metadata from the sources)
    schema = schema.append(pa.field(ZONE_COL, pa.string())).with_metadata({b'geo': geo_meta} if geo_meta else None)

    out_file = Path(out_file).expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix('.parquet.tmp')
    buffers, n_rows, max_pending = {}, 0, 0
    # pass 2: one file at a time
    # statistics only for the lookup columns; per-leaf min/max of every nested asset
    # field (long hrefs) would bloat the footer every reader has to parse
    with pq.ParquetWriter(tmp, schema, compression=compression, write_statistics=[ZONE_COL, TILE_COL]) as w:
        for i, f in enumerate(files):
            table = pq.read_table(f)
            if ZONE_COL in table.column_names:
                table = table.drop_columns([ZONE_COL])
            table = table.append_column(ZONE_COL, pc.utf8_slice_codeunits(table[TILE_COL], 0, 3))
            table = table.select(schema.names).cast(schema)
            for z in pc.unique(table[ZONE_COL]).to_pylist():
                buffers.setdefault(z, []).append(table.filter(pc.equal(table[ZONE_COL], z)))
                pending[z].discard(f)
            del table
            for z in [z for z in buffers if not pending[z]]:
                chunk = pa.concat_tables(buffers.pop(z)).sort_by([(TILE_COL, 'ascending')])
                w.write_table(chunk, row_group_size=chunk.num_rows)
                n_rows += chunk.num_rows
            max_pending = max(max_pending, len(buffers))
            print(f'  [{i + 1}/{len(files)}] {f.name}: {len(buffers)} zones still pending')
    tmp.replace(out_file)
    print(f'Wrote {out_file}: {n_rows} rows, {len(pending)} zones/row groups, '
          f'{out_file.stat().st_size / 2**20:.1f} MiB from {len(files)} files '
          f'(max {max_pending} zones buffered at once)')
    describe_items_store(out_file)
    return out_file


def describe_items_store(items_file, n_show: int = 5):
    """Print the per-row-group zone/tile layout from the footer (no data read)."""
    md = pq.ParquetFile(Path(items_file).expanduser()).metadata
    leaves = [md.schema.column(j).path for j in range(md.num_columns)]
    zc, tc = leaves.index(ZONE_COL), leaves.index(TILE_COL)
    rows = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        z, t = rg.column(zc).statistics, rg.column(tc).statistics
        rows.append(dict(rg=i, rows=rg.num_rows, mib=rg.total_byte_size / 2**20,
                         zone_min=z.min, zone_max=z.max, tile_min=t.min, tile_max=t.max))
    df = pd.DataFrame(rows)
    single = (df['zone_min'] == df['zone_max']).all()
    unique = df['zone_min'].is_unique
    prefix_ok = (df['tile_min'].str[:3].eq(df['zone_min']) & df['tile_max'].str[:3].eq(df['zone_min'])).all()
    with pd.option_context('display.width', 200, 'display.float_format', '{:.2f}'.format):
        print(df.head(n_show).to_string(index=False))
        if len(df) > 2 * n_show:
            print('  ...')
            print(df.tail(n_show).to_string(index=False, header=False))
    print(f'{md.num_row_groups} row groups, {md.num_rows} rows | rows/rg min={df.rows.min()} '
          f'median={int(df.rows.median())} max={df.rows.max()} | uncompressed MiB/rg '
          f'median={df.mib.median():.2f} max={df.mib.max():.2f} | footer={md.serialized_size / 2**10:.0f} KiB '
          f'({md.num_columns} leaf columns)')
    print(f'one zone per row group: {single} | zones unique across row groups: {unique} | '
          f'tiles match their zone: {prefix_ok}')
    return df


def zone_row_group_index(items_file):
    """Map zone -> row group index, read from the footer only (no data read)."""
    pf = pq.ParquetFile(Path(items_file).expanduser())
    # leaf-column index in the parquet schema (nested columns like `assets` expand
    # into many leaves, so the arrow field index does not line up)
    leaves = [pf.metadata.schema.column(j).path for j in range(pf.metadata.num_columns)]
    col = leaves.index(ZONE_COL)
    index = {}
    for i in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(i).column(col).statistics
        if st is None or not st.has_min_max or st.min != st.max:
            raise ValueError(f'Row group {i} of {items_file} does not hold a single zone')
        index[st.min] = i
    return pf, index


def list_tiles(items_file):
    """Sorted unique tiles in the store (reads only the tile column)."""
    tiles = pq.read_table(Path(items_file).expanduser(), columns=[TILE_COL])[TILE_COL]
    return sorted(pc.unique(tiles).to_pylist())


def split_block(tiles, job_id: int, n_jobs: int):
    """Contiguous block `job_id` of `n_jobs` over an ordered tile list.

    Depends only on (tiles, n_jobs), so every array task computes the same split
    without any job file; contiguous blocks keep a job's tiles in few zones.
    """
    if not 0 <= job_id < n_jobs:
        raise ValueError(f'job_id {job_id} out of range for n_jobs {n_jobs}')
    chunk = -(-len(tiles) // n_jobs)
    return list(tiles[job_id * chunk:(job_id + 1) * chunk])


def read_items_for_tiles(items_file, tiles, columns=None):
    """Read STAC items of `tiles`, touching only the row groups of their zones."""
    tiles = list(tiles)
    pf, index = zone_row_group_index(items_file)
    rgs = sorted({index[z] for z in {t[:3] for t in tiles} if z in index})
    if columns is not None:
        columns = list(dict.fromkeys(list(columns) + [TILE_COL, 'geometry']))
    table = pf.read_row_groups(rgs, columns=columns)
    table = table.filter(pc.is_in(table[TILE_COL], value_set=pa.array(tiles)))
    return _arrow_to_geopandas(table)


def _canonical(df):
    """Sort rows and turn geometry / nested columns into comparable strings."""
    # the zone column is derived (and may or may not exist in the sources); compare the rest
    df = pd.DataFrame(df).drop(columns=[ZONE_COL], errors='ignore')
    df = df.sort_values([TILE_COL, 'id']).reset_index(drop=True)
    # naive == UTC (see _utc_timestamps). Concatenating naive and tz-aware parts yields an
    # object column of Timestamps, so detect datetimes by value, not only by dtype.
    for c in df.columns:
        s = df[c]
        is_dt = pd.api.types.is_datetime64_any_dtype(s) or (
            s.dtype == object and s.notna().any() and s.dropna().map(lambda v: isinstance(v, pd.Timestamp)).all())
        if is_dt:
            df[c] = pd.to_datetime(s, utc=True).astype('datetime64[ns, UTC]')
    df['geometry'] = gpd.GeoSeries(df['geometry']).to_wkb(hex=True)
    # string[python] vs object depends on whether pandas metadata survived; compare values only
    to_json = lambda v: None if v is None or v is pd.NA else v if isinstance(v, str) else \
        json.dumps(v, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    for c in df.columns[(df.dtypes == object) | (df.dtypes == 'string')]:
        df[c] = df[c].astype(object).map(to_json)
    return df


def check_items_store(src, out_file, n_sample_tiles: int = 200, seed: int = 0, overwrite: bool = False, **kwargs):
    """Build the zone store from existing part files and verify it against them.

    Checks: one zone per row group, row counts preserved, per-tile items identical
    to the source, and that a job-sized read only touches the expected row groups.
    """
    files = _expand_files(src)
    out_file = Path(out_file).expanduser()
    if overwrite or not out_file.exists():
        t0 = time.time()
        consolidate_items_by_zone(files, out_file)
        print(f'[build] {time.time() - t0:.1f}s')
    else:  # consolidate already prints the layout
        print('[structure]')
        describe_items_store(out_file)

    # 1. structure
    pf, index = zone_row_group_index(out_file)
    md = pf.metadata
    assert len(index) == md.num_row_groups, \
        f'duplicate zone across row groups: {md.num_row_groups} row groups but {len(index)} distinct zones'

    # 2. row counts
    n_src = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    assert md.num_rows == n_src, f'row count mismatch: store {md.num_rows} vs source {n_src}'
    print(f'[rows] {md.num_rows} == source {n_src}')

    # 3. per-tile equality against the source, on a random tile sample
    src_tiles = pa.concat_arrays([pq.read_table(f, columns=[TILE_COL])[TILE_COL].combine_chunks() for f in files])
    all_tiles = np.unique(src_tiles.to_numpy(zero_copy_only=False))
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(all_tiles, size=min(n_sample_tiles, len(all_tiles)), replace=False).tolist())
    src_df = pd.concat([gpd.read_parquet(f, filters=[(TILE_COL, 'in', sample)]) for f in files], ignore_index=True)
    new_df = read_items_for_tiles(out_file, sample)
    assert set(new_df[TILE_COL]) == set(sample), 'missing tiles in store read'
    a = _canonical(src_df)
    b = _canonical(new_df)[a.columns]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    print(f'[equality] {len(sample)} tiles / {len(a)} items identical to source')

    # 4. job-sized read: contiguous block of tiles, as a job would get
    block = all_tiles[: max(1, len(all_tiles) // 21)].tolist()
    zones = {t[:3] for t in block}
    t0 = time.time()
    df = read_items_for_tiles(out_file, block)
    t_store = time.time() - t0
    t0 = time.time()
    gpd.read_parquet(files[0])
    t_part = time.time() - t0
    print(f'[job read] {len(block)} tiles in {len(zones)} zones -> {len(df)} items, '
          f'{len(zones)}/{md.num_row_groups} row groups, {t_store:.2f}s '
          f'(reading one old part file: {t_part:.2f}s)')

    # 5. single-tile read
    t0 = time.time()
    df = read_items_for_tiles(out_file, [sample[0]])
    print(f'[tile read] {sample[0]} -> {len(df)} items in {time.time() - t0:.2f}s')
    print('All checks passed')
