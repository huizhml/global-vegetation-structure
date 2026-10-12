"""Tag GEDI shots with the MGRS 100 km square they fall in.

S2 tiles overlap (~10 km inside a UTM zone, up to most of a tile at zone
edges, since an S2 tile is a full 109.8 km square in its own zone's
projection and does not stop at the zone boundary). MGRS 100 km squares do
not: the zone of a point is decided by its longitude alone, and squares are
cut at the zone edge. So every shot has exactly one `mgrs_tile`, the same in
every S2 tile parquet it was downloaded into, and

    df[df['mgrs_tile'] == tile_name]

keeps each shot once across all tile files (no rewriting / dedup needed).

    python -m tools.run run=add_mgrs_tile \
        run.parq_dir=/projects/dereeco/data/gvs/modeling/gedi/veg_sensitivity_all/all_valid/2019
"""
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

_BANDS = np.array(list('CDEFGHJKLMNPQRSTUVWX'))      # 8° latitude bands, 80°S..84°N
_COLS = ['ABCDEFGH', 'JKLMNPQR', 'STUVWXYZ']          # easting letters, cycle every 3 zones
_ROWS = 'ABCDEFGHJKLMNPQRSTUV'                        # northing letters, cycle every 2000 km


def mgrs_100km(lon, lat) -> np.ndarray:
    """MGRS 100 km square id ('32TQM', zone zero-padded as in S2 tile names).

    Follows NGA.SIG.0012 for the UTM part of the globe. The Norway / Svalbard
    zone exceptions (56°N and up) are not handled; GEDI stops at 51.6°.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if np.any(np.abs(lat) >= 56):
        raise ValueError('|lat| >= 56: zone exceptions / UPS not handled')
    lon = (lon + 180) % 360 - 180
    zone = np.clip(np.floor((lon + 180) / 6).astype(int) + 1, 1, 60)
    band = _BANDS[np.clip(np.floor((lat + 80) / 8).astype(int), 0, 19)]
    south = lat < 0

    easting = np.empty_like(lon)
    northing = np.empty_like(lat)
    for z, s in set(zip(zone.tolist(), south.tolist())):
        m = (zone == z) & (south == s)
        epsg = (32700 if s else 32600) + z
        tr = Transformer.from_crs(4326, epsg, always_xy=True)
        easting[m], northing[m] = tr.transform(lon[m], lat[m])

    # column: zone sets 1,2,3 -> A-H, J-R, S-Z; easting 100-200 km is the first letter
    col_i = np.floor(easting / 100_000).astype(int) - 1
    # row: even zones start 5 letters later (F); southern northings include the
    # 10,000 km false northing, which the letter cycle already accounts for
    row_i = (np.floor(northing / 100_000).astype(int) + np.where(zone % 2 == 0, 5, 0)) % 20

    out = np.empty(len(lon), dtype=object)
    for k in range(len(lon)):
        out[k] = f'{zone[k]:02d}{band[k]}{_COLS[(zone[k] - 1) % 3][col_i[k]]}{_ROWS[row_i[k]]}'
    return out


def _tag_file(path: Path, rewrite: bool, batch_rows: int):
    """-> (tile, n shots, Counter of mgrs ids, error or None); writes the column in place.

    Streams batch by batch, so memory does not grow with the file: unfiltered
    tiles are large, and loading 16 of them whole at once got workers OOM-killed.
    """
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        if not rewrite and 'mgrs_tile' in schema.names:
            ids = pq.read_table(path, columns=['mgrs_tile']).column('mgrs_tile').to_pylist()
            return path.stem, len(ids), Counter(ids), None

        geom_col = json.loads(schema.metadata[b'geo'])['primary_column']
        keep = [n for n in schema.names if n != 'mgrs_tile']
        out_schema = pa.schema([schema.field(n) for n in keep] + [pa.field('mgrs_tile', pa.string())],
                               metadata=schema.metadata)    # keeps the GeoParquet metadata
        tmp = path.with_name(f'.{path.name}.tmp')
        counts, n = Counter(), 0
        with pq.ParquetWriter(tmp, out_schema) as writer:
            for batch in pf.iter_batches(batch_size=batch_rows, columns=keep):
                geom = shapely.from_wkb(batch.column(geom_col).to_numpy(zero_copy_only=False))
                ids = mgrs_100km(shapely.get_x(geom), shapely.get_y(geom))
                writer.write_batch(pa.RecordBatch.from_arrays(
                    batch.columns + [pa.array(ids, pa.string())], schema=out_schema))
                counts.update(ids.tolist())
                n += len(ids)
        os.replace(tmp, path)      # readers never see a half-written file
        return path.stem, n, counts, None
    except Exception as e:
        return path.stem, 0, Counter(), repr(e)


def add_mgrs_tile(parq_dir: str, s2_grid_file: str = None, state_dir: str = None,
                  workers: int = 16, batch_rows: int = 65536, rewrite: bool = False, **kwargs):
    """Add `mgrs_tile` to every tile parquet in parq_dir and report per tile.

    Each row of a tile file is one of:
        owned       mgrs_tile == this tile: the copy to keep in a global union
        copy        owned by another tile that has a file here
        orphan      its MGRS square has no S2 tile (in s2_grid_file, or among
                    the files here if not given): no file owns it
        owner_no_file  owned by an S2 tile that has no file here (e.g. no
                    orbit in that tile's growing season): no file owns it
    Filtering every file to `owned` keeps each shot once, but drops orphan and
    owner_no_file shots entirely.

    The per-tile counts go to <state_dir>/mgrs_summary_<UTC time>.tsv, one
    file per run. Per-shot classes are not stored: they follow from
    mgrs_tile, the file name and the tile list, and would go stale whenever
    tiles are added.
    """
    parq_dir = Path(parq_dir).expanduser()
    for tmp in parq_dir.glob('.*.parquet.tmp'):    # left by a killed run
        tmp.unlink()
    files = sorted(parq_dir.glob('*.parquet'))
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_tag_file, files, [rewrite] * len(files),
                                [batch_rows] * len(files), chunksize=8))
    errors = [(tile, err) for tile, _, _, err in results if err]
    results = [r for r in results if not r[3]]

    have_file = {p.stem for p in files}
    if s2_grid_file:
        s2_names = set(pq.read_table(Path(s2_grid_file).expanduser(), columns=['Name'])
                       .column('Name').to_pylist())
    else:
        s2_names = have_file

    classes = ('owned', 'copy', 'orphan', 'owner_no_file')
    rows, totals = [], Counter()
    orphan_squares = Counter()
    for tile, n, ids, _ in results:
        c = Counter()
        for k, v in ids.items():
            if k == tile:
                c['owned'] += v
            elif k not in s2_names:
                c['orphan'] += v
                orphan_squares[k] += v
            elif k not in have_file:
                c['owner_no_file'] += v
            else:
                c['copy'] += v
        totals.update(c)
        rows.append(f'{tile}\t{n}\t' + '\t'.join(str(c[k]) for k in classes) + '\n')

    total = sum(totals.values())
    print(f'{parq_dir}: {len(files)} files, {total:,} rows')
    if errors:
        print(f'  {len(errors)} files failed (left unchanged, rerun to retry):')
        for tile, err in errors[:20]:
            print(f'    {tile}  {err}')
    for k in classes:
        print(f'  {k:14s} {totals[k]:>14,}  {totals[k] / max(total, 1):6.1%}')
    if orphan_squares:
        print('  top orphan squares:')
        for k, v in orphan_squares.most_common(20):
            print(f'    {k}  {v:,}')

    if state_dir:
        state_dir = Path(state_dir).expanduser()
        state_dir.mkdir(parents=True, exist_ok=True)
        out = state_dir / f'mgrs_summary_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.tsv'
        out.write_text('#tile\trows\t' + '\t'.join(classes) + '\n' + ''.join(rows))
        print(f'per-tile summary: {out}')
