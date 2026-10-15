"""Split the extra columns of a derived GEDI copy into a slim sidecar table.

The gt0p95 subsets were extended by copying the whole table and adding a few
columns (slope, biome, SOTA CHMs, our predictions, ...), so each step stored the
GEDI table again. This keeps the base table once and moves what a derived copy
added into a sidecar with only ``shot_number`` + those columns:

    <base_dir>/<tile>.parquet      base, never rewritten
    <out_dir>/<tile>.parquet       shot_number + columns the derived copy added

A derived copy is either a directory of per-tile parquets with the base's file
names, or one combined parquet with a ``tile_id`` column (what
tools.parq_ops.merge_columns_from_dirs writes, e.g. ``version=masked``).

Per tile, before anything is written:
    * shot_number must be unique and every derived shot must be in the base;
    * columns the derived copy shares with the base (or with sidecars already
      extracted, ``known_dirs``) are compared value by value. A shared column
      that differs was changed by the derived step, so it goes into the sidecar
      too and is listed under ``changed_cols`` in the report;
    * the sidecar is read back and compared with what was meant to be written.

With every check passed, base + sidecars reproduce the derived copy, which can
then be deleted. Rows keep the base order; when the derived copy has the same
shots in the same order (``same_order`` in the report), the sidecar lines up
row by row with the base file and can be appended without a join.

Output, besides the tiles: ``<out_dir>/_report.tsv`` (one row per tile) and
``<out_dir>/README.md`` (source, base, columns, commit).

    python -m tools.run run=gedi_extract_cols run.split=cal \
        run.variant=original_with_sota_chms run.name=sota_chms
    # schemas only, nothing read or written beyond parquet footers
    python -m tools.run run=gedi_extract_cols ... run.dry_run=true
"""
import datetime
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _as_list(x) -> list:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _read(path: Path, columns: list, tile: str = None) -> pa.Table:
    # partitioning=None: a `version=masked` path segment must not become a column
    filters = [('tile_id', '=', tile)] if tile is not None else None
    return pq.read_table(path, columns=columns, filters=filters, partitioning=None)


def _same_values(a: pa.ChunkedArray, b: pa.ChunkedArray) -> bool:
    """Equal values per row, nulls/NaNs in the same places; dtype may differ."""
    a, b = a.to_pandas(), b.to_pandas()
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return np.array_equal(a.to_numpy(np.float64, na_value=np.nan),
                              b.to_numpy(np.float64, na_value=np.nan), equal_nan=True)
    both_null = a.isna().to_numpy() & b.isna().to_numpy()
    return bool(((a.to_numpy() == b.to_numpy()) | both_null).all())


def _derived_tiles(derived: Path) -> list:
    if derived.is_dir():
        return sorted(f.stem for f in derived.glob('*.parquet'))
    return sorted(pc.unique(_read(derived, ['tile_id'])['tile_id']).to_pylist())


def _report_row(tile: str, **kw) -> dict:
    return {'tile': tile, 'status': 'ok', 'base_rows': 0, 'derived_rows': 0,
            'same_order': False, 'missing_in_derived': 0, 'new_cols': '',
            'changed_cols': '', 'note': '', **kw}


def _extract_tile(tile: str, base_dir: Path, derived: Path, out_dir: Path,
                  known_dirs: list, key_col: str, ignore_cols: list,
                  overwrite: bool) -> dict:
    row = _report_row(tile)
    out_fp = out_dir / f'{tile}.parquet'
    base_fp = base_dir / f'{tile}.parquet'
    derived_fp, filt = (derived / f'{tile}.parquet', None) if derived.is_dir() else (derived, tile)

    if not base_fp.exists():
        return {**row, 'status': 'no_base_file'}
    if out_fp.exists() and not overwrite:
        return {**row, 'status': 'exists'}

    # where each column we already have lives: base first, then known sidecars
    have = {c: base_fp for c in pq.read_schema(base_fp).names}
    for d in known_dirs:
        fp = d / f'{tile}.parquet'
        if fp.exists():
            for c in pq.read_schema(fp).names:
                have.setdefault(c, fp)
    derived_cols = [c for c in pq.read_schema(derived_fp).names
                    if c not in ignore_cols and c != key_col]
    new_cols = [c for c in derived_cols if c not in have]
    shared_cols = [c for c in derived_cols if c in have]

    der = _read(derived_fp, [key_col] + derived_cols, filt)
    base_key = _read(base_fp, [key_col])[key_col]
    row.update(base_rows=len(base_key), derived_rows=der.num_rows)

    der_key = der[key_col].to_numpy()
    base_np = base_key.to_numpy()
    if len(np.unique(der_key)) != len(der_key) or len(np.unique(base_np)) != len(base_np):
        return {**row, 'status': 'duplicate_keys'}

    if len(der_key) == len(base_np) and np.array_equal(der_key, base_np):
        row['same_order'] = True
        base_rows = np.arange(len(base_np))
    else:
        # put derived rows in base order; derived may be a subset of the base
        order = np.argsort(base_np)
        pos = np.searchsorted(base_np, der_key, sorter=order).clip(max=len(base_np) - 1)
        base_idx = order[pos]
        if not np.array_equal(base_np[base_idx], der_key):
            n = int((base_np[base_idx] != der_key).sum())
            return {**row, 'status': 'keys_not_in_base', 'note': f'{n} derived shots'}
        sort = np.argsort(base_idx)
        der = der.take(pa.array(sort))
        base_rows = base_idx[sort]
        row['missing_in_derived'] = len(base_np) - len(der_key)

    # shared columns must match what we already have, else they were changed
    changed = []
    by_file = {}
    for c in shared_cols:
        by_file.setdefault(have[c], []).append(c)
    for fp, cols in by_file.items():
        src = _read(fp, [key_col] + cols)
        if fp == base_fp:
            src = src.take(pa.array(base_rows))
        else:
            # a known sidecar may cover other shots: line it up by key
            idx = pd.Index(src[key_col].to_numpy()).get_indexer(der[key_col].to_numpy())
            if (idx < 0).any():
                changed += cols
                row['note'] = f'{int((idx < 0).sum())} shots not in {fp.parent}'
                continue
            src = src.take(pa.array(idx))
        changed += [c for c in cols if not _same_values(src[c], der[c])]

    keep = [c for c in derived_cols if c in new_cols or c in changed]
    row.update(new_cols=','.join(new_cols), changed_cols=','.join(changed))
    if not keep:
        return {**row, 'status': 'nothing_new'}

    side = der.select([key_col] + keep).replace_schema_metadata(None)
    tmp = out_fp.with_suffix('.parquet.tmp')
    pq.write_table(side, tmp)
    if not pq.read_table(tmp, partitioning=None).equals(side):
        tmp.unlink()
        return {**row, 'status': 'readback_mismatch'}
    os.replace(tmp, out_fp)
    return row


def _write_readme(out_dir: Path, derived: Path, base_dir: Path, known_dirs: list,
                  report: pd.DataFrame):
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                text=True, cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        commit = '?'
    cols = lambda s: sorted({c for v in report[s] for c in v.split(',') if c})
    status = report['status'].value_counts().to_dict()
    lines = [
        f'# Sidecar columns: {out_dir.parent.name}/{out_dir.name}',
        '',
        f'shot_number + columns that `{derived}` adds to the base table',
        f'`{base_dir}`. Join on shot_number (rows are in base order;',
        'tiles with same_order=True in _report.tsv line up row by row with the base file).',
        '',
        f'- new columns: {", ".join(cols("new_cols")) or "-"}',
        f'- changed base/known columns (taken from derived): {", ".join(cols("changed_cols")) or "-"}',
        f'- known sidecars compared against: {", ".join(map(str, known_dirs)) or "-"}',
        f'- tiles by status: {status}',
        f'- made by tools/gedi_extract_cols.py @ {commit}, {datetime.date.today()}',
        '',
    ]
    (out_dir / 'README.md').write_text('\n'.join(lines))


def extract_cols(base_dir: str, derived: str, out_dir: str, known_dirs: list = None,
                 key_col: str = 'shot_number', ignore_cols: list = ('geometry', 'tile_id'),
                 workers: int = 8, overwrite: bool = False, dry_run: bool = False,
                 **kwargs) -> pd.DataFrame:
    '''
    Write shot_number + the columns ``derived`` adds to ``base_dir`` into ``out_dir``.

    Args:
        * base_dir: per-tile parquets of the base table (``<tile>.parquet``)
        * derived: per-tile directory with the same file names, or one parquet
          with a ``tile_id`` column
        * out_dir: sidecar directory, one ``<tile>.parquet`` per derived tile
        * known_dirs: sidecars already extracted; their columns are compared
          like base columns instead of being copied again
        * key_col: unique shot id shared by base and derived
        * ignore_cols: derived columns neither compared nor copied
          (geometry stays in the base; tile_id is the file name)
        * workers: tiles processed in parallel (each holds one tile in RAM)
        * overwrite: redo tiles whose sidecar exists
        * dry_run: only compare schemas and row counts (parquet footers)
    Returns:
        * the per-tile report
    '''
    base_dir, derived, out_dir = (Path(p).expanduser() for p in (base_dir, derived, out_dir))
    known_dirs = [Path(p).expanduser() for p in _as_list(known_dirs)]
    ignore_cols = _as_list(ignore_cols)
    if not derived.exists():
        raise FileNotFoundError(derived)
    if not any(base_dir.glob('*.parquet')):
        raise FileNotFoundError(f'no base parquet files in {base_dir}')
    for d in known_dirs:
        if not any(d.glob('*.parquet')):
            raise FileNotFoundError(f'no sidecar parquet files in {d}')
    tiles = _derived_tiles(derived)
    if not tiles:
        raise FileNotFoundError(f'no tiles in {derived} (expects <tile>.parquet directly inside)')
    print(f'{derived}: {len(tiles)} tiles, base {base_dir}')

    if dry_run:
        base_cols = set(pq.read_schema(next(base_dir.glob('*.parquet'))).names)
        for d in known_dirs:
            base_cols |= set(pq.read_schema(next(d.glob('*.parquet'))).names)
        derived_fp = next(derived.glob('*.parquet')) if derived.is_dir() else derived
        der_cols = [c for c in pq.read_schema(derived_fp).names
                    if c not in ignore_cols and c != key_col]
        missing = [t for t in tiles if not (base_dir / f'{t}.parquet').exists()]
        print(f'  new columns:    {[c for c in der_cols if c not in base_cols]}')
        print(f'  shared columns: {[c for c in der_cols if c in base_cols]}')
        print(f'  base columns not in derived: {sorted(base_cols - set(der_cols) - {key_col})}')
        print(f'  derived tiles without base file: {len(missing)} {missing[:5]}')
        if derived.is_dir():
            n_base = sum(pq.ParquetFile(base_dir / f'{t}.parquet').metadata.num_rows
                         for t in tiles if t not in missing)
            n_der = sum(pq.ParquetFile(derived / f'{t}.parquet').metadata.num_rows for t in tiles)
            print(f'  rows: base {n_base:,} (same tiles), derived {n_der:,}')
        return pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    for tmp in out_dir.glob('*.tmp'):
        tmp.unlink()
    args = (base_dir, derived, out_dir, known_dirs, key_col, ignore_cols, overwrite)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_extract_tile, t, *args): t for t in tiles}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:      # keep going; the report says which tile failed
                rows.append(_report_row(futs[fut], status='error', note=repr(e)[:300]))
            if i % 100 == 0 or i == len(tiles):
                print(f'  {i}/{len(tiles)} tiles')

    report = pd.DataFrame(rows).sort_values('tile')
    report.to_csv(out_dir / '_report.tsv', sep='\t', index=False)
    _write_readme(out_dir, derived, base_dir, known_dirs, report)
    print(report['status'].value_counts().to_string())
    for col in ('new_cols', 'changed_cols'):
        print(f'{col}: {sorted({c for v in report[col].dropna() for c in v.split(",") if c})}')
    bad = report[~report['status'].isin(['ok', 'exists', 'nothing_new'])]
    if len(bad):
        print(f'{len(bad)} tiles NOT ok, see {out_dir / "_report.tsv"}')
    return report


def _keys_lonlat(fp: Path, key_col: str, tile: str = None):
    """shot ids and lon/lat of one tile, from lon/lat columns or else the geometry."""
    names = pq.read_schema(fp).names
    if 'lon' in names and 'lat' in names:
        t = _read(fp, [key_col, 'lon', 'lat'], tile)
        return t[key_col].to_numpy(), t['lon'].to_numpy(), t['lat'].to_numpy()
    import geopandas as gpd
    filters = [('tile_id', '=', tile)] if tile is not None else None
    g = gpd.read_parquet(fp, columns=[key_col, 'geometry'], filters=filters, partitioning=None)
    g = g.to_crs('EPSG:4326')
    return g[key_col].to_numpy(), g.geometry.x.to_numpy(), g.geometry.y.to_numpy()


def _nearest_in_group(qk, qlon, qlat, keys, lon, lat) -> np.ndarray:
    """Per query shot: metres to the nearest base shot with the same id (inf if none)."""
    groups = pd.Series(np.arange(len(keys))).groupby(keys).apply(np.array)
    out = np.full(len(qk), np.inf)
    for i, k in enumerate(qk):
        cand = groups.get(k)
        if cand is not None:
            out[i] = (np.hypot((lon[cand] - qlon[i]) * np.cos(np.radians(qlat[i])),
                               lat[cand] - qlat[i]) * 111_320).min()
    return out


def diagnose_keys(base_dir: str, derived: str, n_tiles: int = 20, key_col: str = 'shot_number',
                  near_m: float = 5.0, seed: int = 0, **kwargs) -> pd.DataFrame:
    '''
    Why derived shots are not in the base, on a few random tiles.

    extract_gedi_from_h5 used to pass shot_number through float64, which rounds
    17-digit ids, and a GEE download may have stored them as doubles too. For
    both tables this reports how many ids survive a float64 round trip
    (``*_ids_float_exact``; for real ids only ~1/16 do) and how many ids repeat.
    Matches are then counted three ways, each only when the matched base shot
    lies within ``near_m`` metres:
        * id_match: same id
        * rounded_id_match: same id after rounding the base ids like float64
        * position_match: any base shot within ``near_m``, ids ignored
    ``id_match_far`` counts same-id matches that are in fact another shot.
    '''
    from scipy.spatial import cKDTree
    base_dir, derived = Path(base_dir).expanduser(), Path(derived).expanduser()
    tiles = [t for t in _derived_tiles(derived) if (base_dir / f'{t}.parquet').exists()]
    rng = np.random.default_rng(seed)
    tiles = sorted(rng.choice(tiles, min(n_tiles, len(tiles)), replace=False))
    for fp in (base_dir / f'{tiles[0]}.parquet', derived / f'{tiles[0]}.parquet' if derived.is_dir() else derived):
        try:
            print(f'{key_col} type in {fp}: {pq.read_schema(fp).field(key_col).type}')
        except OSError as e:
            print(f'cannot read {fp}: {e}')
    rows = []
    for t in tiles:
        der_fp, filt = (derived / f'{t}.parquet', None) if derived.is_dir() else (derived, t)
        try:
            dk, dlon, dlat = _keys_lonlat(der_fp, key_col, filt)
        except OSError as e:
            print(f'{t}: cannot read {der_fp}: {e}')
            continue
        try:
            bk, blon, blat = _keys_lonlat(base_dir / f'{t}.parquet', key_col)
        except OSError as e:
            print(f'{t}: cannot read {base_dir / f"{t}.parquet"}: {e}')
            continue
        dk, bk = dk.astype(np.uint64), bk.astype(np.uint64)
        round64 = lambda k: k.astype(np.float64).astype(np.uint64)

        d_id = _nearest_in_group(dk, dlon, dlat, bk, blon, blat)
        d_round = _nearest_in_group(dk, dlon, dlat, round64(bk), blon, blat)
        lat0 = np.radians(np.nanmean(blat))
        xy = lambda lon, lat: np.c_[lon * np.cos(lat0), lat] * 111_320
        d_pos, _ = cKDTree(xy(blon, blat)).query(xy(dlon, dlat))

        rows.append({'tile': t, 'base_shots': len(bk), 'base_repeated_ids': len(bk) - len(np.unique(bk)),
                     'base_ids_float_exact': int((round64(bk) == bk).sum()),
                     'derived_shots': len(dk), 'derived_repeated_ids': len(dk) - len(np.unique(dk)),
                     'derived_ids_float_exact': int((round64(dk) == dk).sum()),
                     'id_match': int((d_id <= near_m).sum()),
                     'id_match_far': int((np.isfinite(d_id) & (d_id > near_m)).sum()),
                     'rounded_id_match': int((d_round <= near_m).sum()),
                     'position_match': int((d_pos <= near_m).sum()),
                     'position_m_median': float(np.median(d_pos))})
    report = pd.DataFrame(rows)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(report.to_string(index=False))
        print(report.drop(columns=['tile', 'position_m_median']).sum().to_string())
    return report
