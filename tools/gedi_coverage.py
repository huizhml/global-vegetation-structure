"""Check whether a `download_all_valid` GEDI run is complete.

`GEDI.download_all_valid` writes one parquet per S2 tile, and only for tiles
that have at least one orbit in their growing season. This op rebuilds that
expected tile list the same way (sjoin of the S2 grid with the orbit index,
then the growing-month filter) and compares it with the parquets on disk.

A tile can be expected and still have no file for a legitimate reason: every
shot of its orbits failed the quality / sensitivity / pft filter. Runs with a
state_dir record that as done ("no valid shots"), so those tiles are not
counted as missing. Without a state (runs from before it existed) the two
cases look the same; rerun `download_all_valid` with state_dir set: it skips
finished tiles, re-queries only the missing ones and records each outcome.

    python -m tools.run run=gedi_coverage run.year=2019 \
        run.save_dir=/projects/dereeco/data/gvs/modeling/gedi/veg_sensitivity_all/all_valid/2019
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pyarrow.parquet as pq

from tools.task_state import TaskState


def _num_rows(path: Path):
    """Row count from the parquet footer, or None if the file is unreadable."""
    try:
        return pq.read_metadata(path).num_rows
    except Exception:
        return None


def _expected_tiles(s2_grid_file: Path, gedi_table_index_file: Path) -> set:
    """Tiles that `download_all_valid` would schedule (see download/products/gedi.py)."""
    s2_table = gpd.read_parquet(s2_grid_file)
    gedi_index = gpd.read_parquet(gedi_table_index_file)
    gedi_index['month'] = gedi_index['time_start'].str[5:7].astype(int)
    gedi_index = gedi_index.set_crs(epsg=4326, allow_override=True)
    pairs = gpd.sjoin(
        s2_table[['Name', 'geometry', 'growing_months']],
        gedi_index[['table_id', 'month', 'geometry']],
        predicate='intersects', how='inner')
    in_growing = pairs.apply(
        lambda r: r['month'] in {int(m) for m in r['growing_months']}, axis=1)
    print(f'S2 tiles in grid: {len(s2_table)}')
    return set(pairs.loc[in_growing, 'Name'])


def check_all_valid_coverage(save_dir: str, s2_grid_file: str, gedi_table_index_file: str,
                             year: int, state_dir: str = None, out_dir: str = None,
                             workers: int = 32, **kwargs):
    save_dir = Path(save_dir).expanduser()
    gedi_table_index_file = Path(gedi_table_index_file).expanduser()
    if not gedi_table_index_file.exists():
        # download_all_valid fetches it from GEE on first run; not having it
        # means the download for this year never got past its first step.
        raise FileNotFoundError(f'{gedi_table_index_file} not found')

    expected = _expected_tiles(Path(s2_grid_file).expanduser(), gedi_table_index_file)

    files = sorted(save_dir.glob('*.parquet'))
    with ThreadPoolExecutor(workers) as pool:
        rows = dict(zip((p.stem for p in files), pool.map(_num_rows, files)))
    ok = {k for k, n in rows.items() if n is not None}
    corrupt = sorted(k for k, n in rows.items() if n is None)
    empty = sorted(k for k, n in rows.items() if n == 0)
    # Only read an existing state: TaskState() would create an empty one.
    state = TaskState(state_dir) if state_dir and Path(state_dir).expanduser().exists() else None
    done = state.done() if state else set()
    failed = state.failed(refresh=False) if state else {}
    no_shots = sorted((expected & done) - ok)
    missing = sorted(expected - ok - done)
    unexpected = sorted(ok - expected)

    print(f'year {year}, {save_dir}')
    print(f'  expected tiles (orbit in growing season): {len(expected)}')
    print(f'  readable parquets: {len(ok)} ({len(ok & expected) / max(len(expected), 1):.2%} of expected)')
    print(f'  total shots: {sum(n for n in rows.values() if n):,}')
    print(f'  corrupt (unreadable footer): {len(corrupt)}')
    print(f'  zero-row files: {len(empty)}')
    if state:
        print(f'  queried, no valid shots (state): {len(no_shots)}')
        print(f'  last attempt failed (state): {len(failed)}')
    else:
        print(f'  no state at {state_dir}: queried-empty tiles count as missing')
    print(f'  missing (expected, no readable file): {len(missing)}')
    print(f'  unexpected (file but not in expected list): {len(unexpected)}')

    out_dir = Path(out_dir).expanduser() if out_dir else save_dir.parent / f'{save_dir.name}_coverage'
    out_dir.mkdir(exist_ok=True, parents=True)
    for name, keys in [('missing', missing), ('corrupt', corrupt), ('unexpected', unexpected),
                       ('failed', sorted(failed))]:
        (out_dir / f'{name}.txt').write_text(''.join(f'{k}\n' for k in keys))
    print(f'tile lists written to {out_dir}')
