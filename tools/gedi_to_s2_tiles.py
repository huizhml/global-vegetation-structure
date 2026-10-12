"""Regroup GEDI shots stored per MGRS zone / orbit into one parquet per S2 tile.

Input is what the legacy `GEDI.download` wrote:

    <src_dir>/<zone>/<orbit>.parquet         zone = MGRS grid zone, e.g. 32T

Output matches `download_all_valid`:

    <save_dir>/<S2 tile>.parquet             + columns orbit, mgrs_tile
    <save_dir>_state/                        TaskState, keys zone/<zone> and <tile>

An S2 tile near a zone edge draws shots from up to four zone dirs, so this
runs in two passes:
  1. split, one task per zone: read the zone's orbit files, sjoin with the
     S2 tile footprints (the polygons, not their bboxes), write each tile's
     share to <save_dir>_parts/<tile>/<zone>.parquet
  2. merge, one task per tile, only once every zone is split: concat the
     parts (plus an existing output, so a rerun is idempotent), drop
     duplicate shot_numbers, add mgrs_tile, write, delete the parts

Like download_all_valid, a shot in an S2 overlap goes to every tile that
covers it; `mgrs_tile == tile` keeps each shot once (see tools/mgrs_tile.py).
Shots outside every S2 tile in the grid (e.g. over water) are dropped and
counted in the zone's state line.

    python -m tools.run run=gedi_to_s2_tiles run.year=2019
"""
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd

from tools.mgrs_tile import mgrs_100km
from tools.task_state import TaskState

_TILES = None   # S2 footprints, loaded once per worker process


def _init_worker(s2_grid_file: str):
    global _TILES
    _TILES = gpd.read_parquet(Path(s2_grid_file).expanduser(), columns=['Name', 'geometry'])
    _TILES = _TILES.to_crs(4326) if _TILES.crs else _TILES.set_crs(4326)


def _write(df: gpd.GeoDataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    df.to_parquet(tmp)
    os.replace(tmp, path)      # readers never see a half-written file


def _split_zone(zone_dir: Path, parts_dir: Path, state: TaskState):
    key = f'zone/{zone_dir.name}'
    try:
        frames = [gpd.read_parquet(f).assign(orbit=f.stem)
                  for f in sorted(zone_dir.glob('*.parquet'))]
        if not frames:
            state.mark_done(key, '0 shots')
            return key, None
        df = pd.concat(frames, ignore_index=True)
        # the legacy download wrote these without a CRS
        df = df.set_crs(4326, allow_override=True)
        joined = gpd.sjoin(df, _TILES, predicate='intersects', how='inner')
        outside = len(df) - joined.index.nunique()
        for tile, g in joined.groupby('Name'):
            _write(g.drop(columns=['index_right', 'Name']), parts_dir / tile / f'{zone_dir.name}.parquet')
        state.mark_done(key, f'{len(df)} shots, {outside} outside S2 grid')
        return key, None
    except Exception as e:
        state.mark_failed(key, repr(e))
        return key, repr(e)


def _merge_tile(tile_parts: Path, save_dir: Path, state: TaskState, keep_parts: bool):
    tile = tile_parts.name
    try:
        out = save_dir / f'{tile}.parquet'
        files = sorted(tile_parts.glob('*.parquet')) + ([out] if out.exists() else [])
        df = pd.concat([gpd.read_parquet(f) for f in files], ignore_index=True)
        # a shot sits in one zone dir, but a zone-edge shot can come back from
        # two GEE bbox queries; an existing output repeats the parts on rerun
        df = df.drop_duplicates('shot_number', ignore_index=True)
        df['mgrs_tile'] = mgrs_100km(df.geometry.x.values, df.geometry.y.values)
        _write(df, out)
        state.mark_done(tile, f'{len(df)} shots')
        if not keep_parts:
            shutil.rmtree(tile_parts)
        return tile, None
    except Exception as e:
        state.mark_failed(tile, repr(e))
        return tile, repr(e)


def regroup_to_s2_tiles(src_dir: str, save_dir: str, s2_grid_file: str, state_dir: str,
                        workers: int = 8, keep_parts: bool = False, **kwargs):
    src_dir = Path(src_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    parts_dir = save_dir.parent / f'{save_dir.name}_parts'
    save_dir.mkdir(parents=True, exist_ok=True)
    state = TaskState(state_dir)

    # ---- pass 1: split each zone into per-tile parts
    zones = sorted(d for d in src_dir.iterdir() if d.is_dir() and not d.name.startswith('.'))
    done = state.done()
    todo = [z for z in zones if f'zone/{z.name}' not in done]
    print(f'{len(zones)} zones, {len(zones) - len(todo)} already split, {len(todo)} to split')
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(s2_grid_file,)) as pool:
        errors = [r for r in pool.map(_split_zone, todo, [parts_dir] * len(todo),
                                      [state] * len(todo)) if r[1]]
    if errors:
        for key, err in errors[:20]:
            print(f'  failed  {key}  {err}')
        print(f'{len(errors)} zones failed; not merging until every zone is split. Rerun.')
        state.compact()
        return

    # ---- pass 2: merge parts into one file per tile
    tile_dirs = sorted(d for d in parts_dir.iterdir() if d.is_dir()) if parts_dir.exists() else []
    print(f'merging {len(tile_dirs)} tiles')
    with ProcessPoolExecutor(workers) as pool:
        errors = [r for r in pool.map(_merge_tile, tile_dirs, [save_dir] * len(tile_dirs),
                                      [state] * len(tile_dirs), [keep_parts] * len(tile_dirs))
                  if r[1]]
    for key, err in errors[:20]:
        print(f'  failed  {key}  {err}')
    if errors:
        print(f'{len(errors)} tiles failed to merge; their parts are kept. Rerun.')
    elif not keep_parts and parts_dir.exists() and not any(parts_dir.iterdir()):
        parts_dir.rmdir()
    state.compact()
    print(f'{len(list(save_dir.glob("*.parquet")))} tile files in {save_dir}, state: {state.path}')
