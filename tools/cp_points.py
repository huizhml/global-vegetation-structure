"""Pair GEDI shots with VSM Q0/Q1/Q2 predictions for conformal calibration.

For one year (2024 first) this makes the calibration and test tables that
scripts/cp/run_cp.py fits and evaluates per-biome offsets on:

    sample_cp_points    one parquet per S2 tile in <save_dir>/<split>/<tile>.parquet
    combine_cp_points   <save_dir>/<split>.parquet, capped per biome

Shots come from the unfiltered all_valid download, filtered to what the
gt0p95 pool keeps (sensitivity >= 0.95; the other quality flags were applied
at download) and to mgrs_tile == tile, so every shot is used once. The split
is per tile (md5 of the tile name), not per shot: neighbouring shots of one
track are near-duplicates, and a per-shot split would put them on both sides.

Predictions are read pointwise, not as whole tiles: Q0/Q2 exist only on LUMI-O
(/vsis3/, credentials from the AWS_* environment, see scripts/hendrix/cp_points.sh),
and a 2024 COG has 1024 px blocks, so a few hundred random shots already touch
every block of every band (~4 s per band over S3, 303 bands per tile). Shots
are therefore taken from a few random blocks per tile (`blocks_per_tile` x
`shots_per_block`), which costs a few blocks per band. Q1 is moving to hendrix
(copy_from_lumio.sh deletes the LUMI-O copy), so each band is read from the
first of `local_cog_dirs` that has it and from LUMI-O otherwise.

Values are written in metres (rasters are int16 decimetres), the unit
src-cp's preprocessing expects.

    python -m tools.run run=sample_cp_points run.year=2024
    python -m tools.run run=combine_cp_points run.split=cal
"""
import hashlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from tools.task_state import TaskState

# GDAL options for reading COGs from LUMI-O; credentials stay in the environment.
_GDAL_ENV = {
    'AWS_S3_ENDPOINT': 'lumidata.eu',
    'AWS_VIRTUAL_HOSTING': 'FALSE',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
    'GDAL_HTTP_MAX_RETRY': '5',
    'GDAL_HTTP_RETRY_DELAY': '2',
}

_ecoregions = None      # per worker process, loaded once by _init_worker


def _init_worker(biome_file: str):
    global _ecoregions
    for k, v in _GDAL_ENV.items():
        os.environ.setdefault(k, v)
    eco = gpd.read_file(Path(biome_file).expanduser())
    _ecoregions = eco[['BIOME', 'ECO_ID', 'geometry']].to_crs(epsg=4326)


def _tile_hash(tile: str) -> int:
    return int(hashlib.md5(tile.encode()).hexdigest()[:8], 16)


def tile_split(tile: str, cal_pct: int, test_pct: int):
    """'cal', 'test' or None, fixed per tile name."""
    h = _tile_hash(tile) % 100
    if h < cal_pct:
        return 'cal'
    if h < cal_pct + test_pct:
        return 'test'
    return None


def _band_path(tile: str, year: int, rh: int, q: int, local_cog_dirs: list[Path]) -> str:
    name = f'RH{rh}_Q{q}.tif'
    for d in local_cog_dirs:
        if (d / tile / name).exists():
            return str(d / tile / name)
    return f'/vsis3/{tile[:3].lower()}-{year}/{tile}/{name}'


def _pick_shots(shots: gpd.GeoDataFrame, src, blocks_per_tile: int, shots_per_block: int,
                min_shots_per_block: int, rng: np.random.Generator):
    """-> (shots subset, rows, cols, blocks [(block row, block col)]) on src's grid."""
    tr = Transformer.from_crs(4326, src.crs, always_xy=True)
    xs, ys = tr.transform(shots.geometry.x.values, shots.geometry.y.values)
    rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
    shots, rows, cols = shots[inside], rows[inside], cols[inside]

    bh, bw = src.block_shapes[0]
    block_id = (rows // bh) * 100_000 + cols // bw
    ids, counts = np.unique(block_id, return_counts=True)
    eligible = ids[counts >= min_shots_per_block]
    chosen = rng.choice(eligible, size=min(blocks_per_tile, len(eligible)), replace=False)

    keep = []
    for b in chosen:
        idx = np.flatnonzero(block_id == b)
        keep.append(rng.choice(idx, size=min(shots_per_block, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)
    blocks = [(int(b // 100_000), int(b % 100_000)) for b in chosen]
    return shots.iloc[keep], rows[keep], cols[keep], blocks


def _read_band(path: str, ref, rows: np.ndarray, cols: np.ndarray, blocks) -> np.ndarray:
    """Values at (rows, cols) in metres, NaN for nodata; reads only `blocks`."""
    out = np.full(len(rows), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        if src.transform != ref.transform or src.shape != ref.shape or src.crs != ref.crs:
            raise ValueError(f'{path}: grid differs from {ref.name}')
        bh, bw = src.block_shapes[0]
        for br, bc in blocks:
            r0, c0 = br * bh, bc * bw
            win = Window(c0, r0, min(bw, src.width - c0), min(bh, src.height - r0))
            data = src.read(1, window=win)
            m = (rows // bh == br) & (cols // bw == bc)
            v = data[rows[m] - r0, cols[m] - c0].astype(np.float32)
            v[v == src.nodata] = np.nan
            out[m] = v / 10.
    return out


def _sample_tile(tile: str, split: str, year: int, gedi_dir: Path, save_dir: Path,
                 local_cog_dirs: list[Path], rh_idxs: list[int], q_idxs: list[int],
                 sensitivity_min: float, blocks_per_tile: int, shots_per_block: int,
                 min_shots_per_block: int, seed: int, threads: int):
    """-> info string for the state; writes <save_dir>/<split>/<tile>.parquet."""
    rh_cols = [f'rh{i}' for i in rh_idxs]
    shots = gpd.read_parquet(
        gedi_dir / f'{tile}.parquet',
        columns=['shot_number', 'beam', 'sensitivity', 'mgrs_tile', 'geometry'] + rh_cols,
        filters=[('mgrs_tile', '==', tile), ('sensitivity', '>=', sensitivity_min)])
    if len(shots) == 0:
        return f'{split}: 0 shots after filter'
    shots = shots.to_crs(epsg=4326)

    rng = np.random.default_rng([seed, _tile_hash(tile)])
    ref_path = _band_path(tile, year, rh_idxs[0], q_idxs[0], local_cog_dirs)
    with rasterio.open(ref_path) as ref:
        shots, rows, cols, blocks = _pick_shots(
            shots, ref, blocks_per_tile, shots_per_block, min_shots_per_block, rng)
        if len(shots) == 0:
            return f'{split}: 0 shots in blocks with >= {min_shots_per_block}'
        bands = [(rh, q) for rh in rh_idxs for q in q_idxs]
        values = {}
        with ThreadPoolExecutor(threads) as pool:
            futs = {pool.submit(_read_band, _band_path(tile, year, rh, q, local_cog_dirs),
                                ref, rows, cols, blocks): (rh, q) for rh, q in bands}
            for f in as_completed(futs):
                rh, q = futs[f]
                values[f'RH{rh}_Q{q}'] = f.result()

    df = shots.drop(columns='mgrs_tile').reset_index(drop=True)
    df.insert(0, 'tile', tile)
    df.insert(1, 'split', split)
    df['lon'], df['lat'] = df.geometry.x, df.geometry.y
    df = df.join(pd.DataFrame({k: values[k] for k in sorted(values, key=_band_order)}))

    # BIOME / ECO_ID from WWF ecoregions; a shot on a shared border keeps one match
    df = gpd.sjoin(df, _ecoregions, how='left', predicate='within').drop(columns='index_right')
    df = df[~df.index.duplicated()]

    out = save_dir / split / f'{tile}.parquet'
    tmp = out.with_name(f'.{out.name}.tmp')
    df.to_parquet(tmp)
    os.replace(tmp, out)        # readers never see a half-written file
    return f'{split}: {len(df)} shots in {len(blocks)} blocks'


def _band_order(col: str):
    rh, q = col[2:].split('_Q')
    return int(rh), int(q)


def sample_cp_points(year: int, gedi_dir: str, save_dir: str, biome_file: str,
                     local_cog_dirs: list[str] = (), state_dir: str = None,
                     cal_pct: int = 50, test_pct: int = 50,
                     rh_idxs: list[int] = tuple(range(101)), q_idxs: list[int] = (0, 1, 2),
                     sensitivity_min: float = 0.95, blocks_per_tile: int = 4,
                     shots_per_block: int = 50, min_shots_per_block: int = 10,
                     seed: int = 42, workers: int = 8, threads: int = 16,
                     tiles_file: str = None, chunk: int = 1, n_chunks: int = 1, **kwargs):
    """Sample shots per tile of gedi_dir and pair them with the year's Q0/Q1/Q2.

    Resumable: tiles already in the state (done or with an output file) are
    skipped. Tiles in neither split (cal_pct + test_pct < 100) are not read.
    `chunk` of `n_chunks` (1-based) takes every n_chunks-th tile, for job arrays.
    """
    gedi_dir = Path(gedi_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    local_cog_dirs = [Path(d).expanduser() for d in local_cog_dirs]
    state = TaskState(state_dir or f'{save_dir}_state')

    if tiles_file:
        tiles = Path(tiles_file).expanduser().read_text().split()
    else:
        tiles = sorted(p.stem for p in gedi_dir.glob('*.parquet'))
    tiles = tiles[chunk - 1::n_chunks]
    splits = {t: tile_split(t, cal_pct, test_pct) for t in tiles}
    for split in ('cal', 'test'):
        (save_dir / split).mkdir(parents=True, exist_ok=True)
        for tmp in (save_dir / split).glob('.*.parquet.tmp'):   # left by a killed run
            tmp.unlink()

    done = state.done()
    todo = [t for t in tiles if splits[t] and t not in done
            and not (save_dir / splits[t] / f'{t}.parquet').exists()]
    print(f'{len(tiles)} tiles in chunk {chunk}/{n_chunks}: '
          f'{sum(s == "cal" for s in splits.values())} cal, '
          f'{sum(s == "test" for s in splits.values())} test, {len(todo)} to do')

    args = dict(year=year, gedi_dir=gedi_dir, save_dir=save_dir, local_cog_dirs=local_cog_dirs,
                rh_idxs=list(rh_idxs), q_idxs=list(q_idxs), sensitivity_min=sensitivity_min,
                blocks_per_tile=blocks_per_tile, shots_per_block=shots_per_block,
                min_shots_per_block=min_shots_per_block, seed=seed, threads=threads)
    n_failed = 0
    # spawn: workers start without the parent's GDAL state
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_worker, initargs=(biome_file,)) as pool:
        futs = {pool.submit(_sample_tile, t, splits[t], **args): t for t in todo}
        for i, f in enumerate(as_completed(futs), 1):
            tile = futs[f]
            try:
                info = f.result()
                state.mark_done(tile, info)
            except Exception as e:
                n_failed += 1
                state.mark_failed(tile, repr(e))
                info = f'FAILED {e!r}'
            print(f'[{i}/{len(todo)}] {tile} {info}')
    print(f'{len(todo) - n_failed} tiles done, {n_failed} failed')


def combine_cp_points(save_dir: str, split: str, out_file: str = None,
                      max_per_biome: int = 200_000, seed: int = 42, **kwargs):
    """Concatenate <save_dir>/<split>/*.parquet into one table, at most
    max_per_biome random shots per BIOME (rows without a biome kept as a group)."""
    save_dir = Path(save_dir).expanduser()
    files = sorted((save_dir / split).glob('*.parquet'))
    df = pd.concat([gpd.read_parquet(f) for f in files], ignore_index=True)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    n_all = df.groupby('BIOME', dropna=False).size()
    df = (df.sample(frac=1, random_state=seed)
            .groupby('BIOME', dropna=False).head(max_per_biome).reset_index(drop=True))

    out_file = Path(out_file).expanduser() if out_file else save_dir / f'{split}.parquet'
    tmp = out_file.with_name(f'.{out_file.name}.tmp')
    df.to_parquet(tmp)
    os.replace(tmp, out_file)
    counts = pd.DataFrame({'shots': n_all, 'kept': df.groupby('BIOME', dropna=False).size()})
    print(f'{len(files)} tile files -> {out_file} ({len(df):,} shots)')
    print(counts.to_string())
