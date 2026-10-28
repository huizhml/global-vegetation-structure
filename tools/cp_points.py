"""Pair GEDI shots with VSM Q0/Q1/Q2 predictions for conformal calibration.

Mirrors the 2020 calibration/test sets (veg_sensitivity_gt0p95/subset_{cal,test},
paper Methods 4.4) for a year that is not in the training data (2024 first):

    select_cp_points     <save_dir>/<split>/<tile>.parquet: GEDI shots of the model's
                         cal / test S2 tiles (tiles_{cal,test}.csv of the training split)
    add_cp_predictions   the same tables plus RH*_Q{q} read at every shot, for the
                         quantiles given; run once on hendrix (Q1, local) and once
                         on LUMI (Q0/Q2, LUMI-O), each step into its own save_dir
    combine_cp_points    <save_dir>/<split>.parquet, one table per split

Shots: the year's all_valid download filtered to sensitivity >= 0.95 and
mgrs_tile == tile, then per tile as many random shots as the 2020 set has in that
tile (all of them when the year has fewer). Tiles without 2020 shots are skipped.
So the spatial density follows the training sampling (0.7034 shots / km² of
land per MGRS zone, then sensitivity >= 0.95) the 2020 sets were cut from.

Predictions are read at every shot, which touches nearly every block of a band,
so a tile costs about a whole band read per RH and quantile. Q1 of 2024 is on
hendrix, Q0/Q2 only on LUMI-O (/vsis3/, credentials from the AWS_* environment,
see scripts/hendrix/cp_points.sh), hence the two add_cp_predictions runs.

Values are written in metres (rasters are int16 decimetres), the unit
src-cp's preprocessing expects.

    python -m tools.run run=select_cp_points
    python -m tools.run run=add_cp_predictions run.q_idxs=[1]
    python -m tools.run run=combine_cp_points run.split=cal
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from tools.task_state import TaskState

SPLITS = ('cal', 'test')

# GDAL options for reading COGs from LUMI-O; credentials stay in the environment.
_GDAL_ENV = {
    'AWS_S3_ENDPOINT': 'lumidata.eu',
    'AWS_VIRTUAL_HOSTING': 'FALSE',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
    'GDAL_HTTP_MAX_RETRY': '5',
    'GDAL_HTTP_RETRY_DELAY': '2',
    'GDAL_CACHEMAX': '256',     # MB per worker
}

_ecoregions = None      # per worker process, loaded once by _init_worker


def _init_worker(biome_file: str = None):
    global _ecoregions
    for k, v in _GDAL_ENV.items():
        os.environ.setdefault(k, v)
    if biome_file:
        eco = gpd.read_file(Path(biome_file).expanduser())
        _ecoregions = eco[['BIOME', 'ECO_ID', 'geometry']].to_crs(epsg=4326)


def _write(df: pd.DataFrame, out: Path):
    tmp = out.with_name(f'.{out.name}.tmp')
    df.to_parquet(tmp)
    os.replace(tmp, out)        # readers never see a half-written file


def _run_tiles(fn, jobs: dict, state: TaskState, workers: int, tiles_per_worker: int,
               biome_file: str = None):
    """Run fn(**kwargs) for {key: kwargs} in worker processes, recording the outcome
    per key in state. A fresh pool per batch of workers x tiles_per_worker keys:
    worker memory creeps by a few GB over a few hundred tiles, and
    max_tasks_per_child hung the pool (job 5681, Python 3.12). If a worker dies
    (e.g. OOM) the run stops with an error; a rerun continues where it stopped."""
    keys = list(jobs)
    n_failed, i = 0, 0
    batch = workers * tiles_per_worker
    for start in range(0, len(keys), batch):
        # spawn: workers start without the parent's GDAL state
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init_worker, initargs=(biome_file,)) as pool:
            futs = {pool.submit(fn, **jobs[k]): k for k in keys[start:start + batch]}
            for f in as_completed(futs):
                i += 1
                key = futs[f]
                try:
                    info = f.result()
                    state.mark_done(key, info)
                except BrokenProcessPool:
                    pool.shutdown(cancel_futures=True)
                    raise RuntimeError(f'worker process died at {i}/{len(keys)} ({key}), '
                                       'probably out of memory; rerun to continue') from None
                except Exception as e:
                    n_failed += 1
                    state.mark_failed(key, repr(e))
                    info = f'FAILED {e!r}'
                print(f'[{i}/{len(keys)}] {key} {info}')
    print(f'{len(keys) - n_failed} done, {n_failed} failed')


def _todo(keys: list, state: TaskState, out_of) -> list:
    """Keys neither done in the state nor with an output file (out_of(key) -> Path)."""
    done = state.done()
    return [k for k in keys if k not in done and not out_of(k).exists()]


def _clean_tmp(save_dir: Path):
    for split in SPLITS:
        (save_dir / split).mkdir(parents=True, exist_ok=True)
        for tmp in (save_dir / split).glob('.*.parquet.tmp'):   # left by a killed run
            tmp.unlink()


# ---------------------------------------------------------------------------
# select


def reference_counts(ref_dirs: dict) -> dict:
    """{split: {tile: number of shots}} of the reference sets, from parquet footers."""
    counts = {}
    for split, d in ref_dirs.items():
        d = Path(d).expanduser()
        counts[split] = {f.stem: pq.ParquetFile(f).metadata.num_rows for f in sorted(d.glob('*.parquet'))}
    return counts


def _select_tile(tile: str, split: str, n_ref: int, gedi_dir: Path, save_dir: Path,
                 rh_idxs: list[int], sensitivity_min: float, seed: int):
    rh_cols = [f'rh{i}' for i in rh_idxs]
    shots = gpd.read_parquet(
        gedi_dir / f'{tile}.parquet',
        columns=['shot_number', 'beam', 'sensitivity', 'mgrs_tile', 'geometry'] + rh_cols,
        filters=[('mgrs_tile', '==', tile), ('sensitivity', '>=', sensitivity_min)])
    n_avail = len(shots)
    if n_avail > n_ref:
        rng = np.random.default_rng([seed, int.from_bytes(tile.encode(), 'little')])
        shots = shots.iloc[np.sort(rng.choice(n_avail, size=n_ref, replace=False))]
    if len(shots) == 0:
        return f'{split}: 0 of {n_avail} shots (2020: {n_ref})'
    shots = shots.to_crs(epsg=4326)

    df = shots.drop(columns='mgrs_tile').reset_index(drop=True)
    df.insert(0, 'tile', tile)
    df.insert(1, 'split', split)
    df['lon'], df['lat'] = df.geometry.x, df.geometry.y
    # BIOME / ECO_ID from WWF ecoregions; a shot on a shared border keeps one match
    df = gpd.sjoin(df, _ecoregions, how='left', predicate='within').drop(columns='index_right')
    df = df[~df.index.duplicated()]
    _write(df, save_dir / split / f'{tile}.parquet')
    return f'{split}: {len(df)} of {n_avail} shots (2020: {n_ref})'


def select_cp_points(year: int, gedi_dir: str, save_dir: str, biome_file: str,
                     split_dir: str, ref_dirs: dict, state_dir: str = None,
                     rh_idxs: list[int] = tuple(range(101)), sensitivity_min: float = 0.95,
                     seed: int = 42, workers: int = 8, tiles_per_worker: int = 20,
                     tiles_file: str = None, chunk: int = 1, n_chunks: int = 1, **kwargs):
    """GEDI shots of the model's cal / test tiles, as many per tile as the 2020 set.

    split_dir holds tiles_{cal,test}.csv of the training split; ref_dirs maps
    cal / test to the per-tile parquets of the 2020 sets. Resumable: tiles done
    in the state or with an output file are skipped. `chunk` of `n_chunks`
    (1-based) takes every n_chunks-th tile, for job arrays.
    """
    gedi_dir = Path(gedi_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    split_dir = Path(split_dir).expanduser()
    state = TaskState(state_dir or f'{save_dir}_state')
    _clean_tmp(save_dir)

    ref = reference_counts(ref_dirs)
    splits = {}
    for split in SPLITS:
        for t in (split_dir / f'tiles_{split}.csv').read_text().split():
            splits[t] = split
    if tiles_file:
        tiles = Path(tiles_file).expanduser().read_text().split()
    else:
        tiles = sorted(splits)
    tiles = [t for t in tiles if t in splits][chunk - 1::n_chunks]
    no_ref = [t for t in tiles if ref[splits[t]].get(t, 0) == 0]
    no_gedi = [t for t in tiles if t not in no_ref and not (gedi_dir / f'{t}.parquet').exists()]
    tiles = [t for t in tiles if t not in no_ref and t not in no_gedi]
    todo = _todo(tiles, state, lambda t: save_dir / splits[t] / f'{t}.parquet')
    print(f'{len(tiles)} tiles ({sum(splits[t] == "cal" for t in tiles)} cal, '
          f'{sum(splits[t] == "test" for t in tiles)} test), {len(todo)} to do; '
          f'skipped {len(no_ref)} without 2020 shots, {len(no_gedi)} without a {year} GEDI file')

    common = dict(gedi_dir=gedi_dir, save_dir=save_dir, rh_idxs=list(rh_idxs),
                  sensitivity_min=sensitivity_min, seed=seed)
    jobs = {t: dict(tile=t, split=splits[t], n_ref=ref[splits[t]][t], **common) for t in todo}
    _run_tiles(_select_tile, jobs, state, workers, tiles_per_worker, biome_file)


# ---------------------------------------------------------------------------
# predictions


def _band_path(tile: str, year: int, rh: int, q: int, local_cog_dirs: list[Path]) -> str:
    name = f'RH{rh}_Q{q}.tif'
    for d in local_cog_dirs:
        if (d / tile / name).exists():
            return str(d / tile / name)
    return f'/vsis3/{tile[:3].lower()}-{year}/{tile}/{name}'


def _read_band(path: str, ref, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Values at (rows, cols) in metres, NaN for nodata or outside; reads only the
    blocks that hold a shot."""
    out = np.full(len(rows), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        if src.transform != ref.transform or src.shape != ref.shape or src.crs != ref.crs:
            raise ValueError(f'{path}: grid differs from {ref.name}')
        bh, bw = src.block_shapes[0]
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        for br, bc in np.unique(np.stack([rows[inside] // bh, cols[inside] // bw], 1), axis=0):
            r0, c0 = br * bh, bc * bw
            win = Window(c0, r0, min(bw, src.width - c0), min(bh, src.height - r0))
            data = src.read(1, window=win)
            m = inside & (rows // bh == br) & (cols // bw == bc)
            v = data[rows[m] - r0, cols[m] - c0].astype(np.float32)
            v[v == src.nodata] = np.nan
            out[m] = v / 10.
    return out


def _band_order(col: str):
    rh, q = col[2:].split('_Q')
    return int(rh), int(q)


def _add_tile(tile: str, split: str, year: int, src_file: Path, save_dir: Path,
              local_cog_dirs: list[Path], rh_idxs: list[int], q_idxs: list[int], threads: int):
    df = gpd.read_parquet(src_file)
    ref_path = _band_path(tile, year, rh_idxs[0], q_idxs[0], local_cog_dirs)
    with rasterio.open(ref_path) as ref:
        tr = Transformer.from_crs(4326, ref.crs, always_xy=True)
        xs, ys = tr.transform(df['lon'].values, df['lat'].values)
        rows, cols = rasterio.transform.rowcol(ref.transform, xs, ys)
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        values = {}
        with ThreadPoolExecutor(threads) as pool:
            futs = {pool.submit(_read_band, _band_path(tile, year, rh, q, local_cog_dirs),
                                ref, rows, cols): f'RH{rh}_Q{q}'
                    for rh in rh_idxs for q in q_idxs}
            for f in as_completed(futs):
                values[futs[f]] = f.result()

    pred_cols = sorted([c for c in df.columns if c.startswith('RH') and '_Q' in c] + list(values),
                       key=_band_order)
    df = df.assign(**values)
    df = df[[c for c in df.columns if c not in pred_cols] + pred_cols]
    _write(df, save_dir / split / f'{tile}.parquet')
    n_nan = int(np.isnan(np.stack(list(values.values()))).any(axis=0).sum())
    return f'{split}: {len(df)} shots, {len(values)} bands, {n_nan} shots with a nodata band'


def add_cp_predictions(year: int, points_dir: str, save_dir: str, q_idxs: list[int],
                       local_cog_dirs: list[str] = (), state_dir: str = None,
                       rh_idxs: list[int] = tuple(range(101)), workers: int = 8,
                       threads: int = 16, tiles_per_worker: int = 20,
                       tiles_file: str = None, chunk: int = 1, n_chunks: int = 1, **kwargs):
    """<points_dir>/<split>/<tile>.parquet plus RH{rh}_Q{q} for q in q_idxs at every
    shot -> <save_dir>/<split>/<tile>.parquet. Each band is read from the first of
    local_cog_dirs that has it, else from LUMI-O. Resumable like select_cp_points.
    """
    points_dir = Path(points_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    local_cog_dirs = [Path(d).expanduser() for d in local_cog_dirs]
    state = TaskState(state_dir or f'{save_dir}_state')
    _clean_tmp(save_dir)

    files = {f.stem: (split, f) for split in SPLITS for f in sorted((points_dir / split).glob('*.parquet'))}
    tiles = sorted(files)
    if tiles_file:
        keep = set(Path(tiles_file).expanduser().read_text().split())
        tiles = [t for t in tiles if t in keep]
    tiles = tiles[chunk - 1::n_chunks]
    todo = _todo(tiles, state, lambda t: save_dir / files[t][0] / f'{t}.parquet')
    print(f'{len(tiles)} tiles in {points_dir}, {len(todo)} to do; quantiles {list(q_idxs)}')

    common = dict(year=year, save_dir=save_dir, local_cog_dirs=local_cog_dirs,
                  rh_idxs=list(rh_idxs), q_idxs=list(q_idxs), threads=threads)
    jobs = {t: dict(tile=t, split=files[t][0], src_file=files[t][1], **common) for t in todo}
    _run_tiles(_add_tile, jobs, state, workers, tiles_per_worker)


# ---------------------------------------------------------------------------
# combine


def combine_cp_points(save_dir: str, split: str, out_file: str = None,
                      max_per_biome: int = None, seed: int = 42, **kwargs):
    """Concatenate <save_dir>/<split>/*.parquet into one table. max_per_biome (off by
    default, the 2020 sets are not capped) keeps at most that many random shots per
    BIOME (rows without a biome kept as a group)."""
    save_dir = Path(save_dir).expanduser()
    files = sorted((save_dir / split).glob('*.parquet'))
    df = pd.concat([gpd.read_parquet(f) for f in files], ignore_index=True)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    n_all = df.groupby('BIOME', dropna=False).size()
    if max_per_biome:
        df = (df.sample(frac=1, random_state=seed)
                .groupby('BIOME', dropna=False).head(max_per_biome).reset_index(drop=True))

    out_file = Path(out_file).expanduser() if out_file else save_dir / f'{split}.parquet'
    _write(df, out_file)
    counts = pd.DataFrame({'shots': n_all, 'kept': df.groupby('BIOME', dropna=False).size()})
    print(f'{len(files)} tile files -> {out_file} ({len(df):,} shots)')
    print(counts.to_string())
