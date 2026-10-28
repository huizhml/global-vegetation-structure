"""Apply per-biome conformal offsets to the VSM Q0 (upper) / Q2 (lower) tiles.

The offsets come from scripts/cp/run_cp.py (cp_results.parquet, decimetres,
one row per method x WWF biome x RH). Per pixel, as in the paper (Methods 4.4)
and as in src-cp's preprocessing:

    lo, hi = min(Q0, Q1, Q2), max(Q0, Q1, Q2)      # quantiles sorted first
    Q2' = lo - c_lo[rh, biome],  Q0' = hi + c_hi[rh, biome]

then clipped so that Q2' <= Q1 <= Q0' (offsets can be negative where the raw
interval overcovers). Q1 is not changed and not written. Pixels of uncalibrated
biomes (Tundra, see `exclude_biomes`), of WWF 98/99 (lakes, rock and ice), outside
every WWF polygon, or with a nodata quantile keep their raw Q0/Q2.

The biome of a pixel is the WWF polygon its centre falls in, the same rule as
the point-in-polygon join that gave the calibration points their BIOME.

    apply_cp_offsets   <save_dir>/<TILE>/RH{k}_Q0.tif, RH{k}_Q2.tif (LERC_ZSTD COG, int16 dm)
    check_cp_tiles     compares a few written tiles against the test points
    cp_upload_tiles    finished tiles not yet on Source Coop (tile whitelist for sc_upload)
    prune_uploaded_tiles  deletes local tiles that are completely on Source Coop

    python -m tools.run run=apply_cp_offsets run.year=2020
    python -m tools.run run=check_cp_tiles run.year=2020 'run.tiles=[01GEL,32UNG]'
"""
import multiprocessing
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.errors import RasterioIOError
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from shapely.affinity import translate
from shapely.geometry import Polygon, box
from shapely.ops import transform as shapely_transform, unary_union

from tools.task_state import TaskState

NODATA = 32767
_ecoregions = None      # per worker process, loaded once by _init_worker


def _init_worker(biome_file: str):
    global _ecoregions
    eco = gpd.read_file(Path(biome_file).expanduser())
    _ecoregions = eco[['BIOME', 'geometry']].to_crs(epsg=4326)


def load_offsets(cp_results: str, method: str, exclude_biomes, rh_idxs):
    """-> (c_lo, c_hi) int16 [101, 256] indexed by rh and biome, calibrated bool [256]."""
    df = pd.read_parquet(Path(cp_results).expanduser()).reset_index()
    df = df[(df['method'] == method) & ~df['biome'].isin(list(exclude_biomes))]
    if df.empty:
        raise ValueError(f'no {method} offsets in {cp_results}')
    if df[['q_lo', 'q_hi']].isna().any().any():
        raise ValueError(f'NaN offsets in {cp_results}:\n{df[df.q_lo.isna() | df.q_hi.isna()]}')
    biomes = sorted(df['biome'].unique())
    missing = [(rh, b) for rh in rh_idxs for b in biomes
               if not ((df['rh'] == rh) & (df['biome'] == b)).any()]
    if missing:
        raise ValueError(f'{len(missing)} (rh, biome) without an offset, e.g. {missing[:5]}')
    c_lo = np.zeros((101, 256), dtype=np.int16)
    c_hi = np.zeros((101, 256), dtype=np.int16)
    # offsets are decimetre quantiles of integer scores, so already whole numbers
    c_lo[df['rh'].values, df['biome'].values] = np.rint(df['q_lo'].values)
    c_hi[df['rh'].values, df['biome'].values] = np.rint(df['q_hi'].values)
    calibrated = np.zeros(256, dtype=bool)
    calibrated[biomes] = True
    return c_lo, c_hi, calibrated


def _polygons(geom):
    """Polygonal part of an intersection (drops touching lines and points)."""
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        return geom
    polys = [g for g in getattr(geom, 'geoms', []) if g.geom_type in ('Polygon', 'MultiPolygon')]
    return unary_union(polys) if polys else Polygon()


def _footprint_4326(ref) -> list:
    """ref's footprint in lon/lat, split in two where it crosses the antimeridian."""
    ring = np.asarray(box(*ref.bounds).segmentize(1000).exterior.coords)
    lon, lat = Transformer.from_crs(ref.crs, 4326, always_xy=True).transform(ring[:, 0], ring[:, 1])
    lon = np.asarray(lon)
    if lon.max() - lon.min() > 180:         # wrapped: continue past 180 instead
        lon = np.where(lon < 0, lon + 360, lon)
    poly = Polygon(zip(lon, lat))
    parts = [poly.intersection(box(-180, -90, 180, 90)),
             translate(poly.intersection(box(180, -90, 540, 90)), xoff=-360)]
    return [p for p in parts if not p.is_empty]


def biome_raster(ref) -> np.ndarray:
    """WWF BIOME (uint8, 0 = no polygon) on ref's grid, by pixel centre."""
    to_tile = Transformer.from_crs(4326, ref.crs, always_xy=True).transform
    shapes = []
    for part in _footprint_4326(ref):
        part = part.buffer(0.01)
        idx = _ecoregions.sindex.query(part, predicate='intersects')
        for geom, biome in zip(_ecoregions.geometry.values[idx], _ecoregions['BIOME'].values[idx]):
            if not geom.is_valid:
                geom = geom.buffer(0)
            clipped = _polygons(geom.intersection(part))
            if not clipped.is_empty:
                shapes.append((shapely_transform(to_tile, clipped), int(biome)))
    if not shapes:
        return np.zeros(ref.shape, dtype=np.uint8)
    return rasterize(shapes, out_shape=ref.shape, transform=ref.transform,
                     fill=0, dtype=np.uint8)


def calibrate(q0, q1, q2, biome, c_lo_rh, c_hi_rh, calibrated):
    """-> new (Q0, Q2) int16, elementwise on arrays of any shape; see the module docstring.

    int16 throughout: valid heights and offsets are a few thousand dm at most,
    and nodata pixels, where the sums can wrap, are replaced by the raw value.
    """
    cal = calibrated[biome] & (q0 != NODATA) & (q1 != NODATA) & (q2 != NODATA)
    lo = np.minimum(np.minimum(q0, q1), q2)
    lo -= c_lo_rh[biome]
    hi = np.maximum(np.maximum(q0, q1), q2)
    hi += c_hi_rh[biome]
    np.minimum(lo, q1, out=lo)
    np.maximum(hi, q1, out=hi)
    return np.where(cal, hi, q0), np.where(cal, lo, q2)


def _write_cog(arr, ref, dst: Path, description: str, tags: dict):
    """Array -> GTiff in memory -> LERC_ZSTD COG at dst (same options as original/tiles/cog)."""
    from postprocessing.core.translate import _translate_one
    profile = dict(driver='GTiff', dtype='int16', count=1, nodata=NODATA, crs=ref.crs,
                   transform=ref.transform, width=ref.width, height=ref.height,
                   tiled=True, blockxsize=1024, blockysize=1024)
    tmp = dst.with_name(f'.{dst.name}.tmp')
    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(arr, 1)
            ds.set_band_description(1, description)
            ds.update_tags(**tags)
        # in_memory: rio-cogeo's intermediate stays in RAM, not on the shared disk
        _translate_one(mem.name, str(tmp), profile='LERC_ZSTD', in_memory=True)
    os.replace(tmp, dst)        # readers never see a half-written file


def _retry_io(fn, *args, attempts: int = 5, wait_s: float = 30):
    """fn(*args), retried on RasterioIOError.

    Under load the shared disk sometimes fails an open ("not recognized as being
    in a supported file format") on a file that reads fine moments later.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except RasterioIOError:
            if attempt == attempts:
                raise
            time.sleep(wait_s * attempt)


def _calibrate_rh(src_dir: dict, rh: int, ref, biome, c_lo_rh, c_hi_rh, calibrated):
    """-> (Q0', Q2') of one RH, whole tile at once (a few GB per worker)."""
    q = []
    for i in (0, 1, 2):
        with rasterio.open(src_dir[i] / f'RH{rh}_Q{i}.tif') as s:
            if s.transform != ref.transform or s.shape != ref.shape or s.crs != ref.crs:
                raise ValueError(f'{s.name}: grid differs from {ref.name}')
            q.append(s.read(1))
    return calibrate(*q, biome, c_lo_rh, c_hi_rh, calibrated)


def _apply_tile(tile: str, roots: dict, products_root: Path, save_dir: Path, c_lo, c_hi,
                calibrated, rh_idxs, tags: dict):
    """-> info string; writes <save_dir>/<tile>/RH{k}_Q{0,2}.tif, skipping RHs already there."""
    src_dir = {q: products_root / roots[q] / tile for q in (0, 1, 2)}
    out_dir = save_dir / tile
    out_dir.mkdir(parents=True, exist_ok=True)
    for tmp in out_dir.glob('.*.tif.tmp'):     # left by a killed run
        tmp.unlink()

    with _retry_io(rasterio.open, src_dir[1] / f'RH{rh_idxs[0]}_Q1.tif') as ref:
        biome = biome_raster(ref)
        counts = np.bincount(biome.ravel(), minlength=256)
        n_written = 0
        for rh in rh_idxs:
            dst0, dst2 = out_dir / f'RH{rh}_Q0.tif', out_dir / f'RH{rh}_Q2.tif'
            if dst0.exists() and dst2.exists():
                continue
            out0, out2 = _retry_io(_calibrate_rh, src_dir, rh, ref, biome, c_lo[rh], c_hi[rh],
                                   calibrated)
            _write_cog(out0, ref, dst0, f'RH{rh}_Q0', tags)
            _write_cog(out2, ref, dst2, f'RH{rh}_Q2', tags)
            n_written += 1

    n_cal = int(counts[calibrated].sum())
    n_none = int(counts[0])
    n_other = int(biome.size - n_cal - n_none)
    return (f'{n_written} RHs written; pixels: {n_cal / biome.size:.3f} calibrated biome, '
            f'{n_other / biome.size:.3f} uncalibrated biome '
            f'({",".join(str(b) for b in np.flatnonzero(counts) if b and not calibrated[b])}), '
            f'{n_none / biome.size:.3f} no biome')


def _job_alive(job_id: str) -> bool:
    """True if the SLURM job is queued/running, or if squeue cannot tell (safe side)."""
    try:
        out = subprocess.run(['squeue', '-h', '-j', job_id, '-o', '%i'], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return True
    if out.returncode != 0:
        return 'Invalid job id' not in out.stderr
    return bool(out.stdout.strip())


def claim_chunk(claims_dir: Path, chunk: int, n_chunks: int, job_id: str) -> bool:
    """Atomically claim a chunk for this job; False if a live job holds it.

    The claim is a directory (mkdir is atomic on NFS) holding the owner's job id.
    A claim whose job is no longer in squeue is stale and taken over, so crashed
    jobs and later redo passes need no manual cleanup.
    """
    claims_dir.mkdir(parents=True, exist_ok=True)
    claim = claims_dir / f'c{chunk}_of_{n_chunks}'
    for _ in range(3):
        try:
            claim.mkdir()
            (claim / 'job_id').write_text(job_id)
            return True
        except FileExistsError:
            try:
                owner = (claim / 'job_id').read_text().strip()
            except FileNotFoundError:
                time.sleep(5)  # being created right now
                continue
            if owner == job_id or _job_alive(owner):
                return owner == job_id
            try:  # stale: only one process wins the rename
                claim.rename(claim.with_name(f'{claim.name}.stale.{owner}.{os.getpid()}'))
            except OSError:
                pass
    return False


def apply_cp_offsets(year: int, manifest: str, products_root: str, cp_results: str,
                     biome_file: str, save_dir: str, method: str = 'SE-CQR',
                     exclude_biomes: list[int] = (11,), rh_idxs: list[int] = tuple(range(101)),
                     state_dir: str = None, workers: int = 8,
                     tiles_file: str = None, chunk: int = 1, n_chunks: int = 1, **kwargs):
    """Calibrated Q0/Q2 for every tile of the release manifest (or tiles_file).

    Resumable: tiles done in the state are skipped, and inside a tile RHs whose
    two files exist. `chunk` of `n_chunks` (1-based) takes every n_chunks-th
    tile, for job arrays.
    """
    products_root = Path(products_root).expanduser()
    save_dir = Path(save_dir).expanduser()
    rh_idxs = list(rh_idxs)
    state = TaskState(state_dir or f'{save_dir}_state')

    man = pd.read_csv(Path(manifest).expanduser(), sep='\t', dtype=str).set_index('tile_id')
    if (man['file_suffix'] != '.tif').any():
        raise ValueError(f'{manifest}: only .tif suffixes are supported')
    tiles = (Path(tiles_file).expanduser().read_text().split() if tiles_file
             else sorted(man.index))
    tiles = tiles[chunk - 1::n_chunks]
    job_id = os.environ.get('SLURM_JOB_ID')
    if n_chunks > 1 and job_id and not claim_chunk(state.path / 'claims', chunk, n_chunks, job_id):
        print(f'chunk {chunk}/{n_chunks} already claimed by a running job, skipping')
        return
    c_lo, c_hi, calibrated = load_offsets(cp_results, method, exclude_biomes, rh_idxs)
    tags = dict(CONFORMAL_METHOD=method, CONFORMAL_OFFSETS=str(Path(cp_results).expanduser()),
                CONFORMAL_UNCALIBRATED='raw quantile kept for Tundra, WWF 98/99, '
                                       'no WWF polygon and nodata')

    done = state.done()
    todo = [t for t in tiles if t not in done]
    print(f'{len(tiles)} tiles in chunk {chunk}/{n_chunks}, {len(todo)} to do, '
          f'{int(calibrated.sum())} calibrated biomes, {method}')

    n_failed = 0
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_worker, initargs=(biome_file,)) as pool:
        futs = {pool.submit(_apply_tile, t,
                            {q: man.at[t, f'q{q}_root'] for q in (0, 1, 2)},
                            products_root, save_dir, c_lo, c_hi, calibrated, rh_idxs,
                            tags): t for t in todo}
        for i, f in enumerate(as_completed(futs), 1):
            tile = futs[f]
            try:
                info = f.result()
                state.mark_done(tile, info)
            except Exception as e:
                n_failed += 1
                state.mark_failed(tile, repr(e))
                info = f'FAILED {e!r}'
            print(f'[{i}/{len(todo)}] {tile} {info}', flush=True)
    print(f'{len(todo) - n_failed} tiles done, {n_failed} failed')


def check_cp_tiles(year: int, tiles: list[str], manifest: str, products_root: str,
                   cp_results: str, biome_file: str, save_dir: str, test_data: str,
                   method: str = 'SE-CQR', exclude_biomes: list[int] = (11,),
                   rh_idxs: list[int] = (10, 25, 50, 75, 95, 98), save_file: str = None,
                   **kwargs):
    """Compare written tiles with the test points that fall in them.

    Per tile and RH: how often the raster's biome equals the point's BIOME, how
    often the written Q0/Q2 equal the offsets applied to the point's sampled
    quantiles, and raw vs calibrated coverage of the point's GEDI rh. Points
    in the overlap of two S2 tiles may have been sampled from the neighbour,
    so a few mismatches there are expected.
    """
    products_root = Path(products_root).expanduser()
    save_dir = Path(save_dir).expanduser()
    rh_idxs = list(rh_idxs)
    _init_worker(biome_file)
    man = pd.read_csv(Path(manifest).expanduser(), sep='\t', dtype=str).set_index('tile_id')
    c_lo, c_hi, calibrated = load_offsets(cp_results, method, exclude_biomes, rh_idxs)

    cols = ['lon', 'lat', 'BIOME'] + [c for k in rh_idxs
                                       for c in (f'rh{k}', f'RH{k}_Q0', f'RH{k}_Q1', f'RH{k}_Q2')]
    pts = pd.read_parquet(Path(test_data).expanduser(), columns=cols)

    rows = []
    for tile in tiles:
        ref_path = products_root / man.at[tile, 'q1_root'] / tile / f'RH{rh_idxs[0]}_Q1.tif'
        with rasterio.open(ref_path) as ref:
            tr = Transformer.from_crs(4326, ref.crs, always_xy=True)
            xs, ys = tr.transform(pts['lon'].values, pts['lat'].values)
            r, c = rasterio.transform.rowcol(ref.transform, xs, ys)
            r, c = np.asarray(r), np.asarray(c)
            inside = (r >= 0) & (r < ref.height) & (c >= 0) & (c < ref.width)
            p, r, c = pts[inside], r[inside], c[inside]
            if len(p) == 0:
                print(f'{tile}: no test points')
                continue
            biome = biome_raster(ref)[r, c]
        pt_biome = p['BIOME'].fillna(0).astype(int).values
        for k in rh_idxs:
            raw = {q: np.rint(p[f'RH{k}_Q{q}'].values * 10) for q in (0, 1, 2)}
            ok = ~np.isnan(raw[0]) & ~np.isnan(raw[1]) & ~np.isnan(raw[2]) & ~np.isnan(p[f'rh{k}'].values)
            q = {q: np.where(ok, raw[q], 0).astype(np.int16) for q in (0, 1, 2)}
            exp0, exp2 = calibrate(q[0], q[1], q[2], pt_biome, c_lo[k], c_hi[k], calibrated)
            got = {}
            for qi in (0, 2):
                with rasterio.open(save_dir / tile / f'RH{k}_Q{qi}.tif') as s:
                    got[qi] = np.array([v[0] for v in s.sample(zip(*rasterio.transform.xy(
                        s.transform, r, c)))])
            y = np.rint(p[f'rh{k}'].values * 10)      # dm, as src-cp's preprocessing
            lo = np.minimum(np.minimum(raw[0], raw[1]), raw[2])     # sorted, as in eval_cp
            hi = np.maximum(np.maximum(raw[0], raw[1]), raw[2])
            m = ok
            rows.append(dict(
                tile=tile, rh=k, n_points=int(m.sum()),
                biome_match=float((biome[m] == pt_biome[m]).mean()),
                q0_match=float((got[0][m] == exp0[m]).mean()),
                q2_match=float((got[2][m] == exp2[m]).mean()),
                coverage_raw=float(((lo[m] <= y[m]) & (y[m] <= hi[m])).mean()),
                coverage_cal=float(((got[2][m] <= y[m]) & (y[m] <= got[0][m])).mean()),
                frac_calibrated=float(calibrated[pt_biome[m]].mean())))
    df = pd.DataFrame(rows)
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(df.round(3).to_string(index=False))
    if save_file:
        df.to_csv(Path(save_file).expanduser(), sep='\t', index=False)
    return df


def _uploaded_per_tile(upload_state: str) -> dict:
    """tile -> number of its files uploaded, from sc_upload's state tsv (run.state_format=tsv).

    Rows are key / status / time / info with keys '<TILE>/RH{k}_Q{q}.tif'; the
    last row of a key wins, so a retry that succeeded counts as done.
    """
    path = Path(upload_state).expanduser()
    status = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 2 and parts[0] != 'key':
                    status[parts[0]] = parts[1]
    counts = {}
    for key, st in status.items():
        if st == 'done':
            tile = key.split('/', 1)[0]
            counts[tile] = counts.get(tile, 0) + 1
    return counts


def cp_upload_tiles(save_dir: str, upload_state: str, out_file: str, state_dir: str = None,
                    n_files: int = 202, **kwargs):
    """Tiles that apply_cp_offsets finished and sc_upload has not fully uploaded -> out_file.

    The tile whitelist for `sc_upload run.src.kind=local`, so a run only lists
    the tiles still to send instead of the whole calibrated tree.
    """
    save_dir = Path(save_dir).expanduser()
    done = TaskState(state_dir or f'{save_dir}_state').done()
    uploaded = _uploaded_per_tile(upload_state)
    todo = sorted(t for t in done if uploaded.get(t, 0) < n_files)
    out = Path(out_file).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(f'{t}\n' for t in todo))
    print(f'{len(done)} tiles calibrated, {sum(v >= n_files for v in uploaded.values())} fully '
          f'uploaded, {len(todo)} to upload -> {out}')
    return len(todo)


def prune_uploaded_tiles(save_dir: str, upload_state: str, n_files: int = 202,
                         dry_run: bool = True, **kwargs):
    """Delete local calibrated tiles whose n_files files are all done in sc_upload's state tsv.

    sc_upload marks a key done only after the byte count it streamed matched
    the local file, so a complete tile there is a complete copy on Source Coop.
    """
    import shutil
    save_dir = Path(save_dir).expanduser()
    uploaded = _uploaded_per_tile(upload_state)
    tiles = sorted(t for t, n in uploaded.items() if n >= n_files and (save_dir / t).is_dir())
    n_bytes, n_tiles = 0, 0
    for t in tiles:
        files = list((save_dir / t).glob('*.tif'))
        if len(files) != n_files:      # rewritten or partial since the upload: keep
            print(f'{t}: {len(files)} local files, expected {n_files}, kept')
            continue
        n_bytes += sum(f.stat().st_size for f in files)
        n_tiles += 1
        if not dry_run:
            shutil.rmtree(save_dir / t)
    print(f'{"would delete" if dry_run else "deleted"} {n_tiles} tiles, {n_bytes / 1e12:.2f} TB'
          f'{" (dry run: run.dry_run=false to delete)" if dry_run else ""}')
    return n_tiles
