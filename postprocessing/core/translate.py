import hydra
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import time
import dask

def _translate_one(src_path, dst_path, profile="ZSTD", profile_options={}, **options):
    """Convert image to COG."""
    # Format creation option (see gdalwarp `-co` option)
    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(
        BIGTIFF="IF_SAFER",
        ZSTD_LEVEL=1,
        PREDICTOR=2,
        BLOCKYSIZE=1024,
        BLOCKXSIZE=1024,
        MAX_Z_ERROR=0
    ))
    output_profile.update(profile_options)

    # Dataset Open option (see gdalwarp `-oo` option)
    config = dict(
        GDAL_NUM_THREADS=2,
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )

    cog_translate(
        src_path,
        dst_path,
        output_profile,
        config=config,
        in_memory=False,
        quiet=True,
        **options,
    )
    return True


# Kept as the delayed form so translate_tile below is unchanged in behaviour.
_translate = dask.delayed(_translate_one)


def translate_tiles(worklist, src_root, dst_root, profile="LERC_ZSTD",
                    pattern="RH*_Q*.tif", chunk=None, n_chunks=None,
                    workers=None, skip_complete=True, **kwargs):
    """Convert many tiles inside ONE process, one thread pool across all files.

    The per-tile entrypoint (translate_tile, driven by run_translate) pays a
    fresh python + hydra + GDAL start per tile, and barriers at every tile
    boundary so the pool drains to a handful of stragglers 567 times per chunk.
    Here the pool is fed continuously from a flat (src, dst) stream, so a job
    holding N cpus keeps N conversions in flight until the chunk is done.

    Args:
        worklist: text file of tile ids, one per line (see tools/stac_cog_audit).
        src_root: dir holding <tile>/RH*_Q*.tif to read.
        dst_root: dir to write <tile>/RH*_Q*.tif into.
        profile: rio-cogeo profile. LERC_ZSTD matches original/tiles/cog.
        pattern: which assets to convert, e.g. RH*_Q1.tif for the median
            quantile only. Completeness is judged against the same pattern, so
            a Q1-only pass is not considered "done" by a later full pass.
        chunk / n_chunks: 1-based slice of the worklist this task owns. Left
            null they are derived from SLURM, so `srun --ntasks=N` (optionally
            inside a job array) partitions the worklist with no bookkeeping in
            the shell:
                n_chunks = SLURM_ARRAY_TASK_COUNT * SLURM_NTASKS
                chunk    = (SLURM_ARRAY_TASK_ID-1) * SLURM_NTASKS + PROCID + 1
            Outside SLURM this is 1/1, i.e. the whole worklist.
        workers: thread count. Defaults to SLURM_CPUS_PER_TASK, else cpu_count.
        skip_complete: skip tiles whose dst already has >= as many tifs as src,
            so a re-run after a timeout costs a directory listing per tile.

    Returns:
        Counts of tiles/files converted, skipped and failed.
    """
    src_root, dst_root = Path(src_root).expanduser(), Path(dst_root).expanduser()
    tiles = [t.strip() for t in Path(worklist).expanduser().read_text().split("\n")
             if t.strip()]
    if not tiles:
        raise RuntimeError(f"empty worklist: {worklist}")

    if chunk is None or n_chunks is None:
        ntasks = int(os.environ.get("SLURM_NTASKS", 1))
        procid = int(os.environ.get("SLURM_PROCID", 0))
        array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        array_n = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
        chunk = (array_id - 1) * ntasks + procid + 1
        n_chunks = array_n * ntasks
    if not 1 <= chunk <= n_chunks:
        raise ValueError(f"chunk {chunk} outside 1..{n_chunks}")

    total = len(tiles)
    mine = tiles[total * (chunk - 1) // n_chunks: total * chunk // n_chunks]
    workers = int(workers or os.environ.get("SLURM_CPUS_PER_TASK", 0)
                  or os.cpu_count())
    print(f"chunk {chunk}/{n_chunks} on {os.uname().nodename.split('.')[0]}: "
          f"{len(mine)} of {total} tiles, {workers} workers, "
          f"profile={profile}, pattern={pattern}", flush=True)

    jobs, skipped, missing = [], 0, []
    for tile in mine:
        src_dir, dst_dir = src_root / tile, dst_root / tile
        if not src_dir.is_dir():
            missing.append(tile)
            continue
        srcs = sorted(src_dir.glob(pattern))
        if skip_complete and dst_dir.is_dir() and \
                len(list(dst_dir.glob(pattern))) >= len(srcs) and srcs:
            skipped += 1
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        jobs.extend((s, dst_dir / s.name) for s in srcs)

    print(f"  {len(jobs)} files to convert, {skipped} tiles already complete, "
          f"{len(missing)} source dirs missing")
    if missing:
        print(f"  [warn] missing source dirs: {missing[:5]}"
              f"{' ...' if len(missing) > 5 else ''}")

    done, failed = 0, []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_translate_one, s, d, profile): (s, d)
                   for s, d in jobs}
        for future in as_completed(futures):
            src_path, _ = futures[future]
            try:
                future.result()
            except Exception as exc:                      # noqa: BLE001
                # One unreadable source must not strand the rest of the chunk;
                # the tile stays off the .done side and a re-run retries it.
                failed.append((str(src_path), f"{type(exc).__name__}: {exc}"))
            done += 1
            # Tagged with the chunk id and paced at 100: dozens of tasks share
            # one log file, and at a few hundred files/hour a 500-file interval
            # leaves each chunk silent for an hour, which reads as a hang.
            if done % 100 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9) / 3600
                print(f"  [chunk {chunk}/{n_chunks}] {done}/{len(jobs)} files  "
                      f"{rate*3600:.0f}/h  eta {eta:.1f}h  "
                      f"{len(failed)} failed  last={src_path.parent.name}",
                      flush=True)

    if failed:
        print(f"  [error] {len(failed)} files failed, first 5:")
        for path, err in failed[:5]:
            print(f"    {path}: {err}")

    return {"tiles": len(mine), "tiles_skipped": skipped,
            "tiles_missing_src": len(missing), "files": len(jobs),
            "files_failed": len(failed), "elapsed_sec": time.time() - t0}


def translate_tile(src_dir, dst_dir, profile="ZSTD", **kwargs):
    src_dir = Path(src_dir).expanduser()
    dst_dir = Path(dst_dir).expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)
    tiff_files = src_dir.glob('RH*_Q*.tif')
    futures = []
    for tiff_file in tiff_files:
        dst_path = dst_dir / tiff_file.name #.replace('_uncompressed.tif', '.cog.tif')
        futures.append(_translate(tiff_file, dst_path, profile))
    # dask defaults to os.cpu_count() threads, which is the NODE's core count,
    # not what the SLURM cgroup allows -- a 64-core node with
    # --cpus-per-task=16 would oversubscribe 4x.
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None
    results = dask.compute(*futures, scheduler="threads", num_workers=num_workers)
    if all(results):
        print(f"Successfully translated {len(results)} files")
    else:
        print(f"Failed to translate {len(results)} files")
        

@dataclass
class TranslateConfig:
    src_dir: str='~/data/gvs/deploy/predictions_2020/09WWQ_fp32_infer'
    dst_dir: str='~/data/gvs/deploy/predictions_2020/09WWQ_fp32_infer_cog'
    profile: str = "LERC_ZSTD"
    
cs = ConfigStore.instance()
cs.store(name="config", node=TranslateConfig)
    
@hydra.main(config_path=None, config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    
    t0 = time.time()
    translate_tile(cfg.src_dir, cfg.dst_dir, cfg.profile)
    print(f"Time taken: {time.time() - t0} seconds")


if __name__ == "__main__":
    main() 