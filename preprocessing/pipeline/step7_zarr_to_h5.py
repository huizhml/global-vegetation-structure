"""
Pipeline step 7 — convert a chunk-by-1 zarr store to a single HDF5 file.

Why: ``init_out_zarr`` writes ``chunks=(1, ...)`` with zarr's default
LocalStore = ONE small file per chunk -> ~10^6 tiny files for the naturalness
data. On a network filesystem (Lustre/NFS/GPFS) the per-file open/stat
metadata cost dominates and starves the (tiny) model's GPU. HDF5 packs
everything into a single file with an internal chunk index, so the same data
loads orders of magnitude faster (this is why "switching to h5 just works").
One-time, offline conversion.

Only the arrays the naturalness pipeline needs are copied by default
(``vsm_median``, ``s2``, ``rowid`` -- used by NaturalnessDataset and step6);
set ``keys=null`` (hydra) to copy every array. Output is CONTIGUOUS +
uncompressed: the consumer reads it once sequentially, so contiguous = maximal
throughput with zero per-chunk overhead.

Idempotent: skips if the output exists unless ``force=True``. Writes to a
``.tmp`` then atomically renames, so an interrupted run can't leave a
half-written .h5 that the skip check would later trust.

    python -m preprocessing.pipeline.step7_zarr_to_h5 \
        data_file=~/.../vsm_patches....zarr \
        out_fp=~/.../vsm_patches....h5
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import h5py
import hydra
import zarr
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm

# Arrays consumed by the naturalness pipeline (NaturalnessDataset + step6).
DEFAULT_KEYS = ('vsm_median', 's2', 'rowid')


def convert_zarr_to_h5(data_file: str, out_fp: str, keys=DEFAULT_KEYS,
                       block: int = 4096, force: bool = False, **kwargs):
    data_file = Path(data_file).expanduser()
    out_fp = Path(out_fp).expanduser()

    if out_fp.exists() and not force:
        print(f'{out_fp} already exists; pass force=true to regenerate. Skipping.')
        return
    if data_file.suffix != '.zarr':
        raise ValueError(f'expected a .zarr input, got {data_file}')
    if not data_file.exists():
        raise FileNotFoundError(f'data_file does not exist: {data_file}')
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open(str(data_file), mode='r')
    all_keys = list(getattr(store, 'array_keys', store.keys)())
    if keys is None or keys == 'all':
        keys = all_keys
    else:
        keys = list(keys)
        missing = [k for k in keys if k not in all_keys]
        if missing:
            raise KeyError(f'{missing} not in {data_file} (have {all_keys})')

    tmp_fp = out_fp.with_suffix(out_fp.suffix + '.tmp')
    with h5py.File(tmp_fp, 'w') as f:
        for k in keys:
            z = store[k]
            n = z.shape[0]
            # contiguous + uncompressed (no chunks/compression args)
            dset = f.create_dataset(k, shape=z.shape, dtype=z.dtype)
            for a in tqdm(range(0, n, block), desc=f'{data_file.name} -> {k}'):
                b = min(a + block, n)
                dset[a:b] = z[a:b]   # sequential block read -> sequential write
    tmp_fp.replace(out_fp)
    print(f'Wrote {out_fp}')
    print(f'  from   : {data_file}')
    print(f'  arrays : {list(keys)}')


@dataclass
class Config:
    # Defaults point at the naturalness vsm patches store; override via hydra.
    # keys=null copies every array (default copies only what naturalness needs).
    data_file: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2017/vsm_patches_ps15_single_h5_72xl3wma_ps31.zarr'
    out_fp: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2017/vsm_patches_ps15_single_h5_72xl3wma_ps31.h5'
    keys: Optional[List[str]] = field(default_factory=lambda: list(DEFAULT_KEYS))
    block: int = 4096
    force: bool = False


cs = ConfigStore.instance()
cs.store(name='config', node=Config)


@hydra.main(config_name='config', version_base='1.2')
def main(cfg: DictConfig):
    convert_zarr_to_h5(
        data_file=cfg.data_file,
        out_fp=cfg.out_fp,
        keys=cfg.keys,
        block=cfg.block,
        force=cfg.force,
    )


if __name__ == '__main__':
    main()
