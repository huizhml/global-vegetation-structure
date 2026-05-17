"""
Pipeline step 6 — naturalness normalisation statistics.

Computes per-channel mean/std for the naturalness model directly from the
naturalness data file itself (the same ``rhs_predictions_*.h5`` / ``.zarr`` that
``datasets.h5_dataset.NaturalnessDataModule`` consumes), so the stats always
match exactly what the model sees:

  - ``vsm_median``  (loc, 101, 15, 15) int16  -> ``mean`` / ``std``      (101,)
  - ``s2``          (loc,  12, 15, 15) int16  -> ``mean_s2`` / ``std_s2`` (12,)

Stats are computed on the *raw stored values* (no /10 rescale) because
``NaturalnessDataset.__getitem__`` returns them raw; the model normalises raw
inputs with these stats. ``VSM_NODATA`` sentinels (border / masked pixels) are
excluded so they don't skew the distribution.

Output is a single ``.npz`` (one ``mean_std_fp`` for the consumer, atomic,
versioned together) with extra ``provenance_*`` keys recording what it was
computed from. Idempotent: skips if the output exists unless ``force=True``.

Run like the other pipeline steps, e.g.::

    python -m preprocessing.pipeline.step6_calculate_naturalness_stats \
        data_file=~/data/gvs/downstream_task_data/rhs_predictions_2017_izraz2av.h5 \
        out_fp=~/data/gvs/downstream_task_data/naturalness/mean_std.npz
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm

from const import VSM_NODATA
# Single source of truth for the diversity formula (same fn the model uses).
from datasets.h5_dataset import diversity_indices_torch, DIVERSITY_NAMES

N_RH = 101
N_S2 = 12


class _ChannelStats:
    """Streaming per-channel mean/std over a (loc, C, H, W) array.

    Accumulates float64 sum / sum-of-squares / valid-count per channel so the
    full array never has to be in memory at once. Pixels equal to ``nodata``
    are excluded from every channel independently.
    """

    def __init__(self, n_channels: int, nodata: int):
        self.n_channels = n_channels
        self.nodata = nodata
        self.s = np.zeros(n_channels, dtype=np.float64)
        self.ss = np.zeros(n_channels, dtype=np.float64)
        self.cnt = np.zeros(n_channels, dtype=np.int64)

    def update(self, chunk: np.ndarray) -> None:
        # chunk: (b, C, H, W). Reduce over everything except the channel axis.
        valid = chunk != self.nodata
        x = np.where(valid, chunk, 0).astype(np.float64)  # invalid -> 0 (no contribution)
        axes = (0, 2, 3)
        self.s += x.sum(axis=axes)
        self.ss += (x * x).sum(axis=axes)
        self.cnt += valid.sum(axis=axes)

    def finalize(self):
        if (self.cnt == 0).any():
            empty = np.where(self.cnt == 0)[0].tolist()
            raise ValueError(
                f'channels {empty} have no valid (non-nodata) pixels; '
                f'cannot compute stats')
        mean = self.s / self.cnt
        var = self.ss / self.cnt - mean ** 2
        std = np.sqrt(np.clip(var, 0.0, None))  # population std; clip rounding < 0
        return mean.astype(np.float32), std.astype(np.float32), self.cnt.copy()


class _NanChannelStats(_ChannelStats):
    """_ChannelStats variant that excludes NaN/inf instead of a nodata
    sentinel -- used for the derived diversity indices, which are NaN on
    invalid pixels by construction.
    """

    def __init__(self, n_channels: int):
        super().__init__(n_channels, nodata=None)

    def update(self, chunk: np.ndarray) -> None:
        valid = np.isfinite(chunk)
        x = np.where(valid, chunk, 0.0).astype(np.float64)
        axes = (0, 2, 3)
        self.s += x.sum(axis=axes)
        self.ss += (x * x).sum(axis=axes)
        self.cnt += valid.sum(axis=axes)


def _open(data_file: Path):
    """Open the naturalness data file (.zarr or .h5), mirroring h5_dataset.

    Returns the store/accessor. For h5py this object is also the closeable
    handle; for a zarr v3 Group there is nothing to close. The caller closes
    it iff it exposes ``.close()`` (see calculate_naturalness_mean_std).
    """
    if data_file.suffix == '.zarr':
        import zarr
        return zarr.open(str(data_file), mode='r')
    import h5py
    return h5py.File(data_file, 'r')


def calculate_naturalness_mean_std(data_file: str, out_fp: str,
                                   batch_size: int = 4096, force: bool = False, **kwargs):
    data_file = Path(data_file).expanduser()
    out_fp = Path(out_fp).expanduser()

    if out_fp.exists() and not force:
        print(f'{out_fp} already exists; pass force=true to regenerate. Skipping.')
        return
    if not data_file.exists():
        raise FileNotFoundError(f'data_file does not exist: {data_file}')

    out_fp.parent.mkdir(parents=True, exist_ok=True)

    store = _open(data_file)
    try:
        n_loc = store['vsm_median'].shape[0]
        if store['vsm_median'].shape[1] != N_RH:
            raise ValueError(
                f"expected vsm_median with {N_RH} RH bands, got "
                f"{store['vsm_median'].shape[1]}")
        if store['s2'].shape[1] != N_S2:
            raise ValueError(
                f"expected s2 with {N_S2} bands, got {store['s2'].shape[1]}")

        rh_stats = _ChannelStats(N_RH, VSM_NODATA)
        s2_stats = _ChannelStats(N_S2, VSM_NODATA)
        div_stats = _NanChannelStats(len(DIVERSITY_NAMES))

        for i in tqdm(range(0, n_loc, batch_size), desc='naturalness stats'):
            j = min(i + batch_size, n_loc)
            vsm_chunk = store['vsm_median'][i:j]   # (b, 101, H, W) int16
            rh_stats.update(vsm_chunk)
            s2_stats.update(store['s2'][i:j])
            # Diversity from the SAME chunk via the model's exact function
            # (CPU torch here); NaN on invalid pixels -> excluded by _Nan stats.
            div = diversity_indices_torch(
                torch.from_numpy(np.ascontiguousarray(vsm_chunk))).numpy()
            div_stats.update(div)
    finally:
        close = getattr(store, 'close', None)
        if callable(close):
            close()

    mean, std, rh_cnt = rh_stats.finalize()
    mean_s2, std_s2, s2_cnt = s2_stats.finalize()
    mean_div, std_div, div_cnt = div_stats.finalize()

    np.savez(
        out_fp,
        mean=mean, std=std,              # (101,) RH, raw stored units
        mean_s2=mean_s2, std_s2=std_s2,  # (12,) S2, raw stored units
        mean_div=mean_div, std_div=std_div,  # (4,) diversity, DIVERSITY_NAMES order
        # --- provenance (ignored by consumers, which read only the arrays above) ---
        provenance_source=np.array(str(data_file)),
        provenance_created=np.array(datetime.now(timezone.utc).isoformat()),
        provenance_n_loc=np.array(n_loc),
        provenance_vsm_nodata=np.array(VSM_NODATA),
        provenance_div_names=np.array(list(DIVERSITY_NAMES)),
        provenance_rh_valid_counts=rh_cnt,
        provenance_s2_valid_counts=s2_cnt,
        provenance_div_valid_counts=div_cnt,
    )
    print(f'Wrote {out_fp}')
    print(f'  source      : {data_file} ({n_loc} loc)')
    print(f'  RH   mean[:3]: {mean[:3]}  std[:3]: {std[:3]}')
    print(f'  S2   mean[:3]: {mean_s2[:3]}  std[:3]: {std_s2[:3]}')
    print(f'  DIV  {list(DIVERSITY_NAMES)} mean: {mean_div}  std: {std_div}')


@dataclass
class Config:
    # Defaults match config/train_naturalness.yaml; override via hydra, e.g.
    #   data_file=... out_fp=... force=true
    data_file: str = '~/data/gvs/downstream_task_data/rhs_predictions_2017_izraz2av.h5'
    out_fp: str = '~/data/gvs/downstream_task_data/naturalness/mean_std.npz'
    batch_size: int = 4096
    force: bool = False


cs = ConfigStore.instance()
cs.store(name='config', node=Config)


@hydra.main(config_name='config', version_base='1.2')
def main(cfg: DictConfig):
    calculate_naturalness_mean_std(
        data_file=cfg.data_file,
        out_fp=cfg.out_fp,
        batch_size=cfg.batch_size,
        force=cfg.force,
    )


if __name__ == '__main__':
    main()
