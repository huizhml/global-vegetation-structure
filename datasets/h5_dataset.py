import time
import os
from typing import Union
from pathlib import Path
import torch.utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch import Tensor
from kornia.enhance import normalize
import xarray as xr
import pandas as pd
from lightning.pytorch import LightningDataModule

from download.core.utils import get_epsg_from_tile, get_dense_latlon
from const import VSM_NODATA, SCL_EXCLUDE_LABELS, SCL_WATER, ESA_BUILT_UP, ESA_WATER, ESA_SNOW, KEY_RHS_EVAL, MAX_HEIGHT_METERS


import zarr
from zarr.codecs import BloscCodec, BloscShuffle

# Named RH-band selectors. 'key_rhs' is the project-wide set from const
# (KEY_RHS); presets keep YAML/hydra configs free of literal index lists,
# while an explicit list is still accepted for ad-hoc experimentation.
RH_PRESETS = {
    'rh98': np.array([98]),
    'key_rhs': np.asarray(KEY_RHS_EVAL),
    'full_profile': np.arange(101),
}


def resolve_rh_idxs(rh_idxs='rh98') -> np.ndarray:
    '''Normalise the RH-band selector to a 1-D int array.

    rh_idxs may be a preset name ('rh98', 'key_rhs', 'full_profile') or an
    explicit list/tuple/array of band indices. Idempotent, so an
    already-resolved array can safely be passed back through it.
    '''
    if isinstance(rh_idxs, str):
        try:
            return RH_PRESETS[rh_idxs]
        except KeyError:
            raise ValueError(
                f'Unknown rh_idxs preset {rh_idxs!r}; choose from '
                f'{sorted(RH_PRESETS)} or pass an explicit list of ints')
    return np.asarray(rh_idxs)


# Diversity indices derived from the full 101-band RH profile, in fixed
# stack order. Selected via the `diversity` knob (None / 'all' / list of names).
DIVERSITY_NAMES = ('fhd', 'enl1d', 'enl2d', 'cr')


def resolve_diversity(diversity) -> np.ndarray:
    '''Normalise the diversity selector to indices into DIVERSITY_NAMES.

    None -> [] (disabled); 'all' -> all four; a name or list of names ->
    their stack positions. Order of the returned indices follows the request.
    '''
    if diversity is None:
        return np.array([], dtype=int)
    if isinstance(diversity, str):
        if diversity == 'all':
            return np.arange(len(DIVERSITY_NAMES))
        diversity = [diversity]
    idx = []
    for name in diversity:
        if name not in DIVERSITY_NAMES:
            raise ValueError(
                f'Unknown diversity index {name!r}; choose from '
                f'{DIVERSITY_NAMES}, "all", or None')
        idx.append(DIVERSITY_NAMES.index(name))
    return np.asarray(idx, dtype=int)


def diversity_indices_torch(vsm, nodata=VSM_NODATA, bin_width: float = 5.0,
                            max_height: float = MAX_HEIGHT_METERS):
    '''Per-pixel FHD / ENL1D / ENL2D / CR from a full RH profile, batched on
    whatever device ``vsm`` is on (designed to run on GPU per training batch).

    Numerically mirrors evaluation.on_diversity_indices.pixel_diversity_indices
    (validated against it): equal-width histogram over [0, max_height] of the
    valid RH values, then Shannon entropy and effective number of layers; CR
    from RH25/RH98. Uniform bins -> a single vectorised scatter_add, no Python
    per-pixel loop (unlike batch_binning's searchsorted), so it is cheap enough
    to recompute every batch.

    Args:
        vsm: (B, 101, H, W) raw stored values (meters * 10), ``nodata`` sentinel.
    Returns:
        (B, 4, H, W) float tensor in DIVERSITY_NAMES order; invalid pixels NaN
        (caller decides how to fill before feeding the backbone).

    NOTE: RH25/RH98 use indices 25/98 (RH0..RH100 -> idx == RH number). The
    legacy _chunk_diversity uses 24/97, which is an off-by-one; this follows
    pixel_diversity_indices (25/98), the semantically correct one.
    '''
    vsm = vsm.float()
    n_bins = int(max_height / bin_width)
    valid = (vsm != nodata) & (vsm > 0)            # finite & >0, mirrors reference
    rh_m = vsm / 10.0                              # raw (m*10) -> meters
    bins = torch.clamp((torch.clamp(rh_m, max=max_height) / bin_width).long(),
                       0, n_bins - 1)             # (B, 101, H, W)
    B, _, H, W = vsm.shape
    hist = torch.zeros(B, n_bins, H, W, device=vsm.device, dtype=torch.float32)
    hist.scatter_add_(1, bins, valid.float())      # valid band count per bin/pixel
    total = hist.sum(dim=1, keepdim=True)          # (B, 1, H, W)
    p = hist / total.clamp(min=1.0)
    logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
    fhd = -(p * logp).sum(dim=1)                   # (B, H, W)
    enl1d = torch.exp(fhd)
    sum_p2 = (p * p).sum(dim=1)
    nan = torch.tensor(float('nan'), device=vsm.device)
    enl2d = torch.where(sum_p2 > 0, 1.0 / sum_p2, nan)
    rh25 = torch.clamp(rh_m[:, 25], min=0.0)
    rh98 = rh_m[:, 98]
    cr = torch.where(rh98 > 0, (rh98 - rh25) / rh98, nan)
    # pixels with no valid band -> all indices NaN
    no_valid = total.squeeze(1) <= 0               # (B, H, W)
    out = torch.stack([fhd, enl1d, enl2d, cr], dim=1)  # (B, 4, H, W)
    out = torch.where(no_valid.unsqueeze(1), nan, out)
    return out


def init_out_zarr(pred_fp: Path = None, length: int = None):
    if not os.path.exists(pred_fp):
        print(f'Creating empty Zarr store {pred_fp}')
        store = zarr.open(str(pred_fp), mode='w', zarr_format=3)
        
        # Zarr v3 expects a bytes->bytes codec, not numcodecs.Blosc.
        compressors = (BloscCodec(cname='zstd', clevel=1, shuffle=BloscShuffle.noshuffle),)
        
        # Chunk along loc in batches of 100 instead of 1
        loc_chunk = 1 #min(100, length)
        
        store.create_dataset('slope',      shape=(length, 15, 15),       chunks=(loc_chunk, 15, 15),       dtype='float32',  fill_value=0)
        store.create_dataset('centroid',   shape=(length, 2),            chunks=(length, 2),               dtype='float32',  fill_value=0)
        store.create_dataset('rowid',      shape=(length,),              chunks=(length,),                 dtype='int32',    fill_value=0)
        store.create_dataset('s2',         shape=(length, 12, 15, 15),   chunks=(loc_chunk, 12, 15, 15),  dtype='int16',    fill_value=VSM_NODATA)
        store.create_dataset('vsm_median', shape=(length, 101, 15, 15),  chunks=(loc_chunk, 101, 15, 15), dtype='int16',    fill_value=VSM_NODATA)
        store.create_dataset('vsm_lower',  shape=(length, 101, 15, 15),  chunks=(loc_chunk, 101, 15, 15), dtype='int16',    fill_value=VSM_NODATA)
        store.create_dataset('vsm_upper',  shape=(length, 101, 15, 15),  chunks=(loc_chunk, 101, 15, 15), dtype='int16',    fill_value=VSM_NODATA)
        
        print(f'Zarr store {pred_fp} created')
        
        
def init_out_h5(pred_fp: Path = None, length: int = None):
    if not os.path.exists(pred_fp):
        # Create empty datasets with xarray
        print(f'Creating empty H5 file {pred_fp}')
        ds = xr.Dataset(
            data_vars={
                'slope': (['loc', 'y', 'x'], np.zeros((length, 15, 15), dtype=np.float32)),
                'centroid': (['loc', 'coord'], np.zeros((length, 2), dtype=np.float32)),
                'rowid': (['loc'], np.zeros(length, dtype=np.int32)),
                's2': (['loc', 'band', 'y', 'x'], np.zeros((length, 12, 15, 15), dtype=np.int16)),
                'vsm_median': (['loc', 'rh', 'y', 'x'], np.zeros((length, 101, 15, 15), dtype=np.int16)),
                'vsm_lower': (['loc', 'rh', 'y', 'x'], np.zeros((length, 101, 15, 15), dtype=np.int16)),
                'vsm_upper': (['loc', 'rh', 'y', 'x'], np.zeros((length, 101, 15, 15), dtype=np.int16)),
            },
            coords={
                'loc': np.arange(length),
                'band': np.arange(12),
                'y': np.arange(15),
                'x': np.arange(15),
                'coord': ['lon', 'lat'],
                'rh': np.arange(101)
            }
        )
        # Save as netCDF file
        comp = {
            'slope': {
                'zlib': False,
                'fletcher32': True,
                'chunksizes': (1, 15, 15)
            },
            'centroid': {
                'zlib': False,
                'fletcher32': True,
                'chunksizes': (1, 2)
            },
            'rowid': {
                'zlib': False,
                'fletcher32': True,
                'chunksizes': (1,)
            },
            's2': {
                'zlib': False,
                'fletcher32': True,
                'chunksizes': (1, 12, 15, 15)
            },
            'vsm_median': {
                'zlib': False,
                'fletcher32': True,
                'chunksizes': (1, 101, 15, 15)
            },
            'vsm_lower': {
                'zlib': False,
                'fletcher32': True,
                'chunksizes': (1, 101, 15, 15)
            },
            'vsm_upper': {
                'zlib': False,
                'fletcher32': True,
                'chunksizes': (1, 101, 15, 15)
            }
        }
        ds.to_netcdf(pred_fp, mode='w', format='NETCDF4', engine='h5netcdf', encoding=comp)
        print(f'H5 file {pred_fp} created')
        
def write_patches_zarr(pred_fp: Path = None, vsm_median: torch.Tensor = None, vsm_lower: torch.Tensor = None, vsm_upper: torch.Tensor = None, image: torch.Tensor = None, coords: torch.Tensor = None, rowid: torch.Tensor = None, batch_size: int = 1, batch_idx: int = 0):
    store = zarr.open(str(pred_fp), mode='r+')
    
    real_batch_size = vsm_median.shape[0]
    idx_start = batch_idx * batch_size
    idx_end = idx_start + real_batch_size
    
    store['vsm_median'][idx_start:idx_end] = vsm_median
    store['vsm_lower'][idx_start:idx_end]  = vsm_lower
    store['vsm_upper'][idx_start:idx_end]  = vsm_upper
    store['s2'][idx_start:idx_end]         = image
    store['centroid'][idx_start:idx_end]   = coords.cpu().numpy().astype(np.float32)
    store['rowid'][idx_start:idx_end]      = rowid.cpu().numpy().astype(np.int32)
    
def write_patches_h5(pred_fp: Path = None, vsm_median: torch.Tensor = None, vsm_lower: torch.Tensor = None, vsm_upper: torch.Tensor = None, image: torch.Tensor = None, coords: torch.Tensor = None, rowid: torch.Tensor = None, batch_size: int = 1, batch_idx: int = 0):
    with h5py.File(pred_fp, 'a') as f:
        real_batch_size = vsm_median.shape[0]
        idx_start = batch_idx * batch_size
        idx_end = idx_start + real_batch_size
        f['vsm_median'][idx_start:idx_end] = vsm_median
        f['vsm_lower'][idx_start:idx_end] = vsm_lower
        f['vsm_upper'][idx_start:idx_end] = vsm_upper
        f['s2'][idx_start:idx_end] = image
        f['centroid'][idx_start:idx_end] = coords.cpu().numpy().astype(np.float32)
        f['rowid'][idx_start:idx_end] = rowid.cpu().numpy().astype(np.int32)
        

class SparsePredDataset(Dataset):
    scl_exclude_labels = torch.tensor(SCL_EXCLUDE_LABELS, dtype=torch.uint16)  # Example SCL labels to exclude

    def __init__(self, h5_file:str, mask_with_scl:bool=True, pred_file_path:str=None, batch_size:int=1, patch_size:int=31, out_file_format:str='zarr') -> None:
        super().__init__()
        self.patch_size = patch_size
        self.h5_file = Path(h5_file).expanduser()
        with h5py.File(self.h5_file) as f:
            self.length = f['s2'].shape[0]
        self.mask_with_scl = mask_with_scl
        self.batch_size = batch_size
        if not pred_file_path:
            self.pred_fp = self.h5_file.parent / 'vs_prediction.h5'
        else:
            self.pred_fp = Path(pred_file_path).expanduser()
        self.out_file_format = out_file_format
        if self.out_file_format == 'h5':
            self.dump_data = write_patches_h5
        elif self.out_file_format == 'zarr':
            self.dump_data = write_patches_zarr
        else:
            raise ValueError(f'Unsupported file format: {self.out_file_format}')

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if not hasattr(self, 'data'):
            self.data = h5py.File(self.h5_file)
        s2 = self.data['s2'][index]
        img = s2[:12, :, :]
        scl = s2[12, :, :]
        slope = self.data['slope'][index]
        coords = self.data['centroid'][index]
        rowid = self.data['rowid'][index]
        epsg = self.data['epsg'][index]
        lon_vector, lat_vector = get_dense_latlon(coords, epsg, resolution=10, grid_size=self.patch_size)
        return torch.from_numpy(img), torch.from_numpy(scl), torch.from_numpy(slope), torch.from_numpy(lon_vector), torch.from_numpy(lat_vector), torch.from_numpy(coords), torch.tensor(rowid)

    def __del__(self):
        if hasattr(self, 'data'):
            self.data.close()
    
    
    def init_out_file(self, run_id:str=None):
        if self.out_file_format == 'h5':
            self.pred_fp = self.pred_fp.with_name(self.pred_fp.stem + f'_{run_id}_ps31').with_suffix('.h5')
            init_out_h5(self.pred_fp, self.length)
        elif self.out_file_format == 'zarr':
            self.pred_fp = self.pred_fp.with_name(self.pred_fp.stem + f'_{run_id}_ps31').with_suffix('.zarr')
            init_out_zarr(self.pred_fp, self.length)
        else:
            raise ValueError(f'Unsupported file format: {self.out_file_format}')
    
    
    
    
    def write_patch_predictions(self, prediction, image, scl, coords, rowid, batch_idx):
        """
        Write the prediction for a batch of patches to the output h5 file.
        prediction: torch.Tensor, shape (B, 303, 15, 15) or (B, 101*3, 15, 15)
        scl: torch.Tensor, shape (B, 15, 15)
        coords: torch.Tensor, shape (B, 2)
        rowid: torch.Tensor, shape (B,)
        """
        if prediction.isnan().any():
            raise ValueError('Prediction contains nan')
        esa_wc = prediction[:, -12:, :, :]
        rhs = prediction[:, :303, :, :]
        
        # Mask invalid SCL values if required
        if self.mask_with_scl:
            scl_mask = torch.isin(scl, self.scl_exclude_labels.to(scl.device))
            scl_mask = scl_mask.unsqueeze(1)  # (B, 1, 15, 15)
        else:
            scl_mask = torch.zeros_like(scl, dtype=torch.bool).unsqueeze(1)
        
        scl = scl.unsqueeze(1)
        nodata_mask = scl == 0

        # ESA class prediction
        esa_wc = torch.argmax(esa_wc, dim=1, keepdim=True)

        # Build combined invalid mask, shape (B, 1, H, W)
        invalid_mask = nodata_mask | scl_mask

        water_mask_scl = scl == SCL_WATER
        built_up_mask = esa_wc == ESA_BUILT_UP
        water_mask_esa = esa_wc == ESA_WATER
        esa_snow_mask = esa_wc == ESA_SNOW

        invalid_mask |= water_mask_scl
        invalid_mask |= built_up_mask
        invalid_mask |= water_mask_esa
        invalid_mask |= esa_snow_mask

        # In-place broadcasted masking; avoids repeat(303) allocation
        rhs.masked_fill_(invalid_mask, torch.nan)

        rhs.mul_(10).round_()
        rhs = torch.nan_to_num(rhs, nan=VSM_NODATA)

        border = (rhs.shape[2] - 15) // 2

        if border > 0:
            rhs = rhs[:, :, border:-border, border:-border]
            image = image[:, :, border:-border, border:-border]

        rhs = rhs.reshape(-1, 101, 3, 15, 15)

        # Move to CPU only after all GPU-side work is done
        rhs = rhs.cpu().numpy().astype(np.int16)
        image = image.cpu().numpy().astype(np.int16)
        vsm_median = rhs[:, :, 1, :, :]  # (B, 101, 15, 15)
        vsm_lower = rhs[:, :, 2, :, :]
        vsm_upper = rhs[:, :, 0, :, :]

        # Write to the output h5 file
        t0 = time.time()
        self.dump_data(self.pred_fp, vsm_median, vsm_lower, vsm_upper, image, coords, rowid, self.batch_size, batch_idx)
        print(f'File {self.pred_fp} updated in {time.time() - t0} seconds')
        
class SparsePredDataModule(LightningDataModule):
    def __init__(self, pred_fp: str = None,  prediction_dir: str = None, batch_size: int = 1, num_workers: int = 4, **kwargs):
        super().__init__()
        self.pred_dataset = SparsePredDataset(pred_fp, pred_file_path=prediction_dir, batch_size=batch_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.pred_dataset, batch_size=self.batch_size, num_workers=self.num_workers)



class NaturalnessDataset(Dataset):
    def __init__(self, rhs_fp: Path = None, target_df: pd.DataFrame = None,
                 transform=None, _read_block: int = 8192):
        super().__init__()
        self.rhs_fp = rhs_fp
        self.target_df = target_df
        self.transform = transform
        # The labelled naturalness set is small (~50 KB/sample at 15x15 int16)
        # but the source is a chunk-by-1 compressed zarr -> per-sample random
        # reads starve the (tiny) model's GPU (util ~0%). Preload ONCE into RAM
        # here; __getitem__ is then pure in-memory indexing: no per-sample
        # decompress, no pandas .loc, no per-worker file handles.
        suffix = Path(self.rhs_fp).suffix
        store = (zarr.open(str(self.rhs_fp), mode='r') if suffix == '.zarr'
                 else h5py.File(self.rhs_fp, 'r'))
        try:
            from tqdm import tqdm
            rowids = store['rowid'][:]
            # target_df is a filtered/split SUBSET; keep only file rows whose
            # rowid is in it. mask is over ALL file rows, in file order.
            mask = np.isin(rowids, target_df.index.values)
            n_total = len(rowids)
            n = int(mask.sum())
            vsm_ds, s2_ds = store['vsm_median'], store['s2']
            # Keep int16 (no upcast) to bound RAM; model casts to float later.
            self.vsm = np.empty((n, *vsm_ds.shape[1:]), dtype=vsm_ds.dtype)
            self.s2 = np.empty((n, *s2_ds.shape[1:]), dtype=s2_ds.dtype)
            # CONTIGUOUS block reads (vsm_ds[a:b]) then in-memory boolean subset
            # -- sequential chunk decompression, far faster than scattered fancy
            # indexing of a chunk-by-1 compressed store. Progress is shown so a
            # slow filesystem doesn't look like a hang.
            w = 0
            for a in tqdm(range(0, n_total, _read_block),
                          desc=f'preload {Path(self.rhs_fp).name}'):
                b = min(a + _read_block, n_total)
                m = mask[a:b]
                k = int(m.sum())
                if k == 0:
                    continue
                self.vsm[w:w + k] = vsm_ds[a:b][m]
                self.s2[w:w + k] = s2_ds[a:b][m]
                w += k
            # Targets aligned to the same (file-order) selection; precomputed
            # once. Assumes target_df.index (rowid) is unique.
            self.targets = target_df.loc[rowids[mask], 'class_idx'].to_numpy()
            # Same file-order selection -> aligned with vsm/s2/targets row i.
            # Always emitted as the 4th item so predictions can be joined back
            # to the reference_data_set CSV (lat/lon, biome, ...) by rowid.
            self.rowids = rowids[mask].astype(np.int64)
        finally:
            close = getattr(store, 'close', None)
            if callable(close):
                close()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Pure in-memory; full 101-band profile (RH selection + diversity happen
        # model-side on GPU). numpy -> default_collate -> int16/int64 tensors.
        # rowid is always the 4th item; train/val steps ignore it, test emits it.
        return self.vsm[idx], self.s2[idx], self.targets[idx], self.rowids[idx]
            

class NaturalnessDataModule(LightningDataModule):
    def __init__(self, data_file: str = None, naturalness_fp: str = None, rh_idxs='rh98',
                 use_s2: bool = False, diversity=None,
                 batch_size: int = 1, num_workers: int = 4, train_val_split: float = 0.8,
                 class_balance: bool = False,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Three independent input knobs: rh_idxs (which RH bands), use_s2 (concat
        # Sentinel-2), diversity (which RH-derived indices). The dataset just
        # serves the full 101 profile + s2; selection / diversity / concat all
        # happen model-side on GPU. backbone.in_channels and mean/std are derived
        # from these (auto-wired in run.py before_instantiate_classes).
        self.rh_idxs = resolve_rh_idxs(rh_idxs)
        self.use_s2 = use_s2
        self.diversity = diversity
        # Map original land use IDs to consecutive class indices
        self.land_use_mapping = {
            0: 0,   # No forest
            11: 1,  # Naturally regenerating forest without any signs of human activities, e.g., primary forests.
            20: 2,  # Naturally regenerating forest with signs of human activities, e.g., logging, clear cuts etc.  
            31: 3,  # Planted forest.
            32: 4,  # Short rotation plantations for timber.  
            40: 5,  # Oil palm plantations.  
            53: 6   # Agroforestry. 
        }
        data_file = Path(data_file).expanduser()
        self.data_file = data_file  # exposed so prediction callbacks can write beside it

        if data_file.suffix == '.zarr':
            rowids_with_valid_vsm = zarr.open(data_file, mode='r')['rowid'][:]
        else:
            with h5py.File(data_file) as f:
                rowids_with_valid_vsm = f['rowid'][:]
        
        naturalness_fp = Path(naturalness_fp).expanduser()
        # NOTE: .with_suffix() can't append '_train.csv' (it needs a leading
        # dot and replaces the existing suffix), so build the names explicitly.
        train_csv = naturalness_fp.with_name(f'{naturalness_fp.stem}_train.csv')
        val_csv = naturalness_fp.with_name(f'{naturalness_fp.stem}_val.csv')

        if not (train_csv.exists() and val_csv.exists()):
            # No pre-split files: split the raw CSV once and persist it, so
            # subsequent runs reuse the same fixed split.
            raw = pd.read_csv(naturalness_fp)
            if class_balance:
                # Balanced split: downsample each label to the rarest class,
                # then split per label so both sides stay balanced.
                raw = raw[~raw['Land_use_ID'].isin([-1,1])]
                raw = raw.dropna(subset='Land_use_ID')
                min_count = raw['Land_use_ID'].value_counts().min()
                balanced = raw.groupby('Land_use_ID', group_keys=False).apply(
                    lambda g: g.sample(min_count, random_state=42))
                train_raw = balanced.groupby('Land_use_ID', group_keys=False).apply(
                    lambda g: g.sample(frac=train_val_split, random_state=42))
                val_raw = balanced.drop(train_raw.index)
            else:
                train_raw = raw.sample(frac=train_val_split, random_state=42)
                val_raw = raw.drop(train_raw.index)
            train_raw.to_csv(train_csv, index=False)
            val_raw.to_csv(val_csv, index=False)
            print(f'No pre-split files; wrote {train_csv.name} / {val_csv.name} '
                  f'({len(train_raw)}/{len(val_raw)} rows, frac={train_val_split})')

        # Both paths now load identically: preprocess_target is the single place
        # that filters labels and maps class_idx.
        self.target_df_train = self.preprocess_target(train_csv, rowids_with_valid_vsm)
        self.target_df_val = self.preprocess_target(val_csv, rowids_with_valid_vsm)


        self.train_dataset = NaturalnessDataset(data_file, self.target_df_train)
        self.val_dataset = NaturalnessDataset(data_file, self.target_df_val)
        # Create reverse mapping for reference
        self.id_to_land_use = {v: k for k, v in self.land_use_mapping.items()}

    @property
    def n_rh_channels(self) -> int:
        '''Number of RH bands fed to the model (one input channel each).'''
        return len(self.rh_idxs)

    @property
    def input_channels(self) -> int:
        '''Total backbone in_channels = s2(12 if use_s2) + #RH + #diversity.

        This is the single derivation; run.py computes the same from the
        config and injects it into backbone.init_args.in_channels.
        '''
        return ((12 if self.use_s2 else 0)
                + self.n_rh_channels
                + len(resolve_diversity(self.diversity)))

    def preprocess_target(self, csv_file, valid_rowids):
        '''
        Filter the original expert annoated naturalness labels, following excluded
        1. no valid vsm predictions
        2. unsure labels [-1,1]
        '''
        df = pd.read_csv(csv_file, index_col='rowid')
        # remove unsure labels (-1) and labels with no high resolution images (1)
        df = df.dropna(subset=['Land_use_ID'])
        df = df[~df['Land_use_ID'].isin([1, -1])]
        df['class_idx'] = df['Land_use_ID'].map(self.land_use_mapping)
        # Fill NaN values with 0 (Unknown/No data class)
        df = df.astype({'class_idx': 'int64'})
        df = df.loc[df.index.intersection(valid_rowids)]
        return df


    def _loader_kwargs(self):
        kw = dict(batch_size=self.batch_size, num_workers=self.num_workers,
                  pin_memory=True)
        if self.num_workers > 0:  # invalid when num_workers == 0
            kw.update(persistent_workers=True, prefetch_factor=4)
        return kw

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, drop_last=True,
                          **self._loader_kwargs())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, drop_last=False,
                          **self._loader_kwargs())

    def test_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, drop_last=False,
                          **self._loader_kwargs())



def check_naturalness_data_distribution(data_file: str):
    data_file = Path(data_file).expanduser()
    
    return

class S2Dataset(Dataset):


    def __init__(self, h5_file: Union[str, Path], index_table, transform=None):
        """
        Initialize the S2 dataset.

        Args:
            h5_file (Union[str, Path]): The path to the HDF5 file of a single zone or a merged HDF5 file. Should be placed under the same folder as the zone h5 files.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.
        """
        self.transform = transform
        # self.h5_file = h5py.File(h5_file, mode='r')
        self.h5_file_path = Path(h5_file).expanduser()
        self.index_table = index_table

    def __len__(self):
        return len(self.index_table)

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_file_path, mode='r')
        row = self.index_table.iloc[idx]
        image = self.h5_file[f'{row.path}/image'][row.in_partition_idx]
        # image = image.astype(np.int16)
        wc = image[13]
        image = image[:12]
        label = self.h5_file[f'{row.path}/rhs'][row.in_partition_idx]
        slope = self.h5_file[f'{row.path}/slope'][row.in_partition_idx]
        slope = slope.astype(np.float32)
        latlon = self.h5_file[f'{row.path}/latlon'][row.in_partition_idx]
        epsg = get_epsg_from_tile(row.s2_tile)
        lon_vector, lat_vector = get_dense_latlon(latlon, epsg, resolution=10, grid_size=15)
        # sensitivity = self.h5_file[f'{row.path}/gedi_attrs'][row.in_partition_idx, 24]
        shot_number = self.h5_file[f'{row.path}/shot_number'][row.in_partition_idx]
        if int(shot_number) != int(row.shot_number):
            raise ValueError(f'Error: something wrong with the index table, {shot_number} != {row.shot_number}')
        # image = image.astype('float')
        if self.transform:
            image = self.transform(image)
        # self.h5_file.close()
        return image, label, wc, slope, lon_vector, lat_vector
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()



