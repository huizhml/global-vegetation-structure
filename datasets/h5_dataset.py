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
from const import VSM_NODATA, SCL_EXCLUDE_LABELS, SCL_WATER, ESA_BUILT_UP, ESA_WATER, ESA_SNOW


import zarr
from zarr.codecs import BloscCodec, BloscShuffle

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
    def __init__(self, rhs_fp: str = None, target_df: pd.DataFrame = None, use_full_profile: bool = False, transform=None):
        super().__init__()
        self.rhs_fp = Path(rhs_fp).expanduser()
        self.target_df = target_df
        self.transform = transform
        if use_full_profile:
            self.idx = slice(None)
        else:
            self.idx = slice(98, 99)
        
    def __len__(self):
        if not hasattr(self, 'data'):
            if self.rhs_fp.suffix == '.zarr':
                self.data = zarr.open(str(self.rhs_fp), mode='r')
            else:
                self.data = h5py.File(self.rhs_fp)
        return len(self.data['vsm_median'])
    
    def __getitem__(self, idx):
        if not hasattr(self, 'data'):
            if self.rhs_fp.suffix == '.zarr':
                self.data = zarr.open(str(self.rhs_fp), mode='r')
            else:
                self.data = h5py.File(self.rhs_fp)
        rhs = self.data['vsm_median'][idx, self.idx]
        s2 = self.data['s2'][idx]
        rowid = self.data['rowid'][idx]
        # Return the mapped class index instead of original land use ID
        target = self.target_df.loc[rowid, 'class_idx']
        return rhs, s2, target
    
    def __del__(self):
        if hasattr(self, 'data'):
            self.data.close()
            

# Not used, integrated into the model, 2025-09-15
class Normalize(nn.Module):
    def __init__(self, rhs_only:bool=False, input_rhs:bool=False, input_top_height:bool=False, mean_std_fp:str=None):
        super().__init__()
        mean_std_fp = Path(mean_std_fp).expanduser()
        if not mean_std_fp.exists():
            raise ValueError(f'Mean and std file {mean_std_fp} does not exist')
        file = np.load(mean_std_fp)
        if rhs_only:
            self.mean = file['mean']
            self.std = file['std']
            return
        self.mean = file['mean_s2']
        self.std = file['std_s2']
        if input_rhs:
            self.mean_rhs = file['mean']
            self.std_rhs = file['std']
            self.mean = np.concatenate([self.mean_rhs, self.mean], axis=0)
            self.std = np.concatenate([self.std_rhs, self.std], axis=0)
        if input_top_height:
            self.mean_top_height = file['mean'][98:99]
            self.std_top_height = file['std'][98:99]
            self.mean = np.concatenate([self.mean, self.mean_top_height], axis=0)
            self.std = np.concatenate([self.std, self.std_top_height], axis=0)

    @torch.no_grad()
    def forward(self, x) -> Tensor:
        return normalize(x.float(), self.mean, self.std)

class NaturalnessDataModule(LightningDataModule):
    def __init__(self, h5_file: str = None, naturalness_fp: str = None, use_full_profile: bool = False, 
                 batch_size: int = 1, num_workers: int = 4, train_val_split: float = 0.8, 
                 class_balance: bool = False,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        
        self.target_df = pd.read_csv(naturalness_fp, index_col='rowid')
        # remove unsure labels (-1) and labels with no high resolution images (1)
        self.target_df = self.target_df[~self.target_df['Land_use_ID'].isin([1, -1])]
        self.target_df['class_idx'] = self.target_df['Land_use_ID'].map(self.land_use_mapping)
        # Fill NaN values with 0 (Unknown/No data class)
        self.target_df['class_idx'] = self.target_df['class_idx'].fillna(0)
        self.target_df['class_idx'] = self.target_df['class_idx'].astype(int)
        self.target_df = self.target_df.astype({'class_idx': 'int64'})

        self.full_dataset = NaturalnessDataset(h5_file, self.target_df, use_full_profile=use_full_profile)

        # Create reverse mapping for reference
        self.id_to_land_use = {v: k for k, v in self.land_use_mapping.items()}
        if class_balance:
            class_counts = self.target_df['class_idx'].value_counts()
            min_count = class_counts.min()
            self.target_df = self.target_df.groupby('class_idx', group_keys=False).apply(lambda x: x.sample(min_count, random_state=42))
        
        self.target_df_train = self.target_df.groupby('class_idx', group_keys=False).apply(lambda x: x.sample(frac=0.8))
        self.target_df_val = self.target_df[~self.target_df.index.isin(self.target_df_train.index)]
        with h5py.File(h5_file) as f:
            rowids = f['rowid'][:]
        self.train_idx = np.where(np.isin(rowids, self.target_df_train.index))[0]
        self.val_idx = np.where(np.isin(rowids, self.target_df_val.index))[0]
        self.train_dataset = torch.utils.data.Subset(self.full_dataset, self.train_idx)
        self.val_dataset = torch.utils.data.Subset(self.full_dataset, self.val_idx)
         
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=False, drop_last=False)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=False, drop_last=False)


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



