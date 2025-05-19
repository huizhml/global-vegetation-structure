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
from utils import get_dense_latlon


coverage_beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011']
power_beams = ['BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']
MASKED_VALUE = 32767

def get_epsg_from_tile(tile_name):
    zone = int(tile_name[:2])
    band = tile_name[2]
    hemisphere = 'south' if band <= 'M' else 'north'
    epsg = 32700 + zone if hemisphere == 'south' else 32600 + zone
    return epsg

class SparsePredDataset(Dataset):
    scl_exclude_labels = torch.tensor([3, 8, 9, 10, 11], dtype=torch.uint16)  # Example SCL labels to exclude

    def __init__(self, h5_file:str, mask_with_scl:bool=True, pred_file_path:str=None, batch_size:int=1) -> None:
        super().__init__()
        self.h5_file = Path(h5_file).expanduser()
        with h5py.File(self.h5_file) as f:
            self.length = f['s2'].shape[0]
        self.mask_with_scl = mask_with_scl
        self.batch_size = batch_size
        if not pred_file_path:
            self.pred_fp = self.h5_file.parent / 'vs_prediction.h5'
        else:
            self.pred_fp = Path(pred_file_path).expanduser()
    
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
        lon_vector, lat_vector = get_dense_latlon(coords, epsg, resolution=10, grid_size=15)
        return torch.from_numpy(img), torch.from_numpy(scl), torch.from_numpy(slope), torch.from_numpy(lon_vector), torch.from_numpy(lat_vector), torch.from_numpy(coords), torch.tensor(rowid)

    def __del__(self):
        if hasattr(self, 'data'):
            self.data.close()
    
    def init_out_h5(self, run_id:str=None):
        self.pred_fp = self.pred_fp.with_name(self.pred_fp.stem + f'_{run_id}').with_suffix('.h5')
        if not os.path.exists(self.pred_fp):
            # Create empty datasets with xarray
            ds = xr.Dataset(
                data_vars={
                    'slope': (['loc', 'y', 'x'], np.zeros((self.length, 15, 15), dtype=np.float32)),
                    'centroid': (['loc', 'coord'], np.zeros((self.length, 2), dtype=np.float32)),
                    'rowid': (['loc'], np.zeros(self.length, dtype=np.int32)),
                    'rhs_median': (['loc', 'rh', 'y', 'x'], np.zeros((self.length, 101, 15, 15), dtype=np.int16)),
                    'rhs_lower': (['loc', 'rh', 'y', 'x'], np.zeros((self.length, 101, 15, 15), dtype=np.int16)),
                    'rhs_upper': (['loc', 'rh', 'y', 'x'], np.zeros((self.length, 101, 15, 15), dtype=np.int16)),
                },
                coords={
                    'loc': np.arange(self.length),
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
                    'zlib': True,
                    'complevel': 7,
                    'fletcher32': True,
                    'chunksizes': (1, 15, 15)
                },
                'centroid': {
                    'zlib': True,
                    'complevel': 7,
                    'fletcher32': True,
                    'chunksizes': (1, 2)
                },
                'rowid': {
                    'zlib': True,
                    'complevel': 7,
                    'fletcher32': True,
                    'chunksizes': (1,)
                },
                'rhs_median': {
                    'zlib': True,
                    'complevel': 7,
                    'fletcher32': True,
                    'chunksizes': (1, 101, 15, 15)
                },
                'rhs_lower': {
                    'zlib': True,
                    'complevel': 7,
                    'fletcher32': True,
                    'chunksizes': (1, 101, 15, 15)
                },
                'rhs_upper': {
                    'zlib': True,
                    'complevel': 7,
                    'fletcher32': True,
                    'chunksizes': (1, 101, 15, 15)
                }
            }
            ds.to_netcdf(self.pred_fp, mode='w', format='NETCDF4', engine='h5netcdf', encoding=comp)
    
    def write_patch_predictions(self, prediction, scl, coords, rowid, batch_idx):
        """
        Write the prediction for a batch of patches to the output h5 file.
        prediction: torch.Tensor, shape (B, 303, 15, 15) or (B, 101*3, 15, 15)
        scl: torch.Tensor, shape (B, 15, 15)
        coords: torch.Tensor, shape (B, 2)
        rowid: torch.Tensor, shape (B,)
        """
        # Mask invalid SCL values if required
        if self.mask_with_scl:
            invalid_mask = torch.isin(scl, self.scl_exclude_labels.to(scl.device))
            invalid_mask = invalid_mask.unsqueeze(1)  # (B, 1, 15, 15)
        else:
            invalid_mask = torch.zeros_like(scl, dtype=torch.bool).unsqueeze(1)

        # Set invalid predictions to nan, then scale and convert to int16, using MASKED_VALUE for nan
        prediction = torch.where(invalid_mask, torch.tensor(np.nan, device=prediction.device, dtype=prediction.dtype), prediction)
        prediction = (prediction * 100).round()
        prediction = np.nan_to_num(prediction.cpu().numpy(), nan=MASKED_VALUE).astype(np.int16)  # (B, 303, 15, 15)

        # Reshape to (B, 101, 3, 15, 15)
        rhs = prediction[:, :303, :, :].reshape(-1, 101, 3, 15, 15)
        rhs_median = rhs[:, :, 1, :, :]  # (B, 101, 15, 15)
        rhs_lower = rhs[:, :, 2, :, :]
        rhs_upper = rhs[:, :, 0, :, :]

        # Write to the output h5 file
        with h5py.File(self.pred_fp, 'a') as f:
            real_batch_size = rhs_median.shape[0]
            idx_start = batch_idx * self.batch_size
            idx_end = idx_start + real_batch_size
            f['rhs_median'][idx_start:idx_end] = rhs_median
            f['rhs_lower'][idx_start:idx_end] = rhs_lower
            f['rhs_upper'][idx_start:idx_end] = rhs_upper
            f['centroid'][idx_start:idx_end] = coords.cpu().numpy()
            f['rowid'][idx_start:idx_end] = rowid.cpu().numpy()
        
        
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
    def __init__(self, rhs_fp: str = None, naturalness_fp: str = None, use_full_profile: bool = False, transform=None):
        super().__init__()
        self.rhs_fp = Path(rhs_fp).expanduser()
        self.naturalness_fp = Path(naturalness_fp).expanduser()
        self.target_df = pd.read_csv(self.naturalness_fp, index_col='rowid')
        self.target_df = self.target_df[~self.target_df['Land_use_ID'].isin([1,-1])]
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
        
        # Create reverse mapping for reference
        self.id_to_land_use = {v: k for k, v in self.land_use_mapping.items()}
        
        # Map the land use IDs to consecutive class indices
        self.target_df['class_idx'] = self.target_df['Land_use_ID'].map(self.land_use_mapping)
        # Fill NaN values with 0 (Unknown/No data class)
        self.target_df['class_idx'] = self.target_df['class_idx'].fillna(0)
        
        self.transform = transform
        if use_full_profile:
            self.idx = slice(None)
        else:
            self.idx = slice(98, 99)
        
    def __len__(self):
        if not hasattr(self, 'data'):
            self.data = h5py.File(self.rhs_fp)
        return len(self.data['rhs_median'])
    
    def __getitem__(self, idx):
        if not hasattr(self, 'data'):
            self.data = h5py.File(self.rhs_fp)
        rhs = self.data['rhs_median'][idx, self.idx]
        rowid = self.data['rowid'][idx]
        # Return the mapped class index instead of original land use ID
        target = self.target_df.loc[rowid, 'class_idx']
        return rhs, target
    
    def __del__(self):
        if hasattr(self, 'data'):
            self.data.close()
            

class Normalize(nn.Module):
    def __init__(self, mean_std_fp:str):
        super().__init__()
        mean_std_fp = Path(mean_std_fp).expanduser()
        file = np.load(mean_std_fp)
        self.mean = file['mean']
        self.std = file['std']

    @torch.no_grad()
    def forward(self, x) -> Tensor:
        return normalize(x.float(), self.mean, self.std)

class NaturalnessDataModule(LightningDataModule):
    def __init__(self, h5_file: str = None, naturalness_fp: str = None, mean_std_fp: str = None, use_full_profile: bool = False, 
                 batch_size: int = 1, num_workers: int = 4, train_val_split: float = 0.8, **kwargs):
        super().__init__()
        self.full_dataset = NaturalnessDataset(h5_file, naturalness_fp, use_full_profile=use_full_profile)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create train/val split
        dataset_size = len(self.full_dataset)
        train_size = int(train_val_split * dataset_size)
        val_size = dataset_size - train_size
        
        # Use random_split to create train and validation datasets
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
    #     self.mean_std_fp = Path(mean_std_fp).expanduser()
    #     if not self.mean_std_fp.exists():
    #         self.mean, self.std = self.calculate_mean_std()
    #     else:
    #         file = np.load(self.mean_std_fp)
    #         self.mean = file['mean']
    #         self.std = file['std']
    
    # def calculate_mean_std(self):
    #     # dataloader = self.train_dataloader()
    #     rhs_list = []
        
    #     for batch in self.train_dataset:
    #         rhs, target = batch
    #         rhs_list.append(rhs)
    #     rhs = np.stack(rhs_list, axis=0)
    #     mean = rhs.mean(axis=(0, 2, 3))
    #     std = rhs.std(axis=(0, 2, 3))
    #     np.savez(self.mean_std_fp, mean=mean, std=std)
    #     return mean, std
    
        
        
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
        self.h5_file_path = h5_file
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


def calculate_mean_std(h5_file: str, index_table: str):
    h5_file = Path(h5_file).expanduser()
    index_table = Path(index_table).expanduser()
    index_table = pd.read_parquet(index_table)
    index_table = index_table.compute()
    rhs_list = []
    

if __name__ == '__main__':
    import pandas as pd
    import dask.dataframe as dd
    h5_file = '~/data/GVS/downstream_task_data/rhs_predictions_2017_izraz2av.h5'
    # index_table = Path.home() / 'data/GVS/index_table_train_subsets/train0_v1.parquet'
    # index_table = [str(p) for p in index_table.glob('*.parquet')]
    # index_table = pd.read_parquet(index_table)#.compute()
    # dataset = S2Dataset(h5_files, index_table)
    dataset = NaturalnessDataModule(h5_file, naturalness_fp='~/data/GVS/downstream_task_data/naturalness/reference_data_set_updated.csv', mean_std_fp='~/data/GVS/downstream_task_data/naturalness/mean_std.npz', use_full_profile=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4096, num_workers=4)
    # for batch in enumerate(dataloader):
        # print(batch[0].shape)
    
    # for i in range(len(dataset)):
    #     lon = dataset[i][4]
    #     import ipdb; ipdb.set_trace()
    #     if np.isinf(lon).any():
    #         print(i)
    # dataset.h5_file.close()



# %%
