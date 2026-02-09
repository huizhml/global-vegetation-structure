import os
import time
import torch
import zarr
from zarr.storage import LocalStore
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import lightning as L
import xarray as xr
import random
import torch
from download.core.utils import get_dense_latlon
from const import MASKED_VALUE

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="zarr.codecs.vlen_utf8")

def zarrdataset_worker_init_fn(worker_id):
    """ZarrDataset multithread workers initialization function.
    """

    worker_info = torch.utils.data.get_worker_info()
    w_sel = slice(worker_id, None, worker_info.num_workers)

    dataset_obj = worker_info.dataset

    # Reset the random number generators in each worker.
    torch_seed = torch.initial_seed()
    random.seed(torch_seed)
    np.random.seed(torch_seed % (2**32 - 1))

    dataset_obj._worker_sel = w_sel
    dataset_obj._worker_id = worker_id
    dataset_obj._num_workers = worker_info.num_workers

class ZarrSentinel2Downstream(Dataset):
    """
    A custom Dataset for prediction on Sentinel-2 data stored in Zarr format.
    Handles overlapping patches and provides recomposition functionality.

    Args:
        zarr_store_path (str): Path to Zarr store
        tile_ids (str): IDs of the tile group to process (e.g., ['32TMT'])
        patch_size (int): Size of square patches
        border (int): Overlap between patches
        time_step (int): Which time dimension index to use (default 0)
        bands (list): Band indices to include (default all)
        input_lat_lon (bool): Include lat/lon as input channels
    """

    def __init__(self, zarr_store_path, pred_zarr_path: str = None,
                 mask_with_scl: bool = True,
                 compression: str = None):
        self.mask_with_scl = mask_with_scl
        self.compression = compression
        self.zarr_store_path = Path(zarr_store_path).expanduser()
        if not pred_zarr_path:
            self.pred_zarr_path = self.zarr_store_path.parent / 'vs_predictions.zarr'
        else:
            self.pred_zarr_path = Path(pred_zarr_path).expanduser()
        
        # store = zarr.open(self.zarr_store_path, mode='r')
        # self.length = store['s2'].shape[0]
        # print(self.length)
        self._initialized = False
        
        # try:
        #     self.store = zarr.open(zarr_store_path, mode='r')
        # except Exception as e:
        #     print(f'Error opening zarr store: {e}, conda activate py3!')
        self.scl_zero_canopy_height = np.array([5, 6])  # "not vegetated", "water"
        # cloud shadows, CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY, SNOW, water, nodata
        self.scl_exclude_labels = torch.tensor(np.array([0, 3, 8, 9, 11, 6]))

    def _iniitialze(self, force=False):
        if self._initialized and not force:
            return
        store = LocalStore(self.zarr_store_path)
        self.store = zarr.open(store, mode='r')
        self._initialized = True
        print('initialized')

    def __len__(self):
        if not hasattr(self, 'length'):
            self._iniitialze()
            self.length = self.store['s2'].shape[0]
            print(self.length)
        return self.length
    
    # def __iter__(self):
    #     self._iniitialze()
        

    def __getitem__(self, idx):
        self._iniitialze()
        # store = zarr.open(self.zarr_store_path, mode='r')
        s2 = self.store['s2'][idx]
        img = s2[:12, :, :]
        scl = s2[12, :, :]
        slope = self.store['slope'][idx]
        coords = self.store['centroid'][idx]
        rowid = self.store['rowid'][idx]
        epsg = self.store['epsg'][idx]
        lon_vector, lat_vector = get_dense_latlon(coords, epsg, resolution=10, grid_size=15)
        
        # # Create random arrays instead
        # img = np.random.rand(12, 15, 15).astype(np.float32)
        # scl = np.random.randint(0, 12, size=(15, 15)).astype(np.uint8)
        # slope = np.random.rand(15, 15).astype(np.float32)
        # coords = np.random.rand(2).astype(np.float32)  # [lat, lon]
        # rowid = np.array([idx], dtype=np.int64)
        # lon_vector = np.random.rand(15).astype(np.float32)
        # lat_vector = np.random.rand(15).astype(np.float32)

        return torch.from_numpy(img), torch.from_numpy(scl), torch.from_numpy(slope), torch.from_numpy(lon_vector), torch.from_numpy(lat_vector), torch.from_numpy(coords), torch.from_numpy(rowid)
        
    def write_patch_predictions(self, prediction, scl, coords, rowid):
        """Write patch prediction to tiff file
        Parameters
        ----------
        prediction : np.array, (time, rh_channels, patch_size, patch_size)
        """
        if self.mask_with_scl:
            # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
            invalid_mask = torch.isin(scl, self.scl_exclude_labels.to(scl.device))
            invalid_mask = invalid_mask.unsqueeze(1) # Add  channel dimensions
            prediction = torch.where(invalid_mask, torch.tensor(np.nan, device=prediction.device), prediction)
        prediction = (prediction * 100).round()
        prediction = np.nan_to_num(prediction.cpu().numpy(), nan=MASKED_VALUE).astype(np.int16)
        rhs = prediction[:, :303, :, :].reshape(-1, 101, 3, 15, 15)
        rhs_median = rhs[:,:,1,:,: ]
        rhs_lower = rhs[:,:,0,:,: ]
        rhs_upper = rhs[:,:,2,:,: ]
        ds = xr.Dataset(data_vars={'rhs_median': (['rowid','rh', 'y', 'x'], rhs_median),
                                   'rhs_lower': (['rowid', 'rh', 'y', 'x'], rhs_lower),
                                   'rhs_upper': (['rowid', 'rh', 'y', 'x'], rhs_upper),
                                   'centroid': (['rowid', 'coords'], coords.cpu().numpy())},
                        coords={'rowid': rowid.cpu().numpy().squeeze(), 'rh': list(range(101)), 'y': list(range(15)), 'x': list(range(15)), 'coords': ['lat', 'lon']})
        if not os.path.exists(self.pred_zarr_path):
            ds.to_zarr(self.pred_zarr_path, mode='w')
        else:
            ds.to_zarr(self.pred_zarr_path, mode='a')




class DeployDataModel(L.LightningDataModule):

    def __init__(self,
                 pred_fp: str = None,
                 batch_size: int = 16,
                 num_workers: int = 8,
                 **kwargs
                 ):
        super().__init__()
        self.pred_dataset = ZarrSentinel2Downstream(zarr_store_path=pred_fp)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.pred_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    dataset = ZarrSentinel2Downstream(zarr_store_path='~/data/gvs/downstream_task_data/s2_2017.zarr')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4096, num_workers=4, worker_init_fn=zarrdataset_worker_init_fn, shuffle=False)
    for batch in dataloader:
        print(batch[0].shape)
        
