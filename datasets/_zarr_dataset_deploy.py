from typing import List
import time
import torch
import zarr
import numpy as np
import math
from pathlib import Path
from torch.utils.data import Dataset
import lightning as L
import rasterio
from rasterio.windows import Window
from rasterio.enums import Compression
from numcodecs.zarr3 import LZMA
import rioxarray
from pyproj import Transformer
from utils import get_dense_latlon
import os
import json
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import box
import h5py
import shutil
from download._const import S2_ITEM_PROPS
from download._utils import get_patch, row_to_stac_item
import dask
# import threading
# import fcntl  # For file locking on Unix systems

MASKED_VALUE = 32767
RH100_idx = 301
RH98_idx = 295
LAT_MEAN = 12.7596
LAT_STD = 25.6075
LON_SIN_MEAN = 0.1098
LON_SIN_STD = 0.7536
LON_COS_MEAN = 0.3072
LON_COS_STD = 0.5706

class ZarrSentinel2Deploy(Dataset):
    """
    A custom Dataset for prediction on Sentinel-2 data stored in Zarr format.
    Handles overlapping patches and provides recomposition functionality.
    NOTE: Not used because loading image into memory will be copied by workers

    Args:
        zarr_store_path (str): Path to Zarr store
        tile_ids (str): IDs of the tile group to process (e.g., ['32TMT'])
        patch_size (int): Size of square patches
        border (int): Overlap between patches
        time_step (int): Which time dimension index to use (default 0)
        bands (list): Band indices to include (default all)
        input_lat_lon (bool): Include lat/lon as input channels
    """

    def __init__(self, zarr_store_path, tile_id: str = None, prediction_dir: str = None,
                 patch_size=512, border=16,
                 img_idx: int = None,
                 predict_full_profile: bool = True,
                 bands: List[int] = None, 
                 input_lat_lon=False,
                 mask_with_scl: bool = True,
                 mask_empty: bool = True,
                 compression: str = None,
                 comp_level: int = 6,
                 use_xarray: bool = False,
                 output_format: str = 'h5',
                 chunk_size: int = 1024,
                 **kwargs
                 ):
        # ********** Set up parameters **********
        self.use_xarray = use_xarray
        self.mask_with_scl = mask_with_scl
        self.mask_empty = mask_empty
        self.compression = compression
        self.comp_level = comp_level
        self.patch_size = patch_size
        self.border = border
        self.patch_size_no_border = self.patch_size - 2 * self.border
        self.input_lat_lon = input_lat_lon
        self.output_format = output_format
        self.chunk_size = chunk_size
        if output_format == 'gtiff':
            self._init_output =  self._initialize_gtiff_output
            self.write_patch_predictions = self.write_patch_predictions_gtiff
            self.finalize_output = lambda: None
        elif output_format == 'h5':
            self._init_output =  self._initialize_h5_output
            self.write_patch_predictions = self.write_patch_predictions_h5
            self.finalize_output = lambda: None
        elif output_format == 'zip':
            self._init_output =  self._initialize_zip_output
            self.write_patch_predictions = self.write_patch_predictions_zip
            self.finalize_output = lambda: None
        elif output_format == 'zarr':
            self._init_output =  self._initialize_zarr_output
            self.write_patch_predictions = self.write_patch_predictions_zarr
            self.finalize_output = lambda: None
        elif output_format == 'cog':
            self._init_output =  self._initialize_cog_output
            self.write_patch_predictions = self.write_patch_predictions_cog
            self.finalize_output = self.translate_to_cog
        else:
            raise ValueError(f'Invalid output format: {output_format}')
        # ********** Set up indices **********
        self.rh_idx = slice(0, 303) if predict_full_profile else (RH100_idx, RH98_idx)
        self.tile_id = tile_id
        self.bands = bands or slice(12)
        
        if img_idx is None:
            self.img_idx = slice(None)
        else:
            self.img_idx = slice(img_idx, img_idx + 1)
    
        # ********** Load image and get metadata **********
        zarr_store_path = Path(zarr_store_path).expanduser()
        try:
            self.store = zarr.open(zarr_store_path, mode='r')
        except Exception as e:
            print(f'Error opening zarr store: {e}, conda activate py3!')
        self.image = self.store[f'{self.tile_id}/s2'][self.img_idx]
        self.transform = self.store[f'{tile_id}'].attrs['transform']
        self.crs = f'EPSG:{self.store[f"{tile_id}/epsg"][()].item()}'
        self.img_width, self.img_height = self.image.shape[2:]
        self.scl = self.image[:, -1:, :, :]
        self.image = self.image[:, :-1, :, :]
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        self.image = np.pad(
            self.image, ((0, 0),
                         (0, 0),
                         (self.border, self.border),
                         (self.border, self.border)),
            mode='symmetric')
        if self.input_lat_lon:
            lon_mask = self.store[f'{self.tile_id}/x'][:]
            lat_mask = self.store[f'{self.tile_id}/y'][:]
            lon_mask, lat_mask = transformer.transform(lon_mask, lat_mask, radians=True)
            lon, lat = np.meshgrid(lon_mask, lat_mask)
            sin_lon = np.sin(lon)
            cos_lon = np.cos(lon)
            lat = (lat - LAT_MEAN) / LAT_STD
            sin_lon = (sin_lon - LON_SIN_MEAN) / LON_SIN_STD
            cos_lon = (cos_lon - LON_COS_MEAN) / LON_COS_STD
            coords = np.stack([lat, sin_lon, cos_lon], axis=0)
            self.coords_input = np.pad(
                coords, (
                            (0, 0),
                            (self.border, self.border),
                            (self.border, self.border)),
                mode='symmetric')

        self.patch_coords_dict = self._get_patch_coords()
        print('--------------------------------')
        print(f"Image shape: {self.image.shape}")
        print(f"Number of patches: {len(self.patch_coords_dict)}")
        print('--------------------------------')
        
        # ********** Prediction Configuration **********
        self.scl_zero_canopy_height = np.array([5, 6])  # "not vegetated", "water"
        # cloud shadows, CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY, SNOW, water, nodata
        self.nodata_value = 65535
        self.scl_exclude_labels = np.array([0, 3, 8, 9, 11, 6, self.nodata_value])
        # Pre-compute exclude labels as tensor to avoid repeated creation
        self.scl_exclude_labels_tensor = None  # Will be set when first used
        self.scl = np.array(self.scl, dtype=np.uint8)
        self.prediction_cache = {}
        
        # Create prediction directory first
        year = zarr_store_path.stem.split('_')[1]
        if not prediction_dir:
            self.prediction_dir = zarr_store_path.parent / f'predictions_{year}'
        else:
            self.prediction_dir = Path(prediction_dir).expanduser()
        self.prediction_dir.mkdir(exist_ok=True)
        
        # Use memory mapping instead of in-memory array to handle large data and multiprocessing
        self.full_pred_path = self.prediction_dir / f"{self.tile_id}_full_pred.dat"
        self.full_pred = np.memmap(
            filename=str(self.full_pred_path),
            dtype=np.int16,
            mode='w+',  # create or overwrite
            shape=(303, self.img_height, self.img_width)
        )
        self.full_pred[:] = MASKED_VALUE  # fill with masked value
        
        # Create lock file for multiprocessing coordination
        self.lock_file_path = self.prediction_dir / f"{self.tile_id}_full_pred.lock"

    def _get_patch_coords(self):
        """Calculate patch coordinates with overlap handling"""
        _, _, y_dim, x_dim = self.image.shape
        col_steps = int(math.ceil(y_dim / self.patch_size_no_border))
        row_steps = int(math.ceil(x_dim / self.patch_size_no_border))
        patch_coords_dict = {}
        patch_idx = 0
        for y in range(col_steps): #col_steps
            y_coord = y * self.patch_size_no_border
            if y_coord > y_dim - self.patch_size:
                # move last patch up if it would exceed the image bottom
                y_coord = y_dim - self.patch_size
            for x in range(row_steps):
                x_coord = x * self.patch_size_no_border
                if x_coord > x_dim - self.patch_size:
                    # move last patch left if it would exceed the image right border
                    x_coord = x_dim - self.patch_size
                patch_coords_dict[patch_idx] = (slice(0,10), y_coord, x_coord)
                patch_coords_dict[patch_idx+1] = (slice(10,20), y_coord, x_coord)
                patch_idx += 2
        return patch_coords_dict

    def __len__(self):
        return len(self.patch_coords_dict)

    def __getitem__(self, idx):
        img_batch_idx, y_topleft, x_topleft = self.patch_coords_dict[idx]
        # img_batch_idx is either :10 or 10:

        # Extract patch with border handling
        patch = self.image[img_batch_idx, :, y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size]
        patch = patch.astype(np.float32)
        # # Add lat/lon channels if needed
        if self.input_lat_lon:
            latlon = self.coords_input[:, y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size]
            latlon = np.tile(latlon[None, :, :,:], (patch.shape[0], 1,1,1))
            return torch.from_numpy(patch), torch.from_numpy(latlon)
        else:
            return torch.from_numpy(patch)


    def set_prediction_fname(self, wandb_run_id):
        if not (self.img_idx.stop and self.img_idx.start) or self.img_idx.stop - self.img_idx.start > 1:
            base_name = f'{self.tile_id}_{wandb_run_id}'
        else:
            # predict for one image for testing
            img_id = self.store[f'{self.tile_id}/id'][self.img_idx].item()
            base_name = f'{self.tile_id}_{img_id}_{wandb_run_id}'
        
        self.prediction_fp = self.prediction_dir / f'{base_name}_{self.compression}_{self.comp_level}.tif'
        self._init_output()
        print(f'Predictions will be saved to {self.prediction_fp}')
        

    def _apply_masks(self, prediction, x_topleft, y_topleft):
        
        prediction_no_border = prediction[:, self.rh_idx,
                                          self.border:self.patch_size - self.border,
                                          self.border:self.patch_size - self.border
                                          ]
        
        
        location_key = f'{y_topleft}_{x_topleft}'
        if location_key not in self.prediction_cache:
            self.prediction_cache[location_key] = prediction_no_border
            return None
        else: # ready to write
            prediction_no_border = torch.cat([self.prediction_cache[location_key], prediction_no_border], dim=0)
            
            # Prepare masks before applying them to reduce repeated operations
            masks_to_apply = []
            
            if self.mask_empty:
                # pixels where all RGB values equal zero are empty (bands B02, B03, B04)
                # note self.image has shape: (height, width, channels)
                img = self.image[:, 1:4, y_topleft:y_topleft + self.patch_size_no_border, x_topleft:x_topleft + self.patch_size_no_border]
                # Use torch operations directly instead of numpy sum
                img_tensor = torch.from_numpy(img).to(prediction_no_border.device)
                invalid_mask = torch.sum(img_tensor, dim=1, keepdim=True) == 0
                masks_to_apply.append(invalid_mask)
            
            if self.mask_with_scl:
                # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
                scl = self.scl[:, :, y_topleft:y_topleft + self.patch_size_no_border,
                            x_topleft:x_topleft + self.patch_size_no_border]
                # Convert to torch tensor once and use isin equivalent with pre-computed tensor
                scl_tensor = torch.from_numpy(scl).to(prediction_no_border.device)
                if self.scl_exclude_labels_tensor is None or self.scl_exclude_labels_tensor.device != prediction_no_border.device:
                    self.scl_exclude_labels_tensor = torch.tensor(self.scl_exclude_labels, device=prediction_no_border.device)
                scl_mask = torch.isin(scl_tensor, self.scl_exclude_labels_tensor)
                masks_to_apply.append(scl_mask)
            
            # Apply all masks at once using logical_or to combine them
            if len(masks_to_apply) > 1:
                combined_mask = torch.logical_or(*masks_to_apply)
            else:
                combined_mask = masks_to_apply[0]
            prediction_no_border = torch.where(combined_mask, torch.nan, prediction_no_border)
            
            # aggregate predictions with median
            prediction_no_border, _ = torch.nanmedian(prediction_no_border, dim=0)          
            prediction_no_border.mul_(100).round_()  # In-place operations
            prediction_no_border = torch.nan_to_num(prediction_no_border, nan=MASKED_VALUE).to(torch.int16)
            
            # Move to CPU once and do all numpy operations together
            prediction_no_border = prediction_no_border.cpu().numpy()
            return prediction_no_border
        
    

        
    def _initialize_cog_output(self):
        """Initialize output as a Cloud-Optimized GeoTIFF using GDAL directly"""
        self.prediction_fp = self.prediction_fp.with_suffix('.tif')
        if self.prediction_fp.exists():
            os.remove(self.prediction_fp)
        
        # self.memfile = []
        # self.dataset = []
        # for i in range(101):
        #     for j in range(3):
        #         self.memfile.append(MemoryFile()) 
        #         self.dataset.append(self.memfile.open(driver='GTiff',
        #                              width=self.img_width,
        #                              height=self.img_height,
        #                              count=1,
        #                              dtype=np.int16,
        #                              crs=self.crs,
        #                              transform=Affine(*self.transform),
        #                              tiled=True,
        #                              blockxsize=self.chunk_size,
        #                              blockysize=self.chunk_size,
        #                              compress=self.compression,
        #                              nodata=MASKED_VALUE,
        #                              level=self.comp_level))
        self.memfile = MemoryFile()
        self.dataset = self.memfile.open(driver='GTiff',
                                         width=self.img_width,
                                         height=self.img_height,
                                         count=1,
                                         dtype=np.int16,
                                         crs=self.crs,
                                         transform=Affine(*self.transform),
                                         tiled=True,
                                         blockxsize=self.chunk_size,
                                         blockysize=self.chunk_size,
                                         compress=self.compression,
                                         nodata=MASKED_VALUE,
                                         level=self.comp_level)


    def write_patch_predictions_cog(self, prediction, idx):
        """Write patch predictions directly to COG using GDAL"""
        img_batch_idx, y_topleft, x_topleft = self.patch_coords_dict[idx]
        prediction_no_border = self._apply_masks(prediction, x_topleft, y_topleft)
        if prediction_no_border is None:
            return
        window = Window(x_topleft, y_topleft, self.patch_size_no_border, self.patch_size_no_border)
        # data = da.from_array(prediction_no_border, chunks=(1, self.patch_size_no_border, self.patch_size_no_border))
        # da.store(data, self.dataset)
        self.dataset.write(prediction_no_border[0], 1, window=window)
        del self.prediction_cache[f'{y_topleft}_{x_topleft}']
          

def collate_batch(batch):
    return batch[0]


class S2CogDataset(Dataset):
    """
    This dataset is used to cache the predictions and write once to a COG/H5 file. 
    Not loading the whole input image into memory.
    """
    def __init__(self, zarr_store_path, tile_id: str = None, prediction_dir: str = None,
                 patch_size=512, border=16,
                 img_idx: int = None,
                 predict_full_profile: bool = True,
                 bands: List[int] = None, 
                 input_lat_lon=False,
                 mask_with_scl=True,
                 mask_empty=True,
                 compression=None,
                 comp_level=6,
                 use_xarray: bool=True,
                 to_zarr: bool=True,
                 output_format: str = 'cog',
                 chunk_size: int = 512,
                 debug: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        # ********** Set up parameters **********
        self.mask_with_scl = mask_with_scl
        self.mask_empty = mask_empty
        self.compression = compression
        self.comp_level = comp_level
        self.use_xarray = use_xarray
        self.to_zarr = to_zarr
        self.patch_size = patch_size
        self.border = border
        self.patch_size_no_border = self.patch_size - 2 * self.border
        self.input_lat_lon = input_lat_lon
        self.output_format = output_format
        self.chunk_size = chunk_size
        self.debug = debug
        self.predict_full_profile = predict_full_profile
        
        # ********** Set up indices **********
        self.rh_idx = slice(0, 303) if predict_full_profile else (RH100_idx, RH98_idx)
        self.tile_id = tile_id
        self.bands = bands or slice(12)
        
        if img_idx is None:
            self.img_idx = slice(None)
        else:
            self.img_idx = slice(img_idx, img_idx + 1)
    
        # ********** Load image and get metadata **********
        zarr_store_path = Path(zarr_store_path).expanduser()
        try:
            self.store = zarr.open(zarr_store_path, mode='r')
        except Exception as e:
            print(f'Error opening zarr store: {e}, conda activate py3!')
        self.img_width, self.img_height = self.store[f'{self.tile_id}/s2'].shape[2:]  # Get shape without loading array
        self.transform = self.store[f'{tile_id}'].attrs['transform']
        self.crs = f'EPSG:{self.store[f"{tile_id}/epsg"][()].item()}'
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        
        if self.input_lat_lon:
            lon_mask = self.store[f'{self.tile_id}/x'][:]
            lat_mask = self.store[f'{self.tile_id}/y'][:]
            lon_mask, lat_mask = transformer.transform(lon_mask, lat_mask, radians=True)
            lon, lat = np.meshgrid(lon_mask, lat_mask)
            sin_lon = np.sin(lon)
            cos_lon = np.cos(lon)
            lat = (lat - LAT_MEAN) / LAT_STD
            sin_lon = (sin_lon - LON_SIN_MEAN) / LON_SIN_STD
            cos_lon = (cos_lon - LON_COS_MEAN) / LON_COS_STD
            self.coords_input = np.stack([lat, sin_lon, cos_lon], axis=0)

        self.patch_coords_dict = self._get_patch_coords()
        print('--------------------------------')
        print(f"Image shape: {self.img_width}, {self.img_height}")
        print(f"Number of patches: {len(self.patch_coords_dict)}")
        print('--------------------------------')
        
        # ********** Prediction Configuration **********
        self.scl_zero_canopy_height = np.array([5, 6])  # "not vegetated", "water"
        # cloud shadows, CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY, SNOW, water, nodata
        self.nodata_value = 65535
        self.scl_exclude_labels = np.array([0, 3, 8, 9, 11, 6, self.nodata_value], dtype=np.uint16)
        # Pre-compute exclude labels as tensor to avoid repeated creation
        self.scl_exclude_labels_tensor = None  # Will be set when first used
        self.prediction_cache = {}
        
        # Create prediction directory first
        year = zarr_store_path.stem.split('_')[1]
        if not prediction_dir:
            self.prediction_dir = zarr_store_path.parent / f'predictions_{year}' / self.tile_id
        else:
            self.prediction_dir = Path(prediction_dir).expanduser()
        self.prediction_dir.mkdir(exist_ok=True)
        # self.full_pred = np.full((303, self.img_height, self.img_width), MASKED_VALUE, dtype=np.int16)
        

    def __len__(self):
        return len(self.patch_coords_dict)
    
    def __getitem__(self, idx):
        img_batch_idx, y_topleft, x_topleft = self.patch_coords_dict[idx]
        # img_batch_idx is either :10 or 10:

        # Extract patch with border handling
        y_end = y_topleft + self.patch_size
        x_end = x_topleft + self.patch_size
        y_topleft = max(y_topleft, 0)
        x_topleft = max(x_topleft, 0)
        patch = self.store[f'{self.tile_id}/s2'][img_batch_idx, :-1, y_topleft:y_end, x_topleft:x_end]
        y_start_no_border = max(y_topleft - self.border, 0)
        x_start_no_border = max(x_topleft - self.border, 0)
        # TODO: not mem efficient
        scl = self.store[f'{self.tile_id}/s2'][:20, -1:, y_start_no_border:y_start_no_border+self.patch_size_no_border, x_start_no_border:x_start_no_border+self.patch_size_no_border]
        patch = patch.astype(np.float32)
        # # Add lat/lon channels if needed
        if self.input_lat_lon:
            latlon = self.coords_input[:, y_topleft:y_end, x_topleft:x_end]
            latlon = np.tile(latlon[None, :, :,:], (patch.shape[0], 1,1,1))
            return torch.from_numpy(patch), torch.from_numpy(scl), torch.from_numpy(latlon)
        else:
            return torch.from_numpy(patch), torch.from_numpy(scl)
    
    def _get_patch_coords(self):
        """Calculate patch coordinates with overlap handling"""
        y_dim = self.img_height + self.border * 2
        x_dim = self.img_width + self.border * 2
        col_steps = int(math.ceil(y_dim / self.patch_size_no_border))
        row_steps = int(math.ceil(x_dim / self.patch_size_no_border))
        col_steps = 5 if self.debug else col_steps
        row_steps = 5 if self.debug else row_steps
        patch_coords_dict = {}
        patch_idx = 0
        for y in range(col_steps): # TODO: col_steps
            y_coord = y * self.patch_size_no_border
            if y_coord > y_dim - self.patch_size:
                # move last patch up if it would exceed the image bottom
                y_coord = y_dim - self.patch_size
            for x in range(row_steps):
                x_coord = x * self.patch_size_no_border
                if x_coord > x_dim - self.patch_size:
                    # move last patch left if it would exceed the image right border
                    x_coord = x_dim - self.patch_size
                patch_coords_dict[patch_idx] = (slice(0,10), y_coord-self.border, x_coord-self.border)
                patch_coords_dict[patch_idx+1] = (slice(10,20), y_coord-self.border, x_coord-self.border)
                patch_idx += 2
        return patch_coords_dict
    
    def cache_predictions(self, prediction, scl, idx):
        y_topleft, x_topleft = self.patch_coords_dict[idx][1:]
        y_topleft = y_topleft + self.border
        x_topleft = x_topleft + self.border     
        prediction_no_border = self._apply_masks(prediction, scl, x_topleft, y_topleft)
        if prediction_no_border is None:
            return
            
        del self.prediction_cache[f'{y_topleft}_{x_topleft}']
            
            
    @dask.delayed
    def _write_prediction_to_cog(self, pred, name, src_profile, dst_profile):
        with MemoryFile() as memfile:
            # Write data to memory file and close it properly
            with memfile.open(**src_profile) as dst:
                dst.write(pred, indexes=1)
            # File is now closed and ready for translation
            
            output_path = self.prediction_fp.with_suffix(f'.{name}.tif')
            print(f'Saving {name} to {output_path}')
            # Save memory file as temporary GeoTIFF
            temp_path = output_path.with_suffix('.geo.tif')
            with rasterio.open(temp_path, 'w', **src_profile) as dst:
                dst.write(pred, indexes=1)
            
            cog_translate(
                memfile,
                output_path,
                dst_profile,
                in_memory=False,
                quiet=False,
                use_cog_driver=False
            )

    def save_predictions_as_cog(self, full_pred,*args, **kwargs):
        """Recompose full tile from cached predictions and save as COG."""
            
        transform = Affine(*self.transform)

        src_profile = dict(
            driver="GTiff",
            dtype="int16",
            count=1,
            height=self.img_height,
            width=self.img_width,
            crs=self.crs,
            transform=transform,
            nodata=MASKED_VALUE,  # Important: add nodata value
            tiled=True,
            compress='none',  # No compression for the intermediate file
            blockxsize=self.chunk_size,
            blockysize=self.chunk_size,
        )
        
        dst_profile = cog_profiles.get('deflate')
        dst_profile.update({
            'blockxsize': self.chunk_size,
            'blockysize': self.chunk_size,
            'tiled': True,
            'interleave': 'band',
            'predictor': 2,
            # 'nodata': MASKED_VALUE,  # Ensure nodata is preserved
        })
        print(dst_profile)
        # Save as COG
        t0 = time.time()
        futures = []
        if self.predict_full_profile:
            for i in range(full_pred.shape[0]):
                rh_idx = i // 3
                q_idx = i % 3
                futures.append(self._write_prediction_to_cog(full_pred[i], f'RH{rh_idx}_Q{q_idx}', src_profile, dst_profile))
            dask.compute(*futures)
        else:
            for i, idx in enumerate(self.rh_idx):
                rh_idx = idx // 3
                futures.append(self._write_prediction_to_cog(full_pred[i], f'RH{rh_idx}_Q1', src_profile, dst_profile))
            dask.compute(*futures)
            
        # for attr_name, name in zip(['full_pred_rh_upper', 'full_pred_rh_lower', 'full_pred_rh_median'], ['upper', 'lower', 'median']):
        #     pred = getattr(self, attr_name)
        #     print(f'Processing {name} - data range: {pred.min()} to {pred.max()}')
        #     print(f'Data shape: {pred.shape}, dtype: {pred.dtype}')
            
        #     # Check if pred has valid data
        #     valid_data = pred[pred != MASKED_VALUE]
        #     if len(valid_data) > 0:
        #         print(f'Valid data range: {valid_data.min()} to {valid_data.max()}')
        #     else:
        #         print('Warning: No valid data found!')
            
        #     futures = []
        #     for i in range(self.full_pred_rh_upper.shape[0]):
        #         futures.append(self._write_prediction_to_cog(pred[i], f'RH{i}_{name}', src_profile, dst_profile))
        #     dask.compute(futures)
        print(f'Time taken to save {self.tile_id} as COG: {time.time() - t0:.2f} seconds')
           
    def set_prediction_fname(self, wandb_run_id):
        if not (self.img_idx.stop and self.img_idx.start) or self.img_idx.stop - self.img_idx.start > 1:
            base_name = f'{self.tile_id}_{wandb_run_id}'
        else:
            # predict for one image for testing
            img_id = self.store[f'{self.tile_id}/id'][self.img_idx].item()
            base_name = f'{self.tile_id}_{img_id}_{wandb_run_id}'
        
        self.prediction_fp = self.prediction_dir / f'{base_name}_{self.compression}_{self.comp_level}.tif'
        # self._init_output()
        print(f'Predictions will be saved to {self.prediction_fp}')

    def _apply_masks(self, prediction, scl, x_topleft, y_topleft):
        prediction_no_border = prediction[:, self.rh_idx,
                                          self.border:self.patch_size - self.border,
                                          self.border:self.patch_size - self.border
                                          ]
        
        
        location_key = f'{y_topleft}_{x_topleft}'
        if location_key not in self.prediction_cache:
            self.prediction_cache[location_key] = prediction_no_border
            return None
        else: # ready to write
            prediction_no_border = torch.cat([self.prediction_cache[location_key], prediction_no_border], dim=0)
            
            # Prepare masks before applying them to reduce repeated operations
            masks_to_apply = []
            
            # if self.mask_empty:
            #     # pixels where all RGB values equal zero are empty (bands B02, B03, B04)
            #     # note self.image has shape: (height, width, channels)
            #     img = self.store[f'{self.tile_id}/s2'][:, 1:4, y_topleft:y_topleft + self.patch_size_no_border, x_topleft:x_topleft + self.patch_size_no_border]
            #     # Use torch operations directly instead of numpy sum
            #     img_tensor = torch.from_numpy(img).to(prediction_no_border.device)
            #     invalid_mask = torch.sum(img_tensor, dim=1, keepdim=True) == 0
            #     masks_to_apply.append(invalid_mask)
            
            if self.mask_with_scl:
                # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
                scl = scl[:, :, y_topleft:y_topleft + self.patch_size_no_border,
                            x_topleft:x_topleft + self.patch_size_no_border]
                # Convert to torch tensor once and use isin equivalent with pre-computed tensor
                scl_tensor = torch.from_numpy(scl).to(prediction_no_border.device)
                if self.scl_exclude_labels_tensor is None or self.scl_exclude_labels_tensor.device != prediction_no_border.device:
                    self.scl_exclude_labels_tensor = torch.tensor(self.scl_exclude_labels, device=prediction_no_border.device)
                scl_mask = torch.isin(scl_tensor, self.scl_exclude_labels_tensor)
                masks_to_apply.append(scl_mask)
            
            # Apply all masks at once using logical_or to combine them
            if len(masks_to_apply) > 1:
                combined_mask = torch.logical_or(*masks_to_apply)
            else:
                combined_mask = masks_to_apply[0]
            prediction_no_border = torch.where(combined_mask, torch.nan, prediction_no_border)
            
            # aggregate predictions with median
            prediction_no_border, _ = torch.nanmedian(prediction_no_border, dim=0)          
            # prediction_no_border.mul_(100).round_()  # In-place operations
            prediction_no_border = torch.nan_to_num(prediction_no_border, nan=MASKED_VALUE)#.to(torch.int16)
            
            # Move to CPU once and do all numpy operations together
            prediction_no_border = prediction_no_border.cpu().numpy()
            return prediction_no_border
        
class S2DatasetStream(Dataset):
    def __init__(self, 
                 metadata_file: str, 
                 tile_id: str = None, 
                 s2_img_file: str = None,
                 prediction_dir: str = None,
                 patch_size=512, border=16,
                 img_idx: int = None,
                 predict_full_profile: bool = True,
                 input_lat_lon=False,
                 mask_with_scl=True,
                 mask_empty=True,
                 compression=None,
                 comp_level=6,
                 use_xarray: bool=True,
                 n_iamges_per_tile: int = 20,
                 **kwargs):
        super().__init__()
        self.metadata_file = Path(metadata_file).expanduser()
        self.s2_img_file = Path(s2_img_file).expanduser()
        self.tile_id = tile_id
        self.n_iamges_per_tile = n_iamges_per_tile
        self.prediction_dir = prediction_dir
        self.patch_size = patch_size
        self.border = border
        self.img_idx = img_idx
        self.predict_full_profile = predict_full_profile
        self.bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]
        
        

    def _get_patch_coords(self):
        """Calculate patch coordinates with overlap handling"""
        _, _, y_dim, x_dim = self.image.shape
        col_steps = int(math.ceil(y_dim / self.patch_size_no_border))
        row_steps = int(math.ceil(x_dim / self.patch_size_no_border))
        patch_coords_dict = {}
        patch_idx = 0
        for y in range(col_steps): #col_steps
            y_coord = y * self.patch_size_no_border
            if y_coord > y_dim - self.patch_size:
                # move last patch up if it would exceed the image bottom
                y_coord = y_dim - self.patch_size
            for x in range(row_steps):
                x_coord = x * self.patch_size_no_border
                if x_coord > x_dim - self.patch_size:
                    # move last patch left if it would exceed the image right border
                    x_coord = x_dim - self.patch_size
                patch_coords_dict[patch_idx] = (slice(0,10), y_coord, x_coord)
                patch_coords_dict[patch_idx+1] = (slice(10,20), y_coord, x_coord)
                patch_idx += 2
        return patch_coords_dict
    
    def download_tile(self):
        s2_df = gpd.read_parquet(self.metadata_file)
        tile_df = s2_df[s2_df['s2:mgrs_tile'] == self.tile_id].set_index('id')
        if len(tile_df)>self.n_iamges_per_tile:
            if (tile_df['s2:nodata_pixel_percentage']==0).sum() > 0:
                tile_df = tile_df.sort_values(['s2:nodata_pixel_percentage', 'eo:cloud_cover']).head(self.n_iamges_per_tile)
            else:
                idx = tile_df.groupby('orbit')['eo:cloud_cover'].nsmallest(self.n_iamges_per_tile//2).index.get_level_values(1)
                tile_df = tile_df.loc[idx]
        tile_df['datetime'] = tile_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        bbox = box(*tile_df.total_bounds)
        items = row_to_stac_item(tile_df, S2_ITEM_PROPS)  
        epsg = items[0].properties['proj:epsg']
        image = get_patch(items, self.bands, dtype='uint16', fill_value=np.uint16(0))
        image.name = 's2'
        self.image = image
        self.patch_coords_dict = self._get_patch_coords()
        
    def __getitem__(self, idx):
        img_batch_idx, y_topleft, x_topleft = self.patch_coords_dict[idx]
        patch = self.image.isel[img_batch_idx, :, y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size]
        t0 = time.time()
        patch = patch.compute()
        print(f'Time taken to compute patch: {time.time() - t0:.2f} seconds')
        patch = patch.astype(np.float32)
        return torch.from_numpy(patch)
        # # Add lat/lon channels if needed
        # if self.input_lat_lon:
        #     latlon = self.coords_input[:, y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size]
        #     latlon = np.tile(latlon[None, :, :,:], (patch.shape[0], 1,1,1))
        #     return torch.from_numpy(patch), torch.from_numpy(latlon)
        # else:
        #     return torch.from_numpy(patch)

        
        
                 
class DeployDataModel(L.LightningDataModule):

    def __init__(self,
                 pred_fp: str = None,
                 tile_id: str = None,
                 prediction_dir: str = None,
                 batch_size: int = 1,
                 num_workers: int = 8,
                 patch_size:int=512, border:int=16,
                 bands: List[int] = None, 
                 input_lat_lon:bool=False,
                 img_idx:int=None,
                 predict_full_profile:bool=True,
                 comp_level:int=6,
                 compression:str=None,
                 **kwargs
                 ):
        super().__init__()
        self.pred_dataset = S2CogDataset(
            zarr_store_path=pred_fp, 
            tile_id=tile_id, 
            prediction_dir=prediction_dir,
            patch_size=patch_size, 
            border=border, 
            img_idx=img_idx, 
            bands=bands, 
            input_lat_lon=input_lat_lon, 
            predict_full_profile=predict_full_profile,
            comp_level=comp_level,
            compression=compression,
            **kwargs
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.pred_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_batch)
    
    def on_predict_end(self):
        """Convert zarr output to GeoTIFF if needed"""
        if self.use_zarr_output:
            self.pred_dataset.convert_zarr_to_tiff()

# Lightning callback for zarr conversion
class ZarrOutputCallback(L.Callback):
    def __init__(self, wandb_run_id="prediction", convert_to_tiff=True):
        super().__init__()
        self.wandb_run_id = wandb_run_id
        self.convert_to_tiff = convert_to_tiff
    
    def on_predict_start(self, trainer, pl_module):
        # Set prediction filename
        trainer.datamodule.pred_dataset.set_prediction_fname(self.wandb_run_id)
    
    def on_predict_end(self, trainer, pl_module):
        # Convert zarr to tiff if requested
        if self.convert_to_tiff and trainer.datamodule.use_zarr_output:
            trainer.datamodule.pred_dataset.convert_zarr_to_tiff()
            print(f"Converted zarr output to GeoTIFF")


def compare_compressors(tiff_file):
    import subprocess
    import os
    import matplotlib.pyplot as plt
    # Initialize lists to store results
    levels = []
    deflate_sizes = []
    deflate_times = []
    zstd_sizes = []
    zstd_times = []
    jpeg2000_size = None
    jpeg2000_time = None
    ccsds_size = None
    ccsds_time = None

    # TEST DEFLATE compression
    for level in range(1, 10):
        t0 = time.time()
        suffix = f'.DEFLATE.{level}.tif'
        output_file = tiff_file.with_suffix(suffix)

        command = [
            'gdal_translate',
            '-co', 'TILED=YES',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'PREDICTOR=2',
            '-co', f'ZLEVEL={level}',
            '-co', 'NUM_THREADS=ALL_CPUS',
            str(tiff_file),
            str(output_file)
        ]
        subprocess.run(command, check=True)
        time_taken = time.time() - t0
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB

        levels.append(level)
        deflate_sizes.append(file_size)
        deflate_times.append(time_taken)
        print(f'DEFLATE level {level}: {file_size:.2f} MB, time taken: {time_taken:.2f} sec')

    # Test ZSTD compression
    for level in range(1, 23):
        t0 = time.time()
        suffix = f'.ZSTD.{level}.tif'
        output_file = tiff_file.with_suffix(suffix)

        command = [
            'gdal_translate',
            '-co', 'TILED=YES',
            '-co', 'COMPRESS=ZSTD',
            '-co', f'ZSTD_LEVEL={level}',
            '-co', 'NUM_THREADS=ALL_CPUS',
            str(tiff_file),
            str(output_file)
        ]
        subprocess.run(command, check=True)
        time_taken = time.time() - t0
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB

        zstd_sizes.append(file_size)
        zstd_times.append(time_taken)
        print(f'ZSTD level {level}: {file_size:.2f} MB, time taken: {time_taken:.2f} sec')

    # Test JPEG2000 compression
    t0 = time.time()
    suffix = '.JPEG2000.tif'
    output_file = tiff_file.with_suffix(suffix)

    command = [
        'gdal_translate',
        '-co', 'TILED=YES',
        '-of', 'JP2KAK', 
        '-co', 'REVERSIBLE=YES',
        '-co', 'QUALITY=95',
        str(tiff_file),
        str(output_file)
    ]
    subprocess.run(command, check=True)
    jpeg2000_time = time.time() - t0
    jpeg2000_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
    print(f'JPEG2000: {jpeg2000_size:.2f} MB, time taken: {jpeg2000_time:.2f} sec')

    # Test CCSDS compression
    t0 = time.time()
    suffix = '.CCSDS.tif'
    output_file = tiff_file.with_suffix(suffix)

    command = [
        'gdal_translate',
        '-of', 'CCSDS',
        '-co', 'BLOCKSIZE=496',
        str(tiff_file),
        str(output_file)
    ]
    subprocess.run(command, check=True)
    ccsds_time = time.time() - t0
    ccsds_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
    print(f'CCSDS: {ccsds_size:.2f} MB, time taken: {ccsds_time:.2f} sec')

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot file sizes
    plt.subplot(1, 2, 1)
    plt.plot(levels, deflate_sizes, marker='o', label='DEFLATE')
    plt.plot(range(1, 23), zstd_sizes, marker='o', label='ZSTD')
    plt.axhline(y=jpeg2000_size, color='r', linestyle='--', label='JPEG2000')
    plt.axhline(y=ccsds_size, color='g', linestyle='--', label='CCSDS')
    plt.xlabel('Compression Level')
    plt.ylabel('File Size (MB)')
    plt.title('File Size vs Compression Level')
    plt.legend()

    # Plot time taken
    plt.subplot(1, 2, 2)
    plt.plot(levels, deflate_times, marker='o', label='DEFLATE')
    plt.plot(range(1, 23), zstd_times, marker='o', label='ZSTD')
    plt.axhline(y=jpeg2000_time, color='r', linestyle='--', label='JPEG2000')
    plt.axhline(y=ccsds_time, color='g', linestyle='--', label='CCSDS')
    plt.xlabel('Compression Level')
    plt.ylabel('Time Taken (sec)')
    plt.title('Time Taken vs Compression Level')
    plt.legend()

    plt.tight_layout()
    plt.savefig('output/compression_comparison.png')


# Example usage
if __name__ == '__main__':
    # deploy_dataset = ZarrSentinel2Deploy(
    #     zarr_store_path='~/data/GVS/Deploy/inference_2017.zarr',
    #     tile_id='32MQE',
    #     prediction_dir='~/data/GVS/Deploy/32MQE',
    #     use_zarr_output=True,
    #     chunk_size=512
    # )
    # deploy_dataset.set_prediction_fname("cg11fpjr")
    # deploy_dataset.finalize_cog()
    dataset = S2DatasetStream(
        metadata_file='~/data/GVS/Deploy/deploy_s2_items_2020_part0.parquet',
        tile_id='11UMP',
        s2_img_file='~/data/GVS/Deploy/inference_2020.zarr',
        prediction_dir='~/data/GVS/Deploy/32MQE',
        patch_size=512,
        border=16,
    )
    for data in dataset:
        print(data.shape)
        



