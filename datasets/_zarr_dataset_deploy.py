from typing import List
import time
import dask.delayed
import torch
import zarr
import numpy as np
import math
from pathlib import Path
from multiprocessing import shared_memory
from torch.utils.data import Dataset
import lightning as L
import rasterio
from rasterio.windows import Window
import torch.nn.functional as F
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
import dask.array as da
from osgeo import gdal
from dask.distributed import get_client


MASKED_VALUE = {
    'int16': 32767,
    'float16': 65500,
}
RH100_idx = 301
RH98_idx = 295
LAT_MEAN = 12.7596
LAT_STD = 25.6075
LON_SIN_MEAN = 0.1098
LON_SIN_STD = 0.7536
LON_COS_MEAN = 0.3072
LON_COS_STD = 0.5706
gdal.UseExceptions()

class BaseDeployDataset(Dataset):
    """
    Base class for deploy datasets.
    Input data (S2) has nodata value of 0.
    """
    def __init__(self, zarr_store_path, tile_id: str = None, prediction_dir: str = None,
                 patch_size=512, border=16,
                 img_idx: int = None,
                 predict_full_profile: bool = True,
                 bands: List[int] = None, 
                 input_lat_lon=False,  
                 output_dtype='int16',
                 mask_with_scl=True,
                 output_format='cog',
                 chunk_size=512,
                 debug=False,
                 year=None,
                 **kwargs):
        # ********** Set up parameters **********
        self.mask_with_scl = mask_with_scl
        self.patch_size = patch_size
        self.border = border
        self.patch_size_no_border = self.patch_size - 2 * self.border
        self.input_lat_lon = input_lat_lon
        self.output_dtype = output_dtype
        self.output_format = output_format
        self.chunk_size = chunk_size
        self.debug = debug
        self.predict_full_profile = predict_full_profile
        self.year = year
        
        # ********** Set up RH & image indices **********
        if predict_full_profile:
            self.rh_idx = slice(0, 303)
            self.rh_dim = 303
        else:
            self.rh_idx = [RH100_idx, RH98_idx]
            self.rh_dim = 2
        self.tile_id = tile_id
        self.bands = bands or slice(12)
        
        if img_idx is None:
            self.img_idx = slice(None)
        else:
            self.img_idx = slice(img_idx, img_idx + 1)
            
        # ********** Open image and get metadata **********
        zarr_store_path = Path(zarr_store_path).expanduser()
        try:
            self.store = zarr.open(zarr_store_path, mode='r')
        except Exception as e:
            print(f'Error opening zarr store: {e}, conda activate py3!')
        if zarr_store_path.name.endswith('.zarr'):
            self.key = f'{self.tile_id}/'
            self.transform = self.store[f'{self.tile_id}'].attrs['transform']
        else:
            self.key = '' # open a group as a store
            self.transform = self.store.attrs['transform']

        self.img_width, self.img_height = self.store[f'{self.key}s2'].shape[2:]  # Get shape without loading array
        self.crs = f'EPSG:{self.store[f"{self.key}epsg"][()].item()}'
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        
        if self.input_lat_lon:
            lon_mask = self.store[f'{self.key}x'][:].astype(np.float32) # in local crs
            lat_mask = self.store[f'{self.key}y'][:].astype(np.float32)
            lon_mask, lat_mask = transformer.transform(lon_mask, lat_mask, radians=True)
            lon, lat = np.meshgrid(lon_mask, lat_mask)
            sin_lon = np.sin(lon)
            cos_lon = np.cos(lon)
            lat = (lat - LAT_MEAN) / LAT_STD
            sin_lon = (sin_lon - LON_SIN_MEAN) / LON_SIN_STD
            cos_lon = (cos_lon - LON_COS_MEAN) / LON_COS_STD
            self.coords_input = np.stack([lat, sin_lon, cos_lon], axis=0).astype(np.float32)

        self.patch_coords_dict = self._get_patch_coords()
        print('--------------------------------')
        print(f"Image shape: {self.img_width}, {self.img_height}")
        print(f"Number of patches: {len(self.patch_coords_dict)}")
        print('--------------------------------')
        
        # ********** Prediction Configuration **********
        self.nodata_value = MASKED_VALUE[self.output_dtype]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # cloud shadows, CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY, SNOW, water, nodata
        self.scl_exclude_labels = torch.tensor([0, 1, 3, 8, 9, 10, 11, 65535], dtype=torch.uint16, device=device) # scl is uint16
        # self.esa_exclude_labels = torch.tensor([5, 8], dtype=torch.uint8, device=device) # built-up, water
        self.scl_water = 6
        self.esa_snow = 7
        self.esa_built_up = 5
        self.esa_water = 8
        # self.scl_zero_canopy_height = torch.tensor([5, 6], dtype=torch.uint16, device=device)  # "not vegetated", "water"
        self.prediction_cache = {}
        
        if not prediction_dir:
            self.prediction_dir = zarr_store_path.parent / f'predictions_{self.year}' / self.tile_id
        else:
            self.prediction_dir = Path(prediction_dir).expanduser()
        self.prediction_dir.mkdir(exist_ok=True)
    
    def __len__(self):
        return len(self.patch_coords_dict)
    
    def __getitem__(self, idx):
        img_batch_idx, y_topleft, x_topleft = self.patch_coords_dict[idx]
        # img_batch_idx is either :10 or 10:
        # Extract patch with border handling
        if y_topleft < 0:
            pad_h = (self.border, 0)
        elif y_topleft + self.patch_size >= self.img_height:
            pad_h = (0, self.border)
        else:
            pad_h = (0, 0)
        if x_topleft < 0:
            pad_w = (self.border, 0)
        elif x_topleft + self.patch_size >= self.img_width:
            pad_w = (0, self.border)
        else:
            pad_w = (0, 0)

        patch = self.store[f'{self.key}s2'][img_batch_idx, :,  max(y_topleft, 0):y_topleft+self.patch_size, max(x_topleft, 0):x_topleft+self.patch_size]
        patch = np.pad(patch, ((0,0), (0,0), pad_h, pad_w), mode='symmetric')
        image = patch[:, :-1, :, :]
        scl = patch[:, -1:, :, :]
        
        if self.input_lat_lon:
            latlon = self.coords_input[:, max(y_topleft, 0):y_topleft+self.patch_size, max(x_topleft, 0):x_topleft+self.patch_size]
            latlon = np.tile(latlon[None, :, :,:], (image.shape[0], 1,1,1))
            latlon = np.pad(latlon, ((0,0), (0,0), pad_h, pad_w), mode='symmetric')
            return torch.from_numpy(image), torch.from_numpy(scl), torch.from_numpy(latlon)
        else:
            return torch.from_numpy(image), torch.from_numpy(scl)

    def _get_patch_coords(self):
        """Calculate patch coordinates with overlap handling"""
        y_dim = self.img_height + 2 * self.border
        x_dim = self.img_width + 2 * self.border
        col_steps = int(math.ceil(y_dim / self.patch_size_no_border))
        row_steps = int(math.ceil(x_dim / self.patch_size_no_border))
        col_steps = 5 if self.debug else col_steps
        row_steps = 5 if self.debug else row_steps
        patch_coords_dict = {}
        patch_idx = 0
        for y in range(col_steps):
            y_coord = y * self.patch_size_no_border
            if y_coord > y_dim - self.patch_size:
                # move last patch up if it would exceed the image bottom
                y_coord = y_dim - self.patch_size
            for x in range(row_steps):
                x_coord = x * self.patch_size_no_border
                if x_coord > x_dim - self.patch_size:
                    # move last patch left if it would exceed the image right border
                    x_coord = x_dim - self.patch_size
                patch_coords_dict[patch_idx] = (slice(0,10), y_coord - self.border, x_coord - self.border)
                patch_coords_dict[patch_idx+1] = (slice(10,20), y_coord - self.border, x_coord - self.border)
                patch_idx += 2
        return patch_coords_dict

    
    def _apply_masks(self, prediction, scl, x_topleft, y_topleft):
        
        prediction_no_border = prediction[:, self.rh_idx, self.border:self.patch_size - self.border, self.border:self.patch_size - self.border]
        esa_wc = prediction[:, -12:, self.border:self.patch_size - self.border, self.border:self.patch_size - self.border]
        esa_wc = torch.argmax(esa_wc, dim=1, keepdim=True).to(torch.float32)
        scl = scl[:, :, self.border:self.patch_size - self.border, self.border:self.patch_size - self.border].to(torch.float32)
        
        location_key = f'{y_topleft}_{x_topleft}'
        if location_key not in self.prediction_cache:
            self.prediction_cache[location_key] = prediction_no_border
            self.prediction_cache[f'{location_key}_scl'] = scl
            self.prediction_cache[f'{location_key}_esa_wc'] = esa_wc
            return None
        else: # ready to write
            prediction_no_border = torch.cat([self.prediction_cache[location_key], prediction_no_border], dim=0)
            scl = torch.cat([self.prediction_cache[f'{location_key}_scl'], scl], dim=0)
            esa_wc = torch.cat([self.prediction_cache[f'{location_key}_esa_wc'], esa_wc], dim=0)
            if self.mask_with_scl:
                # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
                scl_mask = torch.isin(scl, self.scl_exclude_labels)
                
            # masks applied to all input images
            nodata_mask = scl == 0
            scl[nodata_mask] = float('nan')
            esa_wc[nodata_mask] = float('nan')
            nodata_mask = nodata_mask.repeat(1, 303, 1, 1)
            prediction_no_border[nodata_mask] = float('nan') # mask the prediction when input is nodata
            water_mask_scl = torch.mode(scl, dim=0).values == self.scl_water
            
            built_up_mask = torch.mode(esa_wc, dim=0).values == self.esa_built_up
            water_mask_esa = torch.mode(esa_wc, dim=0).values == self.esa_water
            esa_wc_mask = esa_wc == self.esa_snow
            prediction_no_border = torch.where(scl_mask | esa_wc_mask, torch.nan, prediction_no_border)
            prediction_no_border, _ = torch.nanmedian(prediction_no_border, dim=0)
            prediction_no_border = torch.where(water_mask_scl | water_mask_esa | built_up_mask, torch.nan, prediction_no_border)
            prediction_no_border.mul_(10).round_()  # In-place operations
            prediction_no_border = torch.nan_to_num(prediction_no_border, nan=self.nodata_value)            
            # Move to CPU once and do all numpy operations together
            prediction_no_border = prediction_no_border.cpu().numpy().astype(self.output_dtype)
            return prediction_no_border 
        
     
class CachedDeployDataset(BaseDeployDataset):
    """
    CachedDeployDataset is a subclass of BaseDeployDataset that caches the predictions.

    Args:
        BaseDeployDataset (_type_): _description_
    """
    
    def __init__(self, compression=None, comp_level=6, **kwargs):
        super().__init__(**kwargs)
        self.compression = compression
        self.comp_level = comp_level
        self.shm = shared_memory.SharedMemory(name=f'full_pred_{self.tile_id}')
        self.full_pred = np.ndarray((self.rh_dim, self.img_height, self.img_width), dtype=self.output_dtype, buffer=self.shm.buf)
        self.prediction_fp = self.prediction_dir / f'{self.tile_id}'
        self.prediction_fp.mkdir(exist_ok=True)
        
    def write_patch_predictions(self, patch_pred, scl, idx):
        y_topleft, x_topleft = self.patch_coords_dict[idx][1:]
        y_topleft = y_topleft + self.border
        x_topleft = x_topleft + self.border     
        prediction_no_border = self._apply_masks(patch_pred, scl, x_topleft, y_topleft)
        if prediction_no_border is None:
            return
        self.full_pred[:,y_topleft:y_topleft + self.patch_size_no_border, x_topleft:x_topleft + self.patch_size_no_border] = prediction_no_border
        del self.prediction_cache[f'{y_topleft}_{x_topleft}']
        del self.prediction_cache[f'{y_topleft}_{x_topleft}_scl']
            
            
    def save_predictions(self, *args, **kwargs):
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
            nodata=self.nodata_value,  # Important: add nodata value
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
        if self.output_format == 'cog':
            write_func = write_prediction_as_cog
        else:
            write_func = write_prediction_as_tif
            
        print(dst_profile)
        # Save as COG
        t0 = time.time()
        futures = []
        if self.predict_full_profile:
            for i in range(self.full_pred.shape[0]):
                rh_idx = i // 3
                q_idx = i % 3
                pred_fp = self.prediction_fp.with_stem(f'RH{rh_idx}_Q{q_idx}_uncompressed')
                futures.append(write_func(self.full_pred[i], src_profile, dst_profile, pred_fp))
            dask.compute(*futures)
        else:
            for i, idx in enumerate(self.rh_idx):
                rh_idx = idx // 3
                pred_fp = self.prediction_fp.with_stem(f'RH{rh_idx}_Q1_uncompressed')
                futures.append(write_func(self.full_pred[i], src_profile, dst_profile, pred_fp))
            dask.compute(*futures)
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

class ChunkedWriteDataset(BaseDeployDataset):
    """
    ChunkedWriteDataset is a subclass of BaseDeployDataset that writes predictions to a file in chunks.

    Args:
        BaseDeployDataset (_type_): _description_
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prediction_fp = self.prediction_dir / f'{self.tile_id}'
        self.transform = Affine(*self.transform).to_gdal()
        
    def initialize_output(self):
        self.options = [
            'TILED=YES',
            'BLOCKXSIZE={}'.format(self.chunk_size), # a different block size than the actual patch size is slow
            'BLOCKYSIZE={}'.format(self.chunk_size),
            'PREDICTOR=2',
            'NUM_THREADS=ALL_CPUS',
            'COMPRESS=None',
            'INTERLEAVE=BAND'
        ]
        if self.predict_full_profile:
            output_files = [self.prediction_fp.with_stem(f'RH{i}_Q{j}_uncompressed') for i in range(self.rh_dim//3) for j in range(3)]
        else:
            output_files = [self.prediction_fp.with_stem(f'RH{i//3}_Q1_uncompressed') for i in self.rh_idx]
        
        self.tiff_writers = [self.init_gtiff(output_file) for output_file in output_files]
        dtype = np.dtype(
            [("raster_writer", gdal.Dataset), ("array", 'float16', (self.patch_size_no_border, self.patch_size_no_border))]
        )
        self.pred_table = np.full((self.rh_dim), None, dtype=dtype)
        
    def write_patch_predictions(self, prediction,scl, idx):
        y_topleft, x_topleft = self.patch_coords_dict[idx][1:]
        y_topleft = y_topleft + self.border
        x_topleft = x_topleft + self.border
        prediction_no_border = self._apply_masks(prediction, scl, x_topleft, y_topleft)
        if prediction_no_border is None:
            return
        for i in range(self.rh_dim):
            self.pred_table[i] = (self.tiff_writers[i], prediction_no_border[i])
        darr = da.from_array(self.pred_table, chunks=(1,))
        darr = darr.map_blocks(write_patch_predictions, x_topleft, y_topleft, self.nodata_value, meta=np.array((1,2), dtype=darr.dtype))
        darr.compute(scheduler='threads')
        del self.prediction_cache[f'{y_topleft}_{x_topleft}']
        del self.prediction_cache[f'{y_topleft}_{x_topleft}_scl']
        del self.prediction_cache[f'{y_topleft}_{x_topleft}_esa_wc']


    def init_gtiff(self, prediction_fp: Path):
        prediction_fp = prediction_fp.with_suffix('.tif')
        if prediction_fp.exists():
            os.remove(prediction_fp)
        driver = gdal.GetDriverByName('GTiff')
        tiff_output = driver.Create(
            str(prediction_fp),
            xsize=self.img_width,
            ysize=self.img_height,
            bands=1,
            eType=gdal.GDT_Int16,
            options=self.options
        )
        tiff_output.SetGeoTransform(self.transform)
        tiff_output.SetProjection(self.crs)
        tiff_output.SetMetadataItem('Year', str(self.year))
        tiff_output.SetMetadataItem('Sentinel-2 tile', self.tile_id)
        return tiff_output

def write_patch_predictions(entry, x_topleft, y_topleft, nodata_value):
    raster_writer, array = entry[0]
    band = raster_writer.GetRasterBand(1)
    band.WriteArray(array, 
                    xoff=x_topleft, 
                    yoff=y_topleft)
    filename = raster_writer.GetDescription()
    filename = Path(filename)
    rh_idx = int(filename.stem.split('_')[0].split('RH')[1])
    q_idx = int(filename.stem.split('_')[1].split('Q')[1])
    band.SetDescription(f"RH{rh_idx}_Q{q_idx}")
    band.SetNoDataValue(nodata_value)
    raster_writer.FlushCache()


# *************************************************
# **************** For Cached Write ***************
# *************************************************

@dask.delayed
def write_prediction_as_cog(pred, src_profile, dst_profile, prediction_fp):
    with MemoryFile() as memfile:
        # Write data to memory file and close it properly
        with memfile.open(**src_profile) as dst:
            dst.write(pred, indexes=1)
        # File is now closed and ready for translation
        
        output_path = prediction_fp.with_suffix(f'.tif')
        print(f'Saving {output_path}')
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
   
   
def write_prediction_as_tif(pred, src_profile, dst_profile, prediction_fp):
    output_path = prediction_fp.with_suffix(f'.tif')
    with rasterio.open(output_path, 'w', **src_profile) as dst:
        dst.write(pred, indexes=1)
    print('--------------------------------')
    print(f'Saving {output_path}')
    print('--------------------------------')

 
def collate_batch(batch):
    return batch[0]

        
class S2DatasetStream(BaseDeployDataset):
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
                 n_iamges_per_tile: int = 20,
                 **kwargs):
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
        self.img_width, self.img_height = self.image.shape[2:]
        self.transform = self.image.transform
        self.crs = self.image.crs
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
                 cache_predictions: bool = False,
                 stream_input: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.cache_predictions = cache_predictions
        if stream_input:
            self.pred_dataset = S2DatasetStream(
                metadata_file=pred_fp,
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
        elif cache_predictions:
            self.pred_dataset = CachedDeployDataset(
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
        else:   
            self.pred_dataset = ChunkedWriteDataset(
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
        print(f'batch_size: {self.batch_size}, num_workers: {self.num_workers}')
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.pred_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_batch,
            # pin_memory=True,
            prefetch_factor=2)
    

# Example usage
if __name__ == '__main__':
    dataset = BaseDeployDataset(
        zarr_store_path='~/data/GVS/Deploy/inference_2020.zarr',
        tile_id='32MQE',
        prediction_dir='~/data/GVS/Deploy/32MQE',
        patch_size=544,
        border=16,
        comp_level=7,
    )
    # dataset = S2DatasetStream(
    #     metadata_file='~/data/GVS/Deploy/deploy_s2_items_2020_part0.parquet',
    #     tile_id='11UMP',
    #     s2_img_file='~/data/GVS/Deploy/inference_2020.zarr',
    #     prediction_dir='~/data/GVS/Deploy/32MQE',
    #     patch_size=512,
    #     border=16,
    # )
    for data in dataset:
        print(data[0].shape)
        



