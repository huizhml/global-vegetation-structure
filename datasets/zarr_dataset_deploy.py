
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
from osgeo import gdal
import lightning as L
import rasterio
from rasterio.windows import Window
import torch.nn.functional as F
from pyproj import Transformer
import pandas as pd
import os
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rasterio.transform import Affine
import geopandas as gpd
import dask
import dask.array as da
import h5py
import xarray as xr
from pystac_client.stac_api_io import StacApiIO
import planetary_computer
import pystac_client
from download.core.stackstac_lib import stack
from download.core.constants import S2_ITEM_PROPS
from download.core.utils import get_patch, row_to_stac_item, harmonize_to_old
from download.core.items_store import read_items_for_tiles
from datasets.per_image_report import per_image_dir_name, write_per_image_report
from const import SCL_EXCLUDE_LABELS, SCL_WATER, ESA_BUILT_UP, ESA_WATER, ESA_SNOW


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
LON_COS_STD = 0.5706 # TODO: load from file
gdal.UseExceptions()


def rh_channel(var: str) -> int:
    """Model output channel of a variable name, e.g. 'RH98_Q1' -> 98 * 3 + 1 = 295."""
    rh, q = var.split('_')
    return int(rh[2:]) * 3 + int(q[1:])


class BaseDeployDataset(Dataset):
    """
    Base class for deploy datasets.
    Input data (S2) has nodata value of 0.
    """
    def __init__(self, zarr_store_path, tile_id: str = None, prediction_dir: str = None,
                 metadata_file: str = None,
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
                 max_images: int = None,
                 images_per_forward: int = 10,
                 timeseries: bool = False,
                 timeseries_vars: List[str] = None,
                 **kwargs):
        # ********** Set up parameters **********
        self.max_images = max_images # None = use every image in the store
        self.images_per_forward = images_per_forward # model batch size; bounds memory, not the image count
        self.timeseries = timeseries # True: one band per image instead of the per-pixel median
        self.timeseries_vars = list(timeseries_vars or ['RH25_Q1', 'RH98_Q1'])
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
        self._set_output_channels(predict_full_profile)
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
            try:
                self.transform = self.store[f'{self.tile_id}'].attrs['transform']
            except Exception as e:
                print('No ESA WC, attrs in s2')
                self.transform = self.store[f'{self.tile_id}/s2'].attrs['transform']
        else:
            self.key = '' # open a group as a store
            self.transform = self.store.attrs['transform']
        self.img_width, self.img_height = self.store[f'{self.key}s2'].shape[2:]  # Get shape without loading array
        self.crs = f'EPSG:{self.store[f"{self.key}epsg"][()].item()}'
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        n_images = self.store[f'{self.key}s2'].shape[0]
        self.img_sel = self._select_images(n_images, metadata_file)

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
        self.scl_exclude_labels = torch.tensor(SCL_EXCLUDE_LABELS, dtype=torch.uint16, device=device) # scl is uint16
        # self.esa_exclude_labels = torch.tensor([5, 8], dtype=torch.uint8, device=device) # built-up, water
        # self.scl_zero_canopy_height = torch.tensor([5, 6], dtype=torch.uint16, device=device)  # "not vegetated", "water"

        if not prediction_dir:
            self.prediction_dir = zarr_store_path.parent / f'predictions_{self.year}' / self.tile_id
        else:
            self.prediction_dir = Path(prediction_dir).expanduser()
        self.prediction_dir.mkdir(exist_ok=True)
    
    def __len__(self):
        return len(self.patch_coords_dict)

    def _set_output_channels(self, predict_full_profile):
        """Model output channels kept after each forward (crop_prediction)."""
        if self.timeseries:
            # only the requested variables, so GPU memory stays small with many images
            self.rh_idx = [rh_channel(v) for v in self.timeseries_vars]
            self.rh_dim = len(self.rh_idx)
        elif predict_full_profile:
            self.rh_idx = slice(0, 303)
            self.rh_dim = 303
        else:
            self.rh_idx = [RH100_idx, RH98_idx]
            self.rh_dim = 2
    
    def _select_images(self, n_images, metadata_file=None):
        """Store indices of the images to predict on, in the order they are stacked.

        With a metadata file, images are ranked by nodata % then cloud cover and
        duplicates in the store are dropped; otherwise they keep store order.
        `max_images` keeps the first N (None = all), e.g. 20 for the global deploy.
        """
        if metadata_file is None or metadata_file == 'none':
            return slice(0, self.max_images)
        self.metadata_file = Path(metadata_file).expanduser()
        df = read_items_for_tiles(self.metadata_file, [self.tile_id],
                                  columns=['id', 'eo:cloud_cover', 's2:mgrs_tile', 's2:nodata_pixel_percentage', 'orbit'])
        if len(df) == 0:
            raise RuntimeError(f'No images found for tile {self.tile_id} in metadata file {self.metadata_file}')
        ids_from_zarr = self.store[f'{self.key}id'][:]
        df = df[df['id'].isin(ids_from_zarr)]
        df = df.reset_index(drop=True)
        df['s2:nodata_pixel_percentage'] = df['s2:nodata_pixel_percentage'].round()
        df = df.sort_values(['s2:nodata_pixel_percentage', 'eo:cloud_cover']).iloc[:self.max_images]
        return [np.where(ids_from_zarr == i)[0][0] for i in df['id']] # there exists duplicated images in the zarr store

    def __getitem__(self, idx):
        # One sample = every selected image at one patch location; the model
        # splits it into chunks of images_per_forward.
        y_topleft, x_topleft = self.patch_coords_dict[idx]
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

        patch = self.store[f'{self.key}s2'][self.img_sel, :,  max(y_topleft, 0):y_topleft+self.patch_size, max(x_topleft, 0):x_topleft+self.patch_size]
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
                patch_coords_dict[patch_idx] = (y_coord - self.border, x_coord - self.border)
                patch_idx += 1
        return patch_coords_dict

    def crop_prediction(self, prediction):
        """Keep what _apply_masks needs from one forward chunk: the RH channels and
        the ESA WorldCover class, border removed.

        Called right after each forward so the full output (with border and all
        channels) can be freed before the next chunk; contiguous() makes the crop
        its own copy rather than a view that would keep the full tensor alive.
        """
        inner = slice(self.border, self.patch_size - self.border)
        prediction_no_border = prediction[:, self.rh_idx, inner, inner].contiguous()
        esa_wc = torch.argmax(prediction[:, -12:, inner, inner], dim=1, keepdim=True).to(torch.int8)
        return prediction_no_border, esa_wc

    def _apply_masks(self, prediction_no_border, esa_wc, scl, keep_images=False):
        """Mask and take the per-pixel median over all images of one patch location
        (or keep every image when timeseries=True or keep_images=True)."""
        scl = scl[:, :, self.border:self.patch_size - self.border, self.border:self.patch_size - self.border]

        if self.mask_with_scl:
            # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
            scl_mask = torch.isin(scl, self.scl_exclude_labels)
            
        # 1. do majority vote for scl water, esa built-up, esa water, and apply these masks to the final aggregated prediction
        # 2. apply image specific masks from scl about cloud and snow, and esa wc snow mask
        nodata_mask = scl == 0
        scl = scl.to(torch.float32)
        scl[nodata_mask] = float('nan')
        esa_wc = esa_wc.to(torch.float32)
        esa_wc[nodata_mask] = float('nan')
        nodata_mask = nodata_mask.repeat(1, prediction_no_border.shape[1], 1, 1)
        prediction_no_border[nodata_mask] = float('nan') # mask the prediction when input is nodata
        water_mask_scl = torch.mode(scl, dim=0).values == SCL_WATER #!!! this is not nan mode,mode value could be nan, resulting in all-false water mask
        
        # NOTE: this cannot mask all water pixels neither, there are cloud pixels, scl shows as cloud or even vegetation instead of water, 
        # and the predicted land cover also fails because of the cloud pixels.
        # water_mask_scl = (scl == self.scl_water).any(dim=0) 
        built_up_mask = torch.mode(esa_wc, dim=0).values == ESA_BUILT_UP
        water_mask_esa = torch.mode(esa_wc, dim=0).values == ESA_WATER # predicted esa wc
        esa_wc_mask = esa_wc == ESA_SNOW
        prediction_no_border = torch.where(scl_mask | esa_wc_mask, torch.nan, prediction_no_border)
        if self.timeseries or keep_images:
            pass  # keep every image: (n_images, n_vars, h, w); water/built-up masks below are still per tile
        elif self.save_intermediate_tif:
            prediction_no_border = prediction_no_border[:, 295, :, :]
        else:
            prediction_no_border, _ = torch.nanmedian(prediction_no_border, dim=0)
        prediction_no_border = torch.where(water_mask_scl | water_mask_esa | built_up_mask, torch.nan, prediction_no_border)
        prediction_no_border.mul_(10).round_()  # In-place operations
        prediction_no_border = torch.nan_to_num(prediction_no_border, nan=self.nodata_value)            
        # Move to CPU once and do all numpy operations together
        prediction_no_border = prediction_no_border.cpu().numpy().astype(self.output_dtype)
        return prediction_no_border 
        

    def _image_dates_ids(self):
        """Date (YYYY-MM-DD) and S2 id of each predicted image, in the order they are stacked."""
        from cftime import num2date
        tvar = self.store[f'{self.key}time']
        dates = num2date(tvar[:], tvar.attrs['units'], calendar=tvar.attrs.get('calendar', 'standard'))
        ids = self.store[f'{self.key}id'][:]
        img_idx = np.arange(len(dates))[self.img_sel]
        return ([dates[i].strftime('%Y-%m-%d') for i in img_idx],
                [ids[i].decode() if isinstance(ids[i], bytes) else str(ids[i]) for i in img_idx])

    def _init_timeseries_output(self):
        """One GeoTIFF per variable (e.g. RH98_Q1.tif) with one band per image, on the
        same grid, scale and nodata as the yearly product; images.csv maps bands to images."""
        dates, ids = self._image_dates_ids()
        pd.DataFrame({'band': np.arange(1, len(dates) + 1), 'date': dates, 's2_id': ids}).to_csv(
            self.prediction_dir / 'images.csv', index=False)
        writers = []
        for var in self.timeseries_vars:
            writer = self.init_gtiff(self.prediction_fp.with_stem(var), n_bands=len(dates))
            for b, (date, img_id) in enumerate(zip(dates, ids), start=1):
                band = writer.GetRasterBand(b)
                band.SetDescription(f'{date}_{img_id}')
                band.SetNoDataValue(self.nodata_value)
            writers.append(writer)
        print(f'Time series: {len(dates)} images x {self.timeseries_vars} -> {self.prediction_dir}')
        return writers

    def _write_timeseries_patch(self, prediction, esa_wc, scl, idx):
        y_topleft, x_topleft = self.patch_coords_dict[idx]
        prediction_no_border = self._apply_masks(prediction, esa_wc, scl)  # (n_images, n_vars, h, w)
        for v, writer in enumerate(self.tiff_writers):
            for b in range(prediction_no_border.shape[0]):
                writer.GetRasterBand(b + 1).WriteArray(prediction_no_border[b, v],
                                                       xoff=x_topleft + self.border, yoff=y_topleft + self.border)

     
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
        
    def write_patch_predictions(self, patch_pred, esa_wc, scl, idx):
        y_topleft, x_topleft = self.patch_coords_dict[idx]
        y_topleft = y_topleft + self.border
        x_topleft = x_topleft + self.border
        prediction_no_border = self._apply_masks(patch_pred, esa_wc, scl)
        self.full_pred[:,y_topleft:y_topleft + self.patch_size_no_border, x_topleft:x_topleft + self.patch_size_no_border] = prediction_no_border
            
            
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
                pred_fp = self.prediction_fp.with_stem(f'RH{rh_idx}_Q{q_idx}')
                futures.append(write_func(self.full_pred[i], src_profile, dst_profile, pred_fp))
            dask.compute(*futures)
        else:
            for i, idx in enumerate(self.rh_idx):
                rh_idx = idx // 3
                pred_fp = self.prediction_fp.with_stem(f'RH{rh_idx}_Q1')
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
    
    def __init__(self, save_intermediate_tif: bool = False, diagnostics_dir: str = '~/data/gvs/deploy/per_image_predictions', **kwargs):
        super().__init__(**kwargs)
        self.prediction_fp = self.prediction_dir / f'{self.tile_id}'
        self.transform = Affine(*self.transform).to_gdal()
        self.save_intermediate_tif = save_intermediate_tif # for debugging purposes
        self.diagnostics_dir = Path(diagnostics_dir).expanduser() # root of per-image debug reports
        self.zarr_store_path = kwargs.get('zarr_store_path')
        
    def initialize_output(self):
        self.options = [
            'TILED=YES',
            'BLOCKXSIZE={}'.format(self.chunk_size), # a different block size than the actual patch size is slow
            'BLOCKYSIZE={}'.format(self.chunk_size),
            # 'PREDICTOR=2',
            'NUM_THREADS=ALL_CPUS',
            'COMPRESS=None',
            'INTERLEAVE=BAND'
        ]
        if self.timeseries:
            self.tiff_writers = self._init_timeseries_output()
            return
        if self.predict_full_profile:
            output_files = [self.prediction_fp.with_stem(f'RH{i}_Q{j}') for i in range(self.rh_dim//3) for j in range(3)]
        else:
            output_files = [self.prediction_fp.with_stem(f'RH{i//3}_Q1') for i in self.rh_idx]
        if self.save_intermediate_tif:
            output_files = self._init_per_image_output()
        
        self.tiff_writers = [self.init_gtiff(output_file) for output_file in output_files]
        dtype = np.dtype(
            [("raster_writer", gdal.Dataset), ("array", np.int16, (self.patch_size_no_border, self.patch_size_no_border))]
        )
        self.pred_table = np.empty((len(output_files)), dtype=dtype)
        
    def write_patch_predictions(self, prediction, esa_wc, scl, idx):
        if self.timeseries:
            return self._write_timeseries_patch(prediction, esa_wc, scl, idx)
        y_topleft, x_topleft = self.patch_coords_dict[idx]
        y_topleft = y_topleft + self.border
        x_topleft = x_topleft + self.border
        prediction_no_border = self._apply_masks(prediction, esa_wc, scl)
        for i in range(len(self.tiff_writers)):
            self.pred_table[i] = (self.tiff_writers[i], prediction_no_border[i])
        darr = da.from_array(self.pred_table, chunks=(1,))
        darr = darr.map_blocks(write_patch_predictions, x_topleft, y_topleft, self.nodata_value, meta=np.array((1,2), dtype=darr.dtype))
        darr.compute(scheduler='threads')


    def init_gtiff(self, prediction_fp: Path, n_bands: int = 1):
        prediction_fp = prediction_fp.with_suffix('.tif')
        if prediction_fp.exists():
            os.remove(prediction_fp)
        driver = gdal.GetDriverByName('GTiff')
        tiff_output = driver.Create(
            str(prediction_fp),
            xsize=self.img_width,
            ysize=self.img_height,
            bands=n_bands,
            eType=gdal.GDT_Int16,
            options=self.options
        )
        tiff_output.SetGeoTransform(self.transform)
        tiff_output.SetProjection(self.crs)
        tiff_output.SetMetadataItem('Year', str(self.year))
        tiff_output.SetMetadataItem('Sentinel-2 tile', self.tile_id)
        return tiff_output

    def _init_per_image_output(self):
        """One RH98 tif per predicted image, inside a dated report folder (see datasets/per_image_report.py)."""
        from cftime import num2date
        tvar = self.store[f'{self.key}time']
        dates = num2date(tvar[:], tvar.attrs['units'], calendar=tvar.attrs.get('calendar', 'standard'))
        ids = self.store[f'{self.key}id'][:]
        # predictions are stacked in img_sel order, so outputs must follow the same order
        img_idx = np.arange(len(dates))[self.img_sel]
        self.per_image_meta = pd.DataFrame({
            'img_idx': img_idx,
            'date': [dates[i].strftime('%Y-%m-%d') for i in img_idx],
            'id': [ids[i].decode() if isinstance(ids[i], bytes) else str(ids[i]) for i in img_idx],
        })
        # img_idx keeps names unique when two images share a date
        self.per_image_meta['name'] = [f'{self.tile_id}_{d}_img{i:03d}'
                                       for d, i in zip(self.per_image_meta['date'], img_idx)]
        self.per_image_dir = self.diagnostics_dir / per_image_dir_name(self.tile_id, self.year)
        (self.per_image_dir / 'tifs').mkdir(parents=True, exist_ok=True)
        print(f'Per-image predictions will be saved to {self.per_image_dir}')
        return [self.per_image_dir / 'tifs' / name for name in self.per_image_meta['name']]

    def finalize_output(self):
        """Called once after the last batch: write the per-image report, then close all GDAL datasets."""
        if self.save_intermediate_tif and not self.timeseries:
            write_per_image_report(
                out_dir=self.per_image_dir,
                meta=self.per_image_meta,
                tiff_writers=self.tiff_writers,
                s2_array=self.store[f'{self.key}s2'],
                nodata_value=self.nodata_value,
                inputs={
                    'tile_id': self.tile_id,
                    'year': self.year,
                    'zarr_store': self.zarr_store_path,
                    'patch_size': self.patch_size,
                    'border': self.border,
                    'mask_with_scl': self.mask_with_scl,
                    'nodata_value': self.nodata_value,
                },
                title=f'Per-image predictions {self.tile_id} {self.year}',
            )
        for writer in self.tiff_writers:
            writer.FlushCache()
        self.tiff_writers = None
        self.pred_table = None

def write_patch_predictions(entry, x_topleft, y_topleft, nodata_value):
    raster_writer, array = entry[0]
    band = raster_writer.GetRasterBand(1)
    band.WriteArray(array, xoff=x_topleft, yoff=y_topleft)
    filename = raster_writer.GetDescription()
    filename = Path(filename)
    if 'uncompressed' in filename.stem:
        date = filename.stem.split('_')[1]
        band.SetDescription(f"{date}")
    elif not filename.stem.startswith('RH'):
        # per-image debug output: <tile>_<date>_img<idx>
        band.SetDescription(filename.stem)
    else:
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


def newest_processing(df):
    """Keep the newest processing of each acquisition.

    The STAC search returns some acquisitions twice, the original processing and
    the 2024 reprocessing of the archive. Their ids (product URI without the
    N05xx baseline) differ only in the trailing processing time, e.g.
    S2B_MSIL2A_20230115T032059_R118_T48PWV_20230115T141952 and ..._20240810T232727.
    """
    parts = df['id'].str.rsplit('_', n=1, expand=True)  # 0: acquisition, 1: processing time
    return df[parts[1] == parts.groupby(0)[1].transform('max')]

        
class S2DatasetStream(BaseDeployDataset):
    def __init__(self, 
                 metadata_file: str, 
                 s2_grid_file: str,
                 h5_dir: str,
                 tile_id: str = None, 
                 prediction_dir: str = None,
                 patch_size=512, border=16,
                 img_idx: int = None,
                 predict_full_profile: bool = True,
                 debug: bool = False,
                 input_lat_lon=False,
                 mask_with_scl=True,
                 n_iamges_per_tile: int = 20,
                 year: int = 2020,
                 chunk_size: int = 512,
                 output_format: str = 'cog',
                 output_dtype: str = 'int16',
                 impute_cloud_with_mean: bool = False,
                 download_data: bool = False,
                 save_intermediate_tif: bool = False,
                 max_images: int = None,
                 images_per_forward: int = 10,
                 timeseries: bool = False,
                 timeseries_vars: List[str] = None,
                 max_cloud_cover: float = 90,
                 months: List[int] = None,
                 crop_bounds: List[float] = None,
                 crop_epsg: int = None,
                 crop_name: str = None,
                 sample_points_file: str = None,
                 sample_output_file: str = None,
                 sample_per_image_vars: List[str] = None,
                 **kwargs):
        # sample_points_file (parquet with lon, lat): keep the yearly prediction only at
        # these pixels and write it to sample_output_file (the points' columns plus
        # RH<r>_Q<q>, int16 dm, nodata as in the GeoTIFF) instead of writing GeoTIFFs.
        # Same masks / median / order of patches as the GeoTIFF, so the values are equal.
        self.sample_points_file = sample_points_file
        self.sample_output_file = sample_output_file
        # sample_per_image_vars (e.g. [RH98_Q1]): also keep these variables of every image at the
        # points, masked as for the median -> <sample_output_file stem>_per_image.parquet
        self.sample_per_image_vars = list(sample_per_image_vars or [])
        if sample_points_file and not sample_output_file:
            raise ValueError('sample_points_file needs sample_output_file')
        # Optional window instead of the whole tile: [xmin, ymin, xmax, ymax] in the
        # tile's UTM CRS (crop_epsg), saved as <h5_dir>/<crop_name>.h5. Images are
        # still picked for the whole tile, so they match the full-tile product.
        self.crop_bounds = list(crop_bounds) if crop_bounds is not None else None
        self.crop_epsg = crop_epsg
        if self.crop_bounds is not None and not crop_name:
            raise ValueError('crop_bounds needs crop_name, so the cropped h5 does not replace the tile h5')
        self.max_images = max_images
        self.images_per_forward = images_per_forward
        self.timeseries = timeseries # True: one band per image instead of the per-pixel median
        self.timeseries_vars = list(timeseries_vars or ['RH25_Q1', 'RH98_Q1'])
        self.max_cloud_cover = max_cloud_cover # STAC eo:cloud_cover filter when downloading
        self.months = months # image months when downloading; None = the tile's growing months
        if metadata_file != 'none':
            self.metadata_file = Path(metadata_file).expanduser()
        else:
            self.metadata_file = None
        if s2_grid_file != 'none':
            self.s2_grid_file = Path(s2_grid_file).expanduser()
        else:
            self.s2_grid_file = None
        self.h5_file = Path(h5_dir).expanduser() / f'{crop_name or tile_id}.h5'
        self.output_format = output_format
        self.output_dtype = output_dtype
        self.tile_id = tile_id
        self.n_iamges_per_tile = n_iamges_per_tile
        self.prediction_dir = prediction_dir
        self.patch_size = patch_size
        self.border = border
        self.patch_size_no_border = self.patch_size - 2 * self.border
        self.img_idx = img_idx
        self.debug = debug
        self.input_lat_lon = input_lat_lon
        self.mask_with_scl = mask_with_scl
        self.predict_full_profile = predict_full_profile
        self.year = year
        self.chunk_size = chunk_size
        self.impute_cloud_with_mean = impute_cloud_with_mean
        self.save_intermediate_tif = save_intermediate_tif
        if download_data and not self.h5_file.exists():
            self.h5_file.parent.mkdir(parents=True, exist_ok=True)
            print(f'{self.h5_file} does not exist, downloading...')
            if self.metadata_file is None:
                self.download_tile_by_api(max_cloud_cover=self.max_cloud_cover)
            elif self.timeseries:
                raise ValueError('timeseries picks its images from the STAC API; set metadata_file to none')
            else:
                self.download_tile()
        self.store = h5py.File(self.h5_file, 'r')
        if not hasattr(self, 'mean') and self.impute_cloud_with_mean:
            print('Calculating mean and std of the training data')
            t0 = time.time()
            data = xr.open_dataset(self.h5_file, engine='h5netcdf', chunks={'time': 1, 'band': 1, 'y': 1024, 'x': 1024})
            data = data.s2
            cloud_mask = data.sel(band=['SCL']).isin([8,9])
            masked_input = xr.where(cloud_mask.data, np.nan, data.isel(band=slice(12)))
            self.mean = masked_input.mean(dim=('y','x')).compute()
            self.mean = self.mean.data
            # self.mean = np.ones((20, 12))*3000
            print(self.mean)
            print(f'Time taken to calculate mean: {time.time() - t0:.2f} seconds')


        self.key = ""
        self.img_height, self.img_width = self.store['s2'].shape[2:]
        self.transform = Affine(*self.store['s2'].attrs['transform']).to_gdal()
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

        # ********** Set up RH & image indices **********
        self._set_output_channels(predict_full_profile)
        self.tile_id = tile_id
        self.bands = slice(12)
        
        if img_idx is None:
            self.img_idx = slice(None)
        else:
            self.img_idx = slice(img_idx, img_idx + 1)
        # The h5 file already holds the images picked at download time, so no
        # metadata re-ranking here; h5py also needs a slice or increasing indices.
        self.img_sel = slice(0, self.max_images)

        self.patch_coords_dict = self._get_patch_coords()
        print('--------------------------------')
        print(f"Image shape: {self.img_width}, {self.img_height}")
        print(f"Number of patches: {len(self.patch_coords_dict)}")
        print('--------------------------------')
        
        # ********** Prediction Configuration **********
        self.nodata_value = MASKED_VALUE[self.output_dtype]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # cloud shadows, CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY, SNOW, water, nodata
        # self.scl_exclude_labels = torch.tensor([0, 1, 3, 10, 11, 65535], dtype=torch.uint16, device=device) # scl is uint16
        # self.scl_cloud = torch.tensor([8, 9], dtype=torch.uint16, device=device)
        self.scl_exclude_labels = torch.tensor([0, 1, 3, 8, 9, 10, 11, 65535], dtype=torch.uint16, device=device)
        self.scl_cloud_mask_buffer = 7
        y, x = torch.meshgrid(torch.arange(-self.scl_cloud_mask_buffer, self.scl_cloud_mask_buffer+1), torch.arange(-self.scl_cloud_mask_buffer, self.scl_cloud_mask_buffer+1), indexing='ij')
        self.kernel = ((x**2 + y**2) <= self.scl_cloud_mask_buffer**2).float().unsqueeze(0).unsqueeze(0)
        # self.esa_exclude_labels = torch.tensor([5, 8], dtype=torch.uint8, device=device) # built-up, water
        self.scl_water = 6
        self.esa_snow = 7
        self.esa_built_up = 5
        self.esa_water = 8
        # self.scl_zero_canopy_height = torch.tensor([5, 6], dtype=torch.uint16, device=device)  # "not vegetated", "water"

        if not prediction_dir:
            self.prediction_dir = self.h5_file.parent.parent / f'predictions_{self.year}' / self.tile_id
        else:
            self.prediction_dir = Path(prediction_dir).expanduser()
        self.prediction_dir.mkdir(exist_ok=True)
        self.prediction_fp = self.prediction_dir / f'{self.tile_id}'
        if self.sample_points_file:
            self._init_sample_points()

    def _init_sample_points(self):
        """Pixel row / col of each sample point in the tile grid (floor, as rasterio.rowcol)."""
        if self.timeseries or not self.predict_full_profile:
            raise ValueError('sample_points_file needs the yearly full-profile prediction')
        self.sample_points = pd.read_parquet(Path(self.sample_points_file).expanduser())
        x, y = Transformer.from_crs('EPSG:4326', self.crs, always_xy=True).transform(
            self.sample_points.lon.values, self.sample_points.lat.values)
        gt = self.transform     # gdal order: x0, dx, 0, y0, 0, dy
        self.sample_cols = np.floor((np.asarray(x) - gt[0]) / gt[1]).astype(np.int64)
        self.sample_rows = np.floor((np.asarray(y) - gt[3]) / gt[5]).astype(np.int64)
        self.sample_values = np.full((len(self.sample_points), self.rh_dim), self.nodata_value, dtype=self.output_dtype)
        inside = ((self.sample_rows >= 0) & (self.sample_rows < self.img_height)
                  & (self.sample_cols >= 0) & (self.sample_cols < self.img_width))
        if self.sample_per_image_vars:
            self.sample_images = self._image_dates_ids()
            self.sample_per_image = np.full((len(self.sample_points), len(self.sample_images[0]),
                                             len(self.sample_per_image_vars)), self.nodata_value, dtype=self.output_dtype)
        print(f'{len(self.sample_points)} sample points, {int(inside.sum())} inside the tile, no GeoTIFF written')

    def _write_sample_patch(self, prediction, esa_wc, scl, idx):
        y0, x0 = self.patch_coords_dict[idx]
        y0, x0 = y0 + self.border, x0 + self.border
        n = self.patch_size_no_border
        m = ((self.sample_rows >= y0) & (self.sample_rows < y0 + n)
             & (self.sample_cols >= x0) & (self.sample_cols < x0 + n))
        if not m.any():
            return
        if self.sample_per_image_vars:
            ch = [rh_channel(v) for v in self.sample_per_image_vars]
            per = self._apply_masks(prediction[:, ch].clone(), esa_wc.clone(), scl.clone(), keep_images=True)  # (img, var, n, n)
            self.sample_per_image[m] = per[:, :, self.sample_rows[m] - y0, self.sample_cols[m] - x0].transpose(2, 0, 1)
        pred = self._apply_masks(prediction, esa_wc, scl)          # (rh_dim, n, n)
        # later patches overwrite overlapping ones, as WriteArray does for the GeoTIFF
        self.sample_values[m] = pred[:, self.sample_rows[m] - y0, self.sample_cols[m] - x0].T

    def _finalize_samples(self):
        cols = {f'RH{i // 3}_Q{i % 3}': self.sample_values[:, i] for i in range(self.rh_dim)}
        out = pd.concat([self.sample_points.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
        out_file = Path(self.sample_output_file).expanduser()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out.attrs = {}
        out.to_parquet(out_file.with_suffix('.tmp'), index=False)
        out_file.with_suffix('.tmp').rename(out_file)
        print(f'{len(out)} sample points -> {out_file}')
        if self.sample_per_image_vars:
            dates, ids = self.sample_images
            cols = {f'{v}:{d}_{i}': self.sample_per_image[:, k, j]
                    for j, v in enumerate(self.sample_per_image_vars) for k, (d, i) in enumerate(zip(dates, ids))}
            per = pd.concat([self.sample_points.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
            per.attrs = {}
            per_file = out_file.with_name(f'{out_file.stem}_per_image.parquet')
            per.to_parquet(per_file, index=False)
            print(f'{len(dates)} images x {self.sample_per_image_vars} -> {per_file}')

    def download_tile(self):
        '''
        This function is mainly used for streaming Sentinel-2 data (2024) on LUMI.
        I decided to not download ESA World Cover, because:
        1. We don't have ground truth for 2024.
        2. We don't need it for inference.
        '''
        bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]
        tile_df = read_items_for_tiles(self.metadata_file, [self.tile_id])
        tile_df = tile_df.drop_duplicates(subset='id', keep='first')
        tile_df = tile_df.set_index('id')
        if len(tile_df)>self.n_iamges_per_tile:
            tile_df['s2:nodata_pixel_percentage'] = tile_df['s2:nodata_pixel_percentage'].round()
            tile_df = tile_df.sort_values(['s2:nodata_pixel_percentage', 'eo:cloud_cover'])
            if (tile_df['s2:nodata_pixel_percentage']==0).sum() > 0:
                tile_df = tile_df.sort_values(['s2:nodata_pixel_percentage', 'eo:cloud_cover']).head(self.n_iamges_per_tile)
            else:
                # NOTE: I didn't select n_images_per_orbit = n_iamges_per_tile//n_orbits, because there might be too few images for some orbits
                idx = tile_df.groupby('orbit')['eo:cloud_cover'].nsmallest(self.n_iamges_per_tile//2).index.get_level_values(1)
                tile_df = tile_df.loc[idx].head(self.n_iamges_per_tile)
        tile_df['datetime'] = tile_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        # bbox = box(*tile_df.total_bounds) # For esa world cover
        # epsg = items[0].properties['proj:epsg']
        items = row_to_stac_item(tile_df, S2_ITEM_PROPS)  
        image = get_patch(items, bands, dtype='uint16', fill_value=np.uint16(0))
        image = harmonize_to_old(image)
        image.name = 's2'
        del image.attrs['spec']
        del image.attrs['crs']
        t0=time.time()
        image.to_netcdf(self.h5_file, engine='h5netcdf', encoding={'s2': {'zlib': False, 'chunksizes': (1, 1, 1024, 1024)}})
        print(f'Time taken to save image: {time.time() - t0:.2f} seconds')
        
    def download_tile_by_api(self, collection_id='sentinel-2-l2a', max_cloud_cover=90):
        '''
        This function is used to download the tile by api.
        '''
        bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]
        stac_api_io = StacApiIO()
        stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
        api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)
        datetime = f'{self.year}-01-01/{self.year}-12-31'
        s2_df = gpd.read_parquet(self.s2_grid_file)

        row = s2_df[s2_df['Name'] == self.tile_id]#.iloc[0]
        if len(row) == 0:
            raise RuntimeError(f'{self.tile_id} not found in {self.s2_grid_file}')
        bbox = row.geometry.total_bounds
        row = row.iloc[0]
        tile = row['Name']
        
        bbox = [bbox[0], bbox[1], min(bbox[2], 180), bbox[3]]
        search = api.search(collections=collection_id, bbox=bbox, datetime=datetime, 
                            query={'eo:cloud_cover': {'lt': max_cloud_cover},
                                    's2:nodata_pixel_percentage': {'lt': 90},
                                    's2:mgrs_tile': {'eq': tile}})
        items = search.item_collection()
        
        if len(items) == 0:
            print(f'{tile} has no images, bbox={bbox}, skipping...')
            raise RuntimeError(f'{tile} has no images, bbox={bbox}, skipping...')
        df = gpd.GeoDataFrame.from_features(items.to_dict(), crs='epsg:4326')
        df['id'] = df['s2:product_uri'].str.replace(r'_[A-Z]\d{4}', '', regex=True).str.replace('.SAFE', '')
        month = pd.to_datetime(df['datetime']).dt.month
        months = self.months or list(row.growing_months)
        df = df[month.isin(months)]
        if len(df) == 0:
            print(f'{tile} has no images, bbox={bbox}, skipping...')
            raise RuntimeError(f'{tile} has no images in months {months}, bbox={bbox}, skipping...')
        if self.timeseries:
            # every image of the selected months under the cloud threshold (applied in the search),
            # one processing per acquisition
            df = newest_processing(df.drop_duplicates(subset='id'))
        else:
            df = self._pick_best_images(df)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        # df.to_parquet(self.save_dir/ f'{self.year}_{tile}_images.parquet')
        epsg = int(items[0].properties['proj:code'][5:])
        items = [item for item in items.items if item.id in df['id'].values]
        if self.crop_bounds is not None:
            if self.crop_epsg is not None and int(self.crop_epsg) != epsg:
                raise RuntimeError(f'{tile}: crop_epsg {self.crop_epsg} differs from the tile CRS EPSG:{epsg}')
            images = stack(items, bands, dtype='uint16', fill_value=np.uint16(0), epsg=epsg, resolution=10,
                           bounds=self.crop_bounds, rescale=False)
            print(f'{tile}: cropped to {self.crop_bounds} (EPSG:{epsg}), shape {images.shape}')
        else:
            images = stack(items, bands, dtype='uint16', fill_value=np.uint16(0), epsg=epsg, resolution=10, rescale=False)
        images = harmonize_to_old(images)
        if self.crop_bounds is None:
            assert images.shape[2] == images.shape[3] == 10980, f'{tile} has incorrect shape {images.shape}'
        images.name = 's2'
        del images.attrs['spec']
        del images.attrs['crs']
        images = images.reset_coords(['proj:bbox'], drop=True)
        t0=time.time()
        images.to_netcdf(self.h5_file, engine='h5netcdf', encoding={'s2': {'zlib': False, 'chunksizes': (1, 1, 1024, 1024)}})
        print(f'Time taken to save image: {time.time() - t0:.2f} seconds')
        
        
    def _pick_best_images(self, df):
        '''Yearly product: keep up to n_iamges_per_tile images, preferring full-coverage
        orbits and low cloud cover.'''
        # get top 30 images, 10 from the best orbits and 20 from the rest
        df['s2:nodata_pixel_percentage'] = df['s2:nodata_pixel_percentage'].round()
        if (df['s2:nodata_pixel_percentage']==0).sum() > 0:
            best_orbits = df[df['s2:nodata_pixel_percentage']==0]['sat:relative_orbit'].unique()
            best = df[df['sat:relative_orbit'].isin(best_orbits)]
            rest = df[~df['sat:relative_orbit'].isin(best_orbits)]
        else:
            rest = df
            best = pd.DataFrame([], columns=df.columns)
            
        unique_orbits = rest['sat:relative_orbit'].unique()
        if len(unique_orbits) >= 2:
            top_orbits = rest.groupby('sat:relative_orbit').min('s2:nodata_pixel_percentage').sort_values('s2:nodata_pixel_percentage').head(2)
            rest = rest[rest['sat:relative_orbit'].isin(top_orbits.index)]
            idx = rest.groupby('sat:relative_orbit')['eo:cloud_cover'].nsmallest(10).index.get_level_values(1)
            rest = rest.loc[idx]
            df = pd.concat([best, rest])
        else:
            rest = rest.sort_values('eo:cloud_cover').head(10)
            best = best.sort_values('eo:cloud_cover').head(20)
        df = pd.concat([best, rest])
        
        # get top 20 images
        df = df.drop_duplicates(subset='id')
        if len(df)>self.n_iamges_per_tile:
            # df['s2:nodata_pixel_percentage'] = df['s2:nodata_pixel_percentage'].round() # NOTE： maybe we should do this earlier, when getting top 30 images
            df = df.sort_values(['s2:nodata_pixel_percentage', 'eo:cloud_cover'])
            if (df['s2:nodata_pixel_percentage']==0).sum() > 0:
                df = df.head(self.n_iamges_per_tile)
            else:
                idx = df.groupby('orbit')['eo:cloud_cover'].nsmallest(self.n_iamges_per_tile//2).index.get_level_values(1)
                df = df.loc[idx]
        return df

    def initialize_output(self):
        if self.sample_points_file:
            return
        self.options = [
            'TILED=YES',
            'BLOCKXSIZE={}'.format(self.chunk_size), # a different block size than the actual patch size is slow
            'BLOCKYSIZE={}'.format(self.chunk_size),
            'PREDICTOR=2',
            'NUM_THREADS=ALL_CPUS',
            'COMPRESS=None',
            'INTERLEAVE=BAND'
        ]
        if self.timeseries:
            self.tiff_writers = self._init_timeseries_output()
            return
        if self.predict_full_profile:
            output_files = [self.prediction_fp.with_stem(f'RH{i}_Q{j}') for i in range(self.rh_dim//3) for j in range(3)]
        else:
            output_files = [self.prediction_fp.with_stem(f'RH{i//3}_Q1') for i in self.rh_idx]
        
        self.tiff_writers = [self.init_gtiff(output_file) for output_file in output_files]
        dtype = np.dtype(
            [("raster_writer", gdal.Dataset), ("array", np.int16, (self.patch_size_no_border, self.patch_size_no_border))]
        )
        self.pred_table = np.empty((self.rh_dim), dtype=dtype)
        
    def write_patch_predictions(self, prediction, esa_wc, scl, idx):
        if self.timeseries:
            return self._write_timeseries_patch(prediction, esa_wc, scl, idx)
        if self.sample_points_file:
            return self._write_sample_patch(prediction, esa_wc, scl, idx)
        y_topleft, x_topleft = self.patch_coords_dict[idx]
        y_topleft = y_topleft + self.border
        x_topleft = x_topleft + self.border
        prediction_no_border = self._apply_masks(prediction, esa_wc, scl)
        for i in range(self.rh_dim):
            self.pred_table[i] = (self.tiff_writers[i], prediction_no_border[i])
        darr = da.from_array(self.pred_table, chunks=(1,))
        darr = darr.map_blocks(write_patch_predictions, x_topleft, y_topleft, self.nodata_value, meta=np.array((1,2), dtype=darr.dtype))
        darr.compute(scheduler='threads')


    def init_gtiff(self, prediction_fp: Path, n_bands: int = 1):
        prediction_fp = prediction_fp.with_suffix('.tif')
        if prediction_fp.exists():
            os.remove(prediction_fp)
        driver = gdal.GetDriverByName('GTiff')
        tiff_output = driver.Create(
            str(prediction_fp),
            xsize=self.img_width,
            ysize=self.img_height,
            bands=n_bands,
            eType=gdal.GDT_Int16,
            options=self.options
        )
        tiff_output.SetGeoTransform(self.transform)
        tiff_output.SetProjection(self.crs)
        tiff_output.SetMetadataItem('Year', str(self.year))
        tiff_output.SetMetadataItem('Sentinel-2 tile', self.tile_id)
        return tiff_output


    def _image_dates_ids(self):
        with xr.open_dataset(self.h5_file, engine='h5netcdf') as ds:
            img_idx = np.arange(ds.sizes['time'])[self.img_sel]
            dates = pd.to_datetime(ds['time'].values[img_idx]).strftime('%Y-%m-%d').tolist()
            ids = [str(i) for i in ds['id'].values[img_idx]] if 'id' in ds.coords else [''] * len(img_idx)
        return dates, ids

    def finalize_output(self):
        if self.sample_points_file:
            return self._finalize_samples()
        for writer in self.tiff_writers:
            writer.FlushCache()
        self.tiff_writers = None

    def __del__(self):
        if hasattr(self, 'store'):
            self.store.close()

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
                h5_dir=pred_fp,
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
    