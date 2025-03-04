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
import rioxarray
from pyproj import Transformer
from utils import get_dense_latlon

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
                 bands: List[int] = None, 
                 map_lc: bool = False,
                 input_lat_lon=False,
                 mask_with_scl: bool = True,
                 mask_empty: bool = True,
                 compression: str = None):
        self.mask_with_scl = mask_with_scl
        self.mask_empty = mask_empty
        self.compression = compression
        self.img_idx = img_idx
        zarr_store_path = Path(zarr_store_path).expanduser()
        if not prediction_dir:
            self.prediction_dir = zarr_store_path.parent / 'predictions'
        else:
            self.prediction_dir = Path(prediction_dir).expanduser()
        self.prediction_dir.mkdir(exist_ok=True)
        try:
            self.store = zarr.open(zarr_store_path, mode='r')
        except Exception as e:
            print(f'Error opening zarr store: {e}, conda activate py3!')
        self.crs = f'EPSG:{self.store[f"{tile_id}/epsg"][()].item()}'
        self.transform = self.store[f'{tile_id}'].attrs['transform']
        self.tile_id = tile_id
        self.bands = bands or slice(12)
        self.patch_size = patch_size
        self.border = border
        self.patch_size_no_border = self.patch_size - 2 * self.border
        self.input_lat_lon = input_lat_lon
        self.nodata_value = 65535
        if img_idx is None:
            img_idx = slice(0,15) # TODO: was None, 20 images
        else:
            img_idx = slice(self.img_idx, self.img_idx + 1)
        
        self.image = self.store[f'{self.tile_id}/s2'][img_idx]
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
        self.scl_zero_canopy_height = np.array([5, 6])  # "not vegetated", "water"
        # cloud shadows, CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY, SNOW, water, nodata
        self.scl_exclude_labels = np.array([0, 3, 8, 9, 11, 6, self.nodata_value])
        self.scl = np.array(self.scl, dtype=np.uint8)

        print(f"Image shape: {self.image.shape}")
        print(f"Number of patches: {len(self.patch_coords_dict)}")

    def set_prediction_fname(self, wandb_run_id):
        # corrected_postfix = 'corrected' if self.correct_bias else 'uncorrected'
        if self.img_idx is None:
            self.prediction_fp = self.prediction_dir / f'{self.tile_id}_{wandb_run_id}.tif'
        else:
            img_idx = slice(self.img_idx, self.img_idx + 1)
            img_id = self.store[f'{self.tile_id}/id'][img_idx].item()
            self.prediction_fp = self.prediction_dir / f'{self.tile_id}_{img_id}_{wandb_run_id}.tif'
    
    def _get_patch_coords(self):
        """Calculate patch coordinates with overlap handling"""
        _, _, y_dim, x_dim = self.image.shape
        col_steps = int(math.ceil(y_dim / self.patch_size_no_border))
        row_steps = int(math.ceil(x_dim / self.patch_size_no_border))
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
                patch_coords_dict[patch_idx] = (y_coord, x_coord)
                patch_idx += 1
        print('number of patches: ', len(patch_coords_dict))
        return patch_coords_dict

    def __len__(self):
        return len(self.patch_coords_dict)

    def __getitem__(self, idx):
        y_topleft, x_topleft = self.patch_coords_dict[idx]

        # Extract patch with border handling
        patch = self.image[:, :, y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size]
        patch = patch.astype(np.float32)

        # # Add lat/lon channels if needed
        if self.input_lat_lon:
            latlon = self.coords_input[:, y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size]
            latlon = np.tile(latlon[None, :, :,:], (patch.shape[0], 1,1,1))
            return torch.from_numpy(patch), torch.from_numpy(latlon)
        else:
            return torch.from_numpy(patch)

    def write_patch_predictions(self, prediction, idx, wandb_run_id):
        """Write patch prediction to tiff file
        Parameters
        ----------
        prediction : np.array, (time, rh_channels, patch_size, patch_size)
        """

        y_topleft, x_topleft = self.patch_coords_dict[idx]
        prediction_no_border = prediction[:, (RH100_idx, RH98_idx),
                                          self.border:self.patch_size - self.border,
                                          self.border:self.patch_size - self.border
                                          ]
        prediction_no_border = prediction_no_border.cpu().numpy()
        if self.mask_empty:
            # pixels where all RGB values equal zero are empty (bands B02, B03, B04)
            # note self.image has shape: (height, width, channels)
            img = self.image[:, 1:4, y_topleft:y_topleft + self.patch_size_no_border, x_topleft:x_topleft + self.patch_size_no_border]
            invalid_mask = np.sum(img,axis=1, keepdims=True) == 0
            # print('self.image.shape', self.image.shape)
            # print('invalid_mask.shape', invalid_mask.shape)
            # print('number of empty pixels:', np.sum(invalid_mask))
            # mask empty image pixels
            prediction_no_border = np.where(invalid_mask, np.nan, prediction_no_border)
        if self.mask_with_scl:
            # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
            scl = self.scl[:, :, y_topleft:y_topleft + self.patch_size_no_border,
                           x_topleft:x_topleft + self.patch_size_no_border]
            invalid_mask = np.logical_and(np.isin(scl, self.scl_exclude_labels), ~invalid_mask)
            prediction_no_border = np.where(invalid_mask, np.nan, prediction_no_border)
        prediction_no_border = np.nanmedian(prediction_no_border, axis=0)
        prediction_no_border = (prediction_no_border * 100).round()
        prediction_no_border = np.nan_to_num(prediction_no_border, nan=MASKED_VALUE).astype(np.int16)

        window = Window(col_off=x_topleft, row_off=y_topleft,
                        width=self.patch_size_no_border, height=self.patch_size_no_border)
        if not self.prediction_fp.exists():
            mode = 'w'
        else:
            mode = 'r+'
        with rasterio.open(self.prediction_fp, mode,
                           driver='GTiff',
                           width=self.img_width,
                           height=self.img_height,
                           count=prediction_no_border.shape[0],
                           dtype=rasterio.int16,
                           crs=self.crs,
                           transform=self.transform,
                        #    compress=self.compression,
                        #    TILED='YES',
                        #    BIGTIFF='IF_SAFER',
                        #    multithread=True,
                           nodata=MASKED_VALUE,
                           NUM_THREADS=8,) as dst:
            dst.write(prediction_no_border, window=window)


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



# def compare_compressors(tiff_file):
#     import subprocess

    
#     # tiff_file = str(tiff_file)
#     for level in range(1, 10):
#         t0 = time.time()
#         suffix = f'.DEFLATE.{level}.tif'
#         output_file = tiff_file.with_suffix(suffix)

#         command = [
#             'gdal_translate',
#             '-co', 'TILED=YES',
#             '-co', 'COMPRESS=DEFLATE',
#             '-co', 'PREDICTOR=2',
#             '-co', f'ZLEVEL={level}',
#             '-co', 'NUM_THREADS=ALL_CPUS',
#             str(tiff_file),
#             str(output_file)
#         ]
#         subprocess.run(command, check=True)
#         print(f'compressor DEFLATE level {level} time taken: {time.time() - t0}')
#     for level in range(1,23):
#         t0 = time.time()
#         suffix = f'.ZSTD.{level}.tif'
#         output_file = tiff_file.with_suffix(suffix)

#         command = [
#             'gdal_translate',
#             '-co', 'TILED=YES',
#             '-co', 'COMPRESS=ZSTD',
#             '-co', f'ZSTD_LEVEL={level}',
#             '-co', 'NUM_THREADS=ALL_CPUS',
#             str(tiff_file),
#             str(output_file)
#         ]
#         subprocess.run(command, check=True)
#         print(f'compressor ZSTD level {level} time taken: {time.time() - t0}')

#     t0 = time.time()
#     output_file = tiff_file.with_suffix('.JPED2000.tif')

#     command = [
#             'gdal_translate',
#             '-co', 'TILED=YES',
#             '-co', 'COMPRESS=JPEG2000',
#             '-co', 'REVERSIBLE=YES',
#             '-co', 'QUALITY=95',
#             str(tiff_file),
#             str(output_file)
#         ]
#     subprocess.run(command, check=True)
#     print(f'compressor JPEG2000 time taken: {time.time() - t0}')

#     t0 = time.time()
#     suffix = tiff_file.with_suffix('.CCSDS.tif')
#     output_file = tiff_file.with_suffix(suffix)

#     command = [
#             'gdal_translate',
#             '-co', 'CCSDS',
#             '-co', 'BLOCKSIZE=496',
#             str(tiff_file),
#             str(output_file)
#         ]
#     subprocess.run(command, check=True)
#     print(f'compressor CCSDS time taken: {time.time() - t0}')

def collate_batch(batch):
    return batch


class DeployDataModel(L.LightningDataModule):

    def __init__(self,
                 pred_fp: str = None,
                 tile_id: str = None,
                 prediction_dir: str = None,
                 batch_size: int = 1,
                 num_workers: int = 8,
                 patch_size:int=512, border:int=8,
                 bands: List[int] = None, 
                 input_lat_lon:bool=False,
                 img_idx:int=None,
                 **kwargs
                 ):
        super().__init__()
        self.pred_dataset = ZarrSentinel2Deploy(pred_fp, tile_id, prediction_dir,
                                                patch_size, border, img_idx, bands, input_lat_lon=input_lat_lon)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.pred_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_batch)

if __name__ == '__main__':
    compare_compressors(Path('~/data/GVS/Deploy/2020/predictions/32MQE.tif').expanduser())