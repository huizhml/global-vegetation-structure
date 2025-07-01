from typing import List
import torch
import zarr
import numpy as np
import math
from torch.utils.data import Dataset

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
    def __init__(self, zarr_store_path, tile_ids:List[str]=None, 
                 patch_size=128, border=8, time_step=0,
                 bands=None, input_lat_lon=False):
        
        self.store = zarr.open(zarr_store_path, mode='r')
        self.tile_ids = tile_ids or list(self.store.keys())
        self.patch_size = patch_size
        self.border = border
        self.time_step = time_step
        self.input_lat_lon = input_lat_lon
        self.nodata_value = 65535

        self.scl_exclude_labels = np.array([8, 9, 11, 6, self.nodata_value])  # CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY, SNOW, water, nodata
        
        # Load data arrays from Zarr
        self.scl = self.tile_group['SCL'][:]
        self.cloud_mask = self.tile_group['cloud_mask'][:]
        self.lat_mask = self.tile_group['lat_mask'][:]
        self.lon_mask = self.tile_group['lon_mask'][:]

        # Initialize patch coordinates
        self.patch_size_no_border = patch_size - 2 * border
        self.patch_coords = self._calculate_patch_coordinates()
    
        print(f"Image shape: {self.image.shape}")
        print(f"Number of patches: {len(self.patch_coords)}")


    def _calculate_patch_coordinates(self, tile_id):
        """Calculate patch coordinates with overlap handling"""
        coords = []
        n, _, y_dim, x_dim = self.store[f'{tile_id}/s2'].shape
        col_steps = int(math.ceil(y_dim / self.patch_size_no_border))
        row_steps = int(math.ceil(x_dim / self.patch_size_no_border))
        scl = self.store[f'{tile_id}/s2'][:, -1, :, :]
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
            idx = (scl[:, y_coord:y_coord+self.patch_size, x_coord:x_coord + self.patch_size] != self.nodata_value).sum(dim=[1, 2]) > 0
            idx = np.range(n)[idx]
            for i in idx:
                coords.append((i, y_coord, x_coord))

        return coords

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        y_start, x_start, y_end, x_end = self.patch_coords[idx]
        
        # Extract patch with border handling
        patch = self.image[y_start:y_end, x_start:x_end, :]
        patch = self._pad_patch(patch)
        
        # Add lat/lon channels if needed
        if self.input_lat_lon:
            lat = self.lat_mask[y_start:y_end, x_start:x_end]
            lon = self.lon_mask[y_start:y_end, x_start:x_end]
            patch = self._add_geo_channels(patch, lat, lon)

        # Convert to tensor and channel-first format
        patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1)

        return {
            'data': patch_tensor,
            'coordinates': (y_start, x_start, y_end, x_end)
        }

    def _pad_patch(self, patch):
        """Apply symmetric padding if patch is smaller than target size"""
        pad_y = self.patch_size - patch.shape[0]
        pad_x = self.patch_size - patch.shape[1]
        
        if pad_y > 0 or pad_x > 0:
            return np.pad(
                patch,
                pad_width=((0, pad_y), (0, pad_x), (0, 0)),
                mode='symmetric'
            )
        return patch

    def _add_geo_channels(self, patch, lat, lon):
        """Add geographic coordinates as input channels"""
        lon_sin = np.sin(2 * np.pi * lon / 360)
        lon_cos = np.cos(2 * np.pi * lon / 360)
        return np.concatenate([
            patch,
            lat[..., np.newaxis],
            lon_sin[..., np.newaxis],
            lon_cos[..., np.newaxis]
        ], axis=-1)

    def recompose_predictions(self, predictions):
        """Recompose patch predictions into full raster"""
        full_raster = np.zeros(self.image.shape[:2], dtype=np.float32)
        count_matrix = np.zeros_like(full_raster)

        for (y_start, x_start, y_end, x_end), pred in zip(self.patch_coords, predictions):
            # Remove border from prediction
            valid_pred = pred[self.border:-self.border, self.border:-self.border]
            valid_y = y_start + self.border
            valid_x = x_start + self.border
            
            # Accumulate predictions
            full_raster[valid_y:valid_y+valid_pred.shape[0], 
                       valid_x:valid_x+valid_pred.shape[1]] += valid_pred
            count_matrix[valid_y:valid_y+valid_pred.shape[0],
                        valid_x:valid_x+valid_pred.shape[1]] += 1

        # Normalize overlapping regions
        full_raster = np.divide(full_raster, count_matrix, where=count_matrix>0)
        
        # Apply masks
        full_raster = self._apply_masks(full_raster)
        return full_raster

    def _apply_masks(self, raster):
        """Apply quality masks to final prediction"""
        # Mask clouds
        raster[self.cloud_mask > 0] = np.nan
        
        # Mask non-vegetated areas
        vegetation_mask = ~np.isin(self.scl, [5, 6, 8, 9, 11])
        raster[~vegetation_mask] = np.nan

        return raster
    
    