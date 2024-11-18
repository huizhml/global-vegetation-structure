import joblib
import torch
import torch.nn as nn
from typing import Any
import lightning as L


class BaseModel(L.LightningModule):

    def __init__(
            self, 
            feed_slope: bool = False,
            feed_latlon: bool = False,
            loss_fc: nn.Module = None, 
            transform: nn.Module = None, 
            yhat_transform: nn.Module = None, *args: Any, **
            kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.feed_slope = feed_slope
        self.feed_latlon = feed_latlon
        self.loss_fc = loss_fc
        self.transform = transform
        self.yhat_transform = yhat_transform

    def get_dense_latlon(self, central_coords_array, resolution=10, grid_size=15):
        """
        Calculate densified latitude and longitude tensors for grids around multiple central points.

        Parameters:
        - central_coords_array: List of tuples (latitude, longitude) for the central pixels.
        - resolution: Spatial resolution of the pixels in meters.
        - grid_size: Size of the grid (default is 15x15).

        Returns:
        - Two tensors of size (n, 15, 15) for latitude and longitude.
        """
        # Convert central coordinates to a tensor
        n_points = central_coords_array.size(0)
        device = central_coords_array.device

        # Prepare grid offsets
        half_grid = grid_size // 2
        offsets = torch.arange(-half_grid, half_grid + 1, dtype=torch.float32, device=device)
        row_offsets, col_offsets = torch.meshgrid(offsets, offsets, indexing="ij")  # Shape: (15, 15)
        
        # Flatten the grid offsets for easy broadcasting
        row_offsets = row_offsets.flatten()  # Shape: (15*15,)
        col_offsets = col_offsets.flatten()  # Shape: (15*15,)

        # Extract central latitudes and longitudes
        central_lats = central_coords_array[:, 0].unsqueeze(1)  # Shape: (n, 1)
        central_lons = central_coords_array[:, 1].unsqueeze(1)  # Shape: (n, 1)

        # Compute conversion factors for degrees per meter
        central_lats_radians = central_lats * (torch.pi / 180)  # Convert degrees to radians
        meters_per_degree_lat = 111132.92  # Approximate mean value for latitude
        meters_per_degree_lon = 111320 * torch.cos(central_lats_radians)  # Adjust for latitude

        degree_per_pixel_lat = resolution / meters_per_degree_lat  # Shape: (n, 1)
        degree_per_pixel_lon = resolution / meters_per_degree_lon  # Shape: (n, 1)

        # Broadcast and compute lat/lon offsets for all points
        lat_offsets = row_offsets.unsqueeze(0) * degree_per_pixel_lat  # Shape: (n, 15*15)
        lon_offsets = col_offsets.unsqueeze(0) * degree_per_pixel_lon  # Shape: (n, 15*15)

        # Add offsets to central coordinates
        lat_pixels = central_lats + lat_offsets  # Shape: (n, 15*15)
        lon_pixels = central_lons + lon_offsets  # Shape: (n, 15*15)

        # Reshape into (n, 15, 15)
        latitudes = lat_pixels.view(n_points, grid_size, grid_size)
        longitudes = lon_pixels.view(n_points, grid_size, grid_size)

        return latitudes, longitudes

    def training_step(self, sample, batch_idx):
        x = self.transform(sample[0])
        if self.feed_slope:
            slope = torch.nan_to_num(sample[3], nan=0)
            x = torch.cat([x, slope.unsqueeze(1)/90], dim=1)
        if self.feed_latlon:
            latlon = sample[4]
            lat, lon = self.get_dense_latlon(latlon)
            sin_lon = torch.sin(lon*torch.pi/180)
            cos_lon = torch.cos(lon*torch.pi/180)
            x = torch.cat([x,lat.unsqueeze(1), sin_lon.unsqueeze(1), cos_lon.unsqueeze(1)], dim=1)
        y_hat = self.forward(x.float())
        error_metrics, error_metrics_veg, output = self.loss_fc(y_hat, *sample[1:])
        for name, err in error_metrics.items():
            self.log(f'train_{name}', err, on_epoch=True, on_step=False, sync_dist=True)
        for name, err in error_metrics_veg.items():
            self.log(f'veg/train_{name}', err, on_epoch=True, on_step=False, sync_dist=True)
        return output

    def validation_step(self, sample, batch_idx):
        x = self.transform(sample[0])
        if self.feed_slope:
            slope = torch.nan_to_num(sample[3], nan=0)
            x = torch.cat([x, slope.unsqueeze(1)/90], dim=1)
        if self.feed_latlon:
            latlon = sample[4]
            lat, lon = self.get_dense_latlon(latlon)
            sin_lon = torch.sin(lon/180)
            cos_lon = torch.cos(lon/180)
            x = torch.cat([x,lat.unsqueeze(1), sin_lon.unsqueeze(1), cos_lon.unsqueeze(1)], dim=1)
        y_hat = self.forward(x.float())
        
        error_metrics, error_metrics_veg, output = self.loss_fc(y_hat, *sample[1:])
        for name, err in error_metrics.items():
            self.log(f'val_{name}', err, on_epoch=True, on_step=False, sync_dist=True)
        for name, err in error_metrics_veg.items():
            self.log(f'veg/val_{name}', err, on_epoch=True, on_step=False, sync_dist=True)
        return output