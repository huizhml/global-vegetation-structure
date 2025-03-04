from typing import List
from pathlib import Path
from glob import glob
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm
import torch
import hydra
import numpy as np
from ffcv.loader import Loader, OrderOption

def get_dense_latlon(central_coords_array, resolution=10, grid_size=15):
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


class AverageMeter:
    def __init__(self, is_img:bool=False) -> None:
        channel = 12 if is_img else 1
        self.n = torch.zeros(1)
        self.avg = torch.zeros(channel)
        self.x_square = torch.zeros(channel)
        self.is_img = is_img
    
    @property
    def std(self):
        # use bassel's correction
        sum_x = self.avg * self.n
        n = self.n * 225
        var = self.x_square * self.n / (n -1) - (sum_x**2/n/(n-1))
        if self.is_img:
            return torch.sqrt(var) * 1e4
        return torch.sqrt(var)
    
    @property
    def mean(self):
        if self.is_img:
            return self.avg /225 * 1e4
        return self.avg / 225

    def update(self, x) -> None:
        if self.is_img:
            x = x.double()/1e4
            self.avg = (self.avg * self.n + x.sum(axis=(0, 2, 3)))/(self.n + x.shape[0])
            self.x_square = (self.x_square * self.n + (x**2).sum(axis=(0, 2, 3))) / (self.n+x.shape[0])
        else:
            self.avg = (self.avg * self.n + x.sum())/ (self.n+x.shape[0])
            self.x_square = (self.x_square * self.n + (x**2).sum()) / (self.n + x.shape[0])
        self.n += x.shape[0]

    def reset(self):
        self.n = torch.zeros(self.n.shape)
        self.avg = torch.zeros(self.avg.shape)
        self.x_square = torch.zeros(self.x_square.shape)



def calculate_s2_mean_std(beton_fp:str):
    """
    Calculate mean and std for the beton file.
    """
    batch_size = 100 if 'debug' in beton_fp else 4096
    beton_fp = Path(beton_fp).expanduser()
    beton_fp = glob(str(beton_fp))
    
    avg_img = AverageMeter(is_img=True)
    avg_slope = AverageMeter()
    avg_lat = AverageMeter()
    avg_lon_sin = AverageMeter()
    avg_lon_cos = AverageMeter()

    for fp in beton_fp:
        print(f'Processing {fp}')
        loader = Loader(fp, batch_size=batch_size, num_workers=4,
                    distributed=False, batches_ahead=3,
                    order=OrderOption.SEQUENTIAL, os_cache=False)

        for batch in tqdm(loader):
            img = batch[0]
            slope = torch.nan_to_num(batch[3], nan=0)
            lon = batch[4].unsqueeze(1).repeat(1, 15, 1)
            lat = batch[5].unsqueeze(2).repeat(1, 1, 15)
            lon = lon * np.pi / 180
            lon_sin = torch.sin(lon)
            lon_cos = torch.cos(lon)
            avg_slope.update(slope)
            avg_img.update(img)
            
            avg_lat.update(lat)
            avg_lon_sin.update(lon_sin)
            avg_lon_cos.update(lon_cos)

            # img = img.double()/1e4
            # n_old = n
            # n += img.shape[0]
            # avg = (avg * n_old + img.sum(axis=(0, 2, 3)))/ n
            # avg_x_square = (avg_x_square * n_old + (img**2).sum(axis=(0, 2, 3))) / n          
            
    # variance = avg_x_square * n / (n*225 -1) - ((avg*n)**2/(n*225)/(n*225-1))
    print('n: ', avg_img.n)
    print('mean img: ', avg_img.mean)
    print('std img: ', avg_img.std)
    print('mean slope: ', avg_slope.mean)
    print('std slope: ', avg_slope.std)
    print('mean lat: ', avg_lat.mean)
    print('std lat: ', avg_lat.std)
    print('mean lon sin: ', avg_lon_sin.mean)
    print('std lon sin: ', avg_lon_sin.std)
    print('mean lon cos: ', avg_lon_cos.mean)
    print('std lon cos: ', avg_lon_cos.std)

    np.savetxt('output/data_stats/s2_mean_filtered.txt', avg_img.mean)
    np.savetxt('output/data_stats/s2_std_filtered.txt', avg_img.std)
    np.savetxt('output/data_stats/slope_mean_filtered.txt', avg_slope.mean)
    np.savetxt('output/data_stats/slope_std_filtered.txt', avg_slope.std)
    np.savetxt('output/data_stats/lat_mean_filtered.txt', avg_lat.mean)
    np.savetxt('output/data_stats/lat_std_filtered.txt', avg_lat.std)
    np.savetxt('output/data_stats/lon_sin_mean_filtered.txt', avg_lon_sin.mean)
    np.savetxt('output/data_stats/lon_sin_std_filtered.txt', avg_lon_sin.std)
    np.savetxt('output/data_stats/lon_cos_mean_filtered.txt', avg_lon_cos.mean)
    np.savetxt('output/data_stats/lon_cos_std_filtered.txt', avg_lon_cos.std)


def get_rhs_bin_counts(parquet_fp: str, rh_quantiles_fp: str, lower_q: str='1e-05', upper_q: str='0.9999', n_bins: int=20):
    """
    Get the counts of the rhs values in the bins defined by the bin_edges.
    """
    import pandas as pd
    import dask.dataframe as dd
    import dask.array as da
    import dask
    rh_quantiles_fp = Path(rh_quantiles_fp).expanduser()
    df = pd.read_csv(rh_quantiles_fp)
    pmin = df[lower_q].values
    pmax = df[upper_q].values
    bin_edges = np.linspace(pmin, pmax, n_bins+1)
    parquet_fp = Path(parquet_fp).expanduser()
    parquet_fp = parquet_fp.parent.glob(parquet_fp.name)
    parquet_fp = [str(p) for p in parquet_fp] 
    ddf = dd.read_parquet(parquet_fp)
    counts = []
    bins = []
    for i in range(101):
        arr = ddf[f'rh{i}'].to_dask_array()
        arr = arr.compute()
        bin_edg =  [-np.inf] + bin_edges[:, i].tolist() + [np.inf]
        # c, b = da.histogram(arr, bin_edg) 
        c, b = np.histogram(arr, bin_edg)
        # assign the max count to 'outliers'
        c[0] = max(c)
        c[-1] = max(c)
        counts.append(c)
        bins.append(b)
    counts, bins = dask.compute(counts, bins)
    
    import ipdb; ipdb.set_trace()


@dataclass
class Config:
    beton_fp: str = '~/data/GVS/train_subsets/train*_filtered_v1.beton'

cs = ConfigStore.instance()
cs.store(name='config', node=Config)

@hydra.main(config_name='config', version_base='1.2')
def main(cfg: DictConfig):
    # calculate_s2_mean_std(cfg.beton_fp)
    get_rhs_bin_counts('~/data/GVS/train_subsets/train*_filtered_v1.parquet', 'output/data_stats/quantile_distribution.csv')

if __name__ == '__main__':
    main()