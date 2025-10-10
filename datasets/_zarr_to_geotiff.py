import xarray as xr
import zarr
import rioxarray
from pathlib import Path
import time

def zarr_to_geotiff(zarr_path, geotiff_path, tile_id):
    zarr_path = Path(zarr_path).expanduser()
    geotiff_path = Path(geotiff_path).expanduser()
    geotiff_path.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(zarr_path, group=tile_id)
    imgs = ds.s2.compute()
    print(imgs)
    for time in imgs.time:
        print('writing', time.data, imgs.sel(time=time).id)
        imgs.sel(time=time).rio.to_raster(geotiff_path/ f'{str(time.data)[:10]}.tif', driver='COG')
        
def zarr_to_h5(zarr_path, h5_path, tile_id):
    zarr_path = Path(zarr_path).expanduser()
    h5_path = Path(h5_path).expanduser()
    ds = xr.open_dataset(zarr_path, group=tile_id)
    t0 = time.time()
    imgs = ds.s2.compute()
    print(f'Time taken to compute images: {time.time() - t0:.2f} seconds')
    encoding = {
         's2': {'zlib': True, 'complevel': 7, 'fletcher32': False, 'chunksizes': (1,1, 1024, 1024)}  
    }
    imgs.to_netcdf(h5_path, engine='h5netcdf', encoding=encoding)
    print(f'Time taken to save images: {time.time() - t0:.2f} seconds')


def zarr_to_zip(zarr_path, zip_path, tile_id):
    zarr_path = Path(zarr_path).expanduser()
    zip_path = Path(zip_path).expanduser()
    ds = xr.open_dataset(zarr_path, group=tile_id)
    t0 = time.time()
    imgs = ds.s2.compute()
    print(f'Time taken to compute images: {time.time() - t0:.2f} seconds')
    store = zarr.storage.ZipStore(zip_path, mode='w')
    z = zarr.create_array(store=store, shape=imgs.shape, chunks=(1, 1, 1024,1024), dtype=imgs.dtype)
    z[:] = imgs.data
    print(f'Time taken to write to zipstore: {time.time() - t0:.2f} seconds')



if __name__ == '__main__':
    year = 2020
    zarr_path = f'~/data/gvs/deploy/inference_{year}.zarr'
    h5_path = '~/data/gvs/deploy/32MQE/32MQE.zip'
    tile_id = '32MQE'
    zarr_to_zip(zarr_path, h5_path, tile_id)

    