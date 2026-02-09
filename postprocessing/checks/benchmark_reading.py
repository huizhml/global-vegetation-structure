import time
import xarray as xr
import dask.array as da
import numpy as np
import geopandas as gpd
import dask
from pathlib import Path


def test_reading():
    '''
    Test reading from 303 cog files vs. one compressed geotiff file with 303 bands
    '''
    def func(block):
        return xr.DataArray(
                np.zeros((1, 1,1), dtype="bool"),
                dims=('band',"x", "y"),
                name="block_status",
            )
    template = xr.DataArray(
        da.zeros((303, 11,11), dtype="bool", chunks=(1,1,1)),
        dims=('band',"x", "y"),
        name="block_status",
    )
    tile_id = '19NBF'
    gtiff_dir = Path(f'~/data/gvs/deploy/predictions_gtiff_2020/{tile_id}').expanduser()
    gtiff_dir = gtiff_dir.glob('*.tif')
    ds = [xr.open_dataset(gtiff_file, chunks={'band': 1, 'x': 1024, 'y': 1024}) for gtiff_file in gtiff_dir]
    ds = xr.concat(ds, dim='band')
    t0 = time.time()
    ds = ds.map_blocks(func, template=template)
    ds.compute()
    t1 = time.time()
    print(f'Time taken to read from 303 uncompressed geotiff files: {t1 - t0} seconds')
    
    cog_dir = Path(f'~/data/gvs/deploy/predictions_2020/{tile_id}').expanduser()
    cog_files = cog_dir.glob('*.tif')
    ds = [xr.open_dataset(cog_file, chunks={}) for cog_file in cog_files]
    ds = xr.concat(ds, dim='band')
    t0 = time.time()
    ds = ds.map_blocks(func, template=template)
    ds.compute()
    t1 = time.time()
    print(f'Time taken to read from 303 cog files: {t1 - t0} seconds')
    tif_file = Path(f'~/data/gvs/deploy/predictions_zip_2020/{tile_id}.tif').expanduser()
    ds = xr.open_dataset(tif_file, chunks={})
    t1 = time.time()
    ds = ds.map_blocks(func, template=template)
    ds.compute()
    t2 = time.time()
    print(f'Time taken to read from one compressed geotiff file: {t2 - t1} seconds')

    
