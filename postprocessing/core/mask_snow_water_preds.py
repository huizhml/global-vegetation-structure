import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
import pystac_client
import planetary_computer
from pystac_client.stac_api_io import StacApiIO
import pystac
import xarray as xr
import stackstac
import numpy as np
import rasterio
import dask
import shutil
from pyproj import Transformer
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from const import ESA_DATETIME, ESA_UNKNOWN_RAW, ESA_SNOW_RAW, ESA_WATER_RAW

stac_api_io = StacApiIO()
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)

def get_water_snow_mask(tile: pystac.Item):
    tile_id = tile.id.split("_")[0]
    search = api.search(collections='esa-worldcover', bbox=tile.bbox, datetime=ESA_DATETIME)
    items = search.item_collection()
    if len(items) == 0:
        raise ValueError(f'No ESA World Cover items found for tile {tile_id}')

    file = tile.assets['RH98_Q1'].href.replace('file://', '')
    with rasterio.open(file) as src:
        bounds = src.bounds
        epsg = src.crs.to_epsg()
        width = src.width
        height = src.height
    bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    wc_image = stackstac.stack(items, ['map'], bounds=bounds, epsg=epsg, resolution=10, dtype='uint16', fill_value=np.uint16(0),rescale=False)
    wc_image = wc_image.max(dim='time', skipna=True).squeeze()
    assert wc_image.shape == (width, height), f'{wc_image.shape} != ({width}, {height})'
    
    water_snow_mask = wc_image.isin([ESA_UNKNOWN_RAW, ESA_SNOW_RAW, ESA_WATER_RAW]) # nodata, snow and ice, water
    water_snow_mask = water_snow_mask.compute()
    return water_snow_mask


def mask_predictions(pred_file: Path, save_dir: Path, water_snow_mask: np.ndarray):
    with rasterio.open(pred_file) as src:
        nodata = src.nodata
        profile = src.profile
        data = src.read(1)
        data[water_snow_mask] = nodata
        with rasterio.open(save_dir / pred_file.name, 'w', **profile) as dst:
            dst.write(data, indexes=1)
    return str(save_dir/pred_file.name)

def to_cog(geotiff_path: Path, cog_path: Path):
    output_profile = cog_profiles.get("ZSTD")
    output_profile.update(dict(
        BIGTIFF="IF_SAFER",
        interleave="band",
        # zstd_level=1,
        PREDICTOR=2,
        blockxsize=1024,
        blockysize=1024,
        MAX_Z_ERROR=0
    ))
    cog_translate(geotiff_path, cog_path, output_profile,
                  in_memory=False, quiet=True, use_cog_driver=True)
    return True

def mask_snow_water_preds(stac_collection_dir: str, year: int = 2020, tile_id: str = None, filename_pattern: str = '*Q1.tif', save_dir: str = None, translate: bool = False, **kwargs):
    '''
    NOTE: mask & translate 303 files took 1000s, mask 303 files took 240s, mask 101 files took
    '''
    save_dir = Path(save_dir).expanduser()
    geotiff_dir = save_dir / f'geotiff/{tile_id}'
    geotiff_dir.mkdir(parents=True, exist_ok=True)
    cog_dir = save_dir / f'cog/{tile_id}'
    if translate:
        cog_dir.mkdir(parents=True, exist_ok=True)
    skip_dir = cog_dir if translate else geotiff_dir
    if len(list(skip_dir.glob(filename_pattern))) == 101: # TODO: parameterize 101 to match filename_pattern
        print(f'{tile_id} already has 101 files, skipping')
        return
    stac_collection_dir = Path(stac_collection_dir).expanduser()
    stac_item = pystac.Item.from_file(str(stac_collection_dir / f'{tile_id}_{year}/{tile_id}_{year}.json'))
    water_snow_mask = get_water_snow_mask(stac_item)
    pred_dir = Path(stac_item.assets['RH98_Q1'].href.replace('file://', '')).expanduser()
    pred_files = list(pred_dir.parent.glob(filename_pattern))
    
    tasks = []
    for file in pred_files:
        cog_path = cog_dir / file.name
        geotiff_path = geotiff_dir / file.name
        skip_path = cog_path if translate else geotiff_path
        if skip_path.exists():
            print(f'{tile_id}/{file.name} already exists, skipping')
            continue

        geotiff_path = dask.delayed(mask_predictions)(file, geotiff_dir, water_snow_mask)
        if translate:
            tasks.append(dask.delayed(to_cog)(geotiff_path, cog_dir / file.name))
        else:
            tasks.append(geotiff_path)
    results = dask.compute(*tasks)
    if all(results):
        print(f"Successfully masked {len(results)} files")
        if translate:
            shutil.rmtree(geotiff_dir)
    else:
        print(f"Failed to mask {len(results)} files")
        