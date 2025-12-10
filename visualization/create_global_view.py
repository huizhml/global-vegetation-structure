import os
from osgeo import gdal
from osgeo import osr
from glob import glob
from pathlib import Path
from typing import List, Union, Optional
import geopandas as gpd
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
import hydra
import dask
import subprocess
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import pystac
from tqdm import tqdm
try:
    import resource  # Posix: bump soft limit for open files if possible
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        new_soft = min(hard, 65536)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
except Exception:
    pass


# Speed hints
gdal.SetConfigOption("GDAL_NUM_THREADS", "1")
gdal.SetConfigOption("GDAL_MAX_DATASET_POOL_SIZE", "512")
gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.vrt")  # if reading via http(s)
gdal.SetConfigOption("OGR_CT_FORCE_TRADITIONAL_GIS_ORDER", "YES")

gdal.UseExceptions()



# Prefilter: keep only tiles that can transform to EPSG:4326 (avoid PROJ errors)
def is_transformable_to_4326(path):
    try:
        ds = gdal.Open(path)
        if ds is None:
            return False
        gt = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        srs_wkt = ds.GetProjectionRef()
        ds = None
        if not srs_wkt:
            return False
        src = osr.SpatialReference(); src.ImportFromWkt(srs_wkt)
        dst = osr.SpatialReference(); dst.ImportFromEPSG(4326)
        ct = osr.CoordinateTransformation(src, dst)
        ix, iy = w // 2, h // 2
        x = gt[0] + ix * gt[1] + iy * gt[2]
        y = gt[3] + ix * gt[4] + iy * gt[5]
        ct.TransformPoint(x, y)
        return True
    except Exception:
        return False

def get_tiles_in_countries(countries_file: Path, s2_grid_file: str):
    '''
    Get Sentinel-2 tiles covering the list of countries
    Args:
        countries_file: str, path to txt file containing list of countries
    Returns:
        list of Sentinel-2 tiles covering the list of countries
    '''
    with open(countries_file, 'r') as file:
        countries = [line.strip() for line in file]
    if len(countries) == 0:
        return []
    
    countries_url = "~/data/gvs/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    countries_df = gpd.read_file(countries_url)
    regions = countries_df[countries_df['ADMIN'].isin(countries)]
    s2_grid = gpd.read_parquet(s2_grid_file)
    tiles = s2_grid[s2_grid.intersects(regions.union_all())]['Name'].unique()
    return tiles, regions


def resample_and_mosaic(year = 2020, rh_idx = 98, q_idx = 1, countries: str = None, s2_grid_file: str = None, stac_collection_dir: str = None):
    
    stac_collection_dir = Path(stac_collection_dir).expanduser() 
    
    if countries is not None:
        countries = Path(countries).expanduser()
        tiles, regions = get_tiles_in_countries(countries, s2_grid_file)
        regions_gpkg = countries.parent / f"countries.gpkg"
        if not regions_gpkg.exists():
            regions.to_file(regions_gpkg, driver='GPKG')
        # inputs = [f'{pred_dir}/{t}_cog/RH{rh_idx}_Q{q_idx}.tif' for t in tiles]
        save_dir = countries.parent
    else:
        if year == 2024:
            pred_dir = '~/data/gvs/deploy/predictions_gtiff_2024'
            pred_dir = Path(pred_dir).expanduser()
            tiles = os.listdir(pred_dir)
        else:
            pred_dir = '~/data/gvs/deploy/predictions_2020'
            pred_dir = Path(pred_dir).expanduser()
            tiles = os.listdir(pred_dir)
        # inputs = []
        # if year == 2024: # data on lumi-o
        #     flag_dir = f"~/data/gvs/deploy/flags_inference_{year}"
        #     flag_dir = Path(flag_dir).expanduser()
        #     for flag_file in flag_dir.glob(f'*_best_images_done'):
        #         tile_id = flag_file.stem.split('_')[0]
        #         inputs.append(pred_gtif_dir / f'{tile_id}/RH{rh_idx}_Q{q_idx}.tif')
        # else:
        #     tiles = pred_dir.iterdir()
        #     for tile in tiles:
        #         tile_id = tile.stem.split('_')[0]
        #         if (masked_pred_dir / f'{tile_id}/RH{rh_idx}_Q{q_idx}.tif').exists():
        #             inputs.append(masked_pred_dir / f'{tile_id}/RH{rh_idx}_Q{q_idx}.tif')
        #         elif (pred_gtif_dir / f'{tile_id}/RH{rh_idx}_Q{q_idx}.tif').exists():
        #             inputs.append(pred_gtif_dir / f'{tile_id}/RH{rh_idx}_Q{q_idx}.tif')
        #         else:
        #             inputs.append(tile / f'RH{rh_idx}_Q{q_idx}.tif') #COGs
        save_dir = pred_dir.parent / f"global_mosaic_{year}"
    print(f"Found {len(tiles)} tiles for year {year}")
    if len(tiles) == 0:
        raise FileNotFoundError(
            f"No tiles found for year {year}"
        )
        
    save_dir.mkdir(parents=True, exist_ok=True)  
    temp_dir = save_dir / f"tiles_warp_RH{rh_idx}"
    thumb_path = save_dir / f"thumb_RH{rh_idx}_Q{q_idx}.tif"
    temp_dir = Path(temp_dir).expanduser()
    temp_dir.mkdir(parents=True, exist_ok=True)
        
    def warp_tile(tile_id: str, dst_srs="EPSG:4326", xRes=0.01, yRes=0.01, dst_nodata=32767, resampleAlg="average"):
        item = pystac.Item.from_file(str(stac_collection_dir / f'{tile_id}_{year}/{tile_id}_{year}.json'))
        src_path = Path(item.assets[f'RH{rh_idx}_Q{q_idx}'].href.replace('file://', '')).expanduser()
        if year == 2024 and (not src_path.exists()):
            # sync data from lumi-o
            tile_id = src_path.parent.stem.split('_')[0]
            zone = tile_id[:3].lower()
            bucket_name = f"{zone}-{year}"
            remote = f"lumi-465001846-private:{bucket_name}/predictions_GTiff_{year}/{tile_id}/RH{rh_idx}_Q{q_idx}_uncompressed.tif"
            task = subprocess.run(f"rclone copy {remote} {src_path.parent} --transfers=16 --checkers=16 --multi-thread-streams=4", shell=True)
            if task.returncode != 0:
                raise RuntimeError(f"Failed to sync data from lumi-o for tile {tile_id}")
            # rename the file
            (src_path.parent / f'RH{rh_idx}_Q{q_idx}_uncompressed.tif').rename(src_path)
            
        dst_path = temp_dir / f'{src_path.parent.stem}_resampled.tif'
        if dst_path.exists():
            return str(dst_path)
        warp_opts = gdal.WarpOptions(
            dstSRS=dst_srs,
            xRes=xRes,
            yRes=yRes,
            resampleAlg=resampleAlg,
            dstNodata=dst_nodata,
            creationOptions=["COMPRESS=ZSTD", "TILED=YES"],
            warpOptions=["WRAP_DATELINE=YES"]
        )
        gdal.Warp(destNameOrDestDS=str(dst_path), srcDSOrSrcDSTab=str(src_path), options=warp_opts)
        return str(dst_path)

    # for input in inputs:
    #     warp_tile(input, temp_dir)
    tasks = [dask.delayed(warp_tile)(tile_id) for tile_id in tiles]
    warped_paths = dask.compute(*tasks, scheduler="processes", num_workers=8)
    
    warp_options = dict(
        dstSRS="EPSG:4326",
        resampleAlg="average",
        dstNodata=32767,
        creationOptions=["COMPRESS=LERC_ZSTD", "TILED=YES"],
        warpOptions=["WRAP_DATELINE=YES", "INIT_DEST=NO_DATA"],
    )
    if countries is not None:
        countries_boundary = countries.with_name('countries.gpkg')
        warp_options['cutlineDSName'] = str(countries_boundary)
        warp_options['cropToCutline'] = True
        # thumb_path = thumb_path.with_name(thumb_path.stem + "_clipped.tif")
    
    warp_opts_mosaic = gdal.WarpOptions(**warp_options)

    gdal.Warp(
        destNameOrDestDS=str(thumb_path),
        srcDSOrSrcDSTab=list(warped_paths),
        options=warp_opts_mosaic
    )

    # translate to cog
    cog_path = thumb_path.with_suffix('.cog.tif')
    output_profile = cog_profiles.get("LERC_ZSTD")
    output_profile.update(dict(
        BIGTIFF="IF_SAFER",
        ZSTD_LEVEL=1,
        PREDICTOR=2,
        BLOCKYSIZE=1024,
        BLOCKXSIZE=1024,
        MAX_Z_ERROR=0
    ))
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )
    cog_translate(thumb_path, cog_path, output_profile, config=config, in_memory=False,quiet=True, use_cog_driver=True)
    print(f"✅ Global mosaic written to {cog_path}")
    

def run_mosaic_for_key_rhs(year = 2020, rh_idx: str = '98', q_idx = 1, countries: str = None, s2_grid_file: str = None, stac_collection_dir: str = None):
    print(rh_idx)
    rh_idx = [int(rh) for rh in rh_idx.split(' ')]
    for rh_idx in rh_idx:
        resample_and_mosaic(year, rh_idx, q_idx, countries, s2_grid_file, stac_collection_dir)


def run_mosaic_for_all_rhs(year = 2020):
    for rh_idx in range(0, 101):
        resample_and_mosaic(year, rh_idx, 1)
    
@dataclass
class MosaicConfig:
    year: int = 2020
    rh_idx: str = '98'
    q_idx: int = 1
    countries: Optional[str] = None
    s2_grid_file: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    stac_collection_dir: str = '~/data/gvs/deploy/gvsm_stac_catalog/vsm_local'
    task: str = 'run_mosaic_for_key_rhs'
    
cs = ConfigStore.instance()
cs.store(name='mosaic', node=MosaicConfig)
    
@hydra.main(config_name='mosaic', version_base='1.2')
def main(cfg):
    if cfg.task == 'run_mosaic_for_key_rhs':
        run_mosaic_for_key_rhs(cfg.year, cfg.rh_idx, cfg.q_idx, cfg.countries, cfg.s2_grid_file, cfg.stac_collection_dir)
    elif cfg.task == 'run_mosaic_for_all_rhs':
        run_mosaic_for_all_rhs(cfg.year)

if __name__ == "__main__":
    main()