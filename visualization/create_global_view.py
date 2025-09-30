from osgeo import gdal
from osgeo import osr
from glob import glob
from pathlib import Path
from typing import List, Union
import geopandas as gpd
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
import hydra
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
    
    countries_url = "~/data/GVS/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    countries_df = gpd.read_file(countries_url)
    regions = countries_df[countries_df['ADMIN'].isin(countries)]
    s2_grid = gpd.read_parquet(s2_grid_file)
    tiles = s2_grid[s2_grid.intersects(regions.union_all())]['Name'].unique()
    return tiles


def resample_and_mosaic(year = 2020, rh_idx = 98, q_idx = 1, countries: str = None, s2_grid_file: str = None):
    
    dst_srs = "EPSG:4326"         
    resampling = "average"         
    src_nodata = None                 # trust per-tile nodata if present
    dst_nodata = 32767
    
    pred_dir = f"~/data/GVS/Deploy/predictions_{year}"
    pred_dir = Path(pred_dir).expanduser()
    
    if len(countries) > 0:
        countries = Path(countries).expanduser()
        tiles = get_tiles_in_countries(countries, s2_grid_file)
        inputs = [f'{pred_dir}/{t}_cog/RH{rh_idx}_Q{q_idx}.cog.tif' for t in tiles]
        save_dir = countries.parent
    else:
        inputs = sorted(glob(f"{pred_dir}/*/RH{rh_idx}_Q{q_idx}.cog.tif"))   #COGs
        save_dir = pred_dir.parent / f"global_mosaic_{year}"
    if len(inputs) == 0:
        raise FileNotFoundError(
            f"No input COGs found for pattern: {pred_dir}/*/RH{rh_idx}_Q{q_idx}.cog.tif. "
            f"Check year/rh_idx/q_idx and source directory."
        )
    input_paths = [str(p) for p in inputs]
    valid_paths = [p for p in input_paths if is_transformable_to_4326(p)]
    if len(valid_paths) == 0:
        raise RuntimeError("No valid inputs after CRS validation. Check source CRS tags.")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = save_dir / f"thumb_RH{rh_idx}_Q{q_idx}.tif"
    # 2) Warp + downsample to thumbnail
    warp_kwargs = dict(
        dstSRS=dst_srs,
        outputBounds=(-180.0, -90.0, 180.0, 90.0),
        outputBoundsSRS=dst_srs,
        resampleAlg=resampling,
        srcNodata=src_nodata,
        dstNodata=dst_nodata,
        multithread=False,
        targetAlignedPixels=True,
        warpOptions=["INIT_DEST=NO_DATA","WRAP_DATELINE=YES","NUM_THREADS=1","SOURCE_EXTRA=64"],
        errorThreshold=0.0,
        creationOptions=["COMPRESS=LZW", "TILED=YES"]
    )

    # pick size OR res:
    # Prefer explicit resolution to avoid odd aspect results with bounds
    # Use degree-based resolution derived from target km (EPSG:4326)
    warp_opts_create = gdal.WarpOptions(**warp_kwargs, xRes=0.01, yRes=0.01)

    # Mosaic in chunks directly into the final thumbnail to limit open files
    chunk_size = 500
    paths = valid_paths
    first_chunk = paths[:chunk_size]
    gdal.Warp(destNameOrDestDS=str(thumb_path), srcDSOrSrcDSTab=first_chunk, options=warp_opts_create)

    # Subsequent chunks: update destination without reinitializing
    warp_kwargs_update = dict(warp_kwargs)
    warp_kwargs_update["warpOptions"] = ["WRAP_DATELINE=YES","NUM_THREADS=1","SOURCE_EXTRA=64"]
    warp_opts_update = gdal.WarpOptions(**warp_kwargs_update, xRes=0.01, yRes=0.01)

    for idx in range(chunk_size, len(paths), chunk_size):
        chunk = paths[idx: idx + chunk_size]
        dst_ds = gdal.Open(str(thumb_path), gdal.GA_Update)
        try:
            gdal.Warp(destNameOrDestDS=dst_ds, srcDSOrSrcDSTab=chunk, options=warp_opts_update)
        finally:
            if dst_ds is not None:
                dst_ds = None

    print("Wrote", thumb_path)

def run_mosaic_for_all_rhs(year = 2020):
    for rh_idx in range(0, 101):
        resample_and_mosaic(year, rh_idx, 1)
    
@dataclass
class MosaicConfig:
    year: int = 2020
    rh_idx: int = 98
    q_idx: int = 1
    countries: str = ''
    s2_grid_file: str = '~/data/GVS/S2_tiles_with_growing_months.parquet'
    
cs = ConfigStore.instance()
cs.store(name='mosaic', node=MosaicConfig)
    
@hydra.main(config_name='mosaic', version_base='1.2')
def main(cfg):
    resample_and_mosaic(cfg.year, cfg.rh_idx, cfg.q_idx, cfg.countries, cfg.s2_grid_file)

if __name__ == "__main__":
    main()