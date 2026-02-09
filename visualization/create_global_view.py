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
import numpy as np
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
        src = osr.SpatialReference()
        src.ImportFromWkt(srs_wkt)
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4326)
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


@dask.delayed
def downsample_tile(
        item_file: str, save_dir: Union[str, Path], rh_idx:int=98,
        vrt_dir: Union[str, Path] = None, bias_dir: Union[str, Path] = None, bias_cutoff: float = None,
        bias_col: str = 'bias',
        average_across_rhs: bool = False, dst_srs="EPSG:4326", xRes=0.01, yRes=0.01, dst_nodata=32767,
        resampleAlg="average"):
    '''
    Downsample a tile to 1km resolution
    '''
    save_dir = Path(save_dir)
    tile_id = Path(item_file).stem.split('_')[0]
    item = pystac.Item.from_file(item_file)
    src_path = item.assets[f'RH{rh_idx}_Q1'].href.replace('file://', '')
    dst_path = save_dir / f'{tile_id}.tif'

    if dst_path.exists():
        return str(dst_path)
    if bias_dir is not None:
        bias_dir = Path(bias_dir)
        bias_path = bias_dir / f'{tile_id}.npz'
        if bias_path.exists():
            if average_across_rhs:
                offset = np.load(bias_path)[bias_col].mean()
            else:
                offset = np.load(bias_path)[bias_col][rh_idx]  # 98 is the RH index for bias correction
            if bias_cutoff is None or offset.abs() <= bias_cutoff:
                if vrt_dir is None:
                    raise ValueError("vrt_dir must be provided when bias_dir is used")
                vrt_dir = Path(vrt_dir)
                vrt_path = _make_offset_vrt(src_path, vrt_dir, offset=offset, nodata=dst_nodata)
                src_path = str(vrt_path)

    warp_opts = gdal.WarpOptions(
        dstSRS=dst_srs,
        xRes=xRes,
        yRes=yRes,
        resampleAlg=resampleAlg,
        dstNodata=dst_nodata,
        creationOptions=["COMPRESS=ZSTD", "TILED=YES"],
        warpOptions=["WRAP_DATELINE=YES"],
    )
    gdal.Warp(destNameOrDestDS=str(dst_path), srcDSOrSrcDSTab=str(src_path), options=warp_opts)
    return str(dst_path)


def _make_offset_vrt(src_path: Union[str, Path], vrt_dir: Path, offset: float, nodata: float) -> Path:
    """
    Create a tiny VRT that reads src_path and adds `offset` to valid pixels
    via ScaleOffset. Nodata is preserved.
    """
    src_path = Path(src_path)
    ds = gdal.Open(str(src_path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open {src_path}")

    band = ds.GetRasterBand(1)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    dtype = gdal.GetDataTypeName(band.DataType)
    gt = ds.GetGeoTransform()
    srs_wkt = ds.GetProjectionRef()
    ds = None

    vrt_path = vrt_dir / f"{src_path.parent.stem}_{src_path.stem}.offset.vrt"

    # relativeToVRT=1 requires SourceFilename relative to the VRT file location
    # so we write VRT next to the source and use src_path.name
    src_abs = src_path.resolve()

    # Validate GeoTransform and SRS
    if gt is None or len(gt) != 6:
        raise ValueError(f"Invalid GeoTransform for {src_path}: {gt}")
    if not srs_wkt:
        raise ValueError(f"No SRS found for {src_path}")

    # Build GeoTransform string (space-separated, which is standard for GDAL VRT)
    gt_str = ",".join(str(x) for x in gt)

    # Build SRS string - escape XML special characters
    srs_escaped = srs_wkt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    # Include GeoTransform and SRS at VRTDataset level for proper georeferencing
    vrt_xml = f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
    <GeoTransform>{gt_str}</GeoTransform>
    <SRS dataAxisToSRSAxisMapping="1,2">{srs_escaped}</SRS>
    <VRTRasterBand dataType="{dtype}" band="1">
        <NoDataValue>{nodata}</NoDataValue>
        <ComplexSource>
            <SourceFilename relativeToVRT="0">{src_abs}</SourceFilename>
            <SourceBand>1</SourceBand>
            <ScaleRatio>1.0</ScaleRatio>
            <ScaleOffset>{offset}</ScaleOffset>
            <NODATA>{nodata}</NODATA>
        </ComplexSource>
    </VRTRasterBand>
</VRTDataset>
"""
    vrt_path.write_text(vrt_xml)
    return vrt_path


def check_mosaic_after_bias_correction(
        year=2020, rh_idx: int = 98, q_idx: int = 1, stac_collection_dir: str = None, bias_dir: str = 'null', bias_cutoff: float = None,
        bias_col: str = 'bias', average_across_rhs: bool = False, save_dir: str = None):
    '''
    Check how bias correction works on the global mosaic.
    - Get the global mosaic for RH98 before actually applying bias correction
    '''
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = save_dir / f"tmp_tiles_resampled_1km_RH{rh_idx}_Q{q_idx}"
    thumb_path = save_dir / f"global_mosaic_{year}_RH{rh_idx}_Q{q_idx}.tif"
    temp_dir = Path(temp_dir).expanduser()
    temp_dir.mkdir(parents=True, exist_ok=True)
    vrt_dir = save_dir / f"vrt_tmp"
    vrt_dir.mkdir(parents=True, exist_ok=True)
    stac_collection_dir = Path(stac_collection_dir).expanduser()
    tiles_file = Path('~/data/gvs/assets/worklists/total_tiles_2024.txt').expanduser() #TODO: update the path
    tiles = np.loadtxt(tiles_file, dtype=str)
    print(f"tiles: {len(tiles)}")
    items_files = [ str(stac_collection_dir/ f'{tile}_{year}/{tile}_{year}.json') for tile in tiles if (stac_collection_dir/ f'{tile}_{year}/{tile}_{year}.json').exists()]
    # items_dir = list(stac_collection_dir.glob(f'*{year}/*.json')) # NOTE: very slow
    # items_files = [str(item_dir) for item_dir in items_dir]
    print(f"items_files: {len(items_files)}")
    
    text_template = f"""
        Check how the global mosaic looks like after bias correction, use RH98_Q1 as an example\n
        !!! IMPORTANT !!!\n
        The file path in the STAC collection has to be set correctly!\n
        In this case, it should point to those original predictions, before bias correction.
        Given STAC item example: {items_files[0]}
        """
    print(text_template)

    tasks = [
        downsample_tile(
            item_file, temp_dir, rh_idx, vrt_dir, bias_dir, bias_cutoff, bias_col=bias_col, average_across_rhs=average_across_rhs)
        for item_file in items_files]
    downsampled_paths = dask.compute(*tasks, scheduler="processes", num_workers=8)

    warp_options = dict(
        dstSRS="EPSG:4326",
        resampleAlg="average",
        dstNodata=32767,
        creationOptions=["COMPRESS=LERC_ZSTD", "TILED=YES"],
        warpOptions=["WRAP_DATELINE=YES", "INIT_DEST=NO_DATA"],
    )
    warp_opts_mosaic = gdal.WarpOptions(**warp_options)

    gdal.Warp(
        destNameOrDestDS=str(thumb_path),
        srcDSOrSrcDSTab=list(downsampled_paths),
        options=warp_opts_mosaic
    )
    print(f"✅ Global mosaic written to {thumb_path}")


def resample_and_mosaic(year=2020, rh_idx=98, q_idx=1, countries: str = None, s2_grid_file: str = None,
                        pred_dir: str = None, save_dir: str = None, **kwargs):
    pred_dir = Path(pred_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    tiles = os.listdir(pred_dir)
    tiles = [tile for tile in tiles if (pred_dir / f'{tile}/RH{rh_idx}_Q{q_idx}.tif').exists()]
    if countries is not None:
        countries = Path(countries).expanduser()
        tiles, regions = get_tiles_in_countries(countries, s2_grid_file)
        regions_gpkg = countries.parent / f"countries.gpkg"
        if not regions_gpkg.exists():
            regions.to_file(regions_gpkg, driver='GPKG')
        # inputs = [f'{pred_dir}/{t}_cog/RH{rh_idx}_Q{q_idx}.tif' for t in tiles]
        save_dir = countries.parent

    print(f"Found {len(tiles)} tiles for year {year}")
    if len(tiles) == 0:
        raise FileNotFoundError(
            f"No tiles found for year {year}"
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = save_dir / f"tmp_tiles_resampled_1km_RH{rh_idx}_Q{q_idx}"
    thumb_path = save_dir / f"global_mosaic_{year}_RH{rh_idx}_Q{q_idx}.tif"
    temp_dir = Path(temp_dir).expanduser()
    temp_dir.mkdir(parents=True, exist_ok=True)

    def warp_tile(tile_id: str, dst_srs="EPSG:4326", xRes=0.01, yRes=0.01, dst_nodata=32767, resampleAlg="average"):
        # item = pystac.Item.from_file(str(stac_collection_dir / f'{tile_id}_{year}/{tile_id}_{year}.json'))
        # src_path = Path(item.assets[f'RH{rh_idx}_Q{q_idx}'].href.replace('file://', '')).expanduser()
        src_path = pred_dir / f'{tile_id}/RH{rh_idx}_Q{q_idx}.tif'
        if year == 2024 and (not src_path.exists()):
            # sync data from lumi-o
            tile_id = src_path.parent.stem.split('_')[0]
            zone = tile_id[:3].lower()
            bucket_name = f"{zone}-{year}"
            remote = f"lumi-465001846-private:{bucket_name}/{tile_id}/RH{rh_idx}_Q{q_idx}.tif"
            task = subprocess.run(
                f"rclone copy {remote}  {src_path.parent}  --transfers=16 --checkers=16 --multi-thread-streams=4",
                shell=True)
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
    cog_translate(thumb_path, cog_path, output_profile, config=config, in_memory=False, quiet=True, use_cog_driver=True)
    print(f"✅ Global mosaic written to {cog_path}")
