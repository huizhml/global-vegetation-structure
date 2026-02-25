import os
from osgeo import gdal
from osgeo import osr
from pathlib import Path
from typing import List, Union
import geopandas as gpd
import dask
import subprocess
import numpy as np
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import pystac
from const import NO_DATA
import re
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


def get_bias_correction_offset(
        bias_dir: Path, tile_id: str, rh_idx: int = 98, bias_col: str = 'bias', average_across_rhs: bool = False,
        bias_cutoff: float = None):
    '''
    Get the bias correction offset from a bias path
    '''
    bias_dir = Path(bias_dir).expanduser()
    bias_path = bias_dir / f'{tile_id}.npz'
    if bias_path.exists():
        if average_across_rhs:
            offset = np.load(bias_path)[bias_col].mean()
        else:
            offset = np.load(bias_path)[bias_col][rh_idx]
        if bias_cutoff is not None and offset.abs() > bias_cutoff:
            return None
        return offset
    return None


class GlobalMosaicker:
    '''
    Global mosaicker to create global mosaic of Sentinel-2 tile based data.
    Tile-level data is in local UTM projection, this class will transform the data to global projection defined by target_crs.
    And then mosaic the tiles to create a global mosaic.
    '''

    def __init__(self,
                 tiles_dir: Union[str, Path],
                 save_dir: Union[str, Path],
                 rh_idx: int = 98,
                 q_idx: int = 1,
                 left_q_idx: int = 0,
                 right_q_idx: int = 2,
                 tile_id_list_file: Union[str, Path] = None,
                 bias_dir: Union[str, Path] = None,
                 stac_collection_dir: Union[str, Path] = None,
                 bias_cutoff: float = None,
                 bias_col: str = 'bias',
                 average_across_rhs: bool = False,
                 target_res: float = 0.01,
                 target_crs: int = 4326,
                 resample_alg: str = "average",
                 dst_nodata: int = NO_DATA,
                 compression: str = "ZSTD",
                 **kwargs):
        """
        Initialize the Mosaicker with common paths and settings.
        """
        self.tiles_dir = Path(tiles_dir).expanduser()
        self.save_dir = Path(save_dir).expanduser()
        self.year = re.search(r'\d{4}', tiles_dir).group(0)
        self.rh_idx = rh_idx
        self.q_idx = q_idx
        self.tile_id_list_file = tile_id_list_file
        self.target_res = target_res
        self.resample_alg = resample_alg
        self.dst_nodata = dst_nodata
        self.compression = compression
        self.dst_srs = f'EPSG:{target_crs}'
        self.left_q_idx = left_q_idx
        self.right_q_idx = right_q_idx
        self.bias_dir = bias_dir
        self.stac_collection_dir = stac_collection_dir
        self.bias_cutoff = bias_cutoff
        self.bias_col = bias_col
        self.average_across_rhs = average_across_rhs
        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def downsample_warp_options(self):
        '''
        Get the warp options for downsampling
        '''
        return dict(
            dstSRS=self.dst_srs,
            xRes=self.target_res,
            yRes=self.target_res,
            resampleAlg=self.resample_alg,
            dstNodata=self.dst_nodata,
            creationOptions=[f"COMPRESS={self.compression}", "TILED=YES"],
            warpOptions=["WRAP_DATELINE=YES"],
        )
    
    @property
    def downsample_warp_options_mem(self):
        '''
        Get the warp options for downsampling in memory
        '''
        return dict(
            format='MEM',
            dstSRS=self.dst_srs,
            xRes=self.target_res,
            yRes=self.target_res,
            resampleAlg=self.resample_alg,
            dstNodata=self.dst_nodata,
        )

    @property
    def mosaic_warp_options(self):
        '''
        Get the warp options for mosaicking
        '''
        return dict(
            dstSRS=self.dst_srs,
            resampleAlg=self.resample_alg,
            dstNodata=self.dst_nodata,
            creationOptions=[f"COMPRESS={self.compression}", "TILED=YES"],
            warpOptions=["WRAP_DATELINE=YES"],
        )
        
    def get_cog_profile_and_config(self):
        '''
        Get the cog profile and config for the given compression
        '''
        output_profile = cog_profiles.get(self.compression)
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
        return output_profile, config

    def init_tmp_dir(self, path: Path):
        '''
        Initialize the temporary directory
        '''
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_all_tile_ids(self):
        '''
        Get all tile ids from the tile id list file
        '''
        tiles_file = Path(self.tile_id_list_file).expanduser()
        tiles = np.loadtxt(tiles_file, dtype=str)
        return tiles

    def _get_path_from_stac_item(self, tile_id: str):
        '''
        Get the tile id and path from a stac item file
        '''
        item_file = self.stac_collection_dir / f'{tile_id}_{self.year}/{tile_id}_{self.year}.json'
        item = pystac.Item.from_file(str(item_file))
        src_path = item.assets[f'RH{self.rh_idx}_Q{self.q_idx}'].href.replace('file://', '')
        return str(src_path)

    @staticmethod
    def get_reprojected_bbox(tiff_path: str, target_epsg: int = 4326):
        '''
        Get the bounding box of a tile in degrees
        '''
        ds_src = gdal.Open(tiff_path)
        try:
            wkt = ds_src.GetProjection()
            if not wkt:
                # If no projection is found, we can't transform bounds safely.
                return None

            src_srs = osr.SpatialReference()
            # This is where your error happens ("Corrupt data")
            # We catch it below.
            src_srs.ImportFromWkt(wkt)

            tgt_srs = osr.SpatialReference()
            tgt_srs.ImportFromEPSG(target_epsg)
            tgt_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

            transform = osr.CoordinateTransformation(src_srs, tgt_srs)

            gt = ds_src.GetGeoTransform()
            x_size, y_size = ds_src.RasterXSize, ds_src.RasterYSize

            minx = gt[0]
            maxy = gt[3]
            maxx = minx + gt[1] * x_size
            miny = maxy + gt[5] * y_size

            # Transform the four corners to handle rotation/skew edge cases
            corners = [
                transform.TransformPoint(minx, miny),
                transform.TransformPoint(maxx, miny),
                transform.TransformPoint(maxx, maxy),
                transform.TransformPoint(minx, maxy)
            ]

            # Extract min/max from transformed points
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]

            return [min(xs), min(ys), max(xs), max(ys)]

        except Exception as e:
            # This catches RuntimeError: OGR Error: Corrupt data
            # We return None so the worker knows to skip this file
            return None

    def _write_tiff(self, arr: np.ndarray, dst_path: Path, gdal_ds: gdal.Dataset):
        '''
        Write a numpy array to a tif file
        '''
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(str(dst_path),
                               gdal_ds.RasterXSize,
                               gdal_ds.RasterYSize,
                               1, gdal.GDT_Int16, options=[f"COMPRESS={self.compression}", "TILED=YES"])
        out_ds.SetGeoTransform(gdal_ds.GetGeoTransform())
        out_ds.SetProjection(gdal_ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(arr)
        out_band.SetNoDataValue(self.dst_nodata)
        out_band.FlushCache()
        out_ds = None

    def _warp_tile(self, dst_path: Path, src_path: Path, warp_options: dict):
        '''
        Warp a tile to the target projection and resolution
        '''
        if dst_path.exists():
            return str(dst_path)
        warp_opts = gdal.WarpOptions(**warp_options)
        return gdal.Warp(str(dst_path), str(src_path), options=warp_opts)

    def _warp_tile_diff(self, tile_id: str, dst_dir: Path, warp_options: dict):
        '''
        Get downsampled difference between two tiles
        '''
        warp_opts = gdal.WarpOptions(**warp_options)
        path_left = self.tiles_dir / f'{tile_id}/RH{self.rh_idx}_Q{self.left_q_idx}.tif'
        path_right = self.tiles_dir / f'{tile_id}/RH{self.rh_idx}_Q{self.right_q_idx}.tif'

        dst_path = dst_dir / f'{tile_id}_resampled.tif'
        if dst_path.exists():
            return str(dst_path)
        ds1 = gdal.Warp('', str(path_left), options=warp_opts)
        ds2 = gdal.Warp('', str(path_right), options=warp_opts)
        # 2. Read as Arrays (At 1km, these are tiny, e.g., 100x100 pixels)
        arr1 = ds1.GetRasterBand(1).ReadAsArray()
        arr2 = ds2.GetRasterBand(1).ReadAsArray()

        # 3. Calculate Difference with NoData handling
        mask = (arr1 == NO_DATA) | (arr2 == NO_DATA)
        # Use float32 to prevent overflow/underflow
        diff_arr = arr1.astype(np.float32) - arr2.astype(np.float32)
        diff_arr[mask] = NO_DATA
        diff_arr = diff_arr.astype(np.int16)

        # 4. Create the output TIF
        self._write_tiff(diff_arr, dst_path, ds1)
        ds1 = None
        ds2 = None
        return str(dst_path)

    def _warp_tile_bias_correction(self, tile_id: str, dst_dir: Path, vrt_dir: Path, warp_options: dict):
        '''
        Warp a tile to the target projection and resolution with bias correction
        '''
        warp_opts = gdal.WarpOptions(**warp_options)
        dst_path = dst_dir / f'{tile_id}_resampled.tif'
        if dst_path.exists():
            return str(dst_path)

        src_path = self._get_path_from_stac_item(tile_id)
        offset = get_bias_correction_offset(self.bias_dir, tile_id, self.rh_idx,
                                            self.bias_col, self.average_across_rhs, self.bias_cutoff)
        if offset is not None:
            vrt_path = _make_offset_vrt(src_path, vrt_dir, offset=offset, nodata=self.dst_nodata)
            src_path = str(vrt_path)
        return gdal.Warp(str(dst_path), str(src_path), options=warp_opts)

    def _mosaic_tiles(self, paths: List[str], mosaic_path: Path):
        '''
        Mosaic tiles
        '''
        warp_options = self.mosaic_warp_options
        # if countries is not None:
        #     countries_boundary = countries.with_name('countries.gpkg')
        #     warp_options['cutlineDSName'] = str(countries_boundary)
        #     warp_options['cropToCutline'] = True
        #     # thumb_path = thumb_path.with_name(thumb_path.stem + "_clipped.tif")

        warp_opts_mosaic = gdal.WarpOptions(**warp_options)

        gdal.Warp(
            destNameOrDestDS=str(mosaic_path),
            srcDSOrSrcDSTab=list(paths),
            options=warp_opts_mosaic
        )
        
    def _to_cog(self, mosaic_path: Path):
        '''
        Translate the mosaic to cog
        '''
        cog_path = mosaic_path.with_suffix('.cog.tif')
        output_profile, config = self.get_cog_profile_and_config()
        cog_translate(mosaic_path, cog_path, output_profile, config=config,
                      in_memory=False, quiet=True, use_cog_driver=True)
        return cog_path
        
        
    def create_global_mosaic(self):
        '''
        Create a global mosaic
        '''
        temp_dir_name = f"tmp_tiles_resampled_1km_RH{self.rh_idx}_Q{self.q_idx}"
        temp_dir = self.init_tmp_dir(self.save_dir / temp_dir_name)
        tile_ids = self.get_all_tile_ids()
        warp_options = dict(self.downsample_warp_options)
        tasks = [
            dask.delayed(self._warp_tile)(tile_id, temp_dir, warp_options)
            for tile_id in tile_ids]
        downsampled_paths = dask.compute(*tasks, scheduler="processes", num_workers=8)

        mosaic_path = self.save_dir / f"global_mosaic_{self.year}_RH{self.rh_idx}_Q{self.q_idx}.tif"
        self._mosaic_tiles(downsampled_paths, mosaic_path)
        cog_path = self._to_cog(mosaic_path)
        print(f"✅ Global mosaic written to {cog_path}")
    
    def create_global_diff_mosaic(self):
        '''
        Create a global mosaic of the difference between two tiles
        '''
        # Step 1: Warp the tiles to 1km resolution
        temp_dir_name = f"tmp_tiles_resampled_1km_RH{self.rh_idx}_Q{self.left_q_idx}-Q{self.right_q_idx}"
        temp_dir = self.init_tmp_dir(self.save_dir / temp_dir_name)
        tile_ids = self.get_all_tile_ids()
        tasks = [
            dask.delayed(self._warp_tile_diff)(tile_id, temp_dir, self.downsample_warp_options_mem)
            # self._warp_tile_diff(tile_id, temp_dir, self.downsample_warp_options_mem)
            for tile_id in tile_ids]
        diff_paths = dask.compute(*tasks, scheduler="processes", num_workers=8)

        # Step 2: Mosaic the tiles
        mosaic_path = self.save_dir / f"global_mosaic_RH{self.rh_idx}_Q{self.left_q_idx}-Q{self.right_q_idx}.tif"
        self._mosaic_tiles(diff_paths, mosaic_path)

        # Step 3: Translate the mosaic to cog
        cog_path = self._to_cog(mosaic_path)
        print(f"✅ Global mosaic written to {cog_path}")

    def create_global_bias_correction_mosaic(self):
        '''
        Create a global mosaic with bias correction applied
        '''
        downsampled_tmp_dir = self.init_tmp_dir(
            self.save_dir / f"tmp_tiles_resampled_1km_RH{self.rh_idx}_Q{self.q_idx}")
        vrt_tmp_dir = self.init_tmp_dir(self.save_dir / f"vrt_tmp")
        tile_ids = self.get_all_tile_ids()

        warp_options = dict(self.downsample_warp_options)
        tasks = [
            dask.delayed(self._warp_tile_bias_correction)(tile_id, downsampled_tmp_dir, vrt_tmp_dir, warp_options)
            for tile_id in tile_ids]
        downsampled_paths = dask.compute(*tasks, scheduler="processes", num_workers=8)

        thumb_path = self.save_dir / f"global_mosaic_{self.year}_RH{self.rh_idx}_Q{self.q_idx}.tif"
        self._mosaic_tiles(downsampled_paths, thumb_path)
        cog_path = self._to_cog(thumb_path)
        print(f"✅ Global mosaic written to {cog_path}")




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


# def _warp_tile_diff(tile_pred_dir: Path, save_dir: Path, left_q_idx: int = 0, right_q_idx: int = 2, rh_idx: int = 98,
#                     dst_srs="EPSG:4326", xRes=0.01, yRes=0.01, dst_nodata=NO_DATA, resampleAlg="average", **kwargs):
#     '''
#     Get downsampled difference between two tiles
#     '''

#     left_path = tile_pred_dir / f'RH{rh_idx}_Q{left_q_idx}.tif'
#     right_path = tile_pred_dir / f'RH{rh_idx}_Q{right_q_idx}.tif'
#     tile_id = tile_pred_dir.stem
#     dst_path = save_dir / f'{tile_id}_resampled.tif'
#     if dst_path.exists():
#         return str(dst_path)

#     # 4. Warp with correct LOCAL Degree bounds
#     # bbox_degrees = _get_bbox_in_degrees(str(left_path))
#     warp_opts = gdal.WarpOptions(
#         format='MEM',
#         dstSRS="EPSG:4326",
#         xRes=xRes,
#         yRes=yRes,
#         # outputBounds=bbox_degrees,
#         resampleAlg="average",
#         dstNodata=NO_DATA
#     )
#     ds1 = gdal.Warp('', str(left_path), options=warp_opts)
#     ds2 = gdal.Warp('', str(right_path), options=warp_opts)
#     # 2. Read as Arrays (At 1km, these are tiny, e.g., 100x100 pixels)
#     arr1 = ds1.GetRasterBand(1).ReadAsArray()
#     arr2 = ds2.GetRasterBand(1).ReadAsArray()

#     # 3. Calculate Difference with NoData handling
#     mask = (arr1 == NO_DATA) | (arr2 == NO_DATA)
#     # Use float32 to prevent overflow/underflow
#     diff_arr = arr1.astype(np.float32) - arr2.astype(np.float32)
#     diff_arr[mask] = NO_DATA
#     diff_arr = diff_arr.astype(np.int16)

#     # 4. Create the output TIF
#     _write_tif(diff_arr, dst_path, ds1)
#     ds1 = None
#     ds2 = None
#     return str(dst_path)


# def mosaic_tiles(paths: List[str], mosaic_path: Path, **kwargs):
#     '''
#     Mosaic a list of tiles
#     '''
#     warp_options = dict(
#         dstSRS="EPSG:4326",
#         resampleAlg="average",
#         dstNodata=NO_DATA,
#         creationOptions=["COMPRESS=LERC_ZSTD", "TILED=YES"],
#         warpOptions=["WRAP_DATELINE=YES", "INIT_DEST=NO_DATA"],
#     )
#     # if countries is not None:
#     #     countries_boundary = countries.with_name('countries.gpkg')
#     #     warp_options['cutlineDSName'] = str(countries_boundary)
#     #     warp_options['cropToCutline'] = True
#     #     # thumb_path = thumb_path.with_name(thumb_path.stem + "_clipped.tif")

#     warp_opts_mosaic = gdal.WarpOptions(**warp_options)

#     gdal.Warp(
#         destNameOrDestDS=str(mosaic_path),
#         srcDSOrSrcDSTab=list(paths),
#         options=warp_opts_mosaic
#     )



# def create_global_diff_mosaic(
#         pred_dir: str, save_dir: Path, rh_idx: int = 98, left_q_idx: int = 0, right_q_idx: int = 2, **kwargs):
#     '''
#     Create a global mosaic of the difference between two tiles
#     '''
#     save_dir = Path(save_dir).expanduser()
#     save_dir.mkdir(parents=True, exist_ok=True)

#     # Step 1: Warp the tiles to 1km resolution
#     pred_dir = Path(pred_dir).expanduser()
#     temp_dir = save_dir / f"tmp_tiles_resampled_1km_RH{rh_idx}_Q{left_q_idx}-Q{right_q_idx}"
#     temp_dir.mkdir(parents=True, exist_ok=True)
#     tiles = os.listdir(pred_dir)
#     print(f"Found {len(tiles)} tiles for {pred_dir}")
#     if len(tiles) == 0:
#         raise FileNotFoundError(f"No tiles found for {pred_dir}")
#     diff_paths = [
#         dask.delayed(_warp_tile_diff)(pred_dir / tile, temp_dir, left_q_idx, right_q_idx, rh_idx, **kwargs)
#         for tile in tiles]
#     diff_paths = dask.compute(*diff_paths, scheduler="processes", num_workers=8)

#     # Step 2: Mosaic the tiles
#     mosaic_path = save_dir / f"global_mosaic_RH{rh_idx}_Q{left_q_idx}-Q{right_q_idx}.tif"
#     mosaic_tiles(diff_paths, mosaic_path)

#     # Step 3: Translate the mosaic to cog
#     cog_path = mosaic_path.with_suffix('.cog.tif')
#     output_profile, config = get_cog_profile_and_config(compressor="ZSTD")
#     cog_translate(mosaic_path, cog_path, output_profile, config=config,
#                   in_memory=False, quiet=True, use_cog_driver=True)
#     print(f"✅ Global mosaic written to {cog_path}")


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

    temp_dir = save_dir / f"tmp_tiles_resampled_1km_RH{rh_idx}_Q{q_idx}"
    thumb_path = save_dir / f"global_mosaic_{year}_RH{rh_idx}_Q{q_idx}.tif"
    temp_dir.mkdir(parents=True, exist_ok=True)

    def warp_tile(tile_id: str, dst_srs="EPSG:4326", xRes=0.01, yRes=0.01, dst_nodata=NO_DATA, resampleAlg="average"):
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
        dstNodata=NO_DATA,
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
