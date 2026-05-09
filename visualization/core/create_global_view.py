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
from const import NO_DATA, ESA_DATETIME, ESA_UNKNOWN_RAW, ESA_SNOW_RAW, ESA_WATER_RAW
from dask.diagnostics import ProgressBar
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
    Global mosaicker to create global mosaic of Sentinel-2 tile based data. This runs for one global layer, e.g, RH98_Q1
    Tile-level data is in local UTM projection, this class will transform the data to global projection defined by target_crs.
    And then mosaic the tiles to create a global mosaic.
    '''

    def __init__(self,
                 year: int=None,
                 save_dir: Union[str, Path]=None,
                 rh_idx: int = 98,
                 q_idx: int = 1,
                 total_tiles_file: Union[str, Path] = None,
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
        self.save_dir = Path(save_dir).expanduser()
        self.year = year
        self.rh_idx = rh_idx
        self.q_idx = q_idx
        self.target_res = target_res
        self.resample_alg = resample_alg
        self.dst_nodata = dst_nodata
        self.compression = compression
        self.dst_srs = f'EPSG:{target_crs}'
        self.bias_dir = bias_dir
        self.stac_collection_dir = Path(stac_collection_dir).expanduser()
        self.bias_cutoff = bias_cutoff
        self.bias_col = bias_col
        self.average_across_rhs = average_across_rhs
        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tile_ids = self.get_all_tile_ids(total_tiles_file)

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

    def get_all_tile_ids(self, total_tiles_file: Union[str, Path]):
        '''
        Get all tile ids from the tile id list file
        '''
        tiles_file = Path(total_tiles_file).expanduser()
        tiles = np.loadtxt(tiles_file, dtype=str)
        return tiles

    def _get_path_from_stac_item(self, tile_id: str):
        '''
        Get the tile id and path from a stac item file
        '''
        item_file = self.stac_collection_dir / f'{tile_id}_{self.year}/{tile_id}_{self.year}.json'
        item = pystac.Item.from_file(str(item_file))
        src_path = item.assets[f'RH{self.rh_idx}_Q{self.q_idx}'].href.replace('file://', '')
        return Path(src_path).expanduser()

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

    def _write_tiff(self, arr: np.ndarray, dst_path: Path, gdal_ds: gdal.Dataset, band_name: str = None):
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
        if band_name:
            out_band.SetDescription(band_name)
        out_band.WriteArray(arr)
        out_band.SetNoDataValue(self.dst_nodata)
        out_band.FlushCache()
        out_ds = None

    @staticmethod
    def _warp_tile(dst_path: Path, src_path: Path, warp_options: dict, band_name: str = None):
        '''
        Warp a tile to the target projection and resolution
        '''
        warp_opts = gdal.WarpOptions(**warp_options)
        gdal.Warp(str(dst_path), str(src_path), options=warp_opts)
        if band_name:
            ds = gdal.Open(str(dst_path), gdal.GA_Update)
            ds.GetRasterBand(1).SetDescription(band_name)
            ds = None
        return str(dst_path)
    
    def _warp_tile_diff(self, dst_path: Path, src_path: Path,warp_options: dict):
        '''
        Get downsampled difference between two tiles
        '''
        warp_opts = gdal.WarpOptions(**warp_options)
        path_left = src_path.with_name(f'RH{self.rh_idx}_Q0.tif')
        path_right = src_path.with_name(f'RH{self.rh_idx}_Q2.tif')

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
        self._write_tiff(diff_arr, dst_path, ds1, band_name=f'RH{self.rh_idx}_Q0-Q2')
        ds1 = None
        ds2 = None
        return str(dst_path)
    
    def _warp_tile_relative_diff(self, dst_path: Path, src_path: Path, warp_options: dict):
        """
        Compute (RH*_Q0 - RH*_Q2) / RH*_Q1 at downsampled resolution.
        Output is scaled by 1000 and stored as Int16 to preserve precision.
        """
        warp_opts = gdal.WarpOptions(**warp_options)
        path_left = src_path.with_name(f'RH{self.rh_idx}_Q0.tif')
        path_right = src_path.with_name(f'RH{self.rh_idx}_Q2.tif')
        path_mid = src_path.with_name(f'RH{self.rh_idx}_Q1.tif')

        ds_left = gdal.Warp('', str(path_left), options=warp_opts)
        ds_right = gdal.Warp('', str(path_right), options=warp_opts)
        ds_mid = gdal.Warp('', str(path_mid), options=warp_opts)

        arr_left = ds_left.GetRasterBand(1).ReadAsArray().astype(np.float32)
        arr_right = ds_right.GetRasterBand(1).ReadAsArray().astype(np.float32)
        arr_mid = ds_mid.GetRasterBand(1).ReadAsArray().astype(np.float32)

        # Mask: any input is nodata, or denominator is zero
        mask = (
            (arr_left == NO_DATA) |
            (arr_right == NO_DATA) |
            (arr_mid == NO_DATA) |
            (arr_mid <= 0)
        )

        rel_diff = np.where(mask, NO_DATA,
                            ((arr_left - arr_right) / arr_mid) * 1000)
        rel_diff = rel_diff.astype(np.int16)

        self._write_tiff(rel_diff, dst_path, ds_left, band_name=f'RH{self.rh_idx}_Q0-Q2_over_Q1')
        ds_left = ds_right = ds_mid = None
        return str(dst_path)

    def _warp_tile_bias_correction(self, dst_path: Path, src_path: Path, vrt_dir: Path, warp_options: dict):
        '''
        Warp a tile to the target projection and resolution with bias correction
        '''
        warp_opts = gdal.WarpOptions(**warp_options)
        offset = get_bias_correction_offset(self.bias_dir, src_path.stem, self.rh_idx,
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
        warp_options = dict(self.downsample_warp_options)
        tasks = []
        downsampled_paths = []
        for tile_id in self.tile_ids:
            dst_path = temp_dir / f'{tile_id}.tif'
            downsampled_paths.append(str(dst_path))
            if dst_path.exists():
                continue
            src_path = dask.delayed(self._get_path_from_stac_item)(tile_id)
            # self._warp_tile(src_path, dst_path, warp_options)
            tasks.append(dask.delayed(self._warp_tile)(dst_path, src_path, warp_options, band_name=f'RH{self.rh_idx}_Q{self.q_idx}'))
        with ProgressBar():
            dask.compute(tasks, scheduler="processes", num_workers=8)

        geotiff_dir = self.save_dir / 'geotiff'
        geotiff_dir.mkdir(parents=True, exist_ok=True)
        mosaic_path = self.save_dir / f"RH{self.rh_idx}_Q{self.q_idx}.tif"
        self._mosaic_tiles(downsampled_paths, mosaic_path)
        cog_dir = self.save_dir / 'cog'
        cog_dir.mkdir(parents=True, exist_ok=True)
        cog_path = cog_dir / f"RH{self.rh_idx}_Q{self.q_idx}.tif"
        cog_path = self._to_cog(mosaic_path)
        print(f"✅ Global mosaic written to {cog_path}")
    
    def create_global_diff_mosaic(self, left_q_idx: int=0, right_q_idx: int=2):
        '''
        Create a global mosaic of the difference between two tiles
        '''
        # Step 1: Warp the tiles to 1km resolution
        temp_dir_name = f"tmp_tiles_resampled_1km_RH{self.rh_idx}_Q{left_q_idx}-Q{right_q_idx}"
        temp_dir = self.init_tmp_dir(self.save_dir / temp_dir_name)
        warp_options = dict(self.downsample_warp_options_mem)
        tasks = []
        downsampled_paths = []
        for tile_id in self.tile_ids:
            dst_path = temp_dir / f'{tile_id}.tif'
            downsampled_paths.append(str(dst_path))
            if dst_path.exists():
                continue
            src_path = dask.delayed(self._get_path_from_stac_item)(tile_id)
            tasks.append(dask.delayed(self._warp_tile_diff)(dst_path, src_path, warp_options))
        with ProgressBar():
            dask.compute(tasks, scheduler="processes", num_workers=8)

        # Step 2: Mosaic the tiles
        geotiff_dir = self.save_dir / 'geotiff'
        geotiff_dir.mkdir(parents=True, exist_ok=True)
        mosaic_path = geotiff_dir / f"RH{self.rh_idx}_Q{left_q_idx}-Q{right_q_idx}.tif"
        self._mosaic_tiles(downsampled_paths, mosaic_path)

        # Step 3: Translate the mosaic to cog
        cog_dir = self.save_dir / 'cog'
        cog_dir.mkdir(parents=True, exist_ok=True)
        cog_path = cog_dir / f"RH{self.rh_idx}_Q{left_q_idx}-Q{right_q_idx}.tif"
        self._to_cog(mosaic_path, cog_path)
        print(f"✅ Global mosaic written to {cog_path}")

    def create_global_relative_diff_mosaic(self):
        """
        Create a global mosaic of (Q0 - Q2) / Q1, scaled by 1000.
        """
        temp_dir_name = (
            f"tmp_tiles_resampled_1km_RH{self.rh_idx}_Q0-Q2_over_Q1"
        )
        temp_dir = self.init_tmp_dir(self.save_dir / temp_dir_name)
        warp_options = dict(self.downsample_warp_options_mem)
        tasks = []
        downsampled_paths = []
        for tile_id in self.tile_ids:
            dst_path = temp_dir / f'{tile_id}.tif'
            downsampled_paths.append(str(dst_path))
            if dst_path.exists():
                continue
            src_path = dask.delayed(self._get_path_from_stac_item)(tile_id)
            tasks.append(dask.delayed(self._warp_tile_relative_diff)(dst_path, src_path, warp_options))
        with ProgressBar():
            dask.compute(tasks, scheduler="processes", num_workers=8)


        geotiff_dir = self.save_dir / 'geotiff'
        geotiff_dir.mkdir(parents=True, exist_ok=True)
        mosaic_path = geotiff_dir / f"RH{self.rh_idx}_Q0-Q2_over_Q1.tif"
        self._mosaic_tiles(downsampled_paths, mosaic_path)

        cog_dir = self.save_dir / 'cog'
        cog_dir.mkdir(parents=True, exist_ok=True)
        cog_path = cog_dir / f"RH{self.rh_idx}_Q0-Q2_over_Q1.tif"
        self._to_cog(mosaic_path, cog_path)
        print(f"✅ Global relative diff mosaic written to {cog_path}")

    def create_global_bias_correction_mosaic(self):
        '''
        Create a global mosaic with bias correction applied
        NOTE: experimental, may not work
        '''
        downsampled_tmp_dir = self.init_tmp_dir(
            self.save_dir / f"tmp_tiles_resampled_1km_RH{self.rh_idx}_Q{self.q_idx}")
        vrt_tmp_dir = self.init_tmp_dir(self.save_dir / f"vrt_tmp")

        warp_options = dict(self.downsample_warp_options)
        tasks = [
            dask.delayed(self._warp_tile_bias_correction)(tile_id, downsampled_tmp_dir, vrt_tmp_dir, warp_options)
            for tile_id in self.tile_ids]
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

