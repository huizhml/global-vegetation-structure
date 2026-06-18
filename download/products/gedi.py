import os
import ee
import json
import shutil
import logging
from typing import List
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import dask
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgp
import matplotlib.pyplot as plt
from shapely.geometry import shape
import requests
from retry import retry
from io import StringIO
from dotenv import load_dotenv
import pyarrow.parquet as pq
import xarray as xr
import pystac
import pystac_client
import planetary_computer
from xrspatial import slope
from shapely.geometry import box
from dask.distributed import Client, LocalCluster

from download.core import DaskDownloader
from download.core.utils import authenticate, shapely_to_geojson, ee_fc_to_gpd, get_aux_df, row_to_stac_item, get_patch, get_epsg_from_tile, buffer_and_snap_bounds, get_total_bounds, check_unfinished_files, read_parquets_with_file_name
from download.core.constants import dtypes
load_dotenv()


def is_parquet_ok(path):
    try:
        pq.read_metadata(path)   # only reads footer + schema
        return True
    except Exception:
        return False


# authenticate()  # TODO: only authenticate when needed

stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace)

logger = logging.getLogger(__name__)
GEDI_START = pd.Timestamp('2018-01-01')


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def download_gedi_table_index( year: int, out_file: Path, rewrite: bool = False):
    
    '''
    Download the GEDI table index
    Reason to keep this data local:
    1. At the time of writing, we need to sample a small subset of global GEDI data for Sentinel-2 tile level bias correction.
        There are 20406 and 10228 orbits in 2020 and 2024, the table index data has only three columns, table_id, start_time, end_time.
        It's a very small data
    2. We want to do sample the GEDI footprints from the tile-level growing season, it's easier to do the table_id (orbit_id) filtering locally.
    Output:
        * gedi_table_index_file: the path to the GEDI table index file, should have columns: system:index, table_id, start_time, end_time, geometry

    '''
    out_file.parent.mkdir(exist_ok=True, parents=True)
    if out_file.exists() and not rewrite:
        gdf = gpd.read_parquet(out_file)
        return gdf
    gedi_table_index = ee.FeatureCollection("LARSE/GEDI/GEDI02_A_002_INDEX") \
        .filter(ee.Filter.stringContains('time_start', str(year)))
    download_id = ee.data.getTableDownloadId({'table': gedi_table_index, 'fileFormat': 'csv'})
    res = requests.get(ee.data.makeTableDownloadUrl(download_id))
    if res.status_code == 200:
        data = StringIO(res.content.decode('utf-8'))
        df = pd.read_csv(data)
        geom = df['.geo'].apply(lambda x: shape(json.loads(x)))
        gdf = gpd.GeoDataFrame(df, geometry=geom)
        gdf = gdf.drop(columns=['.geo'])
        gdf.to_parquet(out_file)
        return gdf
    else:
        raise requests.HTTPError(f'Failed to download {out_file}')

class GEDI(DaskDownloader):
    """
    A class for filtering, sampling(stratified, per orbit & per cell) and downloading GEDI data from Google Earth Engine.

    Attributes:
        nSampledPerKm2 (float): The number of GEDI points to sample per km^2.
        raster (ee.ImageCollection): The GEDI raster data collection in Google Earth Engine.
        grid (ee.FeatureCollection): The MGRS grid feature collection in GEE, each grid cell has assigned 'count'(filtered GEDI Points), 'landmass'.
        year (int): The year of the GEDI data to download.
        dataFolder (pathlib.Path): The local folder to save the downloaded GEDI data.
        keepLeafOff255 (bool): Whether to keep GEDI points with leaf off flag of 255.

    Methods:
        download_all_valid(s2_grid_file: str, gedi_table_index_file: str, save_dir: str = None, **kwargs):
            Download all valid GEDI points for all S2 tiles.
            Valid: (quality flag == 1, degrade flag == 0, region class > 0, leaf off flag != 1, sensitivity >= 0.95) for the growing season
            NOTE: keep a local copy for future use, better indexing and faster retrieval.
            Args:
                * s2_grid_file: the path to the S2 table file, should have columns: Name, geometry, growing_months
                * gedi_table_index_file: the path to the GEDI table index file, should have columns: system:index, table_id, start_time, end_time, geometry
                * save_dir: the path to save the downloaded GEDI data
            Returns:
                * None

        add_slope(s2_grid_file: str, gedi_table_index_file: str, save_dir: str = None, **kwargs):
            Add slope to the GEDI data, where the slope is calculated from the DEM data (Copernicus DEM + NASADEM).
            NOTE: slope information indicates how representative the GEDI vertical profile is, higher slope means less representative.
            Args:
                * location_dir: the path to the location files
                * dem_meta_file: the path to the DEM metadata file
                * save_dir: the path to save the GEDI data with slope
                NOTE: the save directory should be the same as the location files directory
            Returns:
                * None

        sample_subset(s2_grid_file: str, save_dir: str = None, correction_number_per_tile: int = 2000, exclude_used_gedi_points: bool = False, used_gedi_points_dir: str = None, **kwargs):
            Sample GEDI points for GVS correction. 
            NOTE: slope filter can be added here
            Args:
                * s2_grid_file: the path to the S2 table file, should have columns: Name, geometry, growing_months
                * save_dir: the path to save the sampled GEDI data
                * correction_number_per_tile: the number of GEDI points to sample per tile
                * exclude_used_gedi_points: whether to exclude used GEDI points
                * used_gedi_points_dir: the path to the used GEDI points file
            Returns:
                * None

        check_slope_distribution(save_dir: str):
            Check the distribution of the slope data
            Args:
                * save_dir: the path to save the GEDI data with slope
            Returns:
                * None


        getTableAssetIds(export=False):
            Returns a list of unique table asset IDs for GEDI orbits from GEE rasterized version of GEDI within the specified year.

        getValidZoneCodes(export=False):
            Returns a list of valid zone codes of MRGS grid cells where GEDI points are present.

        download(processes=40, maxTries=3):
            Downloads GEDI data for all valid MGRS grid cells in the specified year.

        filterGEDI(asset_id):
            Returns a filtered GEDI feature collection for specified table asset ID.
            Filter by quality flag, degrade flag, region class, and growing season.

        sampleCell(zone_code):
            Downloads GEDI data for the specified MGRS grid cell.

        getSampleRatio(cell):
            Returns the number of GEDI points to sample for the specified MGRS grid cell.
    """
    

    # this is obtained from the total number of points we want to sample per year(75M) and the total landmass in the world
    nSampledPerKm2 = 0.7033933006651656
    GEDI_START = pd.Timestamp('2018-01-01')
    _ee_initialized = False

    def __init__(self, year=2019,
                 # all needed
                 save_dir: str = None,
                 # original download
                 mgrs_file: str = 'GEDI/mgrs_with_count_and_orbits.parquet',
                 # download all valid
                 s2_grid_file: str = None,
                 gedi_table_index_file: str = None,
                 # add slope
                 location_dir: str = None,
                 dem_meta_file: str = None,
                 # check slope distribution
                 deploy_status_dir: str = None,
                 input_dir: str = None,
                 used_parq_dir: str = None,
                 # optional
                 flag_dir: str = None,
                 key_file: str = None, npartitions=100,
                 sensitivity_threshold: float = 0.95,
                 n_parallel=40, row_group_size=100, random_state: int = 42, rewrite: bool = False, **kwargs):
        """
        Initializes a GEDI object.

        Args:
            year (int, optional): The year of the GEDI data to download. Defaults to 2019.
            dataFolder (str, optional): The local folder to save the downloaded GEDI data. Defaults to 'gedi'.
            keepLeafOff255 (bool, optional): Whether to keep GEDI points with leaf off flag of 255. Defaults to False.
        """
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.mgrs_file = mgrs_file
        self.s2_grid_file = s2_grid_file
        self.gedi_table_index_file = gedi_table_index_file
        self.location_dir = location_dir
        self.dem_meta_file = dem_meta_file
        self.deploy_status_dir = deploy_status_dir
        self.input_dir = input_dir
        self.used_parq_dir = used_parq_dir
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # flag_dir is only used by the legacy download_zone path; other methods
        # (e.g. download_all_valid) don't need it.
        if flag_dir is not None:
            self.flag_dir = Path(flag_dir).expanduser()
            self.flag_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.flag_dir = None
        self.npartitions = npartitions
        self.n_parallel = n_parallel
        self.row_group_size = row_group_size
        self.year = year
        self.sensitivity_threshold = sensitivity_threshold
        base_filter = 'quality_flag==1 && degrade_flag==0 && region_class>0 && leaf_off_flag!=1'
        self.filter = (
            f'{base_filter} && sensitivity>={sensitivity_threshold}'
            if sensitivity_threshold is not None else base_filter
        )
        self.random_state = random_state
        self.rewrite = rewrite
        self.key_file = key_file
        
        self._ensure_authenticated()

    def _ensure_authenticated(self):
        """Ensures Earth Engine is initialized exactly once per process."""
        if not GEDI._ee_initialized:
            authenticate()  # Your existing utility function
            # Or directly: ee.Initialize(project='your-project')
            GEDI._ee_initialized = True

    @dask.delayed
    def download_zone(self, zone):
        print('downloading ', zone['MGRS_UTM'])
        if zone[f'count_{self.year}'] == 0:
            logger.info(f"no GEDI points in {zone['MGRS_UTM']}")
            return None
        flag = self.flag_dir / f'{zone["MGRS_UTM"]}_{self.year}_done'
        if flag.exists() and not self.rewrite:
            logger.info(f"{flag} exists")
            return None
        zone_dir = self.save_dir / zone["MGRS_UTM"]
        zone_dir.mkdir(exist_ok=True, parents=True)
        geom = ee.Geometry.BBox(*zone['geometry'].bounds).toGeoJSON()
        last_coords = geom['coordinates'][0][0].copy()
        geom['coordinates'][0].append(last_coords)
        sample_ratio = zone['landmass'] * self.nSampledPerKm2/zone[f'count_{self.year}'] + 0.001
        total = sampled = 0
        new_tracks = defaultdict(list)
        for track_id in zone[f'tracks_{self.year}']:
            filename = track_id.split('/')[-1]
            if (zone_dir / f'{filename}.parquet').exists() and not self.rewrite:
                continue
            fc = ee.FeatureCollection(track_id).filterBounds(geom).filter(self.filter)
            orbit_size = fc.size().getInfo()
            if orbit_size > 0:
                total += orbit_size
                sampled += self.download_orbit(fc, sample_ratio, zone_dir, filename)
                new_tracks[self.year].append(track_id)

        logger.info(f'{zone["MGRS_UTM"]}, Total: {total}, sampled: {sampled}, #want: {total * sample_ratio}')
        logger.info(zone["MGRS_UTM"], new_tracks)
        if len(os.listdir(zone_dir)) > 0:
            flag.touch()
        else:
            os.rmdir(zone_dir)  # remove empty directory

    @retry(requests.HTTPError, tries=10, delay=1)
    def download_orbit(self, fc, sample_ratio, zone_dir, filename):
        file = zone_dir / f'{filename}.parquet'
        if file.exists() and not self.rewrite:
            df = gpd.read_parquet(file)
            return len(df)
        if sample_ratio < 1:
            fc = fc.randomColumn('random', seed=self.random_state).filter(ee.Filter.lte('random', sample_ratio))
        sampled = fc.size().getInfo()
        if sampled > 0:
            download_id = ee.data.getTableDownloadId({'table': fc, 'fileFormat': 'csv'})
            res = requests.get(ee.data.makeTableDownloadUrl(download_id))
            if res.status_code == 200:
                data = StringIO(res.content.decode('utf-8'))
                df = pd.read_csv(data)
                df['.geo'] = df['.geo'].apply(lambda x: shape(json.loads(x)))
                df = df.rename(columns={'.geo': 'geometry'})
                df = gpd.GeoDataFrame(df, geometry='geometry')
                df = df[['geometry', *dtypes.keys()]]
                df = df.astype(dtypes)
                df['date'] = pd.to_timedelta(df['delta_time'], unit='s') + GEDI_START
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                df.to_parquet(file, row_group_size=self.row_group_size, engine='pyarrow')
                print(f'{filename} saved')
            else:
                raise requests.HTTPError(f'Failed to download {filename}')
        return sampled

    def download(self):
        """
        Downloads GEDI data for all valid MGRS grid cells in the specified year.
        """
        cluster = LocalCluster()
        client = Client(cluster)  # timeout
        print(client)
        mgrs_file = Path(self.mgrs_file).expanduser()
        if not mgrs_file.exists():
            raise FileNotFoundError(f'{mgrs_file} not found')
        mgrs_df = gpd.read_parquet(self.mgrs_file)
        # row = mgrs_df[mgrs_df['MGRS_UTM']=='04L'].iloc[0]
        # self.download_zone(row).compute()
        self.schedule_tasks(mgrs_df, self.download_zone)

    def download_all_valid(self):
        '''
        Resume-aware per-orbit pipeline.

        Pivots from the old per-tile loop (one GEE call per (tile, orbit) pair)
        to one GEE call per orbit serving all overlapping pending tiles. Skips
        tiles whose final parquet already exists & is valid.

        Phase 1 (GEE-bound, parallel): fetch each pending orbit once filtered to
                the union of pending tile geoms it covers, sjoin locally, dedup
                each shot to its canonical (alphabetically-first Name) pending
                tile to avoid S2-overlap duplicates, stage per-tile slices to
                save_dir/.staging/<tile>/<orbit>.parquet.
        Phase 2 (local CPU, parallel): per pending tile, concat all staging
                slices into save_dir/<tile>.parquet and remove the staging dir.

        Args (via __init__ / yaml):
            * s2_grid_file: S2 tile table with columns Name, geometry, growing_months
            * gedi_table_index_file: GEDI orbit index (downloaded if missing)
        '''
        cluster = LocalCluster()
        client = Client(cluster)  # timeout
        print(client)

        gedi_table_index_file = Path(self.gedi_table_index_file).expanduser()
        gedi_index = download_gedi_table_index(year=self.year, out_file=gedi_table_index_file)
        gedi_index = gedi_index.drop(columns=['system:index', 'time_end'])
        gedi_index['month'] = gedi_index['time_start'].str[5:7].astype(int)
        gedi_index = gedi_index.set_crs(epsg=4326)

        s2_grid_file = Path(self.s2_grid_file).expanduser()
        s2_table = gpd.read_parquet(s2_grid_file)

        existing = {p.stem for p in self.save_dir.glob('*.parquet') if is_parquet_ok(p)}
        pending_tiles = s2_table[~s2_table['Name'].isin(existing)].reset_index(drop=True)
        print(f'{len(existing)} tiles done, {len(pending_tiles)} pending')
        if len(pending_tiles) == 0:
            print('All tiles done.')
            return

        # Spatial join (pending tiles × orbits) + growing-month filter, in one pass
        pairs = gpd.sjoin(
            pending_tiles[['Name', 'geometry', 'growing_months']],
            gedi_index[['table_id', 'month', 'geometry']],
            predicate='intersects', how='inner')
        in_growing = pairs.apply(
            lambda r: r['month'] in {int(m) for m in r['growing_months']}, axis=1)
        pairs = pairs[in_growing]
        if len(pairs) == 0:
            print('No (pending_tile, orbit) pairs in growing months.')
            return
        print(f'{pairs["table_id"].nunique()} orbits to fetch covering pending tiles')

        self.staging_dir = self.save_dir / '.staging'
        self.staging_dir.mkdir(exist_ok=True, parents=True)

        orbit_tasks = []
        for orbit_id, group in pairs.groupby('table_id'):
            tile_subset = group[['Name', 'geometry']].drop_duplicates('Name').reset_index(drop=True)
            tile_subset = gpd.GeoDataFrame(tile_subset, geometry='geometry', crs='EPSG:4326')
            orbit_tasks.append(self.fetch_orbit(orbit_id, tile_subset))
        print(f'Phase 1: scheduling {len(orbit_tasks)} orbit fetches')
        self.schedule_tasks(delayed_tasks=orbit_tasks)

        merge_tasks = [self.merge_tile(name) for name in pending_tiles['Name'].unique()]
        print(f'Phase 2: scheduling {len(merge_tasks)} tile merges')
        self.schedule_tasks(delayed_tasks=merge_tasks)

        flag_dir = self.staging_dir / '.orbit_done'
        if flag_dir.exists():
            shutil.rmtree(flag_dir)
        if self.staging_dir.exists() and not any(self.staging_dir.iterdir()):
            self.staging_dir.rmdir()

    @dask.delayed
    @retry(requests.HTTPError, tries=10, delay=1)
    def fetch_orbit(self, orbit_id: str, pending_tiles: gpd.GeoDataFrame):
        '''
        Fetch one GEDI orbit (filtered to union of pending tile geoms),
        assign each shot to its canonical pending tile (alphabetically-first
        Name to dedupe across S2 overlap), and stage per-tile slices.
        '''
        orbit_key = orbit_id.split('/')[-1]
        done_flag = self.staging_dir / '.orbit_done' / orbit_key
        if done_flag.exists() and not self.rewrite:
            return
        done_flag.parent.mkdir(exist_ok=True, parents=True)

        union_geom = pending_tiles.geometry.unary_union
        fc = (ee.FeatureCollection(orbit_id)
              .filterBounds(shapely_to_geojson(union_geom))
              .filter(self.filter)
              .filter(ee.Filter.inList('pft_class', list(range(1, 9)))))
        points = ee_fc_to_gpd(fc)
        if points is None or len(points) == 0:
            done_flag.touch()
            return

        if points.crs is None:
            points = points.set_crs(epsg=4326)
        joined = gpd.sjoin(
            points, pending_tiles[['Name', 'geometry']],
            predicate='intersects', how='inner')
        # Canonical assignment: each shot to its alphabetically-first matched tile
        joined = (joined.sort_values('Name')
                        .drop_duplicates(subset=['shot_number'], keep='first')
                        .drop(columns=['index_right']))

        for tile_name, slice_df in joined.groupby('Name'):
            slice_df = slice_df.drop(columns=['Name']).reset_index(drop=True)
            if 'system:index' in slice_df.columns:
                slice_df['system:index'] = slice_df['system:index'].astype(str)
            out = self.staging_dir / tile_name / f'{orbit_key}.parquet'
            out.parent.mkdir(exist_ok=True, parents=True)
            slice_df.to_parquet(out)

        done_flag.touch()

    @dask.delayed
    def merge_tile(self, tile_name: str):
        '''Concat all staging slices for one tile into the final parquet.'''
        out = self.save_dir / f'{tile_name}.parquet'
        if out.exists() and is_parquet_ok(out) and not self.rewrite:
            return
        tile_staging = self.staging_dir / tile_name
        if not tile_staging.exists():
            return
        parts = sorted(tile_staging.glob('*.parquet'))
        if not parts:
            return
        df = pd.concat([gpd.read_parquet(p) for p in parts], ignore_index=True)
        df.to_parquet(out)
        shutil.rmtree(tile_staging)
        print(f'Saved {len(df)} GEDI points for {tile_name}')

    def add_slope(self):
        '''
        Add slope to the GEDI data, where the slope is calculated from the DEM data (Copernicus DEM + NASADEM).
        NOTE: slope information indicates how representative the GEDI vertical profile is, higher slope means less representative.
        Args:
            * location_dir: the path to the location files
                NOTE: the location files should be in the format of <tile_id>.parquet
            * dem_meta_file: the path to the DEM metadata file
                NOTE: the DEM metadata file should be in the format of <collection_id>_items.parquet
            * save_dir: the path to save the GEDI data with slope
                NOTE: the save directory should be the same as the location files directory
        '''
        cluster = LocalCluster()
        client = Client(cluster)  # timeout
        print(client)
        location_files = Path(self.location_dir).expanduser().glob('*.parquet')
        dem_meta_file = Path(self.dem_meta_file).expanduser()
        self.dem_df = get_aux_df(dem_meta_file, 'cop-dem-glo-30', time_col='datetime')
        self.dem_df['datetime'] = self.dem_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

        location_files = check_unfinished_files(location_files, self.save_dir, check_exists=True)
        test_tile = '10TEL'
        location_files = [file for file in location_files if test_tile in file.stem]
        tasks = []
        for file in location_files:
            tasks.append(self.get_dem(file))
        self.schedule_tasks(delayed_tasks=tasks)

    @dask.delayed
    @retry(requests.HTTPError, tries=10, delay=1)
    def get_dem(self, file: Path):
        '''
        Get the DEM data for the GEDI points
        #TODO: not sure how it works for large tiles (more than 40k points)
        '''
        output_file = self.save_dir / f'{file.stem}.parquet'
        if output_file.exists() and not self.rewrite:
            print(f'{output_file} exists, skipping')
            return
        tile_id = file.stem
        epsg = get_epsg_from_tile(tile_id)
        loc_df = gpd.read_parquet(file)
        loc_ = loc_df.to_crs(epsg)
        total_bounds = buffer_and_snap_bounds(loc_.geometry, 60, 30)  # buffer 2 pixels
        total_bounds = get_total_bounds(total_bounds)
        bbox = box(*loc_df.geometry.total_bounds)
        dem_df = self.dem_df[self.dem_df.geometry.intersects(bbox)]

        if not dem_df.geometry.union_all().covers(bbox):
            dem_items = api.search(collections=['nasadem'], intersects=bbox).item_collection()
            dem_items.asset_name = 'elevation'
        else:
            dem_items = row_to_stac_item(dem_df, ['datetime'])
            dem_items = pystac.item_collection.ItemCollection(dem_items)
            dem_items.asset_name = 'data'

        if len(dem_items.items) == 0:
            loc_df['slope'] = np.nan
            loc_df.to_parquet(output_file)
            print(f'{output_file} does not have DEM data, skipping')
            return

        dem_image = get_patch(
            dem_items.items, [dem_items.asset_name],
            resolution=30, epsg=epsg, bounds=total_bounds, dtype='float32', fill_value=np.float32(np.nan))
        dem_image = dem_image.max(dim='time', skipna=True).squeeze()
        if dem_image.shape[0] == 0:
            loc_df['slope'] = np.nan
            loc_df.to_parquet(output_file)
            print(f'{output_file} does not have DEM data, skipping')
            return
        dem_image.attrs['res'] = 30
        slope_da = slope(dem_image)
        target_x = xr.DataArray(loc_.geometry.x.values, dims="points")
        target_y = xr.DataArray(loc_.geometry.y.values, dims="points")
        sampled_values = slope_da.sel(x=target_x, y=target_y, method='nearest')
        sampled_values = sampled_values.compute()
        loc_df['slope'] = sampled_values.values
        loc_df.to_parquet(output_file)

    def check_slope_distribution(self):
        '''
        Check the distribution of the slope data
        '''
        input_dir = Path(self.input_dir).expanduser()
        columns = ['geometry', 'slope']
        df = read_parquets_with_file_name(input_dir, columns)
        before_slope_filter = df.groupby('Name')['geometry'].count().reset_index(name='count')
        after_slope_filter = df[df['slope'] < 20].groupby('Name')['geometry'].count().reset_index(name='count')
        valid_tiles_before = before_slope_filter[before_slope_filter['count'] > 200]['Name'].unique()
        valid_tiles_after = after_slope_filter[after_slope_filter['count'] > 200]['Name'].unique()
        affected_tiles = set(valid_tiles_before) - set(valid_tiles_after)
        with open(input_dir.parent / 'figures/affected_tiles_by_slope_filter.txt', 'w') as f:
            for tile in affected_tiles:
                f.write(f'{tile}\n')
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        before_slope_filter['count'].hist(ax=axes[0], bins=100, label='before slope filter')
        after_slope_filter['count'].hist(ax=axes[1], bins=100, label='after slope filter')
        for ax in axes:
            ax.set_yscale('log')
            ax.set_ylim(0, 1e5)
            ax.set_xlabel('Number of GEDI points')
            ax.set_ylabel('#tiles')
            ax.legend()
        plt.savefig(input_dir.parent / 'figures/slope_distribution.png')
        
        deploy_status_dir = Path(self.deploy_status_dir).expanduser()
        s2_files = list(deploy_status_dir.glob('deploy_status_*.parquet'))
        latest_s2_file = sorted(s2_files, key=lambda x: pd.to_datetime(x.stem.split('_')[2]))[-1]
        s2_df = gpd.read_parquet(latest_s2_file)
        after_slope_filter = after_slope_filter.set_index('Name')
        after_slope_filter = after_slope_filter.rename(columns={'count': f'gedi_count_after_slope_filter_{self.year}'})
        s2_df = s2_df.join(after_slope_filter)
        s2_df[f'gedi_count_after_slope_filter_{self.year}'] = s2_df[f'gedi_count_after_slope_filter_{self.year}'].fillna(0).astype(int)
        s2_df.to_parquet(latest_s2_file.with_stem(f'deploy_status_{datetime.now().strftime("%Y-%m-%dT%H")}'))
        return 



    def get_used_gedi_points(self):
        '''
        Get the GEDI points used for GVS training, evaluation and calibration
        '''
        cluster = LocalCluster()
        client = Client(cluster)  # timeout
        print(client)
        s2_tile_file = Path(self.s2_grid_file).expanduser()
        used_parq_dir = Path(self.used_parq_dir).expanduser()
        s2_tile = gpd.read_parquet(s2_tile_file, columns=['Name', 'geometry'])
        temp_dir = self.save_dir / 'temp'
        temp_dir.mkdir(exist_ok=True, parents=True)
        gedi_parq_files = list(used_parq_dir.glob('*_v1.parquet'))
        for gedi_parq_file in gedi_parq_files:
            gedi_df = pd.read_parquet(gedi_parq_file, columns=['lat', 'lon'])
            split_name = gedi_parq_file.stem.split('_')[0]
            self.get_used_gedi_points_for_one_parquet(gedi_df, s2_tile, temp_dir, split_name)
        # merge points from all splits into one file

        @dask.delayed
        def process_tile(tile_id):
            parquet_files = list(temp_dir.glob(f'{tile_id}*.parquet'))
            file = self.save_dir / f'{tile_id}.parquet'
            if len(parquet_files) == 0:  # or (file.exists() and is_parquet_ok(file)):
                return
            gedi_df = dgp.read_parquet(parquet_files, gather_spatial_partitions=False).compute()
            gedi_df.to_parquet(file)
            print(f'Saved {len(gedi_df)} GEDI points to {file}')

        tasks = []
        for tile_id in s2_tile['Name']:
            tasks.append(process_tile(tile_id))
        self.schedule_tasks(delayed_tasks=tasks)

    def get_used_gedi_points_for_one_parquet(
            self, df: pd.DataFrame, s2_tile: pd.DataFrame, temp_dir: Path, split_name: str):
        '''
        Get the GEDI points used for GVS training, evaluation and calibration
        '''
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')
        gdf_ = gpd.sjoin(gdf, s2_tile, how='left', predicate='intersects')
        gdf_.groupby('Name').apply(lambda x: x.to_parquet(temp_dir / f'{x.Name.iloc[0]}_{split_name}.parquet'))
        return df

    def get_tiles_covered_by_gedi(self):
        '''
        Get the tiles covered by GEDI

        Args:
            * s2_grid_file: the path to the S2 tiles file, should have columns: Name, geometry
            * gedi_table_index_file: the path to the GEDI table index file, should have columns: system:index, table_id, start_time, end_time, geometry
            * save_dir: the path to save the tiles covered by GEDI

        Output:
            * tiles_covered_by_gedi.txt: a text file with the tile names in the GEDI range
            Each line is a tile name
        '''
        s2_grid_file = Path(self.s2_grid_file).expanduser()
        s2_tiles = gpd.read_parquet(s2_grid_file)
        gedi_table_index_file = Path(self.gedi_table_index_file).expanduser()
        gedi_table_index = gpd.read_parquet(gedi_table_index_file)

        tiles_covered_by_gedi = s2_tiles[s2_tiles.intersects(gedi_table_index['geometry'])]['Name'].unique()
        tiles_covered_by_gedi.to_csv(self.save_dir / 'tiles_covered_by_gedi.txt', header=None, index=None, sep=' ', mode='w')


def get_unique_shotnumbers(tiles: List[Path]):
    
    records = []
    for t in tiles:
        shots = pq.read_table(t, columns=["shot_number"]).to_pandas()
        shots["_tile"] = t.name
        records.append(shots)

    all_shots = pd.concat(records)
    print(f"Total rows: {len(all_shots)}, unique shot_number: {all_shots['shot_number'].nunique()}")

    # 每个 shot_number 只保留第一个出现的 tile
    keep = all_shots.drop_duplicates(subset=["shot_number"], keep="first")
    print(f'Before: {len(all_shots)}, after: {len(keep)}')
    import ipdb; ipdb.set_trace()
    return keep


def dedup_shots(parq_dir: str, **kwargs):
    '''
    Remove duplicated shoots. Not applied, for furture use
    '''
    parq_dir = Path(parq_dir).expanduser()
    tiles = sorted(parq_dir.glob("*.parquet"))
    keep = get_unique_shotnumbers(tiles)
    for t in tiles:
        tile_name = t.name
        keep_shots = keep[keep["_tile"] == tile_name]["shot_number"]
        
        df = pd.read_parquet(t)
        before = len(df)
        df = df[df["shot_number"].isin(keep_shots.values)]
        after = len(df)
        
        if before != after:
            print(f"{tile_name}: {before} → {after} (removed {before - after})")
            df.to_parquet(str(t) + ".tmp")
            os.replace(str(t) + ".tmp", str(t))

