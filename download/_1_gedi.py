import os
import ee
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
import dask
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgp
import matplotlib.pyplot as plt
from shapely.geometry import shape
import requests
from retry import retry
from io import StringIO
from typing import Any
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from download._const import dtypes
from download._utils import authenticate, shapely_to_geojson, ee_fc_to_gpd
from download._dask_downloader import DaskDownloader
from dotenv import load_dotenv
import pyarrow.parquet as pq
load_dotenv()


def is_parquet_ok(path):
    try:
        pq.read_metadata(path)   # only reads footer + schema
        return True
    except Exception:
        return False


authenticate()

logger = logging.getLogger(__name__)
GEDI_START = pd.Timestamp('2018-01-01')


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


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

    def __init__(self, year=2019, save_dir='GEDI/2019', flag_dir: str = None,
                 mgrs_file='GEDI/mgrs_with_count_and_orbits.parquet', key_file: str = None, npartitions=100,
                 n_parallel=40, row_group_size=100, random_state: int = 42, rewrite: bool = False, **kwargs):
        """
        Initializes a GEDI object.

        Args:
            year (int, optional): The year of the GEDI data to download. Defaults to 2019.
            dataFolder (str, optional): The local folder to save the downloaded GEDI data. Defaults to 'gedi'.
            keepLeafOff255 (bool, optional): Whether to keep GEDI points with leaf off flag of 255. Defaults to False.
        """
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.mgrs_file = Path(mgrs_file).expanduser()
        self.save_dir = Path(f'{save_dir}').expanduser()
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.flag_dir = Path(flag_dir).expanduser()
        self.flag_dir.mkdir(exist_ok=True, parents=True)
        self.npartitions = npartitions
        self.n_parallel = n_parallel
        self.row_group_size = row_group_size
        self.year = year
        self.filter = 'quality_flag==1 && degrade_flag==0 && region_class>0 && leaf_off_flag!=1 && sensitivity>=0.95'
        self.random_state = random_state
        self.rewrite = rewrite
        self.key_file = key_file

        if not self.mgrs_file.exists():
            logger.info(f'mgrs file {self.mgrs_file} not found, download from GEE...')
            from download._0_mgrs import MGRS
            mgrs = MGRS(self.mgrs_file, self.save_dir / 'missing.csv')
            mgrs.get_mgrs()

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
        mgrs_df = gpd.read_parquet(self.mgrs_file)
        # row = mgrs_df[mgrs_df['MGRS_UTM']=='04L'].iloc[0]
        # self.download_zone(row).compute()
        self.schedule_tasks(mgrs_df, self.download_zone)

    def sup_download(
            self, mgrs_stats_file: str = None, old_gedi_dir: str = None, new_gedi_dir: str = None, flag_dir: str = None):
        self.old_gedi_dir = Path(old_gedi_dir).expanduser()
        self.new_gedi_dir = Path(f'{new_gedi_dir}/{self.year}').expanduser()
        self.new_gedi_dir.mkdir(exist_ok=True, parents=True)
        self.flag_dir = Path(flag_dir).expanduser()
        self.flag_dir.mkdir(exist_ok=True, parents=True)

        mgrs_df = gpd.read_parquet(self.mgrs_file)
        if 'no_sup' not in mgrs_df.columns:
            mgrs_stats = gpd.read_parquet(mgrs_stats_file)
            mgrs_stats = mgrs_stats.reset_index()
            mgrs_stats['no_sup'] = mgrs_stats['nwant'] - mgrs_stats[f'count_high_sens_{self.year}']
            mgrs_df = mgrs_df.merge(mgrs_stats[['MGRS_UTM', 'no_sup']], on='MGRS_UTM')
            mgrs_df['no_sup'] = mgrs_df['no_sup'].astype(int)
            mgrs_df = mgrs_df[mgrs_df['no_sup'] > 0]
        self.schedule_tasks(mgrs_df, self.download_zone_sup)
        # self.download_zone_sup(mgrs_df[mgrs_df['MGRS_UTM']=='23J'])

    @dask.delayed
    def download_zone_sup(self, zone):
        """
        Get new GEDI points for S2 download
        """
        number = zone['no_sup']  # .item())
        zone_name = zone['MGRS_UTM']  # .item()
        flag = self.flag_dir / f'{zone_name}_{self.year}_done'
        if flag.exists() and not self.rewrite:
            logger.info(f"{flag} exists")
            return
        old_df = gpd.read_parquet(self.old_gedi_dir/f'{zone_name}.parquet')
        files = list(self.save_dir.glob(f'{zone_name}/*.parquet'))
        if len(files) == 0:
            print(f'no files found for {zone_name}')
            print(zone)
            return
        new_df = dgp.read_parquet(files)
        old = old_df[old_df['path'].str.contains(str(self.year))]
        new_df = new_df[~new_df['shot_number'].isin(old['shot_number'])]
        new_df = new_df.compute()
        new_df = dgp.from_geopandas(new_df.iloc[:number], chunksize=600)
        new_df.to_parquet(self.new_gedi_dir/f'{zone_name}', name_function=lambda x: 'partition_' + str(x) + '.parquet')
        flag.touch()

    def get_orbit_for_s2_tiles(
            self, s2_table_file: str, correction_number_per_tile: int = 2000, save_dir: str = None,
            exclude_used_gedi_points: bool = False, used_gedi_points_dir: str = None, **kwargs):
        '''
        Find the orbit id from the growing season for each S2 tile
        Each orbit records ~1.5 hours of GEDI footprints
        Ags:
            * s2_table_file: the path to the S2 table file, should have columns: Name, geometry, growing_months

        '''
        gedi_table_index_file = f'~/data/gvs/GEDI_for_correction/l2a_table_index_{self.year}.parquet'
        self.gedi_table_index = self.download_gedi_table_index(out_file=gedi_table_index_file)
        self.gedi_table_index = self.gedi_table_index.drop(columns=['system:index', 'time_end'])
        self.gedi_table_index['month'] = self.gedi_table_index['time_start'].str[5:7]
        self.gedi_table_index['month'] = self.gedi_table_index['month'].astype(int)
        self.gedi_table_index.set_crs(epsg=4326, inplace=True)
        self.correction_number_per_tile = correction_number_per_tile
        self.exclude_used_gedi_points = exclude_used_gedi_points
        self.used_gedi_points_dir = Path(used_gedi_points_dir).expanduser()
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(exist_ok=True, parents=True)
        s2_table_file = Path(s2_table_file).expanduser()
        s2_table = gpd.read_parquet(s2_table_file)
        tasks = []
        for idx, row in s2_table.iterrows():
            tasks.append(self.get_orbit_for_one_s2_tile(row))
        self.schedule_tasks(delayed_tasks=tasks)

    @dask.delayed
    def get_orbit_for_one_s2_tile(self, s2_tile: pd.DataFrame):
        '''
        Find the orbit id from the growing season for each S2 tile
        Each orbit records ~1.5 hours of GEDI footprints
        Ags:
            * s2_tile: the S2 tile dataframe
        '''
        file = self.save_dir / f'{s2_tile["Name"]}.parquet'
        if file.exists() and not self.rewrite and is_parquet_ok(file):
            print(f'{file} exists and is ok')
            return
        orbits_intersects_tile = self.gedi_table_index[self.gedi_table_index.intersects(s2_tile['geometry'])]#!!!! inside intersects it has to be a geometry object, otherwise it will try to match the index!!!
        if len(orbits_intersects_tile) == 0:
            return
        growing_months = s2_tile['growing_months']
        growing_months = [int(i) for i in growing_months]
        orbits_in_growing_months = orbits_intersects_tile[orbits_intersects_tile['month'].isin(growing_months)]
        data = []
        for orbit_id in orbits_in_growing_months['table_id']:
            fc = ee.FeatureCollection(orbit_id)
            geom = shapely_to_geojson(s2_tile['geometry'])
            fc = fc.filterBounds(geom).filter(self.filter).filter(ee.Filter.inList('pft_class', list(range(1, 9))))
            fc = ee_fc_to_gpd(fc)
            if fc is None:
                continue
            # print(f'Found {len(fc)} GEDI points for {orbit_id}')
            data.append(fc)
        if len(data) == 0:
            return
        data = pd.concat(data)
        if self.exclude_used_gedi_points:
            used_gedi_points_file = self.used_gedi_points_dir / f'{s2_tile["Name"]}.parquet'
            used_gedi_points = gpd.read_parquet(used_gedi_points_file, columns=['geometry'])
            data = data[~data.geometry.isin(used_gedi_points['geometry'])]

        if len(data) > self.correction_number_per_tile:
            print(
                f'Sampling {self.correction_number_per_tile} GEDI points from {len(data)} GEDI points for {s2_tile["Name"]}')
            data = data.sample(self.correction_number_per_tile)
        data = data.reset_index(drop=True)
        data['system:index'] = data['system:index'].astype(str)
        data.to_parquet(file)
        print(f'Saved {len(data)} GEDI points for {s2_tile["Name"]} for GVS correction')

    def download_gedi_table_index(self, out_file: str):
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
        out_file = Path(out_file).expanduser()
        out_file.parent.mkdir(exist_ok=True, parents=True)
        if out_file.exists() and not self.rewrite:
            gdf = gpd.read_parquet(out_file)
            return gdf
        gedi_table_index = ee.FeatureCollection("LARSE/GEDI/GEDI02_A_002_INDEX") \
            .filter(ee.Filter.stringContains('time_start', str(self.year)))
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
            raise requests.HTTPError(f'Failed to download {self.gedi_table_index_file}')

    def get_used_gedi_points(self, s2_tile_file: str, parquet_dir: str, save_dir: str = None):
        '''
        Get the GEDI points used for GVS training, evaluation and calibration
        '''
        s2_tile_file = Path(s2_tile_file).expanduser()
        parquet_dir = Path(parquet_dir).expanduser()
        save_dir = Path(save_dir).expanduser()
        s2_tile = gpd.read_parquet(s2_tile_file, columns=['Name', 'geometry'])
        temp_dir = save_dir / 'temp'
        temp_dir.mkdir(exist_ok=True, parents=True)
        gedi_parq_files = list(parquet_dir.glob('*_v1.parquet'))
        for gedi_parq_file in gedi_parq_files:
            gedi_df = pd.read_parquet(gedi_parq_file, columns=['lat', 'lon'])
            split_name = gedi_parq_file.stem.split('_')[0]
            self.get_used_gedi_points_for_one_parquet(gedi_df, s2_tile, temp_dir, split_name)
        # merge points from all splits into one file

        @dask.delayed
        def process_tile(tile_id):
            parquet_files = list(temp_dir.glob(f'{tile_id}*.parquet'))
            file = save_dir / f'{tile_id}.parquet'
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

    def get_tiles_covered_by_gedi(self, s2_tiles_file: str, gedi_table_index_file: str = None, save_dir: str = None):
        '''
        Get the tiles covered by GEDI
        
        Args:
            * s2_tiles_file: the path to the S2 tiles file, should have columns: Name, geometry
            * gedi_table_index_file: the path to the GEDI table index file, should have columns: system:index, table_id, start_time, end_time, geometry
            * save_dir: the path to save the tiles covered by GEDI
        
        Output:
            * tiles_covered_by_gedi.txt: a text file with the tile names in the GEDI range
            Each line is a tile name
        '''
        s2_tiles_file = Path(s2_tiles_file).expanduser()
        s2_tiles = gpd.read_parquet(s2_tiles_file)
        gedi_table_index_file = Path(gedi_table_index_file).expanduser()
        gedi_table_index = gpd.read_parquet(gedi_table_index_file)
        
        tiles_covered_by_gedi = s2_tiles[s2_tiles.intersects(gedi_table_index['geometry'])]['Name'].unique()
        tiles_covered_by_gedi.to_csv(save_dir / 's2_tiles_covered_by_gedi.txt', header=None, index=None, sep=' ', mode='w')


@dataclass
class MyConfig:
    year: int = 2019
    n_parallel: int = 100
    row_group_size: int = 100
    rewrite: bool = False
    save_dir: str = '~/data/gvs/GEDI_high_sens'
    mgrs_file: str = '~/data/GEDI/mgrs_stats_v3.parquet'
    mgrs_stats_file: str = '~/data/GEDI/mgrs_stats_v1.parquet'
    flag_dir: str = '~/data/gvs/GEDI_extra_flags'
    old_gedi_dir: str = '~/data/gvs/geo_index_table_with_sensitivity'
    new_gedi_dir: str = '~/data/gvs/GEDI_extra'
    key_file: str = 'keys/private-key.json'
    task: str = 'download'
    debug: bool = False
    exclude_used_gedi_points: bool = False
    used_gedi_points_dir: str = '~/data/gvs/fitting_data_coord_partitions'
    correction_number_per_tile: int = 2000


cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)


@hydra.main(config_name='config', version_base='1.2')
def main(cfg):
    from dask.distributed import Client, LocalCluster
    from dask import config
    config.set({'interface': 'lo'})
    cluster = LocalCluster()
    client = Client(cluster)  # timeout
    gedi = GEDI(**cfg)
    if cfg.task == 'download':
        # gedi.download()
        gedi.sup_download(cfg.mgrs_stats_file, cfg.old_gedi_dir, cfg.new_gedi_dir, flag_dir='~/data/gvs/GEDI_sup_flags')
    elif cfg.task == 'download_gedi_for_gvs_correction':
        gedi.get_orbit_for_s2_tiles(
            '~/data/gvs/s2_tiles_with_growing_months.parquet', save_dir=cfg.save_dir,
            correction_number_per_tile=cfg.correction_number_per_tile,
            exclude_used_gedi_points=cfg.exclude_used_gedi_points, used_gedi_points_dir=cfg.used_gedi_points_dir)
    elif cfg.task == 'get_used_gedi_points':
        gedi.get_used_gedi_points('~/data/gvs/s2_tiles_with_growing_months.parquet',
                                  '~/data/gvs/train_subsets', save_dir=cfg.used_gedi_points_dir)
    elif cfg.task == 'get_tiles_covered_by_gedi':
        gedi.get_tiles_covered_by_gedi(s2_tiles_file='~/data/gvs/s2_tiles_with_growing_months.parquet',
                                       gedi_table_index_file=f'~/data/gvs/GEDI_for_correction/l2a_table_index_{cfg.year}.parquet',
                                       save_dir=f'~/data/gvs/deploy/correction_${cfg.year}')
    else:
        logger.error(f'Task not recognized.\nreceived: {cfg.task} \nexpected: download or visualize')


if __name__ == '__main__':
    main()
