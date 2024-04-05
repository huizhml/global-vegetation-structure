
# %%
import os
import gc
import time
import logging
from pathlib import Path
import pystac_client
import planetary_computer
import adlfs
import dask
import dask_geopandas as dgp
from dask.distributed import Variable
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
from shapely.geometry import box

import hydra
from dotenv import load_dotenv
import warnings

from download.dask_downloader import DaskDownloader
from download._const import S2_ITEM_PROPS, STAC_ITEM_KEYS
# %%
load_dotenv('.planetarycomputer/settings.env')
g0, g1, g2 = gc.get_count()
gc.set_threshold(g0*5, g1*5, g2 * 5)

logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
# Set the logger level to WARNING, suppressing INFO logs
logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace)

class S2MetaGather(DaskDownloader):

    def __init__(self,
                 rewrite: bool = False,
                 root_dir: str = None,
                 data_dir: str = 'GEDI',
                 save_dir: str = 'data/GEDI',
                 S2_meta_dir: str = 'scratch/S2_meta',
                 s2_grid_file: str = 'GEDI/Sentinel-2_tilling_shp/sentinel_2_index_shapefile.shp',
                 year: int = 2019,
                 query_days: int = 90,
                 n_parallel: int = 100,
                 max_cloud_cover: int = 50,
                 max_water_percentage: int = 99,
                 max_retries: int = 3,
                 **kwargs
                 ) -> None:
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.gedi_dir = root_dir / data_dir
        self.save_dir = root_dir / save_dir
        self.S2_meta_dir = root_dir / S2_meta_dir
        self.temp_dir = root_dir / f'scratch/tmp/{year}' # save temporary s2_items_*.parquet files
        self.rewrite = rewrite
        self.n_parallel = n_parallel
        self.year_start = pd.Timestamp(f'{year}-01-01')
        self.query_days = pd.to_timedelta(query_days, unit='D')
        self.max_cloud_cover = max_cloud_cover
        self.max_water_percentage = max_water_percentage

        self.s2asset = api.get_collection("sentinel-2-l2a").assets["geoparquet-items"]
        self.fs_df = self._build_parquet_file_table()
        self.s2_grid = gpd.read_file(root_dir / s2_grid_file)

    def _build_parquet_file_table(self):
        """
        Sentinel-2 file is partitioned by week. 
        Builds a table with three columns: file name, start date, and end date.
        File names of Sentinel-2 geoparquet files are retrieved from Azure Blob Storage.
        Each row is a STAC geoparquet item.

        Returns:
        ------------
        * pandas.DataFrame: A DataFrame containing the file names, start dates, and end dates.
        """
        fs = adlfs.AzureBlobFileSystem(
            **self.s2asset.extra_fields["table:storage_options"]).ls("items/sentinel-2-l2a.parquet")
        fs_df = pd.DataFrame(fs, columns=['fname'])
        date_range = fs_df.fname.str.findall(r'\d{4}-\d{2}-\d{2}')
        fs_df['start'] = pd.to_datetime(date_range.str[0])
        fs_df['end'] = pd.to_datetime(date_range.str[1])
        fs_df['fname'] = 'abfs://' + fs_df['fname']
        return fs_df

    def _filter_parquet_files(self, start, end):
        """
        Filter Sentinel-2 STAC geoparquet items based on the given start and end timestamps.

        Parameters:
        ------------
        * start (datetime): The start timestamp for filtering.
        * end (datetime): The end timestamp for filtering.

        Returns:
        ------------
        * list: A list of filtered file names.
        """
        filtered = self.fs_df[(self.fs_df['start'] < end) & (self.fs_df['end'] > start)]
        return filtered['fname'].to_list()

    def process_zone(self, zone):
        """
        Gather Sentinel-2 metadate(STAC geoparquet items) we'll read data from for each MGRS zone.
        For each partition of GEDI data, we'll filter Sentinel-2 items based on the bounding box and time range of the partition.

        Parameters
        ------------
        * zone (str or list): The MGRS zone(s) to filter S2 tiles for. GEDI partitions with S2 candidates added will be saved to self.save_dir / zone.
            * str: A large zone, e.g. '20M'. S2 metadata table will be saved to the zone level folder e.g, GEDI/2019/20M, with name 's2_items.parquet'.
            * list: A list of smaller zones, e.g. ['01G', '01K']. S2 metadata table will be saved to folder smaller_zones, e.g, GEDI/2019/smaller_zones, with name 's2_items.parquet'.
        """
        if isinstance(zone, str):
            files = list(self.gedi_dir.glob(f'{zone}/partition*.parquet'))
            (self.save_dir / zone).mkdir(exist_ok=True, parents=True)
            self.temp_dir = self.temp_dir / zone
            s2_table_file = self.S2_meta_dir / f'{zone}.parquet'
        else:
            # Read GEDI partitions from multiple zones, add paths of partitions in divisions
            self.temp_dir = self.temp_dir / 'small_zones'
            s2_table_file = self.gedi_dir / 'small_zones.parquet'
            files = []
            for z in zone:
                files.extend(list(self.gedi_dir.glob(f'{z}/partition*.parquet')))
                (self.save_dir / z).mkdir(exist_ok=True, parents=True)
        
        files = [str(p) for p in files]
        files = sorted(files) # key=natural_sort_key
        divisions = tuple(files + [files[-1]])
        gedi_df = dgp.read_parquet(files, index=False, gather_spatial_partitions=False)
        gedi_df.divisions = divisions 
            
        # tmp dir to cache small S2 metadata tables
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f'Processing {gedi_df.npartitions} partitions...')

        # Derive time window from GEDI date or leaf on/off dates if in growing season
        # leaf off date is in the next year
        reverse = gedi_df['leaf_on_doy'] > gedi_df['leaf_off_doy']
        gedi_df['leaf_off_doy'] = gedi_df['leaf_off_doy'].mask(reverse, gedi_df['leaf_off_doy'] + 365)

        leaf_on_date = dd.to_datetime(gedi_df['leaf_on_doy'], unit='D', origin=self.year_start)
        leaf_off_date = dd.to_datetime(gedi_df['leaf_on_doy'], unit='D', origin=self.year_start)

        gedi_df['start'] = dd.to_datetime(gedi_df['date']) - self.query_days
        gedi_df['end'] = dd.to_datetime(gedi_df['date']) + self.query_days

        gedi_df['date'] = dd.to_datetime(gedi_df['date'])
        use_growing_season = ((gedi_df['date'].ge(leaf_on_date)) &
                              (gedi_df['date'].le(leaf_off_date)) &
                              (gedi_df['leaf_off_flag'] == 0))
        gedi_df['start'] = gedi_df['start'].mask(use_growing_season, leaf_on_date)
        gedi_df['end'] = gedi_df['end'].mask(use_growing_season, leaf_off_date)
        # gedi_df.crs = 'epsg:4326'

        # number = 39
        # self.rewrite = True
        # test_df = gedi_df.get_partition(number).compute()#[:20]
        # division = gedi_df.divisions[number]
        # df = self.get_meta_for_partition(test_df, partition_info={'number': number, 'division': division})
        # print('test done')

        df = gedi_df.map_partitions(self.get_meta_for_partition, meta=(None, 'object'))
        nfailed = self.schedule_tasks(df)
        if nfailed > 0:
            # one more try
            logger.info('Some partitions failed. Restarting the client...')
            client = dask.distributed.get_client()
            client.restart()
            nfailed = self.schedule_tasks(df)
        
        if nfailed <= 0:
            logger.info('Finish gathering Sentinel-2 metadata. Merging metadata tables...')
            s2_meta_table = dgp.read_parquet(self.temp_dir / 's2_items_*.parquet', gather_spatial_partitions=False)
            s2_meta_table = s2_meta_table.drop_duplicates(subset=['id'])
            s2_meta_table = s2_meta_table.compute()
            s2_meta_table.to_parquet(s2_table_file)
            logger.info('Finish merging metadata tables. Deleting temporary files...')
            os.system(f'rm -rf {self.temp_dir}')


    def get_meta_for_partition(self, partition, partition_info: dict = None):
        """
        Query, filter, and stack S2 and ESA world cover patches for each GEDI partition.
        GEDI data is partitioned to cache a number of locations for the sake of memory efficiency.

        Args:
            partition (pandas.DataFrame): The partition containing the data.
            partition_info (dict, optional): Information about the partition.
        """
        zone = partition_info["division"].split('/')[-2]
        partition_number = int(partition_info["division"].split('_')[-1].split('.')[0])

        s2_items_file = self.temp_dir / f's2_items_{partition_number}.parquet'
        partition_file = self.save_dir / zone / f'partition_{partition_number}.parquet'
        if not self.rewrite and partition_file.exists() and s2_items_file.exists():
            logger.info(f'partition {partition_number} with S2 candidates already exists. Skipping...')
            return

        start = partition['start'].min()
        end = partition['end'].max()
        geom = box(*partition.total_bounds)

        mgrs_tiles = self.s2_grid[self.s2_grid.geometry.intersects(geom)]['Name'].to_list()
        if len(mgrs_tiles) == 0:
            return
        files = self._filter_parquet_files(start, end)
            
        s2_df = dgp.read_parquet(
            files,
            storage_options=self.s2asset.extra_fields["table:storage_options"],
            gather_spatial_partitions=False,
            columns=STAC_ITEM_KEYS + S2_ITEM_PROPS + ['eo:cloud_cover'],
            filters=[('s2:mgrs_tile', 'in', mgrs_tiles),
                    ("eo:cloud_cover", "<", self.max_cloud_cover),
                    ('s2:water_percentage', '<', self.max_water_percentage)]
        )
        s2_df = s2_df.map_partitions(lambda x: x, meta=s2_df).compute()
        if s2_df.empty:
            return
        # TODO: remove invalid s2 items
        s2_df['datetime'] = s2_df['datetime'].dt.tz_localize(None)
        s2_df = s2_df.astype({'proj:epsg': 'uint16', 'eo:cloud_cover': 'float32',
                             'id': 'string[python]'})
        partition.crs = 'epsg:4326'
        df = partition[['geometry', 'start', 'end', 'date']].sjoin(
            s2_df[['id', 'geometry', 'eo:cloud_cover', 'datetime']], how='left')
        df = df.dropna(subset=['id'])
        # NOTE: s2 geoparquet items are pre-filtered using the partition bbox, each point not necessarily intersects with the s2 item, therefore df could by empty
        if df.empty:
            partition['s2_candidates'] = pd.NA
        else:
            df.loc[:, 'delta_day'] = (df['datetime'] - df['date']).dt.days.abs().astype('uint16')
            df = df.groupby(level=0).apply(self.agg_s2_candidate_ids, zone, include_groups=False)
            df = df.droplevel(1)
            partition = partition.merge(df, how='left', left_index=True, right_index=True)

            s2_df = s2_df.drop(columns='eo:cloud_cover').set_index('id')
            s2_candidates = partition['s2_candidates'].explode().drop_duplicates().dropna()
            s2_candidates = s2_df.loc[s2_candidates].reset_index()
            s2_candidates.to_parquet(s2_items_file)

        partition = partition.drop(columns=['start', 'end'])
        partition.to_parquet(partition_file)
        logger.info(f'finish partition {partition_number}')

    def agg_s2_candidate_ids(self, group, zone):
        mask = (group['datetime'] >= group['start']) & (
            group['datetime'] <= group['end'])
        if mask.sum() >= 10:
            group = group[mask]
        group['mgrs_tile'] = group['id'].str.extract(r'T(\d{2}[A-Z]{1})') == zone
        group = group.sort_values(['mgrs_tile', 'eo:cloud_cover', 'delta_day']).iloc[:10] # should we keep all candidates and decide how many to keep in the next step?
        s2_ids = group['id'].drop_duplicates().to_list()
        return pd.DataFrame({'s2_candidates': [s2_ids]})


# #%%
@hydra.main(config_path="../config", config_name="s2_download", version_base="1.2")
def main(cfg):
    root_dir = Path(cfg.root_dir) if cfg.root_dir else Path.home()
    S2_meta_dir = root_dir / cfg.meta.S2_meta_dir
    if isinstance(cfg.zone, str):
        s2_table_file = S2_meta_dir / f'{cfg.zone}.parquet'
    else:
        s2_table_file = S2_meta_dir / 'small_zones.parquet'

    if not cfg.rewrite and s2_table_file.exists():
        logger.info(f'S2 metadata table already exists for {cfg.zone}. Skipping...')
        return

    from dask.distributed import Client, LocalCluster, performance_report
    from distributed.diagnostics import MemorySampler
    from dask import config
    config.set({'distributed.scheduler.locks.lease-timeout': 60}) 
    # might fix the communication error caused by I/O. ref: https://github.com/dask/distributed/issues/3129#issuecomment-1684858307
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 30})
    dask.config.set({'dataframe.query-planning': False})  # NOTE: dask expr causes the computing of s2 geoparqut hanging
    # NOTE: avoid coverting the assets dict to a long string of type string[pyarrow]
    dask.config.set({"dataframe.convert-string": False})
    cluster = LocalCluster()
    client = Client(cluster)
    logger.info(f'processing zone: {cfg.zone}')
    print(client)

    s2_meta_gather = S2MetaGather(cfg.rewrite, root_dir, **cfg.meta)
    t0 = time.time()

    logger.info(f'processing zone: {cfg.zone}')
    with performance_report(f'logs/meta-gather_{cfg.zone}_{cfg.year}.html'):
        s2_meta_gather.process_zone(cfg.zone)
    logger.info(f'time taken for {cfg.zone} {cfg.year}: {time.time() - t0}')
    client.close()


# %%
if __name__ == "__main__":
    main()
