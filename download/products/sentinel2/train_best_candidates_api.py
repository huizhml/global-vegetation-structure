
# %%
from dotenv import load_dotenv
import hydra
import ipdb
from shapely.geometry import shape
import geopandas as gpd
import pandas as pd
from dask.utils import natural_sort_key
from dask.distributed import Semaphore
import dask_geopandas as dgp
import dask
import dask.dataframe as dd
import os
import logging
import numpy as np
import pystac_client
import planetary_computer
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO

from download.core import DaskDownloader
from download.core.utils import get_patch, get_tile_by_id, get_most_common_epsg

# %%
load_dotenv('.planetarycomputer/settings.env')
retry = Retry(
    # too many retries cause worker sleep too long when backoff_factor is 1
    total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None
)
stac_api_io = StacApiIO(max_retries=retry)
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)

defective_SCL = [0, 1, 8, 9, 10, 11]  # keep cloud shadows, model should learn to be invariant to cloud shadows

logger = logging.getLogger(__name__)



class BestS2FinderAPI(DaskDownloader):
    """
    Find the best Sentinel-2 scene for each GEDI location in the given zone using STAC API.
    At the time of writing, some s2 scenes were missing bands (cannot be found in the blob storage). So MPC was regenerating these s2 scenes and updating 
    the geoparquet meta table. And we can find duplicated s2 items in the geoparquet meta table, with the only difference generation time.
    Therefore we might get invalid ids for some locations and need to re-find the best s2 for those locations.

    Retrieves the best Sentinel-2 (S2) scene for each GEDI location within the specified zone using the STAC API.
    Note: At the time this function was implemented, there were instances of missing bands in some S2 scenes. 
    These scenes are being regenerated and their metadata is being updated in the geoparquet meta table by MPC. 
    This update process can result in duplicate S2 scene entries in the meta table, distinguished only by their generation times.

    As a consequence, invalid scene ids might be retrived in the meta gathering step. 
    If an invalid S2 id is found, the function will attempt to identify and return a more recent, valid S2 scene for those locations.

    """
    def __init__(self,
                 rewrite: bool = False,
                 gedi_dir: str = '~/data/gvs/GEDI_extra',
                 zone: str = '23J',
                 year: int = 2019,
                 n_parallel: int = 100,
                 data_dir='GEDI',
                 save_dir: str = 'data/GEDI',
                 S2_meta_dir: str = 'scratch/S2_meta',
                 flag_dir: str=None,
                 patch_size: int = 15,
                 out_res: int = 10,
                 maxCloudCover: int = 50,
                 maxWaterPercentage: int = 100,
                 queryDaysRange: int = 90,
                 extendDays: int = 30,
                 sem_max_release: int = 60,
                 **kwargs
                 ) -> None:
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.zone = zone
        self.gedi_dir = gedi_dir
        self.save_dir = save_dir
        self.S2_meta_dir = S2_meta_dir
        self.flag_dir = flag_dir
        self.year = year
        self.yearStart = pd.Timestamp(f'{self.year}-01-01')
        self.queryDaysRange = pd.to_timedelta(queryDaysRange, unit='D')
        self.extendDays = pd.to_timedelta(extendDays, unit='D')
        self.maxCloudCover = maxCloudCover
        self.maxWaterPercentage = maxWaterPercentage
        self.sem = Semaphore(sem_max_release, name='max_queries', register=True)
        self.n_parallel = n_parallel
        self.patch_size_in_meters = patch_size * out_res
        self.out_res = out_res
        self.buffer_size = patch_size // 2 * out_res  # in meters
        self.patch_size = (self.buffer_size * 2 + out_res) / out_res
        self.rewrite = rewrite

        with open('download/invalid_s2_items.txt', 'r') as file:
            self.invalid_s2_ids = [line.strip() for line in file.readlines()]


    @property
    def unfinished_files(self):
        if self.rewrite:
            return self.files
        files = []
        for i, fp in enumerate(self.files):
            fg = self.flag_dir / f'{self.year}_partition_{i}_done'
            if not fg.exists():
                files.append(fp)
        return files

    def find_best_s2(self):
        """
        Find the best Sentinel-2 scene for each GEDI location in the given GEDI zone.
        Sentinel-2 candidates are pre-filtered based on scene level cloud cover and growing season (if the GEDI footprint is captured in the growing season).
        The best scene is selected based on the patch-level defective cover and relative days between the GEDI acquisition and the Sentinel-2 scene.

        Parameters
        ----------
        * zone (str): The MGRS zone to find the best S2 scene for.
        * from_file (str): The name of the file containing the GEDI data.

        Returns
        -------
        * str: A string indicating the status of the processing. Returns 'done' if the zone and year have already been processed.
        """
        if isinstance(self.zone, str):
            files = list(self.gedi_dir.glob(f'{self.zone}/partition*.parquet'))
            (self.save_dir / self.zone).mkdir(exist_ok=True, parents=True)
            self.s2_table_file = self.S2_meta_dir / f'{self.zone}.parquet'
            self.flag_dir  = self.flag_dir / self.zone
            flag = self.save_dir / f'{self.zone}_done'
        else:
            # Read GEDI partitions from multiple zones, add paths of partitions in divisions
            self.s2_table_file = self.S2_meta_dir / 'small_zones.parquet'
            self.flag_dir  = self.flag_dir / 'small_zones'
            flag = self.save_dir / 'small_zones_done'
            files = []
            for z in self.zone:
                files.extend(list(self.gedi_dir.glob(f'{z}/partition*.parquet')))
                (self.save_dir / z).mkdir(exist_ok=True, parents=True)
        
        # Get unfinished partition files indicated by the done flags
        files = [str(p) for p in files]
        self.files = sorted(files, key=natural_sort_key)
        
        if len(self.unfinished_files) == 0:
            logger.info('All partitions have been processed.')
            return
        
        gedi_df = dgp.read_parquet(self.unfinished_files, gather_spatial_partitions=False)

        logger.info(f'Processing {gedi_df.npartitions} partitions...')

        with_growing_season = (gedi_df['leaf_on_doy'] < 366) & (gedi_df['leaf_off_flag'] == 0)
        # leaf off date is in the next year
        reverse = gedi_df['leaf_on_doy'] > gedi_df['leaf_off_doy']
        gedi_df['leaf_off_doy'] = gedi_df['leaf_off_doy'].mask(reverse, gedi_df['leaf_off_doy'] + 365)

        leaf_on_doy = dd.to_timedelta(gedi_df['leaf_on_doy'], unit='D')
        leaf_off_doy = dd.to_timedelta(gedi_df['leaf_off_doy'], unit='D')

        gedi_df['start'] = dd.to_datetime(gedi_df['date']) - self.queryDaysRange
        gedi_df['end'] = dd.to_datetime(gedi_df['date']) + self.queryDaysRange
        gedi_df['start'] = gedi_df['start'].mask(with_growing_season, self.yearStart + leaf_on_doy)
        gedi_df['end'] = gedi_df['end'].mask(with_growing_season, self.yearStart + leaf_off_doy)

        # *** DEBUG BLOCK
        # number = 7
        # # self.rewrite = True
        # test = gedi_df.get_partition(number).compute()
        # # test = gpd.read_parquet('/users/zhanghui/scratch/GEDI_with_s2_candidates_and_best/2021/21H/partition_53.parquet')
        # df = self.find_best_s2_for_partition(test, partition_info={'number': number})
        # logger.info('test done')
        # ***

        df = gedi_df.map_partitions(self.find_best_s2_for_partition, meta=(None, 'string'))
        nfailed = self.schedule_tasks(df)

        # if nfailed > 0:
        #     # one more try
        #     logger.info('Some partitions failed. Restarting the client...') # restarting client always failed
        #     client = dask.distributed.get_client()
        #     client.restart()
        #     nfailed = self.schedule_tasks(df)

        # if nfailed <= 0:
        #     flag.touch()

        # update s2 meta table.
        s2_meta_table = gpd.read_parquet(self.s2_table_file)
        s2_ids = gedi_df.map_partitions(lambda x: x['best_s2']).compute()
        s2_ids = s2_ids.reset_index().dropna().drop_duplicates(subset='best_s2')
        new_ids = s2_ids[~s2_ids['best_s2'].isin(s2_meta_table['id'])]
        if not new_ids.empty:
            df = []
            for i in new_ids['best_s2']:
                item = get_tile_by_id(i)
                assets = {k: v.to_dict() for k, v in item.assets.items()}
                df.append([item.id, item.bbox, assets, item.properties['datetime'], item.properties['proj:epsg'], shape(item.geometry)])
            df = gpd.GeoDataFrame(df, columns=['id', 'bbox', 'assets', 'datetime', 'proj:epsg', 'geometry'])
            df = df.astype({'proj:epsg': 'uint16'})
            df.crs='epsg:4326'
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            new_meta_table = pd.concat([s2_meta_table, df])
            new_meta_table = gpd.GeoDataFrame(new_meta_table)
            new_meta_table.to_parquet(self.s2_table_file)


    def find_best_s2_for_partition(self, partition, partition_info: dict = None):
        """
        Find the best Sentinel-2 scene for each GEDI location in the given partition.
        A new column 'best_s2' is added to the partition(GeoDataFrame) and saved to the save_dir with name pattern specified by to_file.

        Parameters
        ----------
        * partition (pandas.DataFrame): The partition containing s2_candidates.
        * to_file (str): The name pattern of the output file.
        * rewrite (bool): Whether to overwrite the existing file.
        """
        partition_idx = partition_info["number"] # the index of the partition in the dataframe
        partition_file = self.unfinished_files[partition_idx]
        partition_number = self.files.index(partition_file) # the index of the partition in zone
        zone = partition_file.split('/')[-2]
        
        invalid_mask = partition['best_s2'].isin(self.invalid_s2_ids)
        partition.loc[invalid_mask, 'best_s2'] = pd.NA
        op_part = partition[['date', 'geometry', 's2_candidates', 'best_s2', 'start', 'end', 'date']]
        op_part = op_part[op_part['best_s2'].isna()]
        
        if op_part.empty:
            logger.info(f'All GEDI locations in {zone} {partition_number} have found best Sentinel-2.')
            return

        best = op_part.apply(self.get_best_s2_for_point, axis=1, result_type='expand')
        best = best.dropna()
        best = best.astype({
            'defective_cover': 'float32',
            'delta_day': 'uint16'
        })
        if best.empty:
            partition.to_parquet(partition_file)
            return
        best = best.rename(columns={'id': 'best_s2'})
        best = best[['best_s2', 'delta_day', 'defective_cover']]
        partition.loc[best.index, ['best_s2', 'delta_day', 'defective_cover']] =  best[['best_s2', 'delta_day', 'defective_cover']]
        partition = partition.drop(columns=['start', 'end'])
        partition.to_parquet(partition_file)
        if os.path.exists(f'{str(self.flag_dir)}/{self.year}_partition_{partition_number}_done'):
            os.remove(f'{str(self.flag_dir)}/{self.year}_partition_{partition_number}_done')


    def get_best_s2_for_point(self, point):
        '''
        Qeury and Filter S2 tiles for each GEDI point
        - Query by date, cloud cover, water percentage, and Filter by leaf on/off dates
        - Filter by defective cover (patch level)
        '''
        geom = point.geometry 
        items = self.query_s2_for_p(point.start, point.end, geom)
        
        if items is None:
            return pd.Series([pd.NA, pd.NA, pd.NA], index=['id', 'delta_day', 'defective_cover'])
        items = [item for item in items if item.id not in self.invalid_s2_ids]
        if len(items) == 0:
            return pd.Series([pd.NA, pd.NA, pd.NA], index=['id', 'delta_day', 'defective_cover'])

        # get patch and calculate defective cover
        epsg = get_most_common_epsg(items)
        geom = gpd.GeoSeries(geom, crs='EPSG:4326').to_crs(epsg)[0]
        bounds = geom.buffer(self.buffer_size).bounds
        patch = get_patch(items, ['SCL'], resolution=self.out_res, bounds=bounds, epsg=epsg, dtype='uint8', snap_bounds=True)
        
        if patch.shape[0] == 0 or patch.shape[-2:] != (self.patch_size, self.patch_size): # why there're cases that the output shape is (14,15)? fill_value doesn't work?
            return pd.Series([pd.NA, pd.NA, pd.NA], index=['id', 'delta_day', 'defective_cover'])

        patch = patch.compute() # simplify compute graph, not sure if this is helpful
        scl = patch.data
        defective_cover = np.any([(scl == k) for k in defective_SCL], 0).sum((-2,-1)) / np.prod(scl.shape[-2:])
        if np.isnan(defective_cover).all() or defective_cover.min() > 0.9:
            return pd.Series([pd.NA, pd.NA, pd.NA], index=['id', 'delta_day', 'defective_cover'])

        patch_df = pd.DataFrame({
            'id': patch.id.values,
            'defective_cover': defective_cover.squeeze(),
            'delta_day': [np.abs(t - pd.Timestamp(point.date.iloc[0])).days for t in patch.time.values],
        })
        
        patch_df = patch_df.sort_values(['defective_cover', 'delta_day'])
        return patch_df.iloc[0]

    def query_s2_for_p(self, start, end, geom):
        """
        Query Sentinel-2 data for a given time range and geometry. 
        If no items are found, extend the time range by 60 days and try again, util the time range covers 365 days.

        Args:
            start (datetime): Start date of the time range.
            end (datetime): End date of the time range.
            geom (shapely.geometry): Geometry object representing the area of interest.

        Returns:
            list: List of Sentinel-2 items matching the query criteria, or None if no items found.
        """
        with self.sem: #? what will happen if sem times out, will we get items?
            search = api.search(collections=['sentinel-2-l2a'],
                                query={
                                    "eo:cloud_cover": {
                                        "lt": self.maxCloudCover
                                    },
                                    's2:water_percentage': {
                                        'lt': self.maxWaterPercentage
                                    }
                                },
                                intersects=geom,
                                datetime=f'{str(start)[:10]}/{str(end)[:10]}')
            items = search.item_collection()
        if len(items) > 0:
            return items
        if len(items) == 0 and (end - start).days < 365:
            logger.info(f'No S2 tile found between {start} - {end}, extend the range by {self.extendDays.days*2} days')
            items = self.query_s2_for_p(start - self.extendDays,
                                           end + self.extendDays, geom)
            return items
        else:
            return None


