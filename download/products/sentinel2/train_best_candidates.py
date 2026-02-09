
# %%
from dotenv import load_dotenv
import dask.bag as db
import pandas as pd
from dask.utils import natural_sort_key
from dask.distributed import get_client
import dask_geopandas as dgp
import logging
from pathlib import Path
from itertools import chain

import numpy as np
import pystac_client
import planetary_computer
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO

from download.core import DaskDownloader
from download.core.constants import S2_ITEM_PROPS
from download.core.utils import row_to_stac_item, trim_memory, buffer_and_snap_bounds, get_total_bounds, get_patch

# %%
load_dotenv('.planetarycomputer/settings.env')

# TODO: control from yaml
# g0, g1, g2 = gc.get_count()
# gc.set_threshold(g0*5, g1*5, g2 * 5)

retry = Retry(
    # too many retries cause worker sleep too long when backoff_factor is 1
    total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None
)
stac_api_io = StacApiIO(max_retries=retry)
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)

defective_SCL = [0, 1, 8, 9, 10, 11]  # keep cloud shadows, model should learn to be invariant to cloud shadows

logger = logging.getLogger(__name__)

class BestS2Finder(DaskDownloader):

    def __init__(self,
                 rewrite: bool = False,
                 zone: str = '23J',
                 year: int = 2019,
                 n_parallel: int = 100,
                 gedi_dir='GEDI',
                 save_dir: str = 'data/GEDI',
                 S2_meta_dir: str = 'scratch/S2_meta',
                 patch_size: int = 15,
                 out_res: int = 10,
                 max_retries: int = 3,
                 **kwargs
                 ) -> None:
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.gedi_dir = Path(gedi_dir).expanduser()
        self.save_dir = Path(save_dir).expanduser()
        self.S2_meta_dir = Path(S2_meta_dir).expanduser()
        self.zone = zone
        self.year = year
        self.n_parallel = n_parallel
        self.patch_size_in_meters = patch_size * out_res
        self.out_res = out_res
        self.buffer_size = patch_size // 2 * out_res  # in meters
        self.patch_size = (self.buffer_size * 2 + out_res) / out_res
        self.rewrite = rewrite

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
            files = list(self.gedi_dir.glob(f'{self.year}/{self.zone}/partition*.parquet'))
            (self.save_dir / f'{self.year}/{self.zone}').mkdir(exist_ok=True, parents=True)
            self.s2_table_file = self.S2_meta_dir / f'{self.year}/{self.zone}.parquet'
            flag = self.save_dir / f'{self.zone}_{self.year}_done'
        else:
            # Read GEDI partitions from multiple zones, add paths of partitions in divisions
            self.s2_table_file = self.S2_meta_dir / f'{self.year}/small_zones.parquet'
            flag = self.save_dir / f'small_zones_{self.year}_done'
            files = []
            for z in self.zone:
                files.extend(list(self.gedi_dir.glob(f'{self.year}/{z}/partition*.parquet')))
                (self.save_dir / f'{self.year}{z}').mkdir(exist_ok=True, parents=True)
        if not self.rewrite and flag.exists():
            logger.info(f'{self.zone} has been processed.')
            return 
        
        self.files = [str(p) for p in files]
        self.files = sorted(self.files, key=natural_sort_key)
        gedi_df = dgp.read_parquet(self.files, gather_spatial_partitions=False)
        s2_meta_table = dgp.read_parquet(self.s2_table_file)
        logger.info(f'Processing {gedi_df.npartitions} partitions...')

        # number = 15
        # self.rewrite = True
        # test = gedi_df.get_partition(number).compute()
        # division = gedi_df.divisions[number]
        # df = self.find_best_s2_for_partition(test, partition_info={'number': number, 'division': division})
        # logger.info('test done')
        df = gedi_df.map_partitions(self.find_best_s2_for_partition, s2_meta_table, meta=(None, 'string'))
        nfailed = self.schedule_tasks(df)

        # if nfailed > 0:
        #     # one more try
        #     logger.info('Some partitions failed. Restarting the client...')
        #     client = dask.distributed.get_client()
        #     client.restart()
        #     nfailed = self.schedule_tasks(df)

        if nfailed <= 0:
            flag.touch()


    def find_best_s2_for_partition(self, partition, s2_meta_table, partition_info: dict = None):
        """
        Find the best Sentinel-2 scene for each GEDI location in the given partition.
        A new column 'best_s2' is added to the partition(GeoDataFrame) and saved to the save_dir with name pattern specified by to_file.

        Parameters
        ----------
        * partition (pandas.DataFrame): The partition containing s2_candidates.
        * to_file (str): The name pattern of the output file.
        * rewrite (bool): Whether to overwrite the existing file.
        """
        partition_number = partition_info["number"]
        zone = self.files[partition_number].split('/')[-2]
        partition_number_infile = self.files[partition_number].split('/')[-1].split('_')[-1].split('.')[0]
        partition_file = self.save_dir / f'{self.year}/{zone}' / f'partition_{partition_number_infile}.parquet'
        # if not self.rewrite and partition_file.exists():
        #     logger.info(f'partition {partition_number_infile} with best S2 already exists. Skipping...')
        if partition['s2_candidates'].isna().all():
            partition['best_s2'] = pd.NA
            partition.to_parquet(partition_file)
            logger.info(f'all none parition: {partition_number_infile}, {zone}')
            return

        if 'best_s2' in partition.columns:
            processed_rows = partition[partition['best_s2'].notna()]
            unprocessed_rows = partition[partition['best_s2'].isna()]
            if unprocessed_rows.empty:
                logger.info(f'partition {partition_number_infile} with best S2 already exists. Skipping...')
                return
        else:
            unprocessed_rows = partition
            processed_rows = pd.DataFrame(columns=partition.columns)

        op_part = unprocessed_rows[['date', 'geometry', 's2_candidates']]
        op_part = op_part.dropna(subset='s2_candidates')
        s2_candidates = set(chain.from_iterable(op_part['s2_candidates']))
        s2_meta_table = s2_meta_table[s2_meta_table['id'].isin(s2_candidates)]
        s2_meta_table = s2_meta_table.set_index('id')
        s2_meta_table['datetime'] = s2_meta_table['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

        op_part = op_part.explode('s2_candidates')
        gedi_date = pd.to_datetime(op_part['date'])
        s2_date = op_part['s2_candidates'].str.extract(r'(\d{8})')
        s2_date = pd.to_datetime(s2_date[0], format='%Y%m%d')
        op_part['delta_day'] = (gedi_date - s2_date).dt.days.abs().astype('uint16')
        op_part = op_part.drop(columns=['date'])
        op_part = op_part.reset_index().set_index('s2_candidates')
        op_part = op_part.loc[s2_meta_table.index] # remove s2_candidates that are not in s2_meta_table
        client = get_client()
        items = row_to_stac_item(s2_meta_table, S2_ITEM_PROPS)
        items_table = []
        for item in items:
            bounds = op_part.loc[[item.id]]
            items_table.append((item, bounds))

        items_table = db.from_sequence(items_table, npartitions=10)
        res = items_table.map(self.calculate_defective_cover).compute()
        
        client.run(trim_memory)
        df = pd.concat(res)
        df = df[df['defective_cover'] <= 0.2]
        df = df.sort_values(['defective_cover', 'delta_day']).groupby('index').head(1)
        df = df.reset_index().set_index('index').drop(columns='geometry')
        res = partition.merge(df, how='left', left_index=True, right_index=True)
        res = res.rename(columns={'id': 'best_s2'})
        res = pd.concat([res, processed_rows])
        res.to_parquet(partition_file)


    def calculate_defective_cover(self, entry):
        """
        Calculate patch-level defective cover for all GEDI locations on the same S2 scene.

        Parameters
        ----------
        * entry (tuple):
            * item (pystac.Item): The S2 STAC item.
            * locs (geopandas.DataFrame): GEDI locations.

        Returns
        -------
        * geopandas.DataFrame: locs with a new column defective cover added
        """
        
        item, locs = entry
        epsg = item.properties['proj:epsg']
        locs = locs.to_crs(epsg)
        bounds = buffer_and_snap_bounds(locs.geometry, self.buffer_size, self.out_res)
        total_bounds = get_total_bounds(bounds)

        image = get_patch(item, assets=['SCL'], resolution=self.out_res, bounds=total_bounds, epsg=epsg, dtype='uint8', fill_value=np.uint8(0))
        if image.shape[0] == 0: #NOTE: point and S2 geometry intersects in EPSG:4326 but not in the local crs
            return
        defective_cover = []
        try:
            image = image.load() #NOTE: slicing on lazy object is inefficient
        except Exception as e:
            locs['defective_cover'] = 1.0
            locs = locs.to_crs(4326)
            return locs.astype({'defective_cover': 'float32'})
        for row in bounds.itertuples():
            xrange = range(row.minx, row.maxx, self.out_res) #slice(row.minx, row.maxx)
            yrange = range(row.maxy, row.miny, -self.out_res) #slice(row.maxy, row.miny)
            patch = image.sel(x=xrange, y=yrange).squeeze()
            dc = patch.isin(defective_SCL).sum(dim=['x', 'y']) / np.prod(patch.shape[-2:])
            defective_cover.append(dc.data)
        locs['defective_cover'] = defective_cover
        locs = locs.to_crs(4326)
        del image
        return locs.astype({'defective_cover': 'float32'})