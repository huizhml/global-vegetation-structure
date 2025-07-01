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
import matplotlib.pyplot as plt
from shapely.geometry import shape
import requests
from retry import retry
from io import StringIO
import hydra
from const import dtypes
from download._utils import authenticate
from download.dask_downloader import DaskDownloader
from dotenv import load_dotenv
load_dotenv()

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

    def __init__(self, year=2019, data_dir='GEDI/2019', mgrs_file='GEDI/mgrs_with_count_and_orbits.parquet', key_file:str=None, npartitions=100, n_parallel=40, row_group_size=100, random_state:int=42, rewrite: bool = False, **kwargs):
        """
        Initializes a GEDI object.

        Args:
            year (int, optional): The year of the GEDI data to download. Defaults to 2019.
            dataFolder (str, optional): The local folder to save the downloaded GEDI data. Defaults to 'gedi'.
            keepLeafOff255 (bool, optional): Whether to keep GEDI points with leaf off flag of 255. Defaults to False.
        """
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.data_dir = Path.home() / data_dir
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.mgrs_file = Path.home() / mgrs_file
        self.npartitions = npartitions
        self.n_parallel = n_parallel
        self.row_group_size = row_group_size
        self.year = year
        self.filter = 'quality_flag==1 && degrade_flag==0 && region_class>0 && leaf_off_flag!=1'
        self.random_state = random_state
        self.rewrite = rewrite
        self.key_file = key_file

        if not self.mgrs_file.exists():
            logger.info(f'mgrs file {self.mgrs_file} not found, download from GEE...')
            from download.mgrs import MGRS
            mgrs = MGRS(self.mgrs_file, self.data_dir / 'missing.csv')
            mgrs.get_mgrs()

    @dask.delayed
    def download_zone(self, zone):
        if zone[f'count_{self.year}'] == 0:
            logger.info(f"no GEDI points in {zone['MGRS_UTM']}")
            return None
        flag = self.data_dir / f'{zone["MGRS_UTM"]}_done'
        if flag.exists() and not self.rewrite:
            logger.info(f"{flag} exists")
            return None
        zone_dir = self.data_dir / zone["MGRS_UTM"]
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
            os.rmdir(zone_dir)
    
    @retry(requests.HTTPError, tries=10, delay=1)
    def download_orbit(self, fc, sample_ratio, zone_dir, filename):
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
                df['date'] = pd.to_timedelta(df['delta_time'], unit='S') + GEDI_START
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                df.to_parquet(zone_dir / f'{filename}.parquet', row_group_size=self.row_group_size, engine='pyarrow')
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


    def plotHistogram(self): #TODO:needs update
        ddf = dd.read_csv(self.dataFolder / f'*/*.csv', usecols=['rh98', 'pft_class', '.geo'], blocksize=64e6)
        h, bins = da.histogram(ddf['rh98'], bins=100).compute()
        plt.stairs(h, bins)
        plt.savefig('rh98.png')
        logger.info('plot saved')

    def getSampleTable(self, plot=False):  # TODO:needs update
        pass

    def visualize(self):
        """
        Visualizes the GEDI data.
        """
        pass


@hydra.main(config_path="../config", config_name="gedi_download", version_base="1.2")
def main(cfg):
    if cfg.task == 'download':
        from dask.distributed import Client, LocalCluster
        from dask import config
        config.set({'interface': 'lo'})
        cluster = LocalCluster()
        client = Client(cluster)  # timeout
        gedi = GEDI(**cfg.init)
        gedi.download()

    elif cfg.task == 'visualize':
        gedi.getSampleTable(plot=True)
    else:
        logger.error(f'Task not recognized.\nreceived: {cfg.task} \nexpected: download or visualize')


if __name__ == '__main__':
    main()
