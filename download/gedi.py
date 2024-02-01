import os
import ee
import logging
from pathlib import Path
import dask
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import hydra
from const import dtypes
from download.mgrs import authenticate
from download.dask_downloader import DaskDownloader
from dotenv import load_dotenv
load_dotenv()

authenticate()

logger = logging.getLogger(__name__)


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
    nSampledPerKm2 = 0.66793882312
    GEDI_START = pd.Timestamp('2018-01-01')

    def __init__(self, year=2019, data_dir='GEDI/2019', mgrs_file='GEDI/mgrs_with_tracks.parquet', key_file:str=None, npartitions=100, n_parallel=40, row_group_size=100, random_state:int=42, rewrite: bool = False, **kwargs):
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
        self.filter = 'quality_flag=1 AND degrade_flag=0 AND region_class>0 AND (leaf_off_flag=0 OR leaf_off_flag=255)'
        self.random_state = random_state
        self.rewrite = rewrite
        self.key_file = key_file

        if not self.mgrs_file.exists():
            print(f'mgrs file {self.mgrs_file} not found, download from GEE...')
            from download.mgrs import MGRS
            mgrs = MGRS(self.mgrs_file, self.data_dir / 'missing.csv')
            mgrs.get_mgrs()

    @dask.delayed
    def download_zone(self, zone):
        if zone[f'count_{self.year}'] == 0:
            print(f"no GEDI points in {zone['MGRS_UTM']}")
            return None
        flag = self.data_dir / f'{zone["MGRS_UTM"]}_done'
        if flag.exists() and not self.rewrite:
            print(f"{flag} exists")
            return None
        zone_dir = self.data_dir / zone["MGRS_UTM"]
        zone_dir.mkdir(exist_ok=True, parents=True)
        geom = ee.Geometry.BBox(*zone['geometry'].bounds).toGeoJSON()
        last_coords = geom['coordinates'][0][0].copy()
        geom['coordinates'][0].append(last_coords)
        sample_ratio = zone['landmass'] * self.nSampledPerKm2/zone[f'count_{self.year}']
        for track_id in zone['tracks']:
            if str(self.year) not in track_id:
                continue
            if (zone_dir / f'{track_id}.parquet').exists() and not self.rewrite:
                continue
            track_gdf = ee.data.listFeatures({'assetId': track_id, 'filter': self.filter, 'region': geom,'fileFormat':'GEOPANDAS_GEODATAFRAME'})
            if track_gdf.empty:
                continue
            track_gdf = track_gdf[['geometry', *dtypes.keys()]]
            track_gdf = track_gdf.astype(dtypes)
            n_sample = min(len(track_gdf), int(sample_ratio * len(track_gdf)))
            filename = track_id.split('/')[-1]
            track_gdf.sample(n=n_sample, random_state=self.random_state).to_parquet(zone_dir / f'{filename}.parquet')
            del track_gdf

        print('finish zone', zone['MGRS_UTM'])
        flag.touch()

    def download(self):
        """
        Downloads GEDI data for all valid MGRS grid cells in the specified year.
        """
        mgrs_df = gpd.read_parquet(self.mgrs_file)
        self.schedule_tasks(mgrs_df, self.download_zone)


    def plotHistogram(self): #TODO:needs update
        ddf = dd.read_csv(self.dataFolder / f'*/*.csv', usecols=['rh98', 'pft_class', '.geo'], blocksize=64e6)
        h, bins = da.histogram(ddf['rh98'], bins=100).compute()
        plt.stairs(h, bins)
        plt.savefig('rh98.png')
        print('plot saved')

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
