
from pathlib import Path
import pandas as pd
from retry import retry
import requests
import geopandas as gpd
import ee
from io import StringIO
import numpy as np
import dask
from dask.distributed import Client, LocalCluster

from download.core import DaskDownloader
from download.core.utils import authenticate, check_unfinished_files 
# authenticate()

def set_fc_properties(row):
    geom = row.geometry
    row = row.drop('geometry')
    fc = ee.Feature(ee.Geometry.Point([geom.x, geom.y]), row.to_dict())
    fc = fc.set('index', row.name)
    return fc



class SOTAChmDownloader(DaskDownloader):
    """
    A class for downloading SOTA CHM data for given locations from Google Earth Engine.
    
    Attributes:
        location_files: The file(s) containing the locations to download the SOTA CHM data.
        save_dir: The root directory to save the SOTA CHM data.
        n_parallel: The number of parallel tasks to download the SOTA CHM data.
        debug: Whether to print debug information.
    """
    _ee_initialized = False
    
    def __init__(self, 
                 location_files: str=None,
                 save_dir: str=None, 
                 n_parallel: int=100,  
                 rewrite: bool=False,
                 debug: bool=False, **kwargs):
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.rewrite = rewrite
        self.debug = debug
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.location_files = Path(location_files).expanduser()
        self.location_files = self.location_files.parent.glob(self.location_files.name)
        self.location_files = check_unfinished_files(self.location_files, self.save_dir, check_exists=True)
        self.canopy_height_um = ee.ImageCollection('projects/worldwidemap/assets/canopyheight2020')
        self.canopy_height_umd = ee.ImageCollection("users/potapovpeter/GEDI_V27")
        self.canopy_height_meta = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
        self.canopy_height_eth = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').rename('RH98_ETH')
        self._ensure_authenticated()

    def _ensure_authenticated(self):
        """Ensures Earth Engine is initialized exactly once per process."""
        if not SOTAChmDownloader._ee_initialized:
            authenticate()  # Your existing utility function
            # Or directly: ee.Initialize(project='your-project')
            SOTAChmDownloader._ee_initialized = True

    def download(self):
        cluster = LocalCluster()
        client = Client(cluster)
        tasks = []
        for file in self.location_files:
            tasks.append(self.download_file(file))
        if len(tasks) < self.n_parallel:
            self.n_parallel = len(tasks)
        self.schedule_tasks(delayed_tasks=tasks)

    @dask.delayed
    @retry(requests.HTTPError, tries=10, delay=1)
    def download_file(self, file: Path):
        output_file = self.save_dir / f'{file.stem}.parquet'
        if output_file.exists() and not self.rewrite:
            print(f'{output_file} exists, skipping')
            return
        
        try:
            loc_df = gpd.read_parquet(file)
        except ValueError as e:
            loc_df = pd.read_parquet(file)
            loc_df = gpd.GeoDataFrame(loc_df, geometry=gpd.points_from_xy(loc_df.lon, loc_df.lat, crs="EPSG:4326"))
            
        if loc_df.empty:
            return
        # features more than 10000 would likely fail because of GEE memory limit
        if len(loc_df) > 10000:
            partitions = [loc_df.iloc[i:i+10000] for i in range(0, len(loc_df), 10000)]
        else:
            partitions = [loc_df]
            
        partition_dfs = {
            'eth_umd': [],
            'um': [],
            'meta': []
        }
        for part in partitions:
            eefc = part[['geometry', 'rh98']].apply(set_fc_properties, axis=1)
            eefc = ee.FeatureCollection(eefc.tolist())
            polyCol = eefc.map(lambda f: f.buffer(12.5))
            # points = ee.Geometry.MultiPoint(partition.geometry.apply(lambda x: [x.x, x.y]).tolist())
            img_um = self.canopy_height_um.filterBounds(eefc).mosaic().divide(100).rename('RH100_UM')
            img_umd = self.canopy_height_umd.filterBounds(eefc).mosaic().rename('RH95_UMD')
            img_meta = self.canopy_height_meta.filterBounds(eefc).mosaic().rename('RH95_META')
            group1 = self.canopy_height_eth.addBands(img_umd)
            # group2 = img_um.addBands(img_meta)
            
            # EPSG:4326
            fc_eth_umd = group1.sampleRegions(
                collection=eefc,
                scale=10
            )
            # EPSG:3857
            fc_um = img_um.sampleRegions(
                collection=eefc,
                scale=10
            )
            fc_meta = img_meta.reduceRegions(
                collection=polyCol,
                reducer=ee.Reducer.max(),
                scale=10
            )
            dfs = []
            if fc_eth_umd.size().getInfo() == 0 or fc_um.size().getInfo() == 0 or fc_meta.size().getInfo() == 0:
                partition_dfs['eth_umd'].append(pd.DataFrame(np.nan, columns=['RH95_UMD', 'RH98_ETH'], index=part.index))
                partition_dfs['um'].append(pd.DataFrame(np.nan, columns=['RH100_UM'], index=part.index))
                partition_dfs['meta'].append(pd.DataFrame(np.nan, columns=['RH95_META'], index=part.index))
                continue
            for i, fc in enumerate([fc_eth_umd, fc_um, fc_meta]):
                download_id = ee.data.getTableDownloadId({'table': fc, 'fileFormat': 'CSV'})
                res = requests.get(ee.data.makeTableDownloadUrl(download_id))
                if res.status_code == 200:
                    data = StringIO(res.content.decode('utf-8'))
                    df = pd.read_csv(data)
                    df = df.drop(columns=['.geo', 'system:index']).set_index('index')
                    dfs.append(df)
                else:
                    raise requests.HTTPError(f'Failed to download ')
            
            partition_dfs['eth_umd'].append(dfs[0])
            partition_dfs['um'].append(dfs[1])
            partition_dfs['meta'].append(dfs[2])
        if len(partition_dfs['eth_umd']) == 0 or len(partition_dfs['um']) == 0 or len(partition_dfs['meta']) == 0:
            print(f'No data downloaded for {file.stem}')
            return
        df1s = pd.concat(partition_dfs['eth_umd'])
        df2s = pd.concat(partition_dfs['um'])
        df3s = pd.concat(partition_dfs['meta'])
        ### if nodata in AOI, GEE will return nothing, so we need to fill with NA
        if 'RH95_UMD' not in df1s.columns:
            df1s['RH95_UMD'] = pd.NA
        if 'RH98_ETH' not in df1s.columns:
            df1s['RH98_ETH'] = pd.NA
        if 'RH100_UM' not in df2s.columns:
            df2s['RH100_UM'] = pd.NA
        if 'max' not in df3s.columns:
            df3s['max'] = pd.NA
        
        loc_df.loc[df1s.index, ['RH95_UMD','RH98_ETH']] = df1s[['RH95_UMD','RH98_ETH']]
        loc_df.loc[df2s.index, ['RH100_UM']] = df2s[['RH100_UM']]
        assert loc_df.index.equals(df3s.index)
        loc_df[['RH95_META']] = df3s[['max']]
        loc_df.to_parquet(output_file)
        
def download_sota_chms(location_files: str=None, save_dir: str=None , n_parallel: int=100, rewrite: bool=False, debug: bool=False):
    cluster = LocalCluster()
    client = Client(cluster)
    downloader = SOTAChmDownloader(
        location_files=location_files,
        save_dir=save_dir,
        n_parallel=n_parallel,
        rewrite=rewrite,
        debug=debug
    )
    downloader.download()
