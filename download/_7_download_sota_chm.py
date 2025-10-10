
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
from retry import retry
import requests
import geopandas as gpd
from download._utils import authenticate, check_unfinished_files 
import ee
from io import StringIO
import numpy as np
import dask
from dask.distributed import Client, LocalCluster

from download._dask_downloader import DaskDownloader
authenticate()

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
        output_dir: The root directory to save the SOTA CHM data.
        n_parallel: The number of parallel tasks to download the SOTA CHM data.
        debug: Whether to print debug information.
    """
    def __init__(self, 
                 location_files: str=None,
                 output_dir: str=None, 
                 n_parallel: int=100,  
                 rewrite: bool=False,
                 debug: bool=False, **kwargs):
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.rewrite = rewrite
        self.debug = debug
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.location_files = Path(location_files).expanduser()
        self.location_files = self.location_files.parent.glob(self.location_files.name)
        self.location_files = check_unfinished_files(self.location_files, self.output_dir, check_exists=True)
        self.canopy_height_um = ee.ImageCollection('projects/worldwidemap/assets/canopyheight2020')
        self.canopy_height_umd = ee.ImageCollection("users/potapovpeter/GEDI_V27")
        self.canopy_height_meta = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
        self.canopy_height_eth = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').rename('RH98_ETH')

    def download(self):
        tasks = []
        for file in self.location_files:
            tasks.append(self.download_file(file))
        self.schedule_tasks(delayed_tasks=tasks)

    @dask.delayed
    @retry(requests.HTTPError, tries=10, delay=1)
    def download_file(self, file: Path):
        output_file = self.output_dir / f'{file.stem}.parquet'
        if output_file.exists() and not self.rewrite:
            print(f'{output_file} exists, skipping')
            return
        
        loc_df = gpd.read_parquet(file, columns=['geometry', 'rh95', 'rh98', 'rh100'])
        if loc_df.empty:
            return
        # features more than 10000 would likely fail because of GEE memory limit
        if len(loc_df) > 10000:
            partitions = [loc_df.iloc[i:i+10000] for i in range(0, len(loc_df), 10000)]
        else:
            partitions = [loc_df]
            
        df1s = []
        df2s = []
        for part in partitions:
            eefc = part.apply(set_fc_properties, axis=1)
            eefc = ee.FeatureCollection(eefc.tolist())
            # points = ee.Geometry.MultiPoint(partition.geometry.apply(lambda x: [x.x, x.y]).tolist())
            img_um = self.canopy_height_um.filterBounds(eefc).mosaic().divide(100).rename('RH100_UM')
            img_umd = self.canopy_height_umd.filterBounds(eefc).mosaic().rename('RH95_UMD')
            img_meta = self.canopy_height_meta.filterBounds(eefc).mosaic().rename('RH95_META')
            group1 = self.canopy_height_eth.addBands(img_umd)
            group2 = img_um.addBands(img_meta)
            # EPSG:4326
            fc1 = group1.sampleRegions(
                collection=eefc,
                scale=10
            )
            # EPSG:3857
            fc2 = group2.sampleRegions(
                collection=eefc,
                scale=10
            )
            dfs = []
            
            if fc1.size().getInfo() == 0 or fc2.size().getInfo() == 0:
                print('nothing found')
                df1s.append(pd.DataFrame(np.nan, columns=['RH95_UMD','RH98_ETH'], index=part.index))
                df2s.append(pd.DataFrame(np.nan, columns=['RH100_UM','RH95_META'], index=part.index))
                continue
            for fc in [fc1, fc2]:
                download_id = ee.data.getTableDownloadId({'table': fc, 'fileFormat': 'CSV'})
                res = requests.get(ee.data.makeTableDownloadUrl(download_id))
                if res.status_code == 200:
                    data = StringIO(res.content.decode('utf-8'))
                    df = pd.read_csv(data)
                    df = df.drop(columns=['.geo', 'system:index']).set_index('index')
                    dfs.append(df)
                else:
                    raise requests.HTTPError(f'Failed to download ')
            df1s.append(dfs[0])
            df2s.append(dfs[1])
        if len(df1s) == 0:
            return
        df1s = pd.concat(df1s)
        df2s = pd.concat(df2s)
        if 'RH95_UMD' not in df1s.columns:
            df1s['RH95_UMD'] = pd.NA
        if 'RH98_ETH' not in df1s.columns:
            df1s['RH98_ETH'] = pd.NA
        if 'RH100_UM' not in df2s.columns:
            df2s['RH100_UM'] = pd.NA
        if 'RH95_META' not in df2s.columns:
            df2s['RH95_META'] = pd.NA
        loc_df.loc[df1s.index, ['RH95_UMD','RH98_ETH']] = df1s[['RH95_UMD','RH98_ETH']]
        loc_df.loc[df2s.index, ['RH100_UM','RH95_META']] = df2s[['RH100_UM','RH95_META']]
        loc_df.to_parquet(output_file)


@dataclass
class MyConfig:
    location_files: str='~/data/GVS/GEDI_for_correction/partitions_2020/*.parquet'
    output_dir: str='~/data/GVS/GEDI_for_correction/partitions_with_sota_chm_2020'
    n_parallel: int=40
    rewrite: bool=False
    debug: bool=False
    
cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg):
    cluster = LocalCluster()
    client = Client(cluster)
    
    downloader = SOTAChmDownloader(
        location_files=cfg.location_files,
        output_dir=cfg.output_dir,
        n_parallel=cfg.n_parallel,
        rewrite=cfg.rewrite,
        debug=cfg.debug
    )
    downloader.download()

if __name__ == '__main__':
    main()
