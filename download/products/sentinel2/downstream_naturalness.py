import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, Field
from pathlib import Path
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO
from dotenv import load_dotenv
import planetary_computer
import pystac_client
import pystac
import logging
import dask
from dask.distributed import Semaphore
import dask_geopandas as dgp
import h5py
import zarr
import calendar
from shapely.geometry import box
import xarray as xr
from download.core import DaskDownloader, slope
from download.core.utils import get_most_common_epsg, harmonize_to_old, get_patch, buffer_and_snap_bounds, get_total_bounds
from download.core.constants import STAC_ITEM_KEYS

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

def download_downstream_task_data(crowd_source_data_file: str, output_dir: str):
    """
    Download downstream task data for a given task name and save it to a directory.

    Args:

    """
    crowd_source_data_file = Path(crowd_source_data_file).expanduser()
    df = pd.read_csv(crowd_source_data_file)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    
    
class S2Downloader(DaskDownloader):
    def __init__(self, 
                 s2_parquet: str=None,
                 output_dir: str=None,
                 crowd_source_data_file: str=None,
                 sem_max_release: int=60,
                 maxCloudCover: int=50,
                 year: int=2017,
                 patch_size: int=15,
                 out_res: int=10,
                 n_parallel: int=100,
                 comp_name: str='blosclz',
                 comp_level: int=7,
                 wc_dem_meta_dir: str=None,
                 **kwargs):
        super().__init__(n_parallel=n_parallel, **kwargs)
        self.maxCloudCover = maxCloudCover
        self.year = year
        self.wc_dem_meta_dir = Path(wc_dem_meta_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.s2_parquet = Path(s2_parquet).expanduser()
        self.crowd_source_data_file = Path(crowd_source_data_file).expanduser()
        if not self.crowd_source_data_file.exists():
            self.train_val_split()
        df = pd.read_csv(self.crowd_source_data_file)
        self.gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
        self.gdf = self.assign_growing_months()
        self.sem = Semaphore(sem_max_release, name='max_queries')
        self.patch_size_in_meters = patch_size * out_res
        self.out_res = out_res
        self.buffer_size = patch_size // 2 * out_res  # in meters
        self.patch_size = (self.buffer_size * 2 + out_res) / out_res
        self.dem_res = 30
        self.dem_buffer_size = self.patch_size_in_meters // self.dem_res // 2 + 2  # 2 pixels buffer for slope and upsampling
        self.dem_buffer_size = self.dem_buffer_size * self.dem_res
        self.patch_size = (self.buffer_size * 2 + out_res) / out_res
        self.bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]
        
        # dem_df = self.get_aux_df('cop-dem-glo-30', time_col='datetime')
        # self.dem_df = dem_df[dem_df.geometry.intersects(box(*self.gdf.total_bounds))]
        # self.dem_df['datetime'] = self.dem_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        # chunk_size = patch_size
        # compressor = zarr.codecs.BloscCodec(cname=comp_name, clevel=comp_level)
        # self.comp = { # for zarr
        #     's2': {
        #         "compressors": compressor,
        #         # "shards": (1, 1, shard_size, shard_size),
        #         "chunks": (1, 13, chunk_size, chunk_size)
        #     },
        #     'slope': {
        #         "compressors": compressor,
        #         # "shards": (shard_size, shard_size),
        #         "chunks": (1, chunk_size, chunk_size)
        #     }
        # }
        comp_level = 7
        self.comp = {
                's2': {
                    "zlib": False,
                    # "complevel": comp_level,
                    # "fletcher32": True,
                    "chunksizes": (1, 13, 15, 15)
                },
                'slope': {
                    "zlib": False,
                    # "complevel": comp_level,
                    # "fletcher32": True,
                    "chunksizes": (1, 15, 15)
                }
            }
        
    def train_val_split(self, val_ratio:float=0.1):
        df = pd.read_csv(self.crowd_source_data_file, index_col='rowid')
        original_name = self.crowd_source_data_file.stem[:26]
        df_val = df.sample(frac=val_ratio)
        df_val.to_csv(self.crowd_source_data_file.with_stem(f'{original_name}_val'))
        df_train = df.drop(df_val.index)
        df_train.to_csv(self.crowd_source_data_file.with_stem(f'{original_name}_train'))
        
        
    def get_aux_df(self, collection_id, filters=None, time_col=None):
        """
        Retrieves auxiliary geodataframe for a specified collection (ESA world cover or DEM).

        Parameter
        ---------
        * collection_id (str): The ID of the collection to retrieve data from.
        * filters (dict, optional): Additional filters to apply to the data. Defaults to None.
        * time_col (str, optional): The name of the time column to include in the retrieved data. Defaults to None.

        Returns
        -------
        * pandas.DataFrame: The retrieved auxiliary data as a pandas DataFrame.
        """
        parquet_file = self.wc_dem_meta_dir / f'{collection_id}_items.parquet'
        if parquet_file.exists():
            logger.info(f'Loading STAC parquet files for {collection_id} from: {parquet_file}')
            df = gpd.read_parquet(parquet_file, columns=STAC_ITEM_KEYS+[time_col])
        else:
            logger.info(f'Downloading STAC parquet files for {collection_id}')
            asset = api.get_collection(collection_id).assets["geoparquet-items"]
            df = dgp.read_parquet(
                asset.href, storage_options=asset.extra_fields["table:storage_options"],
                gather_spatial_partitions=False, filters=filters, columns=STAC_ITEM_KEYS+[time_col])
            df = df.compute()
            df.to_parquet(parquet_file)
        df = df.set_index('id')
        return df
    
        
    def assign_growing_months(self):
        s2_df = gpd.read_parquet(self.s2_parquet, columns=['Name', 'geometry', 'growing_months'])
        df = self.gdf.sjoin(s2_df, how='left', predicate='intersects')
        df = df.drop_duplicates(subset=['rowid'])
        df = df.drop(columns=['index_right'])
        return df
        
        
    def download(self, job_id:int=0):
        out_file = self.output_dir / f's2_{self.year}_part{job_id}.h5'
        if out_file.exists():
            logger.info(f'{out_file} already exists, skipping')
            return
        n_per_job = 11572
        if (job_id+1)*n_per_job > len(self.gdf):
            df = self.gdf.iloc[job_id*n_per_job:]
        else:
            df = self.gdf.iloc[job_id*n_per_job:(job_id+1)*n_per_job]
        if df.empty:
            logger.info(f'No data to download for job {job_id}')
            return
        tasks = [self.get_best_s2_for_point(row) for i, row in df.iterrows()]
        self.n_parallel = min(self.n_parallel, len(tasks))
        nfaild, results = self.schedule_tasks(delayed_tasks=tasks)
        if nfaild > 0:
            print(f'{nfaild} tasks failed')
            
        results = [r for r in results if r is not None]
        ds = xr.concat(results, dim='time')
        print('saving as h5')
        print(ds)
        ds.to_netcdf(out_file, format='NETCDF4', engine='h5netcdf', encoding=self.comp, mode='w')
        # ds.to_zarr(out_file, mode='w', encoding=self.comp)
    
    def merge_h5s(self, h5_files, out_h5:str=None):
        h5_files = Path(h5_files).expanduser()
        if out_h5 is None:
            out_h5 = h5_files.parent / f's2_{self.year}.h5'
        else:
            out_h5 = Path(out_h5).expanduser()
        h5_files = h5_files.parent.glob(h5_files.name)
        print('merging h5s, ', h5_files)
        data = []
        for file in h5_files:
            ds = xr.open_dataset(file, engine='h5netcdf')  
            data.append(ds)
        ds = xr.concat(data, dim='time')
        print(ds)
        print('saving to ', out_h5)
        ds.to_netcdf(out_h5, format='NETCDF4', engine='h5netcdf', encoding=self.comp, mode='w')
        
    
    def merge_zarr_stores(self, zarr_paths: list, output_path: str):
        # NOTE: not used, zarr datasets for training too slow
        zarr_paths = Path(zarr_paths).expanduser().glob('*.zarr')
        output_path = Path(output_path).expanduser()
        datasets = [xr.open_zarr(path) for path in zarr_paths]
        merged_ds = xr.concat(datasets, dim='time')
        merged_ds = merged_ds.chunk({'time': 1})
        comp_level = 7
        comp = {
                's2': {
                    "zlib": True,
                    "complevel": comp_level,
                    "fletcher32": True,
                    "chunksizes": (1, 14, 15, 15)
                },
                'slope': {
                    "zlib": True,
                    "complevel": comp_level,
                    "fletcher32": True,
                    "chunksizes": (1, 15, 15)
                }
            }
        merged_ds.to_netcdf(output_path, format='NETCDF4', engine='h5netcdf', encoding=comp,mode='w')
        print(f'Merged zarr stores saved to {output_path}')
        
    @dask.delayed
    def get_best_s2_for_point(self, point: gpd.GeoSeries):
        growing_months = point['growing_months']
        items = self.query_s2_for_point(growing_months, point['geometry'])
        if len(items) == 0:
            return
            # return pd.Series([pd.NA, pd.NA], index=['id', 'defective_cover'])
        
        epsg = get_most_common_epsg(items, key='proj:code')
        epsg = int(epsg[5:])
        geom = gpd.GeoSeries(point.geometry, crs='EPSG:4326').to_crs(epsg)
        bounds = geom[0].buffer(self.buffer_size).bounds
        patch = get_patch(items, ['SCL'], resolution=self.out_res, bounds=bounds, epsg=epsg, dtype='uint8', snap_bounds=True, fill_value=np.uint8(0))
        
        if patch.shape[0] == 0 or patch.shape[-2:] != (self.patch_size, self.patch_size): # why there're cases that the output shape is (14,15)? fill_value doesn't work?
            return
            # return pd.Series([pd.NA, pd.NA], index=['id', 'defective_cover'])

        patch = patch.compute() # simplify compute graph, not sure if this is helpful
        scl = patch.data
        defective_cover = np.any([(scl == k) for k in defective_SCL], 0).sum((-2,-1)) / np.prod(scl.shape[-2:])
        if np.isnan(defective_cover).all() or defective_cover.min() > 0.6:
            return
            # return pd.Series([pd.NA, pd.NA], index=['id', 'defective_cover'])

        patch_df = pd.DataFrame({
            'id': patch.id.values,
            'defective_cover': defective_cover.squeeze(),
        })

        patch_df = patch_df.sort_values(['defective_cover'])
        best_s2_id = patch_df.iloc[0].id
        best_item = [item for item in items if item.id == best_s2_id][0]
        s2_image = get_patch(best_item, self.bands, resolution=self.out_res, bounds=bounds, epsg=epsg, dtype='uint16', snap_bounds=True, fill_value=np.uint16(0))
        
        bbox = box(*point.geometry.bounds)  
        dem_bounds = buffer_and_snap_bounds(geom, self.dem_buffer_size, self.dem_res)
        total_bounds_dem = get_total_bounds(dem_bounds)
        # dem_df = self.dem_df[self.dem_df.geometry.intersects(point.geometry)]
        dem_items = api.search(collections=['cop-dem-glo-30'], intersects=bbox).item_collection()
        asset_name = 'data'
        if len(dem_items) == 0:
            dem_items = api.search(collections=['nasadem'], intersects=bbox).item_collection()
            asset_name = 'elevation'
        # else:
        #     dem_items = row_to_stac_item(dem_df, ['datetime'])
        #     dem_items = pystac.item_collection.ItemCollection(dem_items)
        #     dem_items.asset_name = 'data'

        dem_image = get_patch(dem_items.items, [asset_name], resolution=self.dem_res, bounds=total_bounds_dem, epsg=epsg, dtype='float32', snap_bounds=True, fill_value=np.float32(np.nan))
        
        dem_image = dem_image.max(dim='time', skipna=True)
        s2_image = harmonize_to_old(s2_image)
        s2_image.name = 's2'
        dem_image['epsg'] = dem_image.epsg.astype('uint16')
        dem_image.attrs['res'] = self.dem_res
        slope_da = slope(dem_image)
        w, h = slope_da.shape[-2:]
        # set xy coords to the center of the pixel (to match s2 xrr coords)
        slope_da = slope_da.assign_coords(x=range(1, 3*w, 3), y=range(1, 3*h, 3))
        slope_da = slope_da.interp(x=range(3*w), y=range(3*h))
        w, h = slope_da.shape[-2:]
        border = int((slope_da.shape[-2] - self.patch_size) // 2)
        slope_da = slope_da.isel(x=slice(border, -border), y=slice(border, -border))  # remove nan
        slope_da = slope_da.assign_coords(x=s2_image.x, y=s2_image.y)  # set xy coords back to 0-14
        slope_da = slope_da.squeeze()
        
        ds = xr.merge([s2_image, slope_da], join='inner', combine_attrs='drop')
        ds = ds.drop_vars(['x', 'y'])
        ds = ds.assign_coords(
            centroid=(['time', 'coord'], [[point.geometry.centroid.x, point.geometry.centroid.y]]), # lon, lat
            defective_cover=('time', [patch_df.iloc[0].defective_cover]),
            epsg=('time', [epsg]),
            rowid=('time', [point.rowid])
        )
        ds['epsg'] = ds['epsg'].astype('uint16')
        ds = ds.compute()
        return ds  
        
    def query_s2_for_point(self, growing_months: list, geom: gpd.GeoSeries):
        start_end_pairs = []

        if len(growing_months)>0:
            growing_months = sorted(growing_months)
            start = growing_months[0]
            end = growing_months[0]
            for month in growing_months[1:]:
                if month == end + 1:
                    end = month
                else:
                    start_end_pairs.append((start, end))
                    start = month
                    end = month
            start_end_pairs.append((start, end))
        if len(start_end_pairs) == 0:
            return []
        items = []
        for start, end in start_end_pairs:
            start_date = f'{self.year}-{start:02d}-01'
            end_date = f'{self.year}-{end:02d}-{calendar.monthrange(self.year, end)[1]}'
            search = api.search(collections=['sentinel-2-l2a'],
                                query={
                                    "eo:cloud_cover": {
                                        "lt": self.maxCloudCover
                                    }
                                },
                                intersects=geom,
                                datetime=f'{str(start_date)[:10]}/{str(end_date)[:10]}')
            items.extend((search.item_collection()))            

        return items

@dataclass
class Config:
    task_name: str = ""  # The name of the task for which the data is being downloaded.
    crowd_source_data_file: str = "~/data/gvs/downstream_task_data/naturalness/reference_data_set_updated.csv"  # Path to the CSV file containing crowd-sourced data.
    output_dir: str = "~/data/gvs/downstream_task_data"  # Directory where the downloaded data will be saved.
    s2_parquet: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'  # Path to the Parquet file containing S2 tiles with growing months information.
    wc_dem_meta_dir: str = '~/data/GEDI'
    n_parallel: int = 100
    maxCloudCover: int = 50
    year: int = 2017
    patch_size: int = 31
    out_res: int = 10
    job_id: int = 0

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(config_name="config", version_base='1.2')
def main(cfg: DictConfig) -> None:
    downloader = S2Downloader(**cfg)
    # downloader.download(job_id=cfg.job_id)
    # downloader.merge_zarr_stores(zarr_paths=cfg.output_dir, output_path=f'{cfg.output_dir}/s2_{cfg.year}.zarr')
    downloader.merge_h5s('~/data/gvs/downstream_task_data/s2_2017_part*.h5', '~/data/gvs/downstream_task_data/s2_2017_ps31.h5')

    
if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster, performance_report
    from dask import config
    config.set({'distributed.scheduler.locks.lease-timeout': 60000000000000}) 
    # might fix the communication error caused by I/O. ref: https://github.com/dask/distributed/issues/3129#issuecomment-1684858307
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 60000000000000})
    dask.config.set({'dataframe.query-planning': False})  # NOTE: dask expr causes the computing of s2 geoparqut hanging
    # NOTE: avoid coverting the assets dict to a long string of type string[pyarrow]
    dask.config.set({"dataframe.convert-string": False})
    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)
    print(client)
    main()
