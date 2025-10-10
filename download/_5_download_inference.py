import time
import logging
import pystac
import planetary_computer
import xarray as xr
from pathlib import Path
from utils._stackstac import stack
from dask.utils import natural_sort_key
import dask.array as da
import pystac_client
from pystac_client.stac_api_io import StacApiIO
from shapely.geometry import box, shape
import geopandas as gpd
import dask_geopandas as dgp
import geodatasets
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import hydra
import zarr
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
import warnings
import os
# from stackstac import stack
from utils._stackstac import stack
from download._utils import utm_to_wgs84, build_parquet_file_table, filter_parquet_files, row_to_stac_item, get_patch
from download._const import S2_ITEM_PROPS, STAC_ITEM_KEYS
from download._dask_downloader import DaskDownloader
from download._4_download import harmonize_to_old


logger = logging.getLogger(__name__)

stac_api_io = StacApiIO()
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)

warnings.filterwarnings("ignore", 
                        category=UserWarning,
                        module="zarr.codecs.vlen_utf8",
                        message=".*vlen-utf8.*")

def get_s2_tiles_by_landmass(s2_shfile):
    s2_df = gpd.read_file(s2_shfile)
    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    world = world[world.bounds.miny > -60]
    s2_df = s2_df.sjoin(world) # 
    # S2 shapefile from this repo has tiles with more than one geometry. https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index
    return s2_df.drop_duplicates(subset='Name', keep='first')


class WorldS2(DaskDownloader):
    def __init__(self, year:int=2020, 
                    s2_parquet:str=None,
                    wc_parq_file:str=None,
                    save_dir: str = None,
                    store_name: str = None,
                    max_cloud_cover: int = 50,
                    max_water_percentage: int = 99,
                    comp_name: str = 'blosclz',
                    comp_level: int = 7,
                    n_parallel: int = 8,
                    n_iamges_per_tile: int = 20,
                    total_splits: int = 21,
                    output_format: str = 'h5',
                    debug: bool=False,
                    **kwargs):
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.total_splits = total_splits
        self.n_tiles_per_split = 900 # for total_splits = 21
        self.n_iamges_per_tile = n_iamges_per_tile
        self.output_format = output_format
        self.s2_parquet = Path(s2_parquet).expanduser()
        self.wc_parq_file = Path(wc_parq_file).expanduser()
        self.save_dir = Path(f'{save_dir}').expanduser()
        self.save_dir.mkdir(exist_ok=True, parents=True)
        (self.save_dir / f'{year}').mkdir(exist_ok=True, parents=True) # for download flags
        
        chunk_size = 1024
        if output_format == 'zarr':
            compressor = zarr.codecs.BloscCodec(cname=comp_name, clevel=comp_level)
            self.store_name = f'{store_name}_{year}.zarr'
            self.comp = {
                        's2': {
                            "compressors": compressor,
                            # "shards": (1, 1, shard_size, shard_size),
                            "chunks": (1, 1, chunk_size, chunk_size)
                        },
                        'esa_wc': {
                            "compressors": compressor,
                            # "shards": (shard_size, shard_size),
                            "chunks": (chunk_size, chunk_size)
                        }
                    }
        elif output_format == 'h5':
            self.store_name = f'{store_name}_{year}'
            (self.save_dir/f'{self.store_name}').mkdir(exist_ok=True, parents=True)
            self.comp = { # don't compress, too slow
                's2': {
                    'zlib': False,
                    # "compressors": None,
                    # "shards": (1, 1, shard_size, shard_size),
                    "chunksizes": (1, 1, chunk_size, chunk_size)
                },
                'esa_wc': {
                    'zlib': False,
                    # "compressors": None,
                    # "shards": (shard_size, shard_size),
                    "chunksizes": (chunk_size, chunk_size)
                }
            }
        self.max_cloud_cover = max_cloud_cover
        self.max_water_percentage = max_water_percentage
        self.debug = debug
        self.esa_wc_time = pd.Timestamp(f'2021-01-01', tz='UTC')
        self.year = year
        self.bands = [
            'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
            'B09', 'B11', 'B12', 'SCL'
        ]
        
        
        
    def download_by_api_query(self, specified_tiles_df=None, collection_id='sentinel-2-l2a', **kwargs):
        s2_tiles_df = gpd.read_parquet(self.s2_parquet, columns=['Name', 'growing_months', 'geometry'])
        s2_tiles_df = s2_tiles_df.set_index('Name')
        if specified_tiles_df is not None:
            specified_tiles = specified_tiles_df['Name'].unique()
            s2_tiles_df = s2_tiles_df.loc[specified_tiles]
        # wc_df = self.retrive_wc_items()
        # wc_df = wc_df.sjoin(s2_tiles_df, how='inner')
        
        # for tile, row in s2_tiles_df.iterrows():
            # self.query_and_download_tile(tile, row, wc_df,collection_id).compute()
        # s2_tiles_df = s2_tiles_df.loc[['01UCT']]
        tasks = [self.query_and_download_tile(tile, row, collection_id) for tile, row in s2_tiles_df.iterrows()]
        dask.compute(*tasks)
            
    @delayed
    def query_and_download_tile(self, tile, row, collection_id='sentinel-2-l2a'):
        '''
        For downloading Sentinel-2 images from 2017, no metadata saved for this year. 
        And the bulk downloading was not working becuase Microsoft may move the metadata parquet files to somewhere else.
        '''
        file = self.save_dir / f'{self.store_name}'
        flag = self.save_dir / f'{self.year}/{tile}_done'
        # if flag.exists():
        #     logger.info(f'{tile} exists, skipping...')
        #     return
        api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)
        datetime = f'{self.year}-01-01/{self.year}-12-31'
        bbox = row.geometry.bounds
        bbox = [bbox[0], bbox[1], min(bbox[2], 180), bbox[3]]
        search = api.search(collections=collection_id, bbox=bbox, datetime=datetime, 
                            query={'eo:cloud_cover': {'lt': self.max_cloud_cover},
                                    's2:nodata_pixel_percentage': {'lt': 90},
                                    's2:mgrs_tile': {'eq': tile}})
        items = search.item_collection()
        if len(items) == 0:
            print(f'{tile} has no images, bbox={bbox}, skipping...')
            return
        df = gpd.GeoDataFrame.from_features(items.to_dict(), crs='epsg:4326')
        df['id'] = df['s2:product_uri'].str.replace(r'_[A-Z]\d{4}', '', regex=True).str.replace('.SAFE', '')
        month = pd.to_datetime(df['datetime']).dt.month
        df = df[month.isin(row.growing_months)]
        # get top 30 images, 10 from the best orbits and 20 from the rest
        if (df['s2:nodata_pixel_percentage']==0).sum() > 0:
            best_orbits = df[df['s2:nodata_pixel_percentage']==0]['sat:relative_orbit'].unique()
            best = df[df['sat:relative_orbit'].isin(best_orbits)]
            rest = df[~df['sat:relative_orbit'].isin(best_orbits)]
        else:
            rest = df
            best = pd.DataFrame([], columns=df.columns)
            
        unique_orbits = rest['sat:relative_orbit'].unique()
        if len(unique_orbits) >= 2:
            top_orbits = rest.groupby('sat:relative_orbit').min('s2:nodata_pixel_percentage').sort_values('s2:nodata_pixel_percentage').head(2)
            rest = rest[rest['sat:relative_orbit'].isin(top_orbits.index)]
            idx = rest.groupby('sat:relative_orbit')['eo:cloud_cover'].nsmallest(10).index.get_level_values(1)
            rest = rest.loc[idx]
            df = pd.concat([best, rest])
        else:
            rest = rest.sort_values('eo:cloud_cover').head(10)
            best = best.sort_values('eo:cloud_cover').head(20)
        df = pd.concat([best, rest])
        
        # get top 20 images
        df = df.drop_duplicates(subset='id')
        if len(df)>self.n_iamges_per_tile:
            df = df.sort_values(['s2:nodata_pixel_percentage', 'eo:cloud_cover'])
            if (df['s2:nodata_pixel_percentage']==0).sum() > 0:
                df = df.head(self.n_iamges_per_tile)
            else:
                idx = df.groupby('orbit')['eo:cloud_cover'].nsmallest(self.n_iamges_per_tile//2).index.get_level_values(1)
                df = df.loc[idx]
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        # df.to_parquet(self.save_dir/ f'{self.year}_{tile}_images.parquet')
        epsg = int(items[0].properties['proj:code'][5:])
        items = [item for item in items.items if item.id in df['id'].values]
        images = stack(items, self.bands, dtype='uint16', fill_value=np.uint16(0), epsg=epsg, resolution=10, rescale=False)
        assert images.shape[2] == images.shape[3] == 10980, f'{tile} has incorrect shape {images.shape}'
        images.name = 's2'
        # wc_df = wc_df[wc_df.Name == tile]
        # wc_df['datetime'] = wc_df['start_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        # wc_df = wc_df.set_index('id')
        # if not wc_df.empty:
        #     wc_items = row_to_stac_item(wc_df, ['datetime'])
        #     wc_image = stack(wc_items, ['map'], bounds=images.spec.bounds, epsg=epsg, dtype='uint16', fill_value=np.uint16(0))
        #     import ipdb; ipdb.set_trace()
        #     wc_image = wc_image.max(dim='time', skipna=True).squeeze()
        #     wc_image.name = 'esa_wc'
        #     ds = xr.merge([images, wc_image], join='outer')
        #     comp = self.comp
        # else:
        del images.attrs['spec']
        del images.attrs['crs']
        images = images.reset_coords(['proj:bbox'], drop=True)
        ds = xr.Dataset({'s2': images}, coords=images.coords, attrs=images.attrs)
        comp = {'s2': self.comp['s2']}
        print(ds)
        if file.exists():
            try:
                print('saveing to ', f'{file}/{tile}')
                store = ds.to_zarr(file, mode='a', group=f'{tile}', encoding=comp)
                print(store)
                flag.touch()
            except Exception as err:
                print(err)
                print(f'{tile} failed')
        else:
            ds.to_zarr(file, mode='w', group=f'{tile}', encoding=comp)
            flag.touch()

            
            
    def download(self, job_id, specified_tiles_df=None, **kwargs):
        if specified_tiles_df is not None:
            postfix = '_specified'
            print(f'Downloading S2 images for specified tiles: {specified_tiles_df}')
        else:
            postfix = ''
        
        if specified_tiles_df is None:
            s2_df = gpd.read_parquet(self.save_dir / f'slurm_job_files_{self.year}/deploy_s2_items_{self.year}_part{job_id}.parquet')
            s2_tiles = s2_df['s2:mgrs_tile'].unique() # 15019 tiles
            s2_df = s2_df.set_index('s2:mgrs_tile')
        else:
            s2_tiles = specified_tiles_df['Name'].unique()
            idx = specified_tiles_df[f'meta_file_idx_{self.year}'].unique().astype(int)
            meta_files = [self.save_dir / f'slurm_job_files_{self.year}/deploy_s2_items_{self.year}_part{i}.parquet' for i in idx]
            s2_df = dgp.read_parquet(meta_files, gather_spatial_partitions=False)
            s2_df = s2_df.compute()
            s2_df = s2_df.set_index('s2:mgrs_tile')
            s2_df = s2_df.loc[s2_tiles]
            s2_df['datetime'] = pd.to_datetime(s2_df['datetime'], utc=True)
        # self.store_name = f'inference_part{job_id}.zarr'
        if self.debug:
            process_tiles = s2_tiles[:8] # ['32MQE'] if job_id == 7 else ['32TMT']
        else:
            process_tiles = s2_tiles
        unfinished_tiles = []
        for tile in process_tiles:
            if not (self.save_dir / f'{self.year}/{tile}_done').exists():
                unfinished_tiles.append(tile)
        s2_df = s2_df.loc[unfinished_tiles]
        print(f'Processing {len(unfinished_tiles)} tiles, {len(s2_df)} images')
        wc_df = self.retrive_wc_items()
        wc_df['datetime'] = wc_df['start_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        self.wc_df = wc_df[wc_df.geometry.intersects(box(*s2_df.total_bounds))]
        tasks = [self.download_tile(df) for tile, df in s2_df.groupby('s2:mgrs_tile')]
        if len(tasks) < self.n_parallel:
            self.n_parallel = len(tasks)
        if self.debug:            
            dask.compute(*tasks)
        else:
            nfailed, results = self.schedule_tasks(delayed_tasks=tasks)

            if nfailed <= 0:
                flag = self.save_dir / f'{self.year}_job_{job_id}_done'
                flag.touch()

    @delayed
    def download_tile(self, df):
        tile = df.index[0]
        df = df.drop_duplicates(subset='id')
        df = df.set_index('id')
        if len(df)>self.n_iamges_per_tile:
            df = df.sort_values(['s2:nodata_pixel_percentage', 'eo:cloud_cover'])
            if (df['s2:nodata_pixel_percentage']==0).sum() > 0:
                df = df.head(self.n_iamges_per_tile)
            else:
                idx = df.groupby('orbit')['eo:cloud_cover'].nsmallest(self.n_iamges_per_tile//2).index.get_level_values(1)
                df = df.loc[idx]
        
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        file = self.save_dir / f'{self.store_name}'
        flag = self.save_dir / f'{self.year}/{tile}_done'
        # if flag.exists():
        #     logger.info(f'{tile} exists, skipping...')
        #     return
        bbox = box(*df.total_bounds)
        wc_df = self.wc_df[self.wc_df.geometry.intersects(bbox)]
        items = row_to_stac_item(df, S2_ITEM_PROPS)  
        epsg = items[0].properties['proj:epsg']
        images = stack(items, self.bands, dtype='uint16', fill_value=np.uint16(0), rescale=False)
        images = harmonize_to_old(images)
        images.name = 's2'
        wc_df = wc_df.set_index('id')
        if not wc_df.empty:
            wc_items = row_to_stac_item(wc_df, ['datetime'])
            wc_image = stack(wc_items, ['map'], bounds=images.spec.bounds, epsg=epsg, dtype='uint16', fill_value=np.uint16(0), rescale=False)
            wc_image = wc_image.max(dim='time', skipna=True).squeeze()
            wc_image.name = 'esa_wc'
            del images.attrs['spec']
            del images.attrs['crs']
            ds = xr.merge([images, wc_image], join='outer')
            
            comp = self.comp
        else:
            del images.attrs['spec']
            del images.attrs['crs']
            ds = xr.Dataset({'s2': images}, coords=images.coords, attrs=images.attrs)


        if self.output_format == 'zarr':
            if file.exists():
                try:
                    print('saveing to ', file)
                    store = ds.to_zarr(file, mode='a', group=f'{tile}', encoding=comp)
                    print(store)
                    flag.touch()
                except Exception as err:
                    print(err)
                    print(f'{tile} failed')
            else:
                ds.to_zarr(file, mode='w', group=f'{tile}', encoding=comp)
                flag.touch()
        elif self.output_format == 'h5':
            ds.to_netcdf(file / f'{tile}.h5', engine='h5netcdf', encoding=comp)
            flag.touch()

    def check_images_per_tile(self):
        import matplotlib.pyplot as plt
        s2_tile = gpd.read_parquet(self.save_dir.parent / f's2_tiles_with_growing_months.parquet')
        s2_tile = s2_tile.set_index('Name')
        df = []
        for part in range(21):
            df.append(gpd.read_parquet(self.save_dir / f'deploy_s2_items_{self.year}_part{part}.parquet'))
        df = pd.concat(df)
        print(len(df))
        count_per_tile = df.groupby('s2:mgrs_tile').size()
        s2_images = s2_tile.join(count_per_tile.to_frame('images_count'), how='left')
        s2_images.plot(column='images_count', legend=True, legend_kwds={'label': "Number of images", 'orientation': "horizontal"})
        plt.savefig( 'output/deploy_data/distribution_map_images_per_tile.png')

    def retrive_wc_items(self):
        if self.wc_parq_file.exists():
            return gpd.read_parquet(self.wc_parq_file)
        logger.info(f'Downloading STAC parquet files for ESA WorldCover')
        asset = api.get_collection('esa-worldcover').assets["geoparquet-items"]
        filters = [('start_datetime', '>=', self.esa_wc_time)]
        time_col = 'start_datetime'
        df = dgp.read_parquet(
                asset.href, storage_options=asset.extra_fields["table:storage_options"],
                gather_spatial_partitions=False, filters=filters, columns=STAC_ITEM_KEYS+[time_col])
        df = df.compute()
        df.to_parquet(self.wc_parq_file)
        return df


    def retrive_s2_items(self, total_splits, specified_tiles=None):
        '''
        Retrieve metadata of the top 10 least cloud covered images for each S2 tile using S2 snapshot
        15019 S2 tiles in land area.
        '''
        # check s2 metadata file completeness
        files = self.save_dir.glob(f'deploy_s2_items_{self.year}*{postfix}.parquet')
        if len(list(files)) == total_splits:
            print(f'There are {len(list(files))} previously generated s2 metadata files, which is equal to the desired {total_splits} splits, skipping...')
            return
        if specified_tiles is None and len(list(files)) > 0 and len(list(files)) < total_splits:
            print(f'There are {len(list(files))} previously generated s2 metadata files, but less than desired {total_splits} splits,  removing them and regenerating...')
            os.remove(self.save_dir / f'deploy_s2_items_{self.year}*.parquet')
        print(f'Downloading STAC parquet files for Sentinel-2')
        s2_tiles_df = pd.read_parquet(self.s2_parquet, columns=['Name', 'growing_months'])
        s2_tiles_df = s2_tiles_df.set_index('Name')
        
        pq_files = build_parquet_file_table('sentinel-2-l2a')
        start = f'{self.year}-01-01'
        end = f'{self.year}-12-31'
        pq_files = filter_parquet_files(pq_files, start, end)
        # pq_files = pq_files[30:32]
        s2asset = api.get_collection("sentinel-2-l2a").assets["geoparquet-items"]
        pq_files = sorted(pq_files, key=natural_sort_key)

        if specified_tiles is None:
            tiles = s2_tiles_df.index.drop_duplicates().tolist()
            tiles_list = [tiles[i*self.n_tiles_per_split:(i+1)*self.n_tiles_per_split] for i in range(total_splits)]
            postfix = ''
        else:
            tiles_list = [specified_tiles]
            postfix = f'_specified'
        for i, tiles_p in enumerate(tiles_list):
            file = self.save_dir / f'deploy_s2_items_{self.year}_part{i}{postfix}.parquet'
            if file.exists():
                continue

            s2_df = dgp.read_parquet(
                pq_files,
                storage_options=s2asset.extra_fields["table:storage_options"],
                gather_spatial_partitions=False,
                columns=STAC_ITEM_KEYS + S2_ITEM_PROPS + ['eo:cloud_cover', 's2:mgrs_tile', 's2:nodata_pixel_percentage'],
                filters=[('s2:mgrs_tile', 'in', tiles_p),
                        ("eo:cloud_cover", "<", self.max_cloud_cover),
                        ('s2:nodata_pixel_percentage', '<', 90) # There're images with nodata percentage = 100 and cloud cover = 0
                        # ('s2:water_percentage', '<', self.max_water_percentage)
                        ]
            )
            # s2_df = s2_df.map_partitions(self.remove_duplicate, meta=s2_df.dtypes.to_dict())
            meta = s2_df.dtypes.to_dict()
            meta.update({'growing_months': 'object', 'orbit': 'string'})
            grow_months = s2_tiles_df.loc[tiles_p]
            # s2_df = s2_df.compute()
            s2_df = s2_df.merge(grow_months, left_on='s2:mgrs_tile', right_index=True)
            # s2_df = s2_df.groupby('s2:mgrs_tile').apply(self.get_top_10)
            s2_df = s2_df.groupby('s2:mgrs_tile').apply(self.get_top_10, meta=meta)
            s2_df.crs = 'epsg:4326'
            s2_df = dgp.from_dask_dataframe(s2_df, geometry='geometry')
            s2_df = s2_df.reset_index(drop=True)
            s2_df = s2_df.compute()

            s2_df.to_parquet(file)#, name_fuction=lambda x: f'{x}.parquet')
        
    def remove_duplicate(self, df):
        df[['group_id', 'generation_time']] = df['id'].str.rsplit('_', n=1, expand=True)
        df = df.sort_values('generation_time').groupby('group_id').first()
        df = df.reset_index().drop(columns=['generation_time', 'group_id'])
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        df = df.astype({'proj:epsg': 'uint16', 'eo:cloud_cover': 'float32',
                             'id': 'string[python]'})
        return df


    def get_top_10(self, df):
        if len(df) <= 20:
            return df
        grow_months = df['growing_months'].iloc[0]
        if isinstance(grow_months, str):
            grow_months = [int(m) for m in grow_months[1:-1].split(' ') if m != '']
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        if len(grow_months) < 12:
            idx = df['datetime'].dt.month.isin(grow_months)
            df = df[idx]
        # in_growth_period = (df['datetime'] >= onset) and (df['datetime'] <= end)
        # distance_to_onset = (df['datetime'] - onset).dt.days
        # distance_to_end = (end - df['datetime']).dt.days
        # df.loc[~in_growth_period, 'distance_¨¨to_growth_period'] = pd.DataFrame([distance_to_onset, distance_to_end]).min()
        # df.loc[in_growth_period, 'distance_to_growth_period'] = 0


        df['orbit'] = df['id'].str.extract(r'_R(\d{3})_')
        # if there's a best orbit that covers the whole tile, 
        if (df['s2:nodata_pixel_percentage']==0).sum() > 0: 
            best_orbits = df[df['s2:nodata_pixel_percentage']==0]['orbit'].unique()
            best = df[df['orbit'].isin(best_orbits)]
            rest = df[~df['orbit'].isin(best_orbits)]
        else:
            rest = df
            best = pd.DataFrame([], columns=df.columns)
         
        unique_orbits = rest['orbit'].unique()
        if len(unique_orbits) >= 2:
            # select the best 2 orbits with the least nodata percentage
            top_orbits = rest.groupby('orbit').min('s2:nodata_pixel_percentage').sort_values('s2:nodata_pixel_percentage').head(2)
            rest = rest[rest['orbit'].isin(top_orbits.index)]
            # select the best 10 images from each orbit
            idx = rest.groupby('orbit')['eo:cloud_cover'].nsmallest(10).index.get_level_values(1)
            rest = rest.loc[idx]
            best = best.sort_values('eo:cloud_cover').head(10)
        else:
            rest = rest.sort_values('eo:cloud_cover').head(10)
            best = best.sort_values('eo:cloud_cover').head(20)

        res = pd.concat([best, rest])
        return res

        
@dataclass
class MyConfig:
    year: int = 2020
    s2_parquet: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    save_dir: str = '~/data/gvs/deploy'
    store_name: str = 'inference'
    wc_parq_file:str = '~/data/gvs/deploy/esa_wc.parquet'
    comp_name: str = 'lz4'
    comp_level: int = 7
    max_cloud_cover: int = 90
    max_water_percentage: int = 99
    total_splits: int = 21
    job_id: int = 5
    specified_tiles_file: str='' # download/evaluation_tiles.txt
    n_parallel: int = 8
    output_format: str = 'zarr'
    debug: bool = False
    task: str = 'download'

cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)


@hydra.main(config_name='config', version_base='1.2')
def main(cfg):
    from omegaconf import open_dict
    t0 = time.time()
    if len(cfg.specified_tiles_file) > 0:
        specified_tiles_df = pd.read_csv(cfg.specified_tiles_file)
    else:
        specified_tiles_df = None
    # Add specified_tiles to cfg using OmegaConf merge
    # with open_dict(cfg):
        # cfg.specified_tiles = specified_tiles
    s2 = WorldS2(**cfg)
    getattr(s2, cfg.task)(**cfg, specified_tiles_df=specified_tiles_df)
    # s2.check_images_per_tile()
    print(f"Time taken: {time.time() - t0:.2f}s")

if __name__ == '__main__':

    from dask.distributed import Client, LocalCluster
    from dask import config
    dask.config.set({"distributed.worker.profile.enabled": False})
    config.set({'distributed.scheduler.locks.lease-timeout': 6000000}) 
    # might fix the communication error caused by I/O. ref: https://github.com/dask/distributed/issues/3129#issuecomment-1684858307
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 6000000})
    dask.config.set({'dataframe.query-planning': False})  # NOTE: dask expr causes the computing of s2 geoparqut hanging
    # NOTE: avoid coverting the assets dict to a long string of type string[pyarrow]
    dask.config.set({"dataframe.convert-string": False})
    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)
    print(client)
    main()

# t0 = time.time()
# item_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20240806T102559_R108_T32TMT_20240806T134756"

# # Load the individual item metadata and sign the assets
# item = pystac.Item.from_file(item_url)
# signed_item = planetary_computer.sign(item)

# img = stack(signed_item, resolution=10, properties=False)
# # Open one of the data assets (other asset keys to use: 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP', 'visual')
# img = harmonize_to_old(img)
# img = img.drop_vars(['gsd', 'title','common_name', 'center_wavelength', 'full_width_half_max', 'proj:bbox'])
# img = img.sel(band=[ 'B01', 'B04', 'B03', 'B02',
#      'B05', 'B06', 'B07', 'B08', 'B8A',
#     'B09', 'B11', 'B12'])

# bbox_of_interest = utm_to_wgs84(img.spec.bounds, utm_epsg=img.spec.epsg)
# search = api.search(collections=["esa-worldcover"],
#     bbox=bbox_of_interest,
#     datetime="2021-01-01/2021-12-01")
# items = search.item_collection()
# wc_img = stack(items, ['map'], resolution=10, epsg=img.spec.epsg, bounds=img.spec.bounds, dtype='int16', fill_value=0, 
#                properties=False)
# wc_img = wc_img.max(dim='time', skipna=True).squeeze()
# wc_img = wc_img.assign_coords({'band': 'esa_wc'})
# wc_img = wc_img.drop_vars(['created', 'raster:bands', 'proj:shape', 'title', 'description' ])
# img_da = xr.concat([img, wc_img], dim='band', coords='minimal', compat='override', combine_attrs='drop')
# img_da.name = 'image'

# encoding = {
# "image": {'dtype': 'int16', 'complevel': 7, 'zlib': True }
# } 
# print(img_da)
# img_da.to_netcdf('predict_tile_32TMT.h5', engine='h5netcdf', format='NETCDF4', encoding=encoding)

# print(f"Time taken: {time.time() - t0:.2f}s")