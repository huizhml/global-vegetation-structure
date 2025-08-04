import time
import numpy as np
import geopandas as gpd
from download._const import S2_ITEM_PROPS
from download._utils import get_patch, row_to_stac_item
from pathlib import Path
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore

def download_tile(metadata_file:str=None, tile_id:str=None, output_dir:str=None, n_iamges_per_tile:int=20):
    '''
    This function is mainly used for streaming Sentinel-2 data (2024) on LUMI.
    I decided to not download ESA World Cover, because:
    1. We don't have ground truth for 2024.
    2. We don't need it for inference.
    '''
    bands = [
        'B01', 'B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B8A',
        'B09', 'B11', 'B12', 'SCL'
    ]
    metadata_file = Path(metadata_file).expanduser()
    s2_df = gpd.read_parquet(metadata_file)
    tile_df = s2_df[s2_df['s2:mgrs_tile'] == tile_id].set_index('id')
    if len(tile_df)>n_iamges_per_tile:
        if (tile_df['s2:nodata_pixel_percentage']==0).sum() > 0:
            tile_df = tile_df.sort_values(['s2:nodata_pixel_percentage', 'eo:cloud_cover']).head(n_iamges_per_tile)
        else:
            idx = tile_df.groupby('orbit')['eo:cloud_cover'].nsmallest(n_iamges_per_tile//2).index.get_level_values(1)
            tile_df = tile_df.loc[idx]
    tile_df['datetime'] = tile_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    # bbox = box(*tile_df.total_bounds) # For esa world cover
    # epsg = items[0].properties['proj:epsg']
    items = row_to_stac_item(tile_df, S2_ITEM_PROPS)  
    image = get_patch(items, bands, dtype='uint16', fill_value=np.uint16(0))
    image.name = 's2'
    del image.attrs['spec']
    del image.attrs['crs']
    t0=time.time()
    h5_file = Path(output_dir).expanduser() / f'{tile_id}.h5'
    image.to_netcdf(h5_file, engine='h5netcdf', encoding={'s2': {'zlib': False, 'chunksizes': (1, 1, 1024, 1024)}})
    print(f'Time taken to save image: {time.time() - t0:.2f} seconds')
    
    
@dataclass
class MyConfig:
    metadata_file: str = '~/data/GVS/Deploy/deploy_s2_items_2024.parquet'
    tile_id: str = '32MRE'
    output_dir: str = '~/data/GVS/Deploy/inference_2024/'
    n_iamges_per_tile: int = 20

cs = ConfigStore.instance()
cs.store(name='config', node=MyConfig)
 
@hydra.main(config_name='config', version_base='1.2')
def main(cfg):
    download_tile(cfg.metadata_file, cfg.tile_id, cfg.output_dir, cfg.n_iamges_per_tile)
    
if __name__ == '__main__':
    main()
    
    