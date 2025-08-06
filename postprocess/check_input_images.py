import xarray as xr
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import hydra
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

def normalize_to_list(val):
    if isinstance(val, list):
        return val
    elif val is None:
        return None
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, str):
        if ',' in val:
            return eval(val)        
        else:
            # val = [int(i) for i in val.replace('[', '').replace(']', '').split(' ')]
            return [int(m) for m in val[1:-1].split(' ') if m != '']
    else:
        return [val]  # or None, depending on desired behavior

def check_water_mask(image):
    pass

def check_duplicated_images():
    
    for year in [2020, 2024]:
        total_cnt = 0
        tiles_duplicated_images = []
        for part_idx in range(21):
            print(f'Checking year {year} part {part_idx}')
            cnt = 0
            tiles_need_redownload = []    
            s2_geoparq_file = Path(f'~/data/GVS/Deploy/deploy_s2_items_{year}_part{part_idx}.parquet').expanduser()
            s2_df = gpd.read_parquet(s2_geoparq_file)
            unique_tiles = s2_df['s2:mgrs_tile'].unique()
            for tile_id in unique_tiles:
                tile_df = s2_df[s2_df['s2:mgrs_tile'] == tile_id]
                duplicated = tile_df['id'].duplicated().any()
                if duplicated:
                    cnt += 1
                    tiles_need_redownload.append(tile_id)
            print(f'Total duplicated images in year {year} part {part_idx}: {cnt}/{len(unique_tiles)}')
            tiles_unique_images = s2_df[~s2_df['s2:mgrs_tile'].isin(tiles_need_redownload)]
            tiles_unique_images.to_parquet(f'~/data/GVS/Deploy/s2_deploy_items_{year}_part{part_idx}_unique_images.parquet')
            tiles_duplicated_images.append(s2_df[s2_df['s2:mgrs_tile'].isin(tiles_need_redownload)])
            total_cnt += cnt
        tiles_duplicated_images = pd.concat(tiles_duplicated_images)
        tiles_duplicated_images['growing_months'] = tiles_duplicated_images['growing_months'].apply(normalize_to_list)
        tiles_duplicated_images.drop_duplicates(subset=['id'], inplace=True)
        tiles_duplicated_images.to_parquet(f'~/data/GVS/Deploy/s2_deploy_items_{year}_part{part_idx+1}_unique_images.parquet')
        print(f'Total duplicated tiles in year {year}: {total_cnt}')

def check_images_order():
    for year in [2020, 2024]:
        for part_idx in range(22):
            not_ordered_tiles = []
            print(f'Checking year {year} part {part_idx}')
            s2_geoparq_file = Path(f'~/data/GVS/Deploy/s2_deploy_items_{year}_part{part_idx}_unique_images.parquet').expanduser()
            s2_df = gpd.read_parquet(s2_geoparq_file)
            unique_tiles = s2_df['s2:mgrs_tile'].unique()
            for tile_id in unique_tiles:
                tile_df = s2_df[s2_df['s2:mgrs_tile'] == tile_id]
                if len(tile_df) > 20:
                    nodata_percentage = tile_df['s2:nodata_pixel_percentage']
                    if nodata_percentage.is_monotonic_increasing:
                        not_ordered_tiles.append(tile_id)
            print(f'Total not ordered tiles in year {year} part {part_idx}: {len(not_ordered_tiles)}')
        

def check_n_images_per_tile():
    for year in [2020, 2024]:
        for part_idx in range(22):
            print(f'Checking year {year} part {part_idx}')
            s2_geoparq_file = Path(f'~/data/GVS/Deploy/deploy_s2_items_{year}_part{part_idx}_unique_images.parquet').expanduser()
            s2_df = gpd.read_parquet(s2_geoparq_file)
            df = s2_df.groupby('s2:mgrs_tile').size().reset_index(name='n_images')
            import ipdb; ipdb.set_trace()
            unique_tiles = s2_df['s2:mgrs_tile'].unique()
            for tile_id in unique_tiles:
                tile_df = s2_df[s2_df['s2:mgrs_tile'] == tile_id]
                if len(tile_df) != 12:
                    print(f'Tile {tile_id} has {len(tile_df)} images')

@dataclass
class MyConfig:
    zarr_store_path: str = '~/data/GVS/Deploy/inference_2024.zarr'
    year: int = 2024
    tile_id: str = '20XNR'
    output_dir: str = '~/data/GVS/Deploy/check_input_images_2024'
    
cs = ConfigStore.instance()
cs.store(name="check_input_images", node=MyConfig)

@hydra.main(config_name='check_input_images', version_base="1.2")
def main(cfg: DictConfig):
    # check_duplicated_images()
    check_images_order()


if __name__ == '__main__':
    main()