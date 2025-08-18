import geopandas as gpd
from dataclasses import dataclass
from pathlib import Path
import hydra
import os
from hydra.core.config_store import ConfigStore
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import re

def check_input_images_availability(s2_grid_file: str):
    s2_grid_file = Path(s2_grid_file).expanduser()
    df = gpd.read_parquet(s2_grid_file)
    for year in [2020, 2024]:
        df[f'no_images_{year}'] = False
        input_dir = Path(f'~/data/GVS/Deploy/inference_{year}.zarr').expanduser()
        for tile in df['Name']:
            if not (input_dir / tile).exists():
                df.loc[df['Name'] == tile, f'no_images_{year}'] = True
    df.to_parquet(f'~/data/GVS/Deploy/s2_grid_input_images_availability.parquet')

def check_prediction_status(s2_grid_file: str, flag_dir: str, prediction_dir: str, save_dir: str):
    s2_grid_file = Path(s2_grid_file).expanduser()
    save_dir = Path(save_dir).expanduser()
    prediction_dir = Path(prediction_dir).expanduser()
    deploy_dir = Path(f'~/data/GVS/Deploy').expanduser()
    
    df = gpd.read_parquet(s2_grid_file, columns=['Name', 'geometry', 'growing_months'])
    tiles = df['Name'].unique()
    for year in  [2020, 2024]:
        flag_dir_ = Path(f'{flag_dir}_{year}').expanduser()
        input_s2_dir = Path(f'~/data/GVS/Deploy/inference_{year}.zarr').expanduser()
        config_files = deploy_dir.glob(f'slurm_job_files_{year}/*_items_{year}_part*.txt')
        config_files = [config_file for config_file in config_files]
        df[f'predicted_{year}'] = pd.NA
        df[f'has_s2_images_{year}'] = pd.NA
        df[f'has_s2_metadata_{year}'] = pd.NA
        df[f'broken_s2_images_{year}'] = pd.NA
        df[f'meta_file_idx_{year}'] = pd.NA
        for tile in tiles:
            flag_file_old = flag_dir_ / f'{tile}_done'
            flag_file_new = flag_dir_ / f'{tile}_best_images_done'
            # cog_files = list((prediction_dir / f'{tile}_cog').glob('*.cog.tif'))
            if (flag_file_old.exists() or flag_file_new.exists()): #and len(cog_files) == 303:
                df.loc[df['Name'] == tile, f'predicted_{year}'] = True
            else:
                df.loc[df['Name'] == tile, f'predicted_{year}'] = False
                # check input images availability
                if (input_s2_dir / tile).exists():
                    with xr.open_zarr(input_s2_dir, group=tile) as ds:
                        if hasattr(ds, 's2'):
                            if ds.s2.shape[0] > 0:
                                df.loc[df['Name'] == tile, f'has_s2_images_{year}'] = True
                            else:
                                df.loc[df['Name'] == tile, f'broken_s2_images_{year}'] = True
                        else:
                            df.loc[df['Name'] == tile, f'broken_s2_images_{year}'] = True
                else:
                    df.loc[df['Name'] == tile, f'has_s2_images_{year}'] = False
                #  check metadata availability
                for config_file in config_files:
                    if tile in config_file.read_text().splitlines():
                        df.loc[df['Name'] == tile, f'has_s2_metadata_{year}'] = True
                        df.loc[df['Name'] == tile, f'meta_file_idx_{year}'] = int(re.search(r'part(\d+)', config_file.stem).group(1))
                        break
                # if tile in tiles_with_metadata:
                #     df.loc[df['Name'] == tile, f'has_s2_metadata_{year}'] = True
                # else:
                #     df.loc[df['Name'] == tile, f'has_s2_metadata_{year}'] = False
    df.to_parquet(save_dir / f'deploy_status.parquet')

def get_unfinished_tiles(deploy_status_file: str):
    deploy_status_file = Path(deploy_status_file).expanduser()
    df = gpd.read_parquet(deploy_status_file)
    df = df.drop_duplicates(subset=[f'Name'])
    for year in [2020, 2024]:
        # need to download images by api query
        tiles_without_images_and_metadata = df[(df[f'has_s2_images_{year}'] == False)&(df[f'has_s2_metadata_{year}'].isna())]
        tiles_without_images_and_metadata['Name'].to_csv(f'~/data/GVS/Deploy/tiles_without_images_and_metadata_{year}.txt', index=False)
        # need to download images by saved metadata
        tiles_without_images = df[(df[f'has_s2_images_{year}'] == False)&(~df[f'has_s2_metadata_{year}'].isna())]
        tiles_without_images[['Name', f'meta_file_idx_{year}']].to_csv(f'~/data/GVS/Deploy/tiles_without_images_{year}.txt', index=False)
        # ready to predict tiles        
        unfinished_tiles = df[df[f'has_s2_images_{year}']==True]
        unfinished_tiles[['Name', f'meta_file_idx_{year}']].to_csv(f'~/data/GVS/Deploy/unfinished_tiles_{year}.txt', index=False)



@dataclass
class MyConfig:
    s2_grid_file: str = '~/data/GVS/S2_tiles_with_growing_months.parquet'
    flag_dir: str = '~/data/GVS/Deploy/translate_flags'
    prediction_dir: str = '~/data/GVS/Deploy/predictions'
    save_dir: str = '~/data/GVS/'
    
cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)

@hydra.main(config_name='config', version_base='1.2')
def main(cfg):
    check_prediction_status(cfg.s2_grid_file, cfg.flag_dir, cfg.prediction_dir, cfg.save_dir)
    # check_input_images_availability(cfg.s2_grid_file)
    get_unfinished_tiles(f'{cfg.save_dir}/deploy_status.parquet')

if __name__ == '__main__':
    main()