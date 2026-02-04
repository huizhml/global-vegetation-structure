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
import subprocess
import re
import pyarrow.parquet as pq

def check_input_images_availability(s2_grid_file: str):
    s2_grid_file = Path(s2_grid_file).expanduser()
    df = gpd.read_parquet(s2_grid_file)
    for year in [2020, 2024]:
        df[f'no_images_{year}'] = False
        input_dir = Path(f'~/data/gvs/deploy/inference_{year}.zarr').expanduser()
        for tile in df['Name']:
            if not (input_dir / tile).exists():
                df.loc[df['Name'] == tile, f'no_images_{year}'] = True
    df.to_parquet(f'~/data/gvs/deploy/s2_grid_input_images_availability.parquet')

def check_prediction_status(s2_grid_file: str, flag_dir: str, prediction_dir: str, save_dir: str):
    s2_grid_file = Path(s2_grid_file).expanduser()
    save_dir = Path(save_dir).expanduser()
    prediction_dir = Path(prediction_dir).expanduser()
    deploy_dir = Path(f'~/data/gvs/deploy').expanduser()
    
    df = gpd.read_parquet(s2_grid_file, columns=['Name', 'geometry', 'growing_months'])
    tiles = df['Name'].unique()
    for year in  [2020, 2024]:
        print('-'*100)
        print(f'Checking prediction status for {year}')
        flag_dir_ = Path(f'{flag_dir}_{year}').expanduser()
        input_s2_dir = Path(f'~/data/gvs/deploy/inference_{year}.zarr').expanduser()
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
            predicted = (flag_file_old.exists() or flag_file_new.exists())
            if year == 2020:
                cog_files = list((prediction_dir.with_name(f'predictions_{year}') / f'{tile}_cog').glob('*.cog.tif'))
                predicted = predicted and len(cog_files) == 303
            if predicted:
                df.loc[df['Name'] == tile, f'predicted_{year}'] = True
            else:
                print(f'{tile} not predicted')
                df.loc[df['Name'] == tile, f'predicted_{year}'] = False
                # check input images availability
                if (input_s2_dir / tile).exists():
                    with xr.open_zarr(input_s2_dir, group=tile, consolidated=False, chunks='auto') as ds:
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


def add_gedi_correction_count(s2_grid_file: str, geidi_correction_dir: str):
    s2_grid_file = Path(s2_grid_file).expanduser()
    geidi_correction_dir = Path(geidi_correction_dir).expanduser()
    df = gpd.read_parquet(s2_grid_file)
    s2_tiles_cover_gedi = np.loadtxt(Path.home() / 'data/gvs/deploy/tiles_covered_by_gedi.txt', dtype=str)
    df['covered_by_gedi'] = False
    df.loc[df['Name'].isin(s2_tiles_cover_gedi), 'covered_by_gedi'] = True
    for year in [2020, 2024]:
        df[f'gedi_correction_count_{year}'] = 0
        for tile in df['Name']:
            file_path = geidi_correction_dir / f'partitions_{year}_v1/{tile}.parquet'
            if file_path.exists():
                gedi_correction_count = pq.ParquetFile(file_path).metadata.num_rows
                df.loc[df['Name'] == tile, f'gedi_correction_count_{year}'] = gedi_correction_count
    df.to_parquet(s2_grid_file)
    import ipdb; ipdb.set_trace()
    

def check_prediction_validity(prediction_files: list[Path], min_size: int=209715200):
    '''
    Check if the prediction is valid
    - number of predictions is 303
    - smallest prediction size is >= min_size
    '''
    num_predictions = len(prediction_files)
    if prediction_files:
        smallest_prediction_size = min(f.stat().st_size for f in prediction_files) # 204800 bytes = 200 MB
    else:
        smallest_prediction_size = 0
    return num_predictions == 303 and smallest_prediction_size >= min_size # 204800 bytes = 200 MB

def update_deploy_status(deploy_status_file: str):
    '''
    Update column: predicted_{year}
    '''
    deploy_status_file = Path(deploy_status_file).expanduser()
    df = gpd.read_parquet(deploy_status_file)
    df = df.drop_duplicates(subset=[f'Name'])
    cog_dir = Path(f'~/data/gvs/deploy/predictions_2020').expanduser()
    gtiff_dir = Path(f'~/data/gvs/deploy/predictions_GTiff_2020').expanduser()
    df['type_2020'] = 'cog'
    predicted_wrong_tiles = pd.read_csv(f'~/data/gvs/deploy/predicted_not_ordered_tiles_2020.txt')
    duplicate_tiles = pd.read_csv(f'~/data/gvs/deploy/tiles_duplicated.txt')
    for tile in df['Name']:
        # check for 2020
        if tile in predicted_wrong_tiles['Name'].values:
            df.loc[df['Name'] == tile, f'type_2020'] = 'gtiff'
            df.loc[df['Name'] == tile, f'predicted_2020'] = True #TODO: assume new prediction are all valid, check number of predictions later
            # # check number of predictions
            # prediction_files = list((gtiff_dir / f'{tile}_GTiff').glob('*.tif'))
            # if check_prediction_validity(prediction_files):
            #     df.loc[df['Name'] == tile, f'predicted_2020'] = True
            # else:
            #     df.loc[df['Name'] == tile, f'predicted_2020'] = False
                
        else:
            prediction_files = list((cog_dir / f'{tile}_cog').glob('*.cog.tif'))
            if check_prediction_validity(prediction_files, min_size=0):
                df.loc[df['Name'] == tile, f'predicted_2020'] = True
            else:
                df.loc[df['Name'] == tile, f'predicted_2020'] = False
                # os.system(f'rm {Path.home()/ f'data/gvs/deploy/translate_flags_2020/{tile}_done'}')
                # os.system(f'rm {Path.home()/ f'data/gvs/deploy/inference_flags_2020/{tile}_best_images_done'}')
        # check for 2024
        zone_name = tile[:3].lower()
        num_predictions = int(subprocess.check_output(
            f"rclone ls lumi-465001846-private:{zone_name}-2024/predictions_GTiff_2024/{tile} | wc -l",
                shell=True
            ).decode("utf-8").split()[0])
        if num_predictions == 0:
            smallest_prediction_size = 0
        else:   
            smallest_prediction_size = int(subprocess.check_output(
                f"rclone ls lumi-465001846-private:{zone_name}-2024/predictions_GTiff_2024/{tile} | awk '{{print $1}}' | sort -n | head -n 1",
                shell=True
                ).decode("utf-8").split()[0])
        if num_predictions == 303 and smallest_prediction_size >= 209715200: # 209715200 bytes = 200 MB
            df.loc[df['Name'] == tile, f'predicted_2024'] = True
        else:
            df.loc[df['Name'] == tile, f'predicted_2024'] = False
            # os.system(f'rm {Path.home()/ f'data/gvs/deploy/flags_inference_2024/{tile}_best_images_done'}')
    df.to_parquet(deploy_status_file)
    unpredicted_20 = df[df['predicted_2020'] == False]['Name'].tolist()
    unpredicted_24 = df[df['predicted_2024'] == False]['Name'].tolist()
    unpredicted_20 = unpredicted_20 + duplicate_tiles['Name'].tolist()
    df_20 = pd.DataFrame(unpredicted_20, columns=['Name'])
    df_20['meta_file_idx_2020'] = pd.NA
    df_20.to_csv(f'~/data/gvs/deploy/slurm_job_files_2020/deploy_s2_items_2020_without_images.txt', index=False)
    df_24 = pd.DataFrame(unpredicted_24, columns=['Name'])
    df_24['meta_file_idx_2024'] = pd.NA
    df_24.to_csv(f'~/data/gvs/deploy/slurm_job_files_2024/deploy_s2_items_2024_without_images.txt', index=False)
    print(f'Number of unpredicted 2020 tiles: {len(unpredicted_20)}')
    print(f'Number of unpredicted 2024 tiles: {len(unpredicted_24)}')
    print(f'Unpredicted 2020 tiles: {unpredicted_20}')
    print(f'Unpredicted 2024 tiles: {unpredicted_24}')
    

def get_unfinished_tiles(deploy_status_file: str):
    deploy_status_file = Path(deploy_status_file).expanduser()
    df = gpd.read_parquet(deploy_status_file)
    df = df.drop_duplicates(subset=[f'Name'])
    for year in [2020, 2024]:
        # need to download images by api query
        tiles_without_images_and_metadata = df[(df[f'has_s2_images_{year}'] == False)&(df[f'has_s2_metadata_{year}'].isna())]
        tiles_without_images_and_metadata['Name'].to_csv(f'~/data/gvs/deploy/tiles_without_images_and_metadata_{year}.txt', index=False)
        # need to download images by saved metadata
        tiles_without_images = df[(df[f'has_s2_images_{year}'] == False)&(~df[f'has_s2_metadata_{year}'].isna())]
        tiles_without_images[['Name', f'meta_file_idx_{year}']].to_csv(f'~/data/gvs/deploy/tiles_without_images_{year}.txt', index=False)
        # ready to predict tiles        
        unfinished_tiles = df[df[f'has_s2_images_{year}']==True]
        unfinished_tiles[['Name', f'meta_file_idx_{year}']].to_csv(f'~/data/gvs/deploy/unfinished_tiles_{year}.txt', index=False)


def get_tiles_covered_by_gedi(s2_grid_file: str, save_dir: str = None):
    '''
    Get the tiles covered by GEDI

    Args:
        * s2_grid_file: the path to the S2 tiles file, should have columns: Name, geometry
        * gedi_table_index_file: the path to the GEDI table index file, should have columns: system:index, table_id, start_time, end_time, geometry

    Output:
        * tiles_covered_by_gedi.txt: a text file with the tile names in the GEDI range
        Each line is a tile name
    '''
    save_dir = Path(save_dir).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    s2_tiles = gpd.read_parquet(s2_grid_file)
    # no_images_tiles_24 = ['16XET',  '18XWT',  '20XNT',  '23XNN',  '24XWT',  '25XEN', '17XNN',  '19XEN',  '22XET',  '23XNP',  '24XWU',  '26XNT']
    for year in [2020, 2024]:
        gedi_tiles_file = save_dir / f'tiles_covered_by_gedi_{year}.txt'
        if gedi_tiles_file.exists():
            tiles_covered_by_gedi = np.loadtxt(gedi_tiles_file, dtype=str)
        else:
            gedi_table_index_file = Path(f'~/data/gvs/GEDI_for_correction/l2a_table_index_{year}.parquet').expanduser()
            gedi_table_index = gpd.read_parquet(gedi_table_index_file)
            tiles_covered_by_gedi = s2_tiles[s2_tiles.intersects(gedi_table_index['geometry'].union_all())]['Name'].unique()
            np.savetxt(gedi_tiles_file, tiles_covered_by_gedi, fmt='%s')
        predicted_in_gedi_range =[]
        if year == 2020:
            pred_cog_dir = Path(f'~/data/gvs/deploy/predictions_{year}').expanduser()
            pred_gtif_dir = Path(f'~/data/gvs/deploy/predictions_GTiff_{year}').expanduser()
        else:
            pred_gtif_dir = Path(f'~/data/gvs/deploy/predictions_GTiff_{year}').expanduser()
            pred_cog_dir = Path(f'~/data/gvs/deploy/predictions_{year}').expanduser()
        for tile in tiles_covered_by_gedi:
            if len(list(pred_cog_dir.glob(f'{tile}_cog'))) > 0 or len(list(pred_gtif_dir.glob(f'{tile}_GTiff'))) > 0:
                predicted_in_gedi_range.append(tile)
        np.savetxt(save_dir / f'tiles_predicted_in_gedi_range_{year}.txt', predicted_in_gedi_range, fmt='%s')


@dataclass
class MyConfig:
    s2_grid_file: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    flag_dir: str = '~/data/gvs/deploy/translate_flags'
    prediction_dir: str = '~/data/gvs/deploy/predictions'
    save_dir: str = '~/data/gvs/deploy'
    task: str = 'check_prediction_status'
    
cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)

@hydra.main(config_name='config', version_base='1.2')
def main(cfg):
    if cfg.task == 'check_prediction_status':
        check_prediction_status(cfg.s2_grid_file, cfg.flag_dir, cfg.prediction_dir, cfg.save_dir)
    elif cfg.task == 'update_deploy_status':
        update_deploy_status(f'{cfg.save_dir}/deploy_status.parquet')
    # check_input_images_availability(cfg.s2_grid_file)
    elif cfg.task == 'get_unfinished_tiles':
        get_unfinished_tiles(f'{cfg.save_dir}/deploy_status.parquet')
    elif cfg.task == 'add_gedi_correction_count':
        gedi_correction_dir = cfg.get('gedi_correction_dir', '~/data/gvs/GEDI_for_correction')
        add_gedi_correction_count(cfg.s2_grid_file, gedi_correction_dir)
    elif cfg.task == 'get_tiles_covered_by_gedi':
        get_tiles_covered_by_gedi(cfg.s2_grid_file, cfg.save_dir)

if __name__ == '__main__':
    main()