import xarray as xr
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import hydra
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(
    "ignore",
    r"The codec `vlen-utf8`.*",
    category=UserWarning,
    module=r"zarr\.codecs\.vlen_utf8",
)

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
            s2_geoparq_file = Path(f'~/data/gvs/deploy/deploy_s2_items_{year}_part{part_idx}.parquet').expanduser()
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
            tiles_unique_images.to_parquet(f'~/data/gvs/deploy/s2_deploy_items_{year}_part{part_idx}_unique_images.parquet')
            tiles_duplicated_images.append(s2_df[s2_df['s2:mgrs_tile'].isin(tiles_need_redownload)])
            total_cnt += cnt
        tiles_duplicated_images = pd.concat(tiles_duplicated_images)
        tiles_duplicated_images['growing_months'] = tiles_duplicated_images['growing_months'].apply(normalize_to_list)
        tiles_duplicated_images.drop_duplicates(subset=['id'], inplace=True)
        tiles_duplicated_images.to_parquet(f'~/data/gvs/deploy/s2_deploy_items_{year}_part{part_idx+1}_unique_images.parquet')
        print(f'Total duplicated tiles in year {year}: {total_cnt}')

def check_images_order():
    for year in [2020]:
        not_ordered_tiles = []
        slurm_config_dir = Path(f'~/data/gvs/deploy/slurm_job_files_{year}').expanduser()
        slurm_config_files = slurm_config_dir.glob(f'*_items_{year}_part*.parquet')
        for s2_geoparq_file in slurm_config_files:
            part_idx = int(s2_geoparq_file.stem.split('_')[-1][4:])
            print(f'Checking year {year} part {part_idx}')
            s2_df = gpd.read_parquet(s2_geoparq_file)
            unique_tiles = s2_df['s2:mgrs_tile'].unique()
            for tile_id in unique_tiles:
                done_flag = Path(f'~/data/gvs/deploy/inference_flags_{year}/{tile_id}_done').expanduser()
                if done_flag.exists(): # prediction with {tile_id}_done flag was with incorrect input images order
                    img_df = s2_df[s2_df['s2:mgrs_tile'] == tile_id]
                    if len(img_df) > 20:
                        img_df['s2:nodata_pixel_percentage'] = img_df['s2:nodata_pixel_percentage'].round()
                        ordered_img_df = img_df.sort_values(['s2:nodata_pixel_percentage', 'eo:cloud_cover']).iloc[:20]
                        should_use_imgs = ordered_img_df['id'].tolist()
                        used_imgs = img_df['id'].iloc[:20].tolist()
                        if set(should_use_imgs) != set(used_imgs):
                            not_ordered_tiles.append([tile_id, part_idx])
            print(f'Total not ordered tiles in year {year} part {part_idx}: {len(not_ordered_tiles)}')
        df = pd.DataFrame(not_ordered_tiles, columns=['Name', 'meta_file_idx_2020'])
        df.to_csv(Path(f'~/data/gvs/deploy/predicted_not_ordered_tiles_{year}.txt').expanduser(), index=False)
        
def check_n_images_per_tile():
    for year in [2020, 2024]:
        for part_idx in range(22):
            print(f'Checking year {year} part {part_idx}')
            s2_geoparq_file = Path(f'~/data/gvs/deploy/deploy_s2_items_{year}_part{part_idx}_unique_images.parquet').expanduser()
            s2_df = gpd.read_parquet(s2_geoparq_file)
            df = s2_df.groupby('s2:mgrs_tile').size().reset_index(name='n_images')
            import ipdb; ipdb.set_trace()
            unique_tiles = s2_df['s2:mgrs_tile'].unique()
            for tile_id in unique_tiles:
                tile_df = s2_df[s2_df['s2:mgrs_tile'] == tile_id]
                if len(tile_df) != 12:
                    print(f'Tile {tile_id} has {len(tile_df)} images')


def convert_rgb_to_geotiff(year, tile_id):
    ds = xr.open_zarr(f'~/data/gvs/deploy/inference_{year}.zarr', group=tile_id)
    rgb = ds.s2.sel(band=['B04', 'B03', 'B02'])
    save_dir = Path(f'~/data/gvs/debug/').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    for time in rgb.time:
        img = rgb.sel(time=time)
        date = np.datetime_as_string(time.data, unit='D')
        img.rio.to_raster(save_dir / f'{tile_id}_{date}_rgb.tif')
        

def check_image_statistics(tile_ids, year, plot_type='bar'):
    '''
    There're systematic shifts in predictions (RH98 checked), so we check the input images here.
    '''
    file_path = Path(f'~/data/gvs/deploy/check_input_images_{year}/stats_{'_'.join(tile_ids)}.csv').expanduser()
    if file_path.exists():
        stats = {'tile_id': [],'date': [], 'band': [],'mean': [], 'std': [], 'min': [], 'max': []}
        for tile_id, (col_offset, row_offset, patch_size) in tile_ids.items():
            ds = xr.open_zarr(f'~/data/gvs/deploy/inference_{year}.zarr', group=tile_id)
            for time in ds.s2.time:
                img = ds.s2.sel(time=time).isel(x=slice(col_offset, col_offset+patch_size), y=slice(row_offset, row_offset+patch_size)).compute()
                scl = img.sel(band=['SCL'])
                nodata_mask = scl.isel(band=0, drop=True) != 0
                valid_s2 = img.isel(band=slice(12)).where(nodata_mask, drop=True)
                date = np.datetime_as_string(time.data, unit='D')
                print(f'{tile_id}_{date}')
                if not (valid_s2.shape[1] == 0 and valid_s2.shape[2] == 0):
                    avg = valid_s2.mean(dim=('y', 'x'), skipna=True).data.tolist() # 12 
                    std = valid_s2.std(dim=('y', 'x'), skipna=True).data.tolist()
                    min = valid_s2.min(dim=('y', 'x'), skipna=True).data.tolist()
                    max = valid_s2.max(dim=('y', 'x'), skipna=True).data.tolist()
                    rgb = valid_s2.sel(band=['B04', 'B03', 'B02']).data
                    rgb = (rgb / 2000).clip(0, 1).transpose(1,2,0)
                    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)  # Increased height to add more space for figtext
                    ax.imshow(rgb)
                    ax.set_title(f'{tile_id}_{date}')
                    bands = valid_s2.band.data.tolist()
                    means = [f'{v:.0f}' for v in avg]
                    stds  = [f'{v:.0f}' for v in std]
                    colw = 6  
                    # first column is the row label
                    row_band = f"{'Band':<{5}}" + "".join([f"{b:>{colw}}" for b in bands])
                    row_mean = f"{'Mean':<{5}}" + "".join([f"{m:>{colw}}" for m in means])
                    row_std  = f"{'Std':<{5}}"  + "".join([f"{s:>{colw}}" for s in stds])
                    table_text = "\n".join([row_band, row_mean, row_std])
                    ax.set_xlabel(table_text,fontsize=12, ha='left', family='monospace')
                    ax.xaxis.set_label_coords(-0.06, -0.05) 
                    plt.savefig(file_path.parent / f'rgb_{tile_id}_{date}.png')
                    plt.close()
                    
                    
                else:
                    avg = [np.nan]*12
                    std = [np.nan]*12
                    min = [np.nan]*12
                    max = [np.nan]*12
                
                stats['mean'].extend(avg)
                stats['std'].extend(std)
                stats['min'].extend(min)
                stats['max'].extend(max)
                stats['date'].extend([date]*12)
                stats['tile_id'].extend([tile_id]*12)
                stats['band'].extend(valid_s2.band.data.tolist())
                
        df = pd.DataFrame(stats)
        df.to_csv(file_path)
    visualize_image_statistics(file_path, plot_type=plot_type)
    
def visualize_image_statistics(stats_fp, plot_type='bar'):
    import matplotlib.pyplot as plt
    stats_fp = Path(stats_fp).expanduser()
    df = pd.read_csv(stats_fp)
    for tile_id in df['tile_id'].unique():
        tile_df = df[df['tile_id'] == tile_id]
        if plot_type == 'bar':
            data = []
            for band in tile_df['band'].unique():
                data.append(tile_df[tile_df['band'] == band]['mean'].values)
            df_new = pd.DataFrame(data, index=tile_df['band'].unique(), columns=tile_df['date'].unique())
            df_new.plot.bar(figsize=(16, 8), width=0.8, title=f'Average reflectance of bands for {tile_id}', ylim=(0, 6000))
            fig_name = f'stats_bar_{tile_id}.png'
        elif plot_type == 'boxplot':
            tile_df.boxplot(column='mean', by='band', figsize=(16, 8))
            plt.title(f"Reflectance of bands for {tile_id}")
            plt.ylim(0, 6000)
            fig_name = f'stats_boxplot_{tile_id}.png'
        plt.tight_layout()
        plt.savefig(stats_fp.parent / fig_name)
        plt.close()
        

def check_intermediate_preds(tile_ids, year):
    import matplotlib.pyplot as plt
    out_file = Path(f'~/data/gvs/deploy/check_input_images_{year}/intermediate_preds_{'_'.join(tile_ids.keys())}.csv').expanduser()
    if out_file.exists():
        import rasterio
        stats = {'tile_id': [],'date': [], 'band': [],'mean': [], 'std': [], 'min': [], 'max': []}
        for tile_id, (col_offset, row_offset, patch_size) in tile_ids.items():
            tile_pred_dir = Path(f'~/data/gvs/deploy/predictions_GTiff_{year}_test/{tile_id}_GTiff').expanduser()
            tif_paths = list(tile_pred_dir.glob('*.tif'))
            window = rasterio.windows.Window(col_offset, row_offset, patch_size, patch_size)
            for tif_path in tif_paths:
                with rasterio.open(tif_path) as src:
                    data = src.read(1, window=window)
                    nodata = src.nodata
                data = data.astype(np.float32)
                data[data == nodata] = np.nan
                avg = np.nanmean(data)
                std = np.nanstd(data)
                min = np.nanmin(data)
                max = np.nanmax(data)
                if not np.isnan(data).all():
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(data, cmap='magma', vmin=0, vmax=500)
                    ax.set_title(f'{tile_id}_{tif_path.stem.split('_')[1]}')
                    ax.set_xlabel(f'avg: {avg:.2f}, std: {std:.2f}, min: {min:.2f}, max: {max:.2f}', ha='center', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(out_file.parent / f'RH98_{tile_id}_{tif_path.stem.split('_')[1]}.png')
                    plt.close()
                stats['tile_id'].append(tile_id)
                stats['date'].append(tif_path.stem.split('_')[1])
                stats['band'].append(f'RH98_Q1')
                stats['mean'].append(avg)
                stats['std'].append(std)
                stats['min'].append(min)
                stats['max'].append(max)
        df = pd.DataFrame(stats)
        df.to_csv(out_file)
    else:
        df = pd.read_csv(out_file)
        data = []
        for tile_id in df['tile_id'].unique():
            tile_df = df[df['tile_id'] == tile_id]
            data.append(tile_df['mean'].values)
        df_new = pd.DataFrame(data, index=df['tile_id'].unique())
        df_new.plot.bar(figsize=(16, 8))
        plt.tight_layout()
        plt.savefig(out_file.parent / f'intermediate_preds_{'_'.join(tile_ids.keys())}.png')
        plt.close()


@dataclass
class MyConfig:
    zarr_store_path: str = '~/data/gvs/deploy/inference_2024.zarr'
    year: int = 2024
    tile_id: str = '20XNR'
    output_dir: str = '~/data/gvs/deploy/check_input_images_2024'
    task: str = 'check_images_order'
    
cs = ConfigStore.instance()
cs.store(name="check_input_images", node=MyConfig)

@hydra.main(config_name='check_input_images', version_base="1.2")
def main(cfg: DictConfig):
    # check_duplicated_images()
    if cfg.task == 'check_images_order':
        check_images_order()
    elif cfg.task == 'check_image_statistics':
        check_image_statistics(cfg.tile_id, cfg.year)
    elif cfg.task == 'check_intermediate_preds':
        check_intermediate_preds(cfg.tile_id, cfg.year)
    # tile_configs = {
    #     '20LPP': (10000, 4, 900), 
    #     '20LQP': (4, 4, 900),
    #     '20LQQ': (4, 10006, 900),
    #     '20LPQ': (10000, 10006, 900)}
    # check_image_statistics(tile_configs, 2020)
    # check_intermediate_preds(tile_configs, 2020)


if __name__ == '__main__':
    main()