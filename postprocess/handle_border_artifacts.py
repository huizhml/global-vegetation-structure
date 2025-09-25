import geopandas as gpd
import pandas as pd
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import glob
import h5py
import numpy as np
import rasterio
from rasterio.warp import transform
import time
import matplotlib.pyplot as plt


def get_gedi_from_h5(h5_fp: str, mgrs_tiles:str, s2_fp: str, s2_tiles:str):
    '''
    Get GEDI data from h5 file
    '''
    h5_fp = Path(h5_fp).expanduser()
    s2_fp = Path(s2_fp).expanduser()
    s2_df = gpd.read_parquet(s2_fp, columns=['Name', 'geometry'])
    h5_file = h5py.File(h5_fp, 'r')
    mgrs_tiles = mgrs_tiles.split(',')
    s2_tiles = s2_tiles.split(',')
    data = []
    for mgrs_tile in mgrs_tiles:
        mgrs_zone_data = h5_file[f'{mgrs_tile}/2020']
        for partition in mgrs_zone_data.keys():
            partition_data = mgrs_zone_data[partition]
            rhs = partition_data['rhs']
            latlon = partition_data['latlon']
            data.append(np.concatenate([rhs, latlon], axis=1))
    data = np.concatenate(data, axis=0)
    cols = [f'rh{i}' for i in range(101)] + ['lat', 'lon']
    df = pd.DataFrame(data, columns=cols)
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat, crs="EPSG:4326"))
    df = gpd.sjoin(df, s2_df, how='left', predicate='intersects')
    df = df.drop(columns=['index_right'])
    df = df[df['Name'].isin(s2_tiles)]
    return df

    
def correct_s2_tile_prediction(ref_data_dir: str, tile_id: str, save_dir: str, year: int):
    '''
    Correct S2 tile prediction
    '''
    # Get GEDI reference data
    ref_data_dir = Path(ref_data_dir).expanduser()
    ref_data_dir = ref_data_dir.with_stem(ref_data_dir.stem + f'_{year}')
    save_dir = Path(save_dir).expanduser()
    save_dir = save_dir.with_stem(save_dir.stem + f'_{year}')
    save_dir = save_dir / f'{tile_id}_cog'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    pred_dir = Path(f'~/data/GVS/Deploy/predictions_{year}/{tile_id}_cog').expanduser()
    gedi_ref = gpd.read_parquet(ref_data_dir / f'{tile_id}.parquet') # to check: no duplicates?

    lon = gedi_ref.geometry.x.values
    lat = gedi_ref.geometry.y.values
    
    def correct_one_rh(median_pred_fp: Path):
        rh_idx = median_pred_fp.stem.split('_')[0][2:]
        with rasterio.open(median_pred_fp) as src:
            xs, ys = transform('EPSG:4326', src.crs, lon, lat)
            coords = list(zip(xs, ys))
            rhs = list(rasterio.sample.sample_gen(src, coords))
            profile = src.profile
        pred = np.concatenate(rhs, axis=0)
        rhs = gedi_ref[f'rh{rh_idx}'].values
        pred = pred.astype(np.float64)
        mask = pred != 32767
        pred = pred[mask]
        rhs = rhs[mask]
        # pred += np.random.uniform(-0.5, 0.5, pred.shape)
        # a, b = get_scale_and_shift(pred, rhs*10) # in decimeters
        bias = (rhs*10 - pred).mean()
        a = 1
        b = bias
        plt.scatter(pred, rhs*10)
        xvalues = np.linspace(0, 500)
        yvalues = xvalues*a + b
        plt.plot(xvalues, yvalues)
        file = Path(f'~/data/GVS/Deploy/plots_bias_correction/{median_pred_fp.stem}_linear_fit.png').expanduser()
        plt.savefig(file)
        print(f'scale: {a}, shift: {b}')
        for q_idx in range(3):
            correct_fp = save_dir / f'RH{rh_idx}_Q{q_idx}.cog.tif'
            old_fp = pred_dir / f'RH{rh_idx}_Q{q_idx}.cog.tif'
            # profile.update(dtype=np.float32)
            with rasterio.open(correct_fp, 'w', **profile) as dst:
                with rasterio.open(old_fp) as src:
                    raw_pred = src.read(1)
                raw_pred = np.where(raw_pred == src.nodata, np.nan, raw_pred)
                correct_pred_float = a * raw_pred + b
                correct_pred = correct_pred_float.round()
                correct_pred = np.nan_to_num(correct_pred, nan=src.nodata)
                correct_pred = correct_pred.astype(np.int16)
                dst.write(correct_pred, indexes=1)
            print(f'saved to {correct_fp}')
    
    correct_one_rh(pred_dir / f'RH98_Q1.cog.tif')



def get_scale_and_shift(pred: np.ndarray, rhs: np.ndarray):
    '''
    Get scale and shift from S2 tile prediction and GEDI point
    '''
    A = np.vstack([pred, np.ones_like(pred)]).T
    a, b = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return a, b

def agg_gedi_to_s2(gedi_fps: str, s2_fp: str, output_dir: str):
    '''
    Add S2 tile name to the GEDI point
    But no time info.
    '''
    gedi_fps = Path(gedi_fps).expanduser()
    s2_fp = Path(s2_fp).expanduser()
    s2_df = gpd.read_parquet(s2_fp, columns=['Name', 'geometry'])
    gedi_fps = glob.glob(str(gedi_fps))
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for gedi_fp in gedi_fps:
        gedi_fp = Path(gedi_fp)
        gedi_df = pd.read_parquet(gedi_fp)
        gedi_df = gpd.GeoDataFrame(gedi_df, geometry=gpd.points_from_xy(gedi_df.lon, gedi_df.lat, crs="EPSG:4326"))
        gedi_df = gedi_df.set_crs(epsg=4326)
        print('Before sjoin, gedi_df has', len(gedi_df), 'rows')
        gedi_df = gpd.sjoin(gedi_df, s2_df, how='left', predicate='intersects')
        gedi_df = gedi_df.drop(columns=['index_right'])
        print('After sjoin, gedi_df has', len(gedi_df), 'rows')
        gedi_df.to_parquet(output_dir / f'{gedi_fp.stem}.parquet')
        print(f'saved to {output_dir / f"{gedi_fp.stem}.parquet"}')



@dataclass
class AggGediToS2:
    gedi_fps: str = '~/data/GVS/train_subsets/train*_filtered_v1.parquet'
    s2_fp: str = '~/data/GVS/S2_tiles_with_growing_months.parquet'
    output_dir: str = '~/data/GVS/train_gedi_agg_by_s2/'
    ref_data_dir: str = '~/data/GVS/GEDI/GVS_correction_set'
    tile_id: str = '20MRS'
    save_dir: str = '~/data/GVS/Deploy/predictions_corrected'
    year: int = 2020
    # test config
    mgrs_tiles: str = '20M,21M,20L,21L'
    s2_tiles: str = '20MRS,21MTM,20LRR,21LTL'
    

cs = ConfigStore.instance()
cs.store(name='agg_gedi_to_s2', node=AggGediToS2)


@hydra.main(config_name='agg_gedi_to_s2', version_base="1.2")
def main(cfg):
    # agg_gedi_to_s2(cfg.gedi_fps, cfg.s2_fp, cfg.output_dir)
    time_start = time.time()
    correct_s2_tile_prediction(cfg.ref_data_dir, cfg.tile_id, cfg.save_dir, cfg.year)
    time_end = time.time()
    print(f'Time taken: {time_end - time_start} seconds')

if __name__ == '__main__':
    main()