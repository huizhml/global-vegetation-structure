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
import time
import matplotlib.pyplot as plt
import dask
from dask.diagnostics import ProgressBar
import pystac
import stackstac
from rasterio.warp import transform
from rasterio.transform import rowcol
import xarray as xr
import numpy as np
import dask.dataframe as dd


def sample_locs_for_all_rhs(tile_id: str, stac_collection_dir:str=None, gedi_ref_df: pd.DataFrame=None, rh_size: int = 101, chunks: int = 1024):
    '''
    Sample locations for all RHS (bands) at once
    10s faster than the for loop, but takes more memory (OOM-kill when mem=64GB)
    '''
    item_json_path = Path(stac_collection_dir).expanduser() / tile_id / f"{tile_id}.json"
    item = pystac.Item.from_file(str(item_json_path))
    assets = [f"RH{i}" for i in range(rh_size)]
    # GEDI points -> target CRS
    lon = gedi_ref_df.geometry.x.values
    lat = gedi_ref_df.geometry.y.values
    crs_epsg = item.properties["proj:epsg"]
    xs, ys = transform("EPSG:4326", f"EPSG:{crs_epsg}", lon, lat)
    # IMPORTANT: compute (row,col) using rasterio.index() from an actual RH file
    # (this matches rasterio.sample pixel selection exactly)
    rh0_href = item.assets["RH0"].href.replace("file://", "")
    with rasterio.open(rh0_href) as src0:
        rows, cols = rowcol(src0.transform, xs, ys, op=np.floor)
        H, W = src0.height, src0.width
        nodata = src0.nodata
    # bounds check
    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    lin_idx = rows[valid] * W + cols[valid]
    
    # Build a single (band, y, x) stack, native grid, lazy
    da = stackstac.stack(
        [item],
        assets=assets,
        epsg=item.properties["proj:epsg"],
        resolution=None,
        dtype="int16",
        rescale=False,
        fill_value=np.int16(nodata),
        chunksize=chunks,
    ).isel(time=0)  # (band, y, x)



    # Gather via single-axis fancy indexing
    flat = da.stack(spatial=("y", "x"))         # (band, spatial)
    pts = xr.DataArray(lin_idx, dims=("points",))
    gathered = flat.isel(spatial=pts)           # (band, points)
    preds_valid = gathered.transpose("points", "band").data.compute().astype(np.int16)

    # Re-expand and fill nodata
    N = len(gedi_ref_df)
    preds = np.full((N, rh_size), nodata, dtype=np.int16)
    preds[valid] = preds_valid
    return preds, nodata

def check_correction_performance(
        ref_data_dir: str = None, year: int = None, prediction_dir: str = None, tiles_list_file: str = None,
        correction_result_dir: str = None, sota_chm_dir: str = None, sota_and_ours_dir: str = None,
        stac_collection_dir: str = None, **kwargs):
    '''
    Check the correction performance (RMSE, MAE and ME) with GEDI reference data and sota chm
    Split the correction data into 2 parts:
    - Part 1: used for correction
    - Part 2: used for evaluation
    '''
    correction_result_dir = Path(f'{correction_result_dir}/tile_stats').expanduser()
    correction_result_dir.mkdir(parents=True, exist_ok=True)
    ref_data_dir = Path(f'{ref_data_dir}').expanduser()
    stac_collection_dir = Path(f'{stac_collection_dir}').expanduser()
    sota_chm_dir = Path(f'{sota_chm_dir}').expanduser()
    sota_and_ours_dir = Path(sota_and_ours_dir).expanduser()
    sota_and_ours_dir.mkdir(parents=True, exist_ok=True)
    if tiles_list_file != '':
        with open(tiles_list_file, 'r') as f:
            all_tiles = f.read().splitlines()
    else:
        all_tiles = [tile.stem for tile in ref_data_dir.glob('*.parquet')]
    # all_tiles = ['32SNA']
    print(f'{len(all_tiles)} tiles have predictions')
    rh_size = 101

    @dask.delayed
    def test_correction_for_one_tile(tile_id: str, min_n_points: int = 200):
        if not (stac_collection_dir/tile_id).exists():
            print(f'{tile_id} not found in stac collection')
            return None, None, None
        gedi_ref_df = gpd.read_parquet(ref_data_dir / f'{tile_id}.parquet')

        if len(gedi_ref_df) <= min_n_points:
            print(f'{tile_id} has less than {min_n_points} points')
            return None, None, None
        # preds, nodata = sample_locs_for_all_rhs(tile_id, stac_collection_dir, gedi_ref_df, rh_size)
        lon = gedi_ref_df.geometry.x.values
        lat = gedi_ref_df.geometry.y.values
        preds = []
        for rh_idx in range(rh_size):
            pred_fp = Path(f'{prediction_dir}/{tile_id}_cog/RH{rh_idx}_Q1.cog.tif').expanduser()
            with rasterio.open(pred_fp) as src:
                xs, ys = transform('EPSG:4326', src.crs, lon, lat)
                coords = list(zip(xs, ys))
                rh = list(rasterio.sample.sample_gen(src, coords))
                pred = np.concatenate(rh, axis=0).reshape(-1, 1)
                preds.append(pred)  # (n_points, 1)
                nodata = src.nodata

        preds = np.concatenate(preds, axis=1)  # (n_points, rh_size)
        # mask nodata (non-vegetation) for pred and ref
        mask = (preds == nodata).any(axis=1)
        preds = preds[~mask]
        preds = preds.astype(np.float32)
        rh_cols = [f'rh{i}' for i in range(rh_size)]
        gedi_ref = gedi_ref_df[rh_cols].values
        gedi_ref = gedi_ref[~mask]
        if len(gedi_ref) <= min_n_points:
            print(f'{tile_id} has less than {min_n_points} valid (not nodata) points')
            return None, None, None

        # split data into correct and eval
        n = gedi_ref.shape[0]
        idx = np.random.permutation(gedi_ref.shape[0])
        correct_idx = idx[:int(n*0.5)]
        eval_idx = idx[int(n*0.5):]

        correct_true = gedi_ref[correct_idx]
        correct_pred = preds[correct_idx]
        eval_true = gedi_ref[eval_idx]
        eval_pred = preds[eval_idx]

        # apply linear fit for all rhs
        a, b = get_scale_and_shift(correct_pred, correct_true*10)  # in decimeters

        eval_pred_linear_corrected = (a * eval_pred + b).round()  # in decimeters
        residuals_linear_corrected = eval_pred_linear_corrected/10 - eval_true  # (n_points, rh_size) # In meters

        # apply bias correction for all rhs
        bias = (correct_true*10 - correct_pred).mean(axis=0)  # >0 means under-estimation, <0 means over-estimation
        eval_pred_bias_corrected = (eval_pred + bias).round()
        residuals_bias_corrected = eval_pred_bias_corrected/10 - eval_true  # (n_points, rh_size)

        # raw residuals
        residuals = eval_pred/10 - eval_true  # (n_points, rh_size) # In meters

        # add sota chm residuals
        sota_chm_df = gpd.read_parquet(sota_chm_dir / f'{tile_id}.parquet')
        assert (sota_chm_df['rh95'] == gedi_ref_df['rh95']).all(), 'index mismatch'

        sota_chm_df = sota_chm_df[~mask].iloc[eval_idx]  # drop points we don't have prediction for (masked area)
        # perhaps also drop locations where sota chm is nan
        
        
        sota_chm_df[['RH95_raw', 'RH98_raw', 'RH100_raw']] = eval_pred[:, [95, 98, 100]]
        sota_chm_df[['RH95_linear_corrected', 'RH98_linear_corrected',
                        'RH100_linear_corrected']] = eval_pred_linear_corrected[:, [95, 98, 100]]
        sota_chm_df[['RH95_bias_corrected', 'RH98_bias_corrected',
                        'RH100_bias_corrected']] = eval_pred_bias_corrected[:, [95, 98, 100]]
        sota_chm_df = sota_chm_df.dropna(subset=['RH95_UMD', 'RH95_META', 'RH98_ETH', 'RH100_UM'])
        if not sota_chm_df.empty:
            sota_chm_df.to_parquet(sota_and_ours_dir / f'{tile_id}.parquet')

        stats = {
            'n': len(eval_pred),
            'scale': a,
            'shift': b,
            'bias': bias,
            'rmse': np.sqrt(np.mean(residuals**2, axis=0)),
            'mae': np.mean(np.abs(residuals), axis=0),
            'me': np.mean(residuals, axis=0),
            'rmse_linear_corrected': np.sqrt(np.mean(residuals_linear_corrected**2, axis=0)),
            'mae_linear_corrected': np.mean(np.abs(residuals_linear_corrected), axis=0),
            'me_linear_corrected': np.mean(residuals_linear_corrected, axis=0),
            'rmse_bias_corrected': np.sqrt(np.mean(residuals_bias_corrected**2, axis=0)),
            'mae_bias_corrected': np.mean(np.abs(residuals_bias_corrected), axis=0),
            'me_bias_corrected': np.mean(residuals_bias_corrected, axis=0),
        }
        np.savez(f'{correction_result_dir}/correction_stats_{year}_{tile_id}.npz', **stats)
        if np.isnan(residuals_linear_corrected).any() or np.isnan(residuals_bias_corrected).any() or np.isnan(residuals).any():
            print(f'{tile_id} has nan in residuals')
        return residuals, residuals_linear_corrected, residuals_bias_corrected
    
    # all_tiles = ['48RWN']
    tasks = []
    for tile_id in all_tiles:
        tasks.append(test_correction_for_one_tile(tile_id))
    with ProgressBar():
        res = dask.compute(*tasks)
    if tiles_list_file is None:
        # no part files, all tiles are processed, aggregate the results
        residuals = []
        residuals_linear_corrected = []
        residuals_bias_corrected = []
        for r in res:
            if r[0] is None:
                continue
            residuals.append(r[0])
            residuals_linear_corrected.append(r[1])
            residuals_bias_corrected.append(r[2])

        rmse = (np.concatenate(residuals)**2).mean(axis=0)**0.5
        mae = np.abs(np.concatenate(residuals)).mean(axis=0)
        me = np.concatenate(residuals).mean(axis=0)
        rmse_linear_corrected = (np.concatenate(residuals_linear_corrected)**2).mean(axis=0)**0.5
        mae_linear_corrected = np.abs(np.concatenate(residuals_linear_corrected)).mean(axis=0)
        me_linear_corrected = np.concatenate(residuals_linear_corrected).mean(axis=0)
        rmse_bias_corrected = (np.concatenate(residuals_bias_corrected)**2).mean(axis=0)**0.5
        mae_bias_corrected = np.abs(np.concatenate(residuals_bias_corrected)).mean(axis=0)
        me_bias_corrected = np.concatenate(residuals_bias_corrected).mean(axis=0)
        data = [rmse, rmse_linear_corrected, rmse_bias_corrected, mae, mae_linear_corrected,
                mae_bias_corrected, me, me_linear_corrected, me_bias_corrected]
        cols = [f'rh{i}' for i in range(rh_size)]

        df = pd.DataFrame(
            data,
            index=['RMSE_raw', 'RMSE_linear_corrected', 'RMSE_bias_corrected', 'MAE_raw', 'MAE_linear_corrected',
                   'MAE_bias_corrected', 'ME_raw', 'ME_linear_corrected', 'ME_bias_corrected'],
            columns=cols)
        df['avg'] = df.mean(axis=1)
        df.to_csv(f'{correction_result_dir}/correction_performance_{year}_all_tiles.csv')
        print(df)


#
def aggregate_correction_performance(correction_result_dir: str, year: int):
    '''
    Aggregate correction performance
    Args:
        correction_result_dir: str, the directory of tile-level correction statistics
        year: int
    '''
    correction_result_dir = Path(f'{correction_result_dir}').expanduser()
    # compare with SOTA CHM
    df = dd.read_parquet(f'{correction_result_dir}/partitions_with_sota_and_ours_2020/*.parquet')
    df = df.dropna(subset=['RH95_UMD', 'RH95_META', 'RH98_ETH', 'RH100_UM'])
    df = df.compute()
    rmse, mae, me = [], [], []
    products = [('RH95_UMD', 'rh95', 1), ('RH95_META', 'rh95', 1), ('RH98_ETH', 'rh98', 1), ('RH100_UM', 'rh100', 1),
                ('RH95_raw', 'rh95', 10), ('RH98_raw', 'rh98', 10), ('RH100_raw', 'rh100', 10),
                ('RH95_linear_corrected', 'rh95', 10), ('RH98_linear_corrected', 'rh98', 10), ('RH100_linear_corrected', 'rh100', 10),
                ('RH95_bias_corrected', 'rh95', 10), ('RH98_bias_corrected', 'rh98', 10), ('RH100_bias_corrected', 'rh100', 10)]
    for product, name, scale in products:
        rmse.append(((df[product]/scale - df[name])**2).mean()**0.5)
        mae.append((df[product]/scale - df[name]).abs().mean())
        me.append((df[product]/scale - df[name]).mean())
    df = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'ME': me}, index=[name for name, _ in products])
    df.to_csv(f'{correction_result_dir}/correction_performance_{year}_compared_with_sota_chm.csv')
    print(df)
    
    # for all RHs
    n = 0
    sum_me = {
        'raw': np.zeros(101),
        'linear_corrected': np.zeros(101),
        'bias_corrected': np.zeros(101)
    }
    sum_mae = {
        'raw': np.zeros(101),
        'linear_corrected': np.zeros(101),
        'bias_corrected': np.zeros(101)
    }
    sum_mse = {
        'raw': np.zeros(101),
        'linear_corrected': np.zeros(101),
        'bias_corrected': np.zeros(101)
    }
    for file in correction_result_dir.glob('tile_stats/*.npz'):
        data = np.load(file)
        n += data[f'n']
        for postfix in ['', '_linear_corrected', '_bias_corrected']:
            group = 'raw' if postfix == '' else 'linear_corrected' if postfix == '_linear_corrected' else 'bias_corrected'
            sum_me[group] += data[f'me{postfix}'] * data[f'n']
            sum_mae[group] += data[f'mae{postfix}'] * data[f'n']
            sum_mse[group] += data[f'rmse{postfix}']**2 * data[f'n']
    # {'raw': (101,), 'linear_corrected': (101,), 'bias_corrected': (101,)}
    me = {group: sum_me[group] / n for group in sum_me}
    mae = {group: sum_mae[group] / n for group in sum_mae}
    rmse = {group: (sum_mse[group] / n)**0.5 for group in sum_mse}
    data = {}
    for matric_name, matric in zip(['ME', 'MAE', 'RMSE'], [me, mae, rmse]):
        for group in matric.keys():
            data[f'{matric_name}_{group}'] = matric[group]
    df = pd.DataFrame(data)
    df = df.T
    df['avg'] = df.mean(axis=1)
    df = df.rename(columns={i: f'rh{i}' for i in range(101)})
    df.to_csv(correction_result_dir.parent / f'correction_performance_{year}_all_tiles_aggregated.csv')
    # verify
    # df_old = pd.read_csv(correction_result_dir / f'correction_performance_{year}_all_tiles.csv')
    # df_old = df_old.set_index('Unnamed: 0')
    # df_old = df_old.sort_index()
    # df = df.sort_index()
    # df.index.name = 'Unnamed: 0'
    # pd.testing.assert_frame_equal(df, df_old) # NOTE: verified, no assertion


def correction_performance_distribution(correction_result_dir: str, year: int):
    '''
    Plot the distribution of correction performance for each tile
    Args:
        correction_result_dir: str, the root directory of correction results of the year
        year: int
    '''
    correction_result_dir = Path(f'{correction_result_dir}').expanduser()
    table = pd.read_csv(correction_result_dir / f'correction_performance_{year}_all_tiles_aggregated.csv')
    table = table.set_index('Unnamed: 0')
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    cols = [f'rh{i}' for i in range(101)]
    for i, metric in enumerate(['RMSE', 'MAE', 'ME']):
        for postfix in ['raw', 'linear_corrected', 'bias_corrected']:
            axs[i].plot(table.loc[f'{metric}_{postfix}', cols], label=f'{postfix}')
        axs[i].set_xticks(np.arange(0, 101, 10))
        axs[i].legend()
        axs[i].set_ylabel(f'{metric}')
    plt.xlabel('Relative Height (0-100)')
    # plt.title(f'Correction performance for {year}')
    plt.savefig(correction_result_dir / f'correction_performance_{year}_all_tiles.png')
    plt.close()

    stats_file = correction_result_dir / f'correction_performance_{year}_per_tile_distribution.csv'
    if not stats_file.exists():
        rows = []
        for file in correction_result_dir.glob('tile_stats/*.npz'):
            tile_id = file.stem.split('_')[-1]
            data = np.load(file)
            for postfix in ['', '_linear_corrected', '_bias_corrected']:
                group = 'raw' if postfix == '' else 'linear_corrected' if postfix == '_linear_corrected' else 'bias_corrected'
                rows.append({'tile_id': tile_id, 'group': group,
                            'RMSE_RH98': data[f'rmse{postfix}'][98],
                             'RMSE_all': data[f'rmse{postfix}'].mean(),
                             'MAE_RH98': data[f'mae{postfix}'][98],
                             'MAE_all': data[f'mae{postfix}'].mean(),
                             'ME_RH98': data[f'me{postfix}'][98],
                             'ME_all': data[f'me{postfix}'].mean()})
        df = pd.DataFrame(rows)
        df.to_csv(stats_file)
    else:
        df = pd.read_csv(stats_file)
    groups = df.groupby('group', sort=False)
    for metric in ['MAE', 'RMSE', 'ME']:
        for rh_idx in ['RH98', 'all']:
            plt.figure(figsize=(8, 6))
            y_positions = {'raw': 100, 'linear_corrected': 100,
                           'bias_corrected': 500}  # Different y positions for each group
            colors = {'raw': 'blue', 'linear_corrected': 'orange',
                      'bias_corrected': 'green'}  # Define colors for each group
            for name, group in groups:
                color = colors.get(name, 'black')  # Default to black if group name not in colors
                group[f'{metric}_{rh_idx}'].hist(bins=100, alpha=0.7, label=f'Group: {name}', color=color)
                max_value = group[f'{metric}_{rh_idx}'].max()
                y_position = y_positions.get(name, 5)  # Default to 5 if group name not in y_positions
                plt.annotate(f'Max: {max_value:.2f}',
                             xy=(max_value, 0),
                             xytext=(max_value, y_position),  # Offset x position for better readability
                             arrowprops=dict(facecolor=color, shrink=0.1),  # Use the same color as the histogram
                             fontsize=10, color='black')
            plt.title(
                f'Distribution of tile-level {metric} ({rh_idx}) for raw, linear_corrected, and bias_corrected predictions')
            plt.xlabel(f'{metric} ({rh_idx})')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'{correction_result_dir}/tile_level_performance_distribution_{metric}_{rh_idx}.png')


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

    pred_dir = Path(f'~/data/gvs/deploy/predictions_{year}/{tile_id}_cog').expanduser()
    # rh_idx = 98
    # share_id ='cTWnFfMN97' # evze6lxv0t (2020)
    # q_idx = 1 # median prediction
    # erda_link=f'https://sid.erda.dk/cgi-sid/ls.py?share_id={share_id}&current_dir={tile_id}&flags=f'
    # file_url = f'https://sid.erda.dk/share_redirect/{share_id}/{tile_id}_cog/'
    # pred_dir = Path(file_url)
    gedi_ref = gpd.read_parquet(ref_data_dir / f'{tile_id}.parquet')  # to check: no duplicates?

    lon = gedi_ref.geometry.x.values
    lat = gedi_ref.geometry.y.values

    @dask.delayed
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
        # pred += np.random.uniform(-1, 1, pred.shape)
        if pred.shape[0] == 0:
            print(f'No valid data for {median_pred_fp}')
            return
        a, b = get_scale_and_shift(pred, rhs*10)  # in decimeters
        # bias = (rhs*10 - pred).mean()
        # a = 1
        # b = bias
        plt.scatter(pred, rhs*10)
        xvalues = np.linspace(0, 500)
        yvalues = xvalues*a + b
        plt.plot(xvalues, yvalues)
        file = Path(f'~/data/gvs/deploy/plots_bias_correction/{median_pred_fp.stem}_linear_fit.png').expanduser()
        plt.savefig(file)
        print(f'scale: {a}, shift: {b}')
        # for q_idx in range(3):
        q_idx = 1
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

    tasks = []
    for rh_idx in range(98, 99):
        tasks.append(correct_one_rh(pred_dir / f'RH{rh_idx}_Q1.cog.tif'))
    with ProgressBar():
        dask.compute(*tasks)
    # correct_one_rh(pred_dir / f'RH98_Q1.cog.tif')


def get_scale_and_shift(pred: np.ndarray, rhs: np.ndarray, eps: float = 1e-6):
    '''
    Get scale and shift from S2 tile prediction and GEDI point
    '''
    x_mean = pred.mean(axis=0)
    y_mean = rhs.mean(axis=0)
    cov = ((pred - x_mean) * (rhs - y_mean)).mean(axis=0)
    var = ((pred - x_mean)**2).mean(axis=0)
    a = cov / (var + eps)
    b = y_mean - a * x_mean
    # A = np.vstack([pred, np.ones_like(pred)]).T
    # a, b = np.linalg.lstsq(A, rhs, rcond=None)[0]
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
    gedi_fps: str = '~/data/gvs/train_subsets/train*_filtered_v1.parquet'
    s2_fp: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    output_dir: str = '~/data/gvs/gedi_with_biome_slope_s2_tile_train_partitions/'
    ref_data_dir: str = '~/data/gvs/GEDI_for_correction/partitions_2020'
    sota_chm_dir: str = '~/data/gvs/GEDI_for_correction/partitions_with_sota_chm_2020' # output dir
    sota_and_ours_dir: str = '~/data/gvs/deploy/correction_2020/partitions_with_sota_and_ours_2020'
    tile_id: str = '20MRS'
    prediction_dir: str = '~/data/gvs/deploy/predictions_2020'
    stac_collection_dir: str = '~/data/gvs/deploy/gvsm_stac_catalog/vsm_2020'
    correction_result_dir: str = '~/data/gvs/deploy/correction_2020'
    corrected_pred_dir: str = '~/data/gvs/deploy/predictions_corrected_2020'
    tiles_list_file: str = ''
    year: int = 2020
    # test config
    mgrs_tiles: str = '20M,21M,20L,21L'
    s2_tiles: str = '20MRS,21MTM,20LRR,21LTL'
    task: str = 'check_correction_performance'


cs = ConfigStore.instance()
cs.store(name='agg_gedi_to_s2', node=AggGediToS2)


@hydra.main(config_name='agg_gedi_to_s2', version_base="1.2")
def main(cfg):
    # agg_gedi_to_s2(cfg.gedi_fps, cfg.s2_fp, cfg.output_dir)
    time_start = time.time()
    # correct_s2_tile_prediction(cfg.ref_data_dir, cfg.tile_id, cfg.corrected_pred_dir, cfg.year)
    # agg_correction_performance('~/data/gvs/deploy/correction', cfg.year)
    # check_nan_tiles(cfg.correction_result_dir, cfg.year)
    if cfg.task == 'check_correction_performance':
        check_correction_performance(**cfg)
    elif cfg.task == 'correction_performance_distribution':
        correction_performance_distribution(cfg.correction_result_dir, cfg.year)
    elif cfg.task == 'aggregate_correction_performance':
        aggregate_correction_performance(cfg.correction_result_dir, cfg.year)
    time_end = time.time()
    print(f'Time taken: {time_end - time_start} seconds')


if __name__ == '__main__':
    # from dask.distributed import Client, LocalCluster
    # from dask import config
    # cluster = LocalCluster(processes=False)
    # client = Client(cluster)  # timeout
    # print(client)
    main()
