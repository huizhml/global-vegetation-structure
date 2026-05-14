import os
from osgeo import gdal
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
from rasterio.io import MemoryFile
from rasterio.warp import transform
from rasterio.transform import rowcol
import xarray as xr
import numpy as np
import dask.dataframe as dd
from rasterio.crs import CRS
from const import VSM_NODATA


def sample_locs_for_all_rhs(tile_id: str, stac_collection_dir: str = None, gedi_ref_df: pd.DataFrame = None, rh_size: int = 101, chunks: int = 1024):
    '''
    Sample locations from predictions for all RHS (bands) at once
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


class BiasCorrection:
    def __init__(
            self, year: int, root_save_dir: str = None, corrected_pred_dir: str = None,
            stac_collection_dir: str = None, min_n_points: int = 2000, debug: bool = False, diagnose_folder: str = None, **kwargs):
        self.root_save_dir = Path(root_save_dir).expanduser()
        self.corrected_pred_dir = Path(corrected_pred_dir).expanduser()
        self.stac_collection_dir = Path(stac_collection_dir).expanduser()
        self.min_n_points = min_n_points
        self.nodata = VSM_NODATA
        self.year = year
        self.debug = debug
        self.diagnose_folder = Path(diagnose_folder).expanduser()
        self.diagnose_folder.mkdir(parents=True, exist_ok=True)
        

    def get_correction_stats(self, tile_id: str, save_dir: str = None, ref_data_dir: str = None):
        '''
        Check the correction performance for the current year
        '''
        ref_data_dir = Path(ref_data_dir).expanduser()
        save_dir = Path(save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        file = save_dir / f'{tile_id}.npz'
        if file.exists() and not self.debug:
            print(f'{tile_id} correction stats already exists')
            return
        item = pystac.Item.from_file(str(self.stac_collection_dir / f'{tile_id}_{self.year}/{tile_id}_{self.year}.json'))
        old_pred_dir = item.assets['RH98_Q1'].href.replace('file://', '')
        old_pred_dir = Path(old_pred_dir).parent
        if 'home' not in str(Path.home()):
            old_pred_dir = Path.home() / 'flash' / Path(*old_pred_dir.parts[3:])
        
        gedi_ref_df = gpd.read_parquet(ref_data_dir / f'{tile_id}.parquet')
        gedi_ref_df = gedi_ref_df[gedi_ref_df['slope'] < 20]

        if len(gedi_ref_df) <= self.min_n_points:
            print(f'{tile_id} has less than {self.min_n_points} points')
            return
        # preds, nodata = sample_locs_for_all_rhs(tile_id, stac_collection_dir, gedi_ref_df, rh_size)
        lon = gedi_ref_df.geometry.x.values
        lat = gedi_ref_df.geometry.y.values
        rh_cols = [f'rh{i}' for i in range(101)]
        @dask.delayed
        def extract_preds(rh_idx: str):
            pred_fp = Path(f'{old_pred_dir}/RH{rh_idx}_Q1.tif').expanduser()
            with rasterio.open(pred_fp) as src:
                xs, ys = transform('EPSG:4326', src.crs, lon, lat)
                coords = list(zip(xs, ys))
                pred = list(rasterio.sample.sample_gen(src, coords))
                pred = np.concatenate(pred, axis=0).reshape(-1, 1)
                return pred
        
        tasks = []
        for rh_idx in range(101):
            tasks.append(extract_preds(rh_idx))
        with ProgressBar():
            preds = dask.compute(*tasks)
        preds = np.concatenate(preds, axis=1)
        mask = (preds == self.nodata).any(axis=1)
        preds = preds[~mask]
        preds = preds.astype(np.float32)
        gedi_ref_df = gedi_ref_df[~mask]
        gedi_ref = gedi_ref_df[rh_cols].values
        if len(gedi_ref) <= self.min_n_points:
            print(f'{tile_id} has less than {self.min_n_points} valid (not nodata) points')
            return
        
        # apply bias correction for all rhs
        residuals = gedi_ref*10 - preds
        median_bias = np.median(residuals, axis=0)
        p5 = np.percentile(residuals, 5, axis=0)
        p95 = np.percentile(residuals, 95, axis=0)
        valid = (residuals >= p5) & (residuals <= p95)
        residuals_trimmed_5_95 = np.where(valid, residuals, np.nan)
        mean_bias_trimmed_5_95 = np.nanmean(residuals_trimmed_5_95, axis=0)
        
        bias = residuals.mean(axis=0)  # >0 means under-estimation, <0 means over-estimation
        stats = {
            'n': len(preds),
            'bias': bias,
            'median_bias': median_bias,
            'mean_bias_trimmed_5_95': mean_bias_trimmed_5_95,
        }
        if np.isnan(bias).any():
            raise ValueError(f'{tile_id} has nan in bias')
        np.savez(f'{save_dir}/{tile_id}.npz', **stats)
        if self.debug:
            diagnose_folder = self.diagnose_folder / f'{tile_id}'
            diagnose_folder.mkdir(parents=True, exist_ok=True)
            pred_rh_cols = [f'RH{i}_Q1' for i in range(101)]
            preds_df = pd.DataFrame(preds/10, columns=pred_rh_cols)
            gedi_ref_df = gedi_ref_df.reset_index(drop=True)
            df = pd.concat([gedi_ref_df, preds_df], axis=1)
            df = gpd.GeoDataFrame(df, geometry=gedi_ref_df.geometry)
            df.to_file(diagnose_folder / f'{tile_id}_pred_and_ref.fgb', driver='FlatGeobuf')
            plt.figure(figsize=(10, 6))
            for rh_profile_name, rh_profile in zip(['GEDI', 'Ours'], [gedi_ref, preds/10]):
                plt.plot(range(101), rh_profile.mean(axis=0), label=rh_profile_name)
            plt.xlabel('Relative height (0-100)')
            plt.ylabel('Relative height (m)')
            plt.title(f'Average RH profile for {tile_id}, n={len(gedi_ref_df)}, average bias={bias.mean()/10:.2f} m')
            plt.legend()
            plt.grid(True)
            plt.savefig(diagnose_folder / f'average_rh_profile_GEDI_and_Ours.pdf')
            plt.close()
        return

    def plot_bias_distribution(self, rh_idxs: list[int], bias_dir: str, bias_col: str = 'bias', save_dir: str=None, average_across_rhs: bool = False):
        save_dir = Path(f'{save_dir}').expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        bias_dir = Path(f'{bias_dir}').expanduser()
        bias_files = list(bias_dir.glob('*.npz'))
        biases = []
        for bias_file in bias_files:
            bias = np.load(bias_file)[bias_col] 
            biases.append(bias)
        biases = np.stack(biases)
        biases = biases / 10  # covert to meters
        if average_across_rhs:
            biases = biases.mean(axis=1, keepdims=True)
            rh_idxs = [0]
        idx1,idx2 = np.where(biases >=1.42) # 95% of tiles have bias > 1.42 m
        gt1p42_tiles = [[bias_files[i].stem, biases[i]] for i in idx1]
        gt1p42_tiles = pd.DataFrame(gt1p42_tiles, columns=['tile', 'average_bias'])
        gt1p42_tiles.to_csv(save_dir / f'tiles_with_average_bias_gt1p42.csv')
        
        for rh_idx in rh_idxs:
            p50 = np.percentile(biases[:, rh_idx], 50)
            p75 = np.percentile(biases[:, rh_idx], 75)
            p95 = np.percentile(biases[:, rh_idx], 95)
            plt.figure(figsize=(10, 6))
            plt.hist(biases[:, rh_idx], bins=100)
            plt.axvline(p50, color='red', label=f'50th percentile: {p50:.2f} m')
            plt.axvline(p75, color='orange', label=f'75th percentile: {p75:.2f} m')
            plt.axvline(p95, color='green', label=f'95th percentile: {p95:.2f} m')
            plt.xlabel('Bias (m)')
            plt.ylabel('# of tiles')
            plt.legend()
            if average_across_rhs:
                plt.title(f'Bias distribution averaged across all RHS')
                plt.savefig(save_dir / f'bias_distribution_averaged_across_rhs.pdf')
            else:
                plt.title(f'Bias distribution for RH{rh_idx}')
                plt.savefig(save_dir / f'bias_distribution_{bias_col}_RH{rh_idx}.pdf')
            plt.close()


def create_vrt_for_rhs(file_paths: list[str]):
    with rasterio.open(file_paths[0]) as src0:
        meta = src0.meta.copy()
        width = meta['width']
        height = meta['height']
        transform_vals = src0.transform
        crs = src0.crs
        nodata_val = src0.nodata  # <--- Capture the nodata value here

    # 3. Construct the VRT XML string
    vrt_xml = f"""
    <VRTDataset rasterXSize="{width}" rasterYSize="{height}">
    <SRS dataAxisToSRSAxisMapping="2,1">{crs.wkt}</SRS>
    <GeoTransform>{transform_vals.c}, {transform_vals.a}, {transform_vals.b}, {transform_vals.f}, {transform_vals.d}, {transform_vals.e}</GeoTransform>
    """

    for i, path in enumerate(file_paths, start=1):
        # Add the <NoDataValue> tag inside each band
        # We use 'nodata_val' if it exists, otherwise we skip that tag
        nodata_xml = f"<NoDataValue>{nodata_val}</NoDataValue>" if nodata_val is not None else ""
        
        vrt_xml += f"""
    <VRTRasterBand dataType="Float32" band="{i}">
        {nodata_xml}
        <SimpleSource>
        <SourceFilename relativeToVRT="0">{path}</SourceFilename>
        <SourceBand>1</SourceBand>
        <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
        <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
        </SimpleSource>
    </VRTRasterBand>
    """
    vrt_xml += "</VRTDataset>"
    return vrt_xml       
        
    
def evaluate_bias_correction_against_sota_chm(
        gedi_chm_ours_dir: str = None, year: int = None,
        correction_stats_dir: str = None,
        save_dir: str = None,
        slope_lt20: bool = False,
        **kwargs):
    '''
    Evaluate the bias correction performance against SOTA CHM
    '''
    correction_stats_dir = Path(f'{correction_stats_dir}').expanduser()
    gedi_chm_ours_dir = Path(f'{gedi_chm_ours_dir}').expanduser()
    save_dir = Path(f'{save_dir}').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    tiles = [tile.stem for tile in gedi_chm_ours_dir.glob('*.parquet')]
    ours_rh_cols = [f'RH{i}_Q1_raw' for i in range(101)]
    gedi_rh_cols = [f'rh{i}' for i in range(101)]

    bias_cols = ['bias', 'median_bias', 'mean_bias_trimmed_5_95']
    residuals_ours_order = ['original', 'avg_bias'] +bias_cols 
    residuals_sota_order = ['UMD', 'ETH', 'UM',  'META']
    
    @dask.delayed
    def get_residuals_for_one_tile(tile_id: str):
        gedi_chm_ours_df = pd.read_parquet(gedi_chm_ours_dir / f'{tile_id}.parquet')
        gedi_chm_ours_df = gedi_chm_ours_df.dropna()
        if slope_lt20:
            gedi_chm_ours_df = gedi_chm_ours_df[gedi_chm_ours_df['slope']<20]
        original_residuals = gedi_chm_ours_df[ours_rh_cols].values - gedi_chm_ours_df[gedi_rh_cols].values
        if not (correction_stats_dir / f'{tile_id}.npz').exists():
            bias = {
                'bias': np.zeros(101),
                'median_bias': np.zeros(101),
                'mean_bias_trimmed_5_95': np.zeros(101)
                }
        else:
            bias = np.load(correction_stats_dir / f'{tile_id}.npz')
        residuals_ours = {} # np.zeros((len(bias_cols) + 2, len(gedi_chm_ours_df), 101)) # n*101*(len(bias_cols) + 2)
        residuals_sota = {} #np.zeros((4, len(gedi_chm_ours_df))) # 4*n
        residuals_ours['original'] = original_residuals # n*101
        residuals_ours['avg_bias'] = original_residuals + bias['bias'].mean() / 10 # n*101
        for i, bias_col in enumerate(bias_cols):
            residuals_ours[bias_col] = original_residuals + bias[bias_col] / 10 # n*101
        
        residuals_sota['UMD'] = gedi_chm_ours_df['RH95_UMD'].values - gedi_chm_ours_df['rh95'].values # n*1
        residuals_sota['ETH'] = gedi_chm_ours_df['RH98_ETH'].values - gedi_chm_ours_df['rh98'].values # n*1
        residuals_sota['UM'] = gedi_chm_ours_df['RH100_UM'].values - gedi_chm_ours_df['rh100'].values # n*1
        residuals_sota['META'] = gedi_chm_ours_df['RH95_META'].values - gedi_chm_ours_df['rh95'].values # n*1
        return np.concatenate([residuals_ours[col][None, :, :] for col in residuals_ours_order], axis=0), np.concatenate([residuals_sota[col][None, :] for col in residuals_sota_order], axis=0)

    tasks = []
    for tile_id in tiles:
        tasks.append(get_residuals_for_one_tile(tile_id))
    with ProgressBar():
        res = dask.compute(*tasks)
    residuals_ours = np.concatenate([r[0] for r in res], axis=1) # 5*n*101
    print('residuals_ours.shape:', residuals_ours.shape)
    me_ours = residuals_ours.mean(axis=1) # 5 * 101
    mae_ours = np.abs(residuals_ours).mean(axis=1) # 5 * 101
    rmse_ours = (residuals_ours**2).mean(axis=1)**0.5 # 5 * 101

    residuals_sota = np.concatenate([r[1] for r in res], axis=1) # 4*n
    me_sota = residuals_sota.mean(axis=1) # 4 * 1
    mae_sota = np.abs(residuals_sota).mean(axis=1) # 4 * 1
    rmse_sota = (residuals_sota**2).mean(axis=1)**0.5 # 4 * 1
    
    rh_cols = [f'RH{i}' for i in range(101)]
    me_ours_df = pd.DataFrame(me_ours, index=residuals_ours_order, columns=rh_cols)
    mae_ours_df = pd.DataFrame(mae_ours, index=residuals_ours_order, columns=rh_cols)
    rmse_ours_df = pd.DataFrame(rmse_ours, index=residuals_ours_order, columns=rh_cols)
    me_sota_df = pd.DataFrame(index=residuals_sota_order, columns=rh_cols)
    mae_sota_df = pd.DataFrame(index=residuals_sota_order, columns=rh_cols)
    rmse_sota_df = pd.DataFrame(index=residuals_sota_order, columns=rh_cols)
    
    zip_order = [('UMD', 'RH95'), ('ETH', 'RH98'), ('UM', 'RH100'), ('META', 'RH95')]
    for i, (product, rh) in enumerate([('UMD', 'RH95'), ('ETH', 'RH98'), ('UM', 'RH100'), ('META', 'RH95')]): # !! pay attention to the order!
        me_sota_df.loc[product, rh] = me_sota[i]
        mae_sota_df.loc[product, rh] = mae_sota[i]
        rmse_sota_df.loc[product, rh] = rmse_sota[i]
    
    
    # 1. Concatenate the 'Ours' metrics into one block
    ours_all = pd.concat(
        [me_ours_df, mae_ours_df, rmse_ours_df], 
        keys=['ME', 'MAE', 'RMSE'], 
        axis=0
    )

    # 2. Concatenate the 'SOTA' metrics into one block
    sota_all = pd.concat(
        [me_sota_df, mae_sota_df, rmse_sota_df], 
        keys=['ME', 'MAE', 'RMSE'], 
        axis=0
    )

    # 3. Concatenate both groups into the final DataFrame
    df_final = pd.concat(
        [sota_all, ours_all], 
        keys=['SOTA', 'Ours'], 
        axis=0
    )

    # 4. Assign clear names to the index levels for easy querying
    df_final.index.names = ['Model', 'Metric', 'Methods']
    # Moves 'Metric' from row index to column headers
    df_unstacked = df_final.unstack(level='Metric')
    print(df_unstacked)
    df_unstacked.to_csv(save_dir / f'correction_performance_val_tiles_{year}_all.csv')
    idx = pd.IndexSlice
    for metric in ['ME', 'MAE', 'RMSE']:
        df = df_unstacked.loc[('Ours'), idx[:, metric]]
        ax = df.T.plot(figsize=(10, 6), linewidth=2)
        plt.title(f"{metric} Curves for different bias calculation methods")
        plt.xlabel("Relative Height (0-100)")
        plt.ylabel(f"{metric}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Relative Height (0-100)")
        plt.savefig(save_dir / f'correction_performance_val_tiles_{year}_{metric}.pdf')
        plt.close()
    compare_with_sota_df = df_unstacked.loc[:, idx[['RH95', 'RH98', 'RH100'], :]].round(2)
    compare_with_sota_df = compare_with_sota_df.swaplevel(0, 1, axis=1)
    rh_labels = compare_with_sota_df.columns.get_level_values(1).unique()
    rh_order = sorted(rh_labels, key=lambda x: int(x.replace('RH', '')))
    # metrics_order = pd.unique(compare_with_sota_df.columns.get_level_values(0))
    metrics_order = ['RMSE', 'MAE', 'ME']
    new_columns = pd.MultiIndex.from_product(
        [metrics_order, rh_order], 
        names=compare_with_sota_df.columns.names
    )
    new_index = [('SOTA', var) for var in residuals_sota_order] +[('Ours', var) for var in residuals_ours_order]
    compare_with_sota_df = compare_with_sota_df.reindex(columns=new_columns, index=new_index)
    n_cols = compare_with_sota_df.shape[1]
    compare_with_sota_df.to_latex(save_dir / f'correction_performance_val_tiles_{year}_compared_with_sota_chm.tex', float_format=f"%.2f", na_rep='', column_format=f'l *{{{n_cols}}}{{S}}')

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
    products = [
        ('RH95_UMD', 'rh95', 1),
        ('RH95_META', 'rh95', 1),
        ('RH98_ETH', 'rh98', 1),
        ('RH100_UM', 'rh100', 1),
        ('RH95_raw', 'rh95', 10),
        ('RH98_raw', 'rh98', 10),
        ('RH100_raw', 'rh100', 10),
        ('RH95_linear_corrected', 'rh95', 10),
        ('RH98_linear_corrected', 'rh98', 10),
        ('RH100_linear_corrected', 'rh100', 10),
        ('RH95_bias_corrected', 'rh95', 10),
        ('RH98_bias_corrected', 'rh98', 10),
        ('RH100_bias_corrected', 'rh100', 10)]
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
        mask = pred != VSM_NODATA
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


def assign_s2_tile_name_to_gedi(gedi_fps: str, s2_fp: str, output_dir: str):
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
class BiasCorrectionConfig:
    root_save_dir: str = '~/data/gvs/deploy/correction'
    corrected_pred_dir: str = '~/data/gvs/deploy/predictions_corrected_2020'
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'
    tile_id: str = '20MRS'

    gedi_fps: str = '~/data/gvs/train_subsets/train*_filtered_v1.parquet'
    s2_fp: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    output_dir: str = '~/data/gvs/gedi_with_biome_slope_s2_tile_train_partitions/'
    sota_chm_dir: str = '~/data/gvs/GEDI_for_correction/partitions_with_sota_chm_2020'  # output dir
    sota_and_ours_dir: str = '~/data/gvs/deploy/correction_2020/partitions_with_sota_and_ours_2020'
    

    tiles_list_file: str = ''
    year: int = 2020

    # for get_tiles_covered_by_gedi
    gedi_table_index_file: str = f'~/data/gvs/GEDI_for_correction/l2a_table_index_{year}.parquet'

    # test config
    diagnose_folder: str = f'~/data/gvs/diagnostics/bias_correction/'
    mgrs_tiles: str = '20M,21M,20L,21L'
    s2_tiles: str = '20MRS,21MTM,20LRR,21LTL'
    debug: bool = False
    task: str = 'check_correction_performance'


cs = ConfigStore.instance()
cs.store(name='bias_correction', node=BiasCorrectionConfig)


@hydra.main(config_name='bias_correction', version_base="1.2")
def main(cfg):
    time_start = time.time()
    vsm_correction = BiasCorrection(**cfg)
    if cfg.task == 'get_correction_stats':
        save_dir = cfg.get('save_dir', f'~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/{cfg.year}/stats_by_tile')
        ref_data_dir = cfg.get('ref_data_dir', f'~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_correction/subset_4k/{cfg.year}/with_slope_col')
        vsm_correction.get_correction_stats(cfg.tile_id, save_dir=save_dir, ref_data_dir=ref_data_dir)
    elif cfg.task == 'plot_bias_distribution':
        bias_dir = cfg.get('bias_dir', f'~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/{cfg.year}/stats_by_tile')
        save_dir = cfg.get('save_dir', f'~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/{cfg.year}/figures')
        rh_idxs = cfg.get('rh_idxs', [10, 25, 50, 98])
        average_across_rhs = cfg.get('average_across_rhs', False)
        bias_col = cfg.get('bias_col', 'bias')
        vsm_correction.plot_bias_distribution(rh_idxs, bias_dir, bias_col, save_dir, average_across_rhs=average_across_rhs)
    elif cfg.task == 'pair_predictions_with_gedi_ref_data':
        gedi_chm_reference_dir = cfg.get('gedi_chm_reference_dir', f'~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_val/with_sota_chms/2020')
        save_dir = cfg.get('save_dir', f'~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_val/with_sota_chms_ours/{cfg.year}')
        stac_collection_dir = cfg.get('stac_collection_dir', f'~/data/gvs/products/gvsm_stac_catalog/vsm_local')
        pair_predictions_with_gedi_ref_data(gedi_chm_reference_dir=gedi_chm_reference_dir, year=cfg.year, save_dir=save_dir, stac_collection_dir=stac_collection_dir)
        
    elif cfg.task == 'evaluate_bias_correction_against_sota_chm':
        gedi_chm_ours_dir = cfg.get('gedi_chm_ours_dir', f'~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_val/with_sota_chms_ours/{cfg.year}')
        correction_stats_dir = cfg.get('correction_stats_dir', f'~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/{cfg.year}/stats_with_median_and_trimmed_5_95_by_tile')
        save_dir = cfg.get('save_dir', f'~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/{cfg.year}/figures/')
        evaluate_bias_correction_against_sota_chm(gedi_chm_ours_dir=gedi_chm_ours_dir, year=cfg.year, correction_stats_dir=correction_stats_dir, save_dir=save_dir)
    
    # if cfg.task == 'get_tiles_covered_by_gedi':
    #     vsm_correction.get_tiles_covered_by_gedi(cfg.s2_fp, cfg.gedi_table_index_file)
    # if cfg.task == 'check_correction_performance':
    #     check_correction_performance(**cfg)
    # elif cfg.task == 'correction_performance_distribution':
    #     correction_performance_distribution(cfg.correction_result_dir, cfg.year)
    # elif cfg.task == 'aggregate_correction_performance':
    #     aggregate_correction_performance(cfg.correction_result_dir, cfg.year)
    time_end = time.time()
    print(f'Time taken: {time_end - time_start} seconds')


if __name__ == '__main__':
    # from dask.distributed import Client, LocalCluster
    # from dask import config
    # cluster = LocalCluster(processes=False)
    # client = Client(cluster)  # timeout
    # print(client)
    main()
