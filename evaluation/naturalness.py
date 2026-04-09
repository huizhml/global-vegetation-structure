from pathlib import Path
import pandas as pd
import geopandas as gpd
import dask
import scipy
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, precision_score
import numpy as np
import h5py
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


MAX_HEIGHT = 100.0
NODATA_IN = 32767

def _chunk_diversity(tile, bin_width=5):
    """
    Vectorized Shannon entropy for a batch of spatial chunks.

    Parameters
    ----------
    tile : ndarray, shape (n, 101, rows, cols)

    Returns
    -------
    entropy, enl1d, enl2d, cr : each ndarray, shape (n, rows, cols), float32
    """
    
    n_batch, n_bands, n_rows, n_cols = tile.shape
    n_pixels = n_batch * n_rows * n_cols
    n_bins = int(MAX_HEIGHT / bin_width)
    valid = np.isfinite(tile) & (~np.isnan(tile) & (tile > 0))
    nodata_mask = valid.sum(axis=1) == 0  # (n, rows, cols)

    tile_clean = np.where(valid, tile, 0.0)
    bin_idx = np.clip((tile_clean / bin_width).astype(np.int32), 0, n_bins - 1)
    bin_idx = np.where(valid, bin_idx, -1)

    # Reshape: merge batch, rows, cols into one pixel dimension
    bin_flat = bin_idx.reshape(n_batch, n_bands, n_rows * n_cols)       # (n, 101, rows*cols)
    bin_flat = bin_flat.transpose(1, 0, 2).reshape(n_bands, n_pixels)   # (101, n_pixels)

    pixel_indices = np.broadcast_to(
        np.arange(n_pixels)[np.newaxis, :], (n_bands, n_pixels)
    )

    hist = np.zeros((n_pixels, n_bins), dtype=np.float32)
    flat_valid = bin_flat != -1
    np.add.at(hist, (pixel_indices[flat_valid], bin_flat[flat_valid]), 1.0)

    total = hist.sum(axis=-1, keepdims=True)
    total = np.where(total == 0, 1, total)  # avoid division by zero
    p = hist / total
    log_p = np.where(p > 0, np.log(p), 0.0)

    entropy = -np.sum(p * log_p, axis=-1).astype(np.float32)
    enl1d = np.exp(entropy).astype(np.float32)
    enl2d = (1 / (p**2).sum(axis=-1)).astype(np.float32)

    entropy = entropy.reshape(n_batch, n_rows, n_cols)
    enl1d = enl1d.reshape(n_batch, n_rows, n_cols)
    enl2d = enl2d.reshape(n_batch, n_rows, n_cols)

    entropy[nodata_mask] = np.nan
    enl1d[nodata_mask] = np.nan
    enl2d[nodata_mask] = np.nan

    cr = (tile[:, 98] - tile[:, 25]) / (tile[:, 98] + 1e-6)
    cr[nodata_mask] = np.nan

    return entropy, enl1d, enl2d, cr


def prepare_loc_parqs(naturalness_csv: str, s2_grid_file: str, save_dir: str, **kwargs):
    '''
    Partition the naturalness csv file into Sentinel-2 tile based parquets. 
    Making it easier to sample VSM patches/points for each tile.
    Args:
        naturalness_csv: path to the naturalness csv file
        s2_grid_file: path to the S2 grid file
        save_dir: path to save the location parquets
        year: year of the naturalness dataset
    Returns:
        None
    '''
    naturalness_csv = Path(naturalness_csv).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(naturalness_csv)
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
    s2_grid = gpd.read_parquet(s2_grid_file, columns=['Name', 'geometry', 'covered_by_gedi'])
    s2_grid = s2_grid.to_crs(epsg=4326)
    df = gpd.sjoin(df, s2_grid, how='left', predicate='intersects')
    df = df.drop_duplicates(subset=['geometry'])
    df = df.drop(columns=['index_right'])
    tile_ids = df['Name'].unique()
    
    @dask.delayed
    def _save_tile(df_tile: gpd.GeoDataFrame, tile_id: str):
        df_tile.to_parquet(save_dir / f'{tile_id}.parquet')
        return
    
    tasks = []
    for tile_id in tile_ids:
        tasks.append(dask.delayed(_save_tile)(df[df['Name'] == tile_id], tile_id))
    dask.compute(*tasks)
    return

def _cal_vsm_patch_stats(h5_file: str, out_file: str, cols: list, **kwargs):
    '''
    Calculate the statistics (mean and std) of the VSM patches
    Args:
        vsm_patches_dir: path to the VSM patches directory
        save_dir: path to save the VSM patch statistics
    Returns:
        None
    '''
    if out_file.exists():
        return
    with h5py.File(h5_file, 'r') as f:
        vsm_patches = f['data'][:]
        naturalness = f['land_use_id'][:]
        rowids = f['rowid'][:]
        mask = np.isin(naturalness, [-1,1]) # unsure and no high resolution images
        vsm_patches = vsm_patches[~mask]
        naturalness = naturalness[~mask][:, np.newaxis]
        rowids = rowids[~mask]
    if rowids.size == 0:
        return
    fhd, enl1d, enl2d, cr = _chunk_diversity(vsm_patches, bin_width=5)
    diversity_indices = np.stack([fhd, enl1d, enl2d, cr], axis=1)
    stats_indices = scipy.stats.describe(diversity_indices, axis=(2,3), nan_policy='omit')
    avg_indices = stats_indices.mean # (n_points, 4)
    std_indices = stats_indices.variance**0.5 # (n_points, 4)
    res = scipy.stats.describe(vsm_patches, axis=(2,3), nan_policy='omit') # (n_points, n_rhs*n_q)
    avg = res.mean # (n_points, n_rhs*n_q)
    std = res.variance**0.5 # (n_points, n_rhs*n_q)
    
    data = np.concatenate([std, avg, naturalness, std_indices, avg_indices], axis=1)
    # add S2 patch statistics
    s2_patch_file = kwargs.get('s2_patch_file', None)
    if s2_patch_file is not None:
        s2_patch_file = Path(s2_patch_file).expanduser()
        with h5py.File(s2_patch_file, 'r') as f:
            all_rowids = f['rowid'][:]
            mask = np.isin(rowids, all_rowids)
            idx = np.where(np.isin(all_rowids, rowids))[0]
            s2_patch = f['s2'][idx, :12, 10:21, 10:21]
            slope = f['slope'][idx, 15:16, 15]
            avg_s2 = s2_patch.mean(axis=(2,3))
            std_s2 = s2_patch.std(axis=(2,3))
            
            # there may be locations where there is no S2 patch, so we need to fill with NaN
            full_avg_s2 = np.full((len(rowids), avg_s2.shape[1]), np.nan)
            full_std_s2 = np.full((len(rowids), std_s2.shape[1]), np.nan)
            full_slope = np.full((len(rowids), 1), np.nan)

            full_avg_s2[mask] = avg_s2
            full_std_s2[mask] = std_s2
            full_slope[mask] = slope   
            data = np.concatenate([data, full_std_s2, full_avg_s2, full_slope], axis=1)
    
    stats = pd.DataFrame(data, columns=cols, index=rowids) # (n_points, n_rhs*n_q + 1)
    stats.to_parquet(out_file)

def cal_vsm_patch_stats(vsm_patches_dir: str, save_dir: str, **kwargs):
    '''
    Calculate the statistics (mean and std) of the VSM patches
    Args:
        vsm_patches_dir: path to the VSM patches directory
        save_dir: path to save the VSM patch statistics
    Returns:
        None
    '''
    vsm_patches_dir = Path(vsm_patches_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    h5_files = vsm_patches_dir.glob('*.h5')
    std_cols = [f'std_RH{i}_Q1' for i in range(101)]
    avg_cols = [f'avg_RH{i}_Q1' for i in range(101)]
    if kwargs.get('s2_patch_file', None) is not None:
        s2_cols = [f'std_s2_band{i}' for i in range(12)] + [f'avg_s2_band{i}' for i in range(12)]
    else:
        s2_cols = []
    indices_cols = [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['fhd', 'enl1d', 'enl2d', 'cr']]
    cols = std_cols + avg_cols + ['land_use_id'] + indices_cols + s2_cols + ['slope']
    tasks = []
    for h5_file in h5_files:
        out_file = save_dir / f'{h5_file.name}.parquet'
        # _cal_vsm_patch_stats(h5_file, out_file, cols, **kwargs)
        tasks.append(dask.delayed(_cal_vsm_patch_stats)(h5_file, out_file, cols, **kwargs))
    with ProgressBar():
        dask.compute(*tasks)
    return


def logistic_regression(vsm_patch_stats_dir: str, vsm_patch_stats_dir_val: str, save_dir: str, **kwargs):
    '''
    Perform logistic regression on the VSM patches
    Args:
        vsm_patch_stats_dir: path to the VSM patch statistics directory
        save_dir: path to save the logistic regression model
    Returns:
        None
    '''
    vsm_patch_stats_dir = Path(vsm_patch_stats_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    files = list(vsm_patch_stats_dir.glob('*.parquet'))
    files_val = list(vsm_patch_stats_dir_val.glob('*.parquet'))
    ddf = dd.read_parquet(files[:1000])
    ddf = ddf.dropna(subset=['avg_s2_band0'])
    ddf = ddf.compute()
    valid = ~ddf['land_use_id'].isin([-1, 1])
    ddf = ddf[valid]
    
    ddf_val = dd.read_parquet(files_val[:1000])
    ddf_val = ddf_val.dropna(subset=['avg_s2_band0'])
    ddf_val = ddf_val.compute()
    valid_val = ~ddf_val['land_use_id'].isin([-1, 1])
    ddf_val = ddf_val[valid_val]
    
    y = ddf['land_use_id'].values.flatten().astype(int)
    y_val = ddf_val['land_use_id'].values.flatten().astype(int)
    classes = np.unique(y)
    
    groups = {
        'full_profile': [f'std_RH{i}_Q1' for i in range(101)] + [f'avg_RH{i}_Q1' for i in range(101)],
        's2_only': [f'std_s2_band{i}' for i in range(12)] + [f'avg_s2_band{i}' for i in range(12)],
        'key_rhs': [f'std_RH{i}_Q1' for i in [25, 50, 75, 90, 95, 98]] + [f'avg_RH{i}_Q1' for i in [25, 50, 75, 90, 95, 98]],
        'rh98': ['std_RH98_Q1', 'avg_RH98_Q1'],
        's2_rh98': [f'std_s2_band{i}' for i in range(12)] + [f'avg_s2_band{i}' for i in range(12)] + ['std_RH98_Q1', 'avg_RH98_Q1'],
        'fhd_enl1d_enl2d_cr': [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['fhd', 'enl1d', 'enl2d', 'cr']],
        'fhd': [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['fhd']],
        'enl2d': [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['enl2d']],
        'cr': [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['cr']],
    }
    # Store results for comparison plots
    all_metrics = {}
    all_cms = {}
    for name, cols in groups.items():
        print(f'{name}')
        x = ddf[cols].values
        x_val = ddf_val[cols].values
        x = sm.add_constant(x)
        model = sm.MNLogit(y, x)
        result = model.fit()
        print(f'{name}: AIC={result.aic:.0f}, BIC={result.bic:.0f}, Pseudo R²={result.prsquared:.4f}')
        y_pred = result.predict(x_val).argmax(axis=1)
        y_pred_classes = classes[y_pred]

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        print(pd.DataFrame(cm, index=classes, columns=classes))

        # Per-class precision, recall, f1
        print(classification_report(y_val, y_pred_classes))
        with open(save_dir / f'logistic_regression_summary_{name}.txt', 'w') as f:
            f.write(result.summary().as_text())
        # Store metrics
        all_metrics[name] = {
            'Macro F1': f1_score(y_val, y_pred_classes, average='macro'),
            'Weighted F1': f1_score(y_val, y_pred_classes, average='weighted'),
            'Accuracy': accuracy_score(y_val, y_pred_classes),
            'Macro Recall': recall_score(y_val, y_pred_classes, average='macro'),
            'Weighted Recall': recall_score(y_val, y_pred_classes, average='weighted'),
            'Macro Precision': precision_score(y_val, y_pred_classes, average='macro', zero_division=0),
            'Weighted Precision': precision_score(y_val, y_pred_classes, average='weighted', zero_division=0),
        }
        all_cms[name] = cm

    # --- Bar plot: metrics comparison ---
    metric_names = list(next(iter(all_metrics.values())).keys())
    model_names = list(all_metrics.keys())
    x_pos = np.arange(len(metric_names))
    n_models = len(model_names)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model_name in enumerate(model_names):
        values = [all_metrics[model_name][m] for m in metric_names]
        ax.bar(x_pos + i * width - (n_models - 1) * width / 2, values, width, label=model_name)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

# --- Confusion matrix heatmaps ---
    class_labels = ['Not forest', 'Natural', 'Managed', 'Planted', 'Plantation', 'Oil palm', 'Agroforestry']
    n_models = len(all_cms)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    for i, (name, cm) in enumerate(all_cms.items()):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_labels if i >= n_models - ncols else False,
                    yticklabels=class_labels if i % ncols == 0 else False,
                    ax=axes[i], vmin=0, vmax=1, cbar=False)
        axes[i].set_title(name)
        if i >= n_models - ncols:
            axes[i].set_xlabel('Predicted')
        if i % ncols == 0:
            axes[i].set_ylabel('Actual')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Single shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm_cbar = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 1))
    fig.colorbar(sm_cbar, cax=cbar_ax, label='Recall')

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(save_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
        

    
if __name__ == '__main__':
    vsm_patches_dir = '~/data/gvs/downstream_tasks/naturalness/vsm_patches_ps11_train/'
    vsm_patch_stats_dir = '~/data/gvs/downstream_tasks/naturalness/vsm_patch_stats_ps11_train/'
    save_dir = '~/data/gvs/downstream_tasks/naturalness/logistic_regression_ps11/'
    s2_patch_file = '~/data/gvs/downstream_tasks/naturalness/s2_2017_ps31.h5'
    # cal_vsm_patch_stats(vsm_patches_dir, vsm_patch_stats_dir, s2_patch_file=s2_patch_file)
    logistic_regression(vsm_patch_stats_dir, save_dir)