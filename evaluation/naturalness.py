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
LAND_USE_NAMES = {
    0: {'name':'No forest', 'short_name':'No Forest'},
    11: {'name':'Naturally regenerating forest without any signs of human activities', 'short_name':'Natural Forest'},
    20: {'name':'Naturally regenerating forest with signs of human activities', 'short_name':'Natural Forest (Secondary)'},
    31: {'name':'Planted forest', 'short_name':'Planted Forest'},
    32: {'name':'Short rotation plantations for timber', 'short_name':'Short Rotation Plantation'},
    40: {'name':'Oil palm plantations', 'short_name':'Oil Palm'},
    53: {'name':'Agroforestry', 'short_name':'Agroforestry'},
}
MODEL_NAMES = {
    'alpha_em': 'AlphaEarth',
    'rh98': 'RH98',
    'full_profile': 'Full Profile',
    'rh98_s2': 'RH98 + S2',
    'key_rhs': 'Key RHs',
    'rh98_cr': 'RH98 + CR',
    'rh98_fhd': 'RH98 + FHD',
    'rh98_enl2d': 'RH98 + ENL2D',
    'rh98_fhd_enl1d_enl2d_cr': 'RH98 + FHD + ENL1D + ENL2D + CR',
}


# ---------------------------------------
#   Helper functions
# ---------------------------------------

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

# ---------------------------------------
#   Plot functions
# ---------------------------------------
def plot_bars(summary_df: pd.DataFrame, per_class_df: pd.DataFrame, metric: str='Recall', avg:str='macro', save_dir: str=None, show_improve: bool=True, include_alpha_em: bool=False, **kwargs):
    '''
    Plot the summary reports
    Args:
        summary_df: dataframe of summary reports
        per_class_df: dataframe of per-class reports
        save_dir: path to save the plots
    Returns:
        None
    '''
    
    assert avg in ['macro', 'weighted']
    mask = summary_df.Metric == f'{avg} avg'
    summary_df = summary_df.loc[mask,[metric]]
    per_class_df = per_class_df.loc[:, ['Class', metric]]
    if not include_alpha_em:
        summary_df = summary_df.drop(index='alpha_em')
        per_class_df = per_class_df.drop(index='alpha_em')
        
    if show_improve:
        baseline = summary_df.loc['rh98']
        summary_df = summary_df - baseline
        summary_df = summary_df.drop(index='rh98')
        baseline_per_class = per_class_df.loc['rh98']
        baseline_map = baseline_per_class.set_index('Class')[metric]
        per_class_df = per_class_df.drop(index='rh98')
        per_class_df[metric] = per_class_df[metric].values - per_class_df['Class'].map(baseline_map).values
        
    class_names = list(per_class_df.Class.unique()) + ['ALL']
    model_names = list(per_class_df.index.unique()) 
    n_models = len(model_names)
    x_pos = np.arange(len(class_names))
    width = 0.8 / n_models
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model_name in enumerate(model_names[:-1]):
        values = per_class_df.loc[model_name, metric].values.tolist() + [summary_df.loc[model_name, metric]]
        ax.bar(x_pos + i * width - (n_models - 1) * width / 2, values, width, label=MODEL_NAMES[model_name])
    # ax.bar(x_pos[-1], summary_df[metric].values, width, label='Overall')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_ylabel(f'{metric}')
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / f'barplot_{metric}_{avg}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
def plot_improve_heatmap(summary_df: pd.DataFrame, per_class_df: pd.DataFrame, metric: str='Recall', avg: str='macro', save_dir: str=None, show_improve: bool=True, include_alpha_em: bool=False, **kwargs):
    '''
    Plot the improvement heatmap: rows = models, columns = land cover classes + ALL
    Args:
        summary_df: dataframe of summary reports
        per_class_df: dataframe of per-class reports
        metric: metric to plot
        avg: 'macro' or 'weighted'
        save_dir: path to save the plots
        show_improve: if True, show delta relative to rh98 baseline
    '''
    assert avg in ['macro', 'weighted']
    mask = summary_df.Metric == f'{avg} avg'
    summary_df = summary_df.loc[mask, [metric]]
    per_class_df = per_class_df.loc[:, ['Class', metric]]
    if not include_alpha_em:
        summary_df = summary_df.drop(index='alpha_em')
        per_class_df = per_class_df.drop(index='alpha_em')

    if show_improve:
        baseline = summary_df.loc['rh98']
        summary_df = summary_df - baseline
        summary_df = summary_df.drop(index='rh98')
        baseline_per_class = per_class_df.loc['rh98']
        baseline_map = baseline_per_class.set_index('Class')[metric]
        per_class_df = per_class_df.drop(index='rh98')
        per_class_df[metric] = per_class_df[metric].values - per_class_df['Class'].map(baseline_map).values

    class_names = list(per_class_df.Class.unique())
    model_names = list(per_class_df.index.unique())

    # Build the heatmap matrix (models x classes+ALL)
    heat_data = []
    for model_name in model_names:
        row = per_class_df.loc[model_name].set_index('Class')[metric].reindex(class_names).tolist()
        row.append(summary_df.loc[model_name, metric])  # ALL column
        heat_data.append(row)

    columns = class_names + ['ALL']
    heat_df = pd.DataFrame(heat_data, index=model_names, columns=columns)

    # Plot
    fig, ax = plt.subplots(figsize=(12, max(3, len(model_names) * 0.6 + 1)))
    vmax = heat_df.abs().values.max()
    cmap = 'RdBu_r' if show_improve else 'YlOrRd_r'
    sns.heatmap(
        heat_df,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        center=0 if show_improve else None,
        vmin=-vmax if show_improve else 0,
        vmax=vmax if show_improve else 1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': f'{metric} {"improvement" if show_improve else ""}'}
    )
    ax.set_ylabel('Model')
    ax.set_xlabel('')
    ax.set_title(f'{metric} {"improvement over RH98 baseline" if show_improve else ""} ({avg} avg)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels([MODEL_NAMES[name] for name in model_names], rotation=0, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / f'heatmap_{metric}_{avg}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    
def plot_confusion_matrix(all_cms: dict, save_dir: Path, **kwargs):
    '''
    Plot the confusion matrix
    Args:
        all_cms: dictionary of confusion matrices
        save_dir: path to save the confusion matrix plots
    Returns:
        None
    '''
    # --- Confusion matrix heatmaps ---
    class_labels = [l['short_name'] for l in LAND_USE_NAMES.values()]
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
    


# ----------------------------------------------------------------------------------------
#  Step 0. Prepare location parquets
# - Partitioned by tile for sampling VSM patches/points
# ----------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------
#  Step 1. Extract VSM patches/points
# ----------------------------------------------------------------------------------------

def _cal_s2_patch_stats(s2_patch_file: str, rowids: list):
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
        
    return full_std_s2, full_avg_s2, full_slope


def _cal_alpha_em_patch_stats(alpha_em_patch_file: str, rowids: list):
    alpha_em_patch_file = Path(alpha_em_patch_file).expanduser()
    with h5py.File(alpha_em_patch_file, 'r') as f:
        all_rowids = f['rowid'][:]
        mask = np.isin(rowids, all_rowids)
        idx = np.where(np.isin(all_rowids, rowids))[0]
        alpha_em = f['data'][idx, :, 2:-2, 2:-2]
        avg_alpha_em = alpha_em.mean(axis=(2,3))
        std_alpha_em = alpha_em.std(axis=(2,3))
        
        # there may be locations where there is no S2 patch, so we need to fill with NaN
        full_avg_alpha_em = np.full((len(rowids), avg_alpha_em.shape[1]), np.nan)
        full_std_alpha_em = np.full((len(rowids), std_alpha_em.shape[1]), np.nan)
        full_avg_alpha_em[mask] = avg_alpha_em
        full_std_alpha_em[mask] = std_alpha_em
    return full_std_alpha_em, full_avg_alpha_em

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
        std_s2, avg_s2, slope = _cal_s2_patch_stats(s2_patch_file, rowids)
        data = np.concatenate([data, std_s2, avg_s2, slope], axis=1)
        s2_dim = std_s2.shape[1]
        cols = cols + [f'std_s2_band{i}' for i in range(s2_dim)] + [f'avg_s2_band{i}' for i in range(s2_dim)] + ['slope']
    
    alpha_em_patch_file = kwargs.get('alpha_em_patch_file', None)
    if alpha_em_patch_file is not None:
        std_alpha_em, avg_alpha_em = _cal_alpha_em_patch_stats(alpha_em_patch_file, rowids)
        data = np.concatenate([data, std_alpha_em, avg_alpha_em], axis=1)
        alpha_em_dim = std_alpha_em.shape[1]
        cols = cols + [f'std_alpha_em_band{i}' for i in range(alpha_em_dim)] + [f'avg_alpha_em_band{i}' for i in range(alpha_em_dim)]
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
    indices_cols = [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['fhd', 'enl1d', 'enl2d', 'cr']]
    cols = std_cols + avg_cols + ['land_use_id'] + indices_cols
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
    vsm_patch_stats_dir_val = Path(vsm_patch_stats_dir_val).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    debug = kwargs.get('debug', False)
    if debug:
        files = list(vsm_patch_stats_dir.glob('*.parquet'))[:10]
        files_val = list(vsm_patch_stats_dir_val.glob('*.parquet'))[:10]
    else:
        files = list(vsm_patch_stats_dir.glob('*.parquet'))
        files_val = list(vsm_patch_stats_dir_val.glob('*.parquet'))
    ddf = dd.read_parquet(files)
    ddf = ddf.dropna(subset=['avg_s2_band0', 'std_alpha_em_band0'])
    ddf = ddf.compute()
    valid = ~ddf['land_use_id'].isin([-1, 1])
    ddf = ddf[valid]
    
    ddf_val = dd.read_parquet(files_val)
    ddf_val = ddf_val.dropna(subset=['avg_s2_band0'])
    ddf_val = ddf_val.compute()
    valid_val = ~ddf_val['land_use_id'].isin([-1, 1])
    ddf_val = ddf_val[valid_val]
    
    y = ddf['land_use_id'].values.flatten().astype(int)
    y_val = ddf_val['land_use_id'].values.flatten().astype(int)
    classes = np.unique(y)
    
    groups = {
        # 'alpha_em': [f'std_alpha_em_band{i}' for i in range(64)] + [f'avg_alpha_em_band{i}' for i in range(64)],
        'rh98': ['std_RH98_Q1', 'avg_RH98_Q1'],
        'full_profile': [f'std_RH{i}_Q1' for i in range(101)] + [f'avg_RH{i}_Q1' for i in range(101)],
        'rh98_s2': [f'std_s2_band{i}' for i in range(12)] + [f'avg_s2_band{i}' for i in range(12)] + ['std_RH98_Q1', 'avg_RH98_Q1'],
        # 's2_only': [f'std_s2_band{i}' for i in range(12)] + [f'avg_s2_band{i}' for i in range(12)],
        'key_rhs': [f'std_RH{i}_Q1' for i in [25, 50, 75, 90, 95, 98]] + [f'avg_RH{i}_Q1' for i in [25, 50, 75, 90, 95, 98]],
        'rh98_cr': ['std_RH98_Q1', 'avg_RH98_Q1'] + [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['cr']],
        'rh98_fhd': ['std_RH98_Q1', 'avg_RH98_Q1'] + [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['fhd']],
        'rh98_enl2d': ['std_RH98_Q1', 'avg_RH98_Q1'] +[f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['enl2d']],
        'rh98_fhd_enl1d_enl2d_cr': ['std_RH98_Q1', 'avg_RH98_Q1'] + [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['fhd', 'enl1d', 'enl2d', 'cr']],
    }
    # Store results for comparison plots
    all_summary_reports = {}
    all_per_class_reports = {}
    all_cms = {}
    for name, cols in groups.items():
        print(f'{name}')
        x = ddf[cols].values
        x_val = ddf_val[cols].values
        x = sm.add_constant(x)
        x_val = sm.add_constant(x_val)
        model = sm.MNLogit(y, x)
        result = model.fit()
        print(f'{name}: AIC={result.aic:.0f}, BIC={result.bic:.0f}, Pseudo R²={result.prsquared:.4f}')
        y_pred = result.predict(x_val).argmax(axis=1)
        y_pred_classes = classes[y_pred]

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        all_cms[name] = cm
        
        # Per-class precision, recall, f1
        per_class_report = classification_report(y_val, y_pred_classes, output_dict=True)
        accuracy = per_class_report['accuracy']
        per_class_report = pd.DataFrame(per_class_report).T
        df = per_class_report.rename(columns={'f1-score': 'F1', 'support': 'Support'})
        df.columns = df.columns.str.title()
        df['Support'] = df['Support'].astype(int)

        # Split into per-class and overall summary
        summary_keys = ['macro avg', 'weighted avg']
        df_summary = df.loc[summary_keys]
        df_summary.loc['macro avg', 'accuracy'] = accuracy
        all_summary_reports[name] = df_summary
        
        # Rename class indices to human-readable names
        df_per_class = df.drop(index=summary_keys + ['accuracy'])
        df_per_class.index = [LAND_USE_NAMES.get(int(idx), idx)['short_name'] for idx in df_per_class.index]
        
        all_per_class_reports[name] = df_per_class

    all_summary_df = pd.concat(all_summary_reports, names=['Model', 'Metric'])
    all_per_class_df = pd.concat(all_per_class_reports, names=['Model', 'Class'])
    all_summary_df.to_csv(save_dir / 'logistic_regression_summary_reports.csv')
    all_per_class_df.to_csv(save_dir / 'logistic_regression_per_class_reports.csv')
    np.savez(save_dir / 'logistic_regression_confusion_matrices.npz', **all_cms)
    return all_summary_df, all_per_class_df, all_cms


        

    
if __name__ == '__main__':
    split = 'train'
    vsm_patches_dir = f'~/data/gvs/downstream_tasks/naturalness/vsm_patches_ps11_{split}/'
    vsm_patch_stats_dir = f'/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/vsm_s2_alpha_patch_stats_ps11_{split}/'
    vsm_patch_stats_dir_val = f'/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/vsm_s2_alpha_patch_stats_ps11_val/'
    save_dir = f'/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/logistic_regression_ps11/'
    s2_patch_file = '~/data/gvs/downstream_tasks/naturalness/s2_2017_ps31.h5'
    # cal_vsm_patch_stats(vsm_patches_dir, vsm_patch_stats_dir, s2_patch_file=s2_patch_file)
    # all_summary_df, all_per_class_df, all_cms = logistic_regression(vsm_patch_stats_dir, vsm_patch_stats_dir_val, save_dir, debug=False)
    save_dir = Path(save_dir).expanduser()
    all_summary_df = pd.read_csv(save_dir / 'logistic_regression_summary_reports.csv', index_col=0)
    all_per_class_df = pd.read_csv(save_dir / 'logistic_regression_per_class_reports.csv', index_col=0)
    all_cms = np.load(save_dir / 'logistic_regression_confusion_matrices.npz')
    plot_bars(all_summary_df, all_per_class_df, metric='Recall', avg='macro', save_dir=save_dir)
    plot_bars(all_summary_df, all_per_class_df, metric='Precision', avg='macro', save_dir=save_dir)
    plot_bars(all_summary_df, all_per_class_df, metric='F1', avg='macro', save_dir=save_dir)
    plot_improve_heatmap(all_summary_df, all_per_class_df, metric='Recall', avg='macro', save_dir=save_dir)
    plot_improve_heatmap(all_summary_df, all_per_class_df, metric='Precision', avg='macro', save_dir=save_dir)
    plot_improve_heatmap(all_summary_df, all_per_class_df, metric='F1', avg='macro', save_dir=save_dir)
    plot_confusion_matrix(all_cms, save_dir)