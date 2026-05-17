import time
from pathlib import Path
import pandas as pd
import geopandas as gpd
import dask
import scipy
import statsmodels.api as sm
import zarr
from patsy import dmatrices
from shapely import wkb
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, precision_score
import numpy as np
import h5py
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings
from evaluation.utils import load_vsm_naturalness
from evaluation.on_diversity_indices import _chunk_diversity
from const import VSM_NODATA, KEY_RHS_EVAL
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)



LAND_USE_NAMES = {
    0: {'name':'No forest', 'short_name':'No Forest'},
    11: {'name':'Naturally regenerating forest without any signs of human activities', 'short_name':'Natural Forest'},
    20: {'name':'Naturally regenerating forest with signs of human activities', 'short_name':'Natural Forest\n(Secondary)'},
    31: {'name':'Planted forest', 'short_name':'Planted Forest'},
    32: {'name':'Short rotation plantations for timber', 'short_name':'Short Rotation\nPlantation'},
    40: {'name':'Oil palm plantations', 'short_name':'Oil Palm'},
    53: {'name':'Agroforestry', 'short_name':'Agroforestry'},
}
MODEL_NAMES = {
    'alpha_em': 'AlphaEarth',
    'rh98': 'RH98',
    'full_profile': 'Full Profile',
    'rh98_s2': 'RH98 + S2',
    'rh98_cr': 'RH98 + CR',
    'rh98_fhd': 'RH98 + FHD',
    'rh98_enl2d': 'RH98 + ENL2D',
    'key_rhs': 'RH98 + RH25, RH50, RH75, RH90, RH95',
    'rh98_fhd_enl1d_enl2d_cr': 'RH98 + FHD + ENL1D + ENL2D + CR',
    'rh98_center': 'RH98 (Center Pixel)',
    'full_profile_center': 'Full Profile (Center Pixel)',
    'full_profile_s2': 'Full Profile + S2',
}


# ---------------------------------------
#   Helper functions
# ---------------------------------------



def _load_patch_stats(patch_stats_dir: str, name_pattern: str='train', **kwargs):
    '''
    Load the VSM patch statistics. One big parquet file for one data/split
    Args:
        patch_stats_dir: path to the VSM patch statistics directory
    Returns:
        None
    '''
    patch_stats_dir = Path(patch_stats_dir).expanduser()
    patch_stats_file = list(patch_stats_dir.glob(f'*{name_pattern}.parquet'))[0]
    df = pd.read_parquet(patch_stats_file)
    df = df.dropna()
    valid = ~df['Land_use_ID'].isin([-1, 1])
    df = df[valid]
    
    y = df['Land_use_ID'].values.flatten().astype(int)
    return df, y

# ---------------------------------------
#   Plot functions
# ---------------------------------------
def plot_bars(summary_df: pd.DataFrame, per_class_df: pd.DataFrame, groups: tuple[str], metric: str='Recall', avg:str='macro', baseline_name: str='rh98', save_dir: str=None, show_improve: bool=True, include_alpha_em: bool=False, **kwargs):
    '''
    Plot the summary reports
    Args:
        summary_df: dataframe of summary reports
        per_class_df: dataframe of per-class reports
        groups: tuple of group names
        metric: metric to plot
        avg: 'macro' or 'weighted'
        save_dir: path to save the plots
        show_improve: if True, show delta relative to rh98 baseline
    '''
    assert avg in ['macro', 'weighted']
    mask = summary_df.Metric == f'{avg} avg'
    summary_df = summary_df.loc[mask,[metric]]
    per_class_df = per_class_df.loc[:, ['Class', metric]]
    if not include_alpha_em:
        summary_df = summary_df.drop(index='alpha_em')
        per_class_df = per_class_df.drop(index='alpha_em')
        
    # forest_types = list(per_class_df.Class.unique()) 
    forest_types = [c['short_name'] for c in LAND_USE_NAMES.values()]
    # model_names = ['rh98', 'rh98_cr', 'rh98_enl2d', 'rh98_fhd', 'rh98_fhd_enl1d_enl2d_cr', 'key_rhs', 'rh98_s2', 'full_profile'] #list(per_class_df.index.unique()) 
    print(groups)
    n_models = len(groups)

    # Build values dict: model_name -> list of values (per class + ALL)
    values_dict = {}
    for model_name in groups:
        values_dict[model_name] = (
            per_class_df.loc[model_name, metric].values.tolist()
            + [summary_df.loc[model_name, metric]]
        )

    # Sort class categories (except ALL) by Full Profile improvement
    if show_improve and 'full_profile' in values_dict and baseline_name in values_dict:
        baseline_vals = values_dict[baseline_name]
        fp_vals = values_dict['full_profile']
        non_all_indices = list(range(len(forest_types)))
        improvements = [max(fp_vals[ci] - baseline_vals[ci], 0) for ci in non_all_indices]
        sorted_pairs = sorted(zip(non_all_indices, improvements), key=lambda x: x[1])
        sorted_indices = [p[0] for p in sorted_pairs] + [len(forest_types)]  # append ALL
        # Reorder
        forest_types = [forest_types[i] for i in sorted_indices[:-1]]
        for m in values_dict:
            values_dict[m] = [values_dict[m][i] for i in sorted_indices]

    class_names = forest_types + ['ALL']
    baseline_vals = np.array(values_dict[baseline_name])

    # Plot setup
    bar_width = 0.5
    group_spacing = 0.5
    x_pos = np.arange(len(class_names)) * (n_models * bar_width + group_spacing)

    fig, ax = plt.subplots(figsize=(20, 12))

    # First pass: draw bars, collect annotations
    annotations = {j: [] for j in range(len(class_names))}

    for i, model_name in enumerate(groups):
        values = np.array(values_dict[model_name])
        offsets = x_pos + i * bar_width

        ax.bar(offsets, values, bar_width, label=MODEL_NAMES[model_name])

        if show_improve and model_name != baseline_name:
            improvement = np.maximum(values - baseline_vals, 0)
            base_part = np.minimum(values, baseline_vals)
            ax.bar(offsets, improvement, bar_width, bottom=base_part,
                   color='none', edgecolor='white', hatch='///',
                   linewidth=0.5)

            for j, imp in enumerate(improvement):
                if imp > 0:
                    annotations[j].append((offsets[j], values[j], f"+{imp:.2f}"))

    # Second pass: resolve label overlaps within each group
    if show_improve:
        min_gap = 0.04
        for j in range(len(class_names)):
            labels = annotations[j]
            if not labels:
                continue
            labels.sort(key=lambda t: t[1])
            y_positions = []
            for k, (lx, ly, txt) in enumerate(labels):
                desired_y = ly + 0.01
                if y_positions:
                    desired_y = max(desired_y, y_positions[-1] + min_gap)
                y_positions.append(desired_y)
            # for k, (lx, ly, txt) in enumerate(labels):
            #     label_y = y_positions[k]
            #     ax.text(lx, label_y, txt, ha='center', va='bottom', fontsize=10)
            #     if label_y - ly > 0.02:
            #         ax.plot([lx, lx], [ly, label_y],
            #                 color='gray', linewidth=0.5, alpha=0.5)

    ax.set_xticks(x_pos + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(class_names, fontsize=18)
    ax.set_ylabel(f'{metric}', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(0.66, 1), loc='upper left', fontsize=18)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'barplot_{metric}_{avg}_baseline_{baseline_name}.pdf', dpi=150, bbox_inches='tight')
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
    
    
def plot_confusion_matrix(all_cms: dict, save_dir: Path, include_alpha_em: bool = False, **kwargs):
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
    if not include_alpha_em:
        all_cms = {k: all_cms[k] for k in all_cms.files if k != 'alpha_em'}
        
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
    
def add_accuracy_from_cms(summary_df, per_class_df, all_cms, labels):
    summary_df = summary_df.copy()
    per_class_df = per_class_df.copy()
    summary_df['Accuracy'] = np.nan
    per_class_df['Accuracy'] = np.nan

    for model_name, cm in all_cms.items():
        cm = np.asarray(cm)
        total = cm.sum()
        if total == 0:
            continue
        tp = np.diag(cm).astype(float)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        tn = total - tp - fn - fp
        per_class_acc = (tp + tn) / total
        overall_acc = np.trace(cm) / total

        for cls, acc in zip(labels, per_class_acc):
            mask = (per_class_df.index == model_name) & (per_class_df['Class'] == cls)
            per_class_df.loc[mask, 'Accuracy'] = acc

        for avg_row in ('macro avg', 'weighted avg'):
            mask = (summary_df.index == model_name) & (summary_df['Metric'] == avg_row)
            summary_df.loc[mask, 'Accuracy'] = overall_acc

    return summary_df, per_class_df

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

# ----------------------------------------------------------------------------------------
#  Step 2. Calculate VSM patch statistics (mean and std, center pixel)
# ----------------------------------------------------------------------------------------
def _read_s2(patch_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(patch_file, 'r') as f:
        s2_patch = f['s2'][:, :12, 10:21, 10:21]
        slope = f['slope'][:, 15:16, 15]
        rowids = f['rowid'][:]
    return s2_patch, slope, rowids

def _read_alpha_em(patch_file: Path):
    with h5py.File(patch_file, 'r') as f:
        alpha_em = f['data'][:, :, 2:-2, 2:-2]
        rowids = f['rowid'][:]
    return alpha_em, rowids

def _read_vsm(patch_file: Path, ps: int=15):
    if patch_file.suffix == '.zarr':
        store = zarr.open(str(patch_file), mode='r')
        vsm = store['vsm_median'][:]
        vsm = vsm.astype(np.float32)
        vsm[vsm == VSM_NODATA] = np.nan
        rowids = store['rowid'][:]
    else:
        with h5py.File(patch_file, 'r') as f:
            vsm = f['vsm_median'][:]
            vsm = vsm.astype(np.float32)
            vsm[vsm == VSM_NODATA] = np.nan
            rowids = f['rowid'][:]
    if ps < 15:
        border = (15 - ps) // 2
        vsm = vsm[:, :, border:-border, border:-border]
    return vsm, rowids
    
def cal_patch_stats(patch_file: str, ref_csv_train: str, out_file: str, product: str='s2', **kwargs):
    '''
    Calculate the statistics (mean and std) of the patches (S2 patches, VSM patches or AlphaEarth embeddings)
    Args:
        patch_file: path to the patch file
        ref_csv_train: path to the reference csv file for training
        out_file: path to save the patch statistics
    Returns:
        None
    '''
    patch_file = Path(patch_file).expanduser()
    ref_csv_train = Path(ref_csv_train).expanduser()
    out_file = Path(out_file).expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ref_df_train = pd.read_csv(ref_csv_train)
    ref_csv_val = ref_csv_train.with_name(ref_csv_train.name.replace('train', 'val'))
    ref_df_val = pd.read_csv(ref_csv_val)
    
    rowids_train = ref_df_train['rowid'].unique()
    rowids_val = ref_df_val['rowid'].unique()
    rowids = np.concatenate([rowids_train, rowids_val])
    ref_df = pd.concat([ref_df_train, ref_df_val])
    ref_df = ref_df.astype({'rowid': 'int32', 'ID': 'int32', 'Land_use_ID': 'int8', 'flag': 'int8'})
    ref_df = ref_df.set_index('rowid')

    if product == 's2':
        data_patch, slope, rowids = _read_s2(patch_file)
        ref_df.loc[rowids, 'slope'] = slope    
    elif product == 'alpha_em':
        data_patch, rowids = _read_alpha_em(patch_file)
    elif product == 'vsm':
        data_patch, rowids = _read_vsm(patch_file)
        fhd, enl1d, enl2d, cr = _chunk_diversity(data_patch, bin_width=5)
        diversity_indices = np.stack([fhd, enl1d, enl2d, cr], axis=1)
        avg_indices = np.nanmean(diversity_indices, axis=(2,3), dtype=np.float32)
        std_indices = np.nanstd(diversity_indices, axis=(2,3), dtype=np.float32)
        indices_cols = ['fhd', 'enl1d', 'enl2d', 'cr']
        for col in indices_cols:
            ref_df.loc[rowids, f'avg_{col}'] = avg_indices[:, indices_cols.index(col)]
            ref_df.loc[rowids, f'std_{col}'] = std_indices[:, indices_cols.index(col)]
        
    else:
        raise ValueError(f'Invalid data type: {product}')
    
    avg = np.nanmean(data_patch, axis=(2,3), dtype=np.float32) # return nan if all values are nan
    std = np.nanstd(data_patch, axis=(2,3), dtype=np.float32)
    for b in range(avg.shape[1]):
        ref_df.loc[rowids, f'avg_{product}_band{b}'] = avg[:, b]
        ref_df.loc[rowids, f'std_{product}_band{b}'] = std[:, b]
    center = data_patch[:, :, data_patch.shape[2]//2, data_patch.shape[3]//2]
    for b in range(center.shape[1]):
        ref_df.loc[rowids, f'center_{product}_band{b}'] = center[:, b]
    
    for split, rowids_split in zip(['train', 'val'], [rowids_train, rowids_val]):
        ref_df_split = ref_df.loc[rowids_split]
        ref_df_split = gpd.GeoDataFrame(ref_df_split, geometry=gpd.points_from_xy(ref_df_split.Longitude, ref_df_split.Latitude), crs="EPSG:4326")
        file_name = out_file.with_stem(f'{out_file.stem.replace("train", split)}')
        ref_df_split.to_parquet(file_name)
        ref_df_split.to_file(file_name.with_suffix('.fgb'), driver='FlatGeobuf')
        
    
# ------------------------------
# When the patches are tiled, use dask for parallel computation
# ------------------------------
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
    vsm_patches, naturalness, rowids = load_vsm_naturalness(h5_file)
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
    stats = pd.DataFrame(data, columns=cols, index=rowids) # (n_points, n_rhs*n_q + 1)
    stats.to_parquet(out_file)


def cal_vsm_patch_stats(ref_by_tile_dir: str, vsm_patches_dir: str, save_dir: str, **kwargs):
    '''
    Calculate the statistics (mean and std) of the VSM patches (tiled)
    Args:
        vsm_patches_dir: path to the VSM patches directory
        save_dir: path to save the VSM patch statistics
    Returns:
        None
    '''
    ref_by_tile_dir = Path(ref_by_tile_dir).expanduser()
    vsm_patches_dir = Path(vsm_patches_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    std_cols = [f'std_vsm_band{i}' for i in range(101)]
    avg_cols = [f'avg_vsm_band{i}' for i in range(101)]
    indices_cols = [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['fhd', 'enl1d', 'enl2d', 'cr']]
    cols = std_cols + avg_cols + ['land_use_id'] + indices_cols
    h5_files = vsm_patches_dir.glob('*.h5')
    h5_files = list(h5_files)
    
    ref_by_tile_files = list(ref_by_tile_dir.glob('*.parquet'))
    if len(ref_by_tile_files) == 0:
        raise ValueError('ref_by_tile_dir does not contain any parquet files')
    if len(h5_files) == 1: # for case where we only have one big h5 file
        vsm_patches_file = str(h5_files[0])
    else:
        vsm_patches_file = str(vsm_patches_dir) + '/{tile_id}.h5'
        
    
    tasks = []
    for ref_by_tile_file in ref_by_tile_files:
        tile_id = ref_by_tile_file.stem
        out_file = save_dir / f'{tile_id}.parquet'
        vsm_patches_file = vsm_patches_file.format(tile_id=tile_id)
        _cal_vsm_patch_stats(vsm_patches_file, out_file, cols, **kwargs)
        tasks.append(dask.delayed(_cal_vsm_patch_stats)(vsm_patches_file, out_file, cols, **kwargs))
    with ProgressBar():
        dask.compute(*tasks)
    return

# ----------------------------------------------------------------------------------------
#  Step 3. Merge patch statistics from different data -- tools.parq_ops.merge_columns_from_files
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
#  Step 4. Run classification
# ----------------------------------------------------------------------------------------

def logistic_regression(x, x_val, y, **kwargs):
    '''
    Perform logistic regression on the VSM patches
    Args:
        vsm_patch_stats_dir: path to the VSM patch statistics directory
        save_dir: path to save the logistic regression model
    Returns:
        None
    '''
    x = sm.add_constant(x)
    x_val = sm.add_constant(x_val)
    model = sm.MNLogit(y, x)
    clf = model.fit()
    print(f'AIC={clf.aic:.0f}, BIC={clf.bic:.0f}, Pseudo R²={clf.prsquared:.4f}')
    return clf, x_val


def xgboost_classification(x, x_val, y, y_val, **kwargs):
    '''
    Perform XGBoost regression on the VSM patches
    Args:
        vsm_patch_stats_dir: path to the VSM patch statistics directory
        save_dir: path to save the XGBoost regression model
    Returns:
        clf: XGBoost classifier
        x_val: validation data
    '''
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)
    clf.fit(x, y, eval_set=[(x_val, y_val)])
    
    return clf, x_val

def run_classification(classifier: str, patch_stats_dir: str,  save_dir: str, **kwargs):
    '''
    Perform logistic regression on the VSM patches
    Args:
        patch_stats_dir: path to the VSM patch statistics directory, should contain *_train.parquet and *_val.parquet
        save_dir: path to save the logistic regression model
    Returns:
        None
    '''
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    ddf, y = _load_patch_stats(patch_stats_dir, 'train', **kwargs)
    ddf_val, y_val = _load_patch_stats(patch_stats_dir, 'val', **kwargs)
    ddf_ref_val = ddf_val[['Land_use_ID', 'slope', 'geometry', 'flag']]
    classes = np.unique(y)
    
    groups = {
        'alpha_em': [f'std_alpha_em_band{i}' for i in range(64)] + [f'avg_alpha_em_band{i}' for i in range(64)],
        'rh98': ['std_vsm_band98', 'avg_vsm_band98'],
        'full_profile': [f'std_vsm_band{i}' for i in range(101)] + [f'avg_vsm_band{i}' for i in range(101)],
        'rh98_s2': [f'std_s2_band{i}' for i in range(12)] + [f'avg_s2_band{i}' for i in range(12)] + ['std_vsm_band98', 'avg_vsm_band98'],
        # 's2_only': [f'std_s2_band{i}' for i in range(12)] + [f'avg_s2_band{i}' for i in range(12)],
        'key_rhs': [f'std_vsm_band{i}' for i in KEY_RHS_EVAL] + [f'avg_vsm_band{i}' for i in KEY_RHS_EVAL],
        'rh98_cr': ['std_vsm_band98', 'avg_vsm_band98'] + [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['cr']],
        'rh98_fhd': ['std_vsm_band98', 'avg_vsm_band98'] + [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['fhd']],
        'rh98_enl2d': ['std_vsm_band98', 'avg_vsm_band98'] +[f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['enl2d']],
        'rh98_fhd_enl1d_enl2d_cr': ['std_vsm_band98', 'avg_vsm_band98'] + [f'{metric}_{var}' for metric in ['std', 'avg'] for var in ['fhd', 'enl1d', 'enl2d', 'cr']],
        'rh98_center': ['center_vsm_band98'],
        'full_profile_center': [f'center_vsm_band{i}' for i in range(101)],
        'full_profile_s2': [f'std_s2_band{i}' for i in range(12)] + [f'avg_s2_band{i}' for i in range(12)] + [f'std_vsm_band{i}' for i in range(101)] + [f'avg_vsm_band{i}' for i in range(101)],
    }
    # Store results for comparison plots
    all_summary_reports = {}
    all_per_class_reports = {}
    all_cms = {}
    for name, cols in groups.items():
        print(f'{name}')
        x = ddf[cols].values
        x_val = ddf_val[cols].values
        if classifier == 'logistic_regression':
            clf, x_val = logistic_regression(x, x_val, y)
        elif classifier == 'xgboost':
            clf, x_val = xgboost_classification(x, x_val, y, y_val)
        y_pred = clf.predict(x_val).argmax(axis=1)
        y_pred_classes = classes[y_pred]
        # save predictions
        ddf_ref_val[name] = y_pred_classes.astype(np.uint8)
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

    # save predictions
    ddf_ref_val['geometry'] = ddf_ref_val['geometry'].apply(wkb.loads)
    ddf_ref_val = gpd.GeoDataFrame(ddf_ref_val, geometry='geometry', crs="EPSG:4326")
    ddf_ref_val = gpd.GeoDataFrame(ddf_ref_val, geometry=ddf_ref_val['geometry'])
    ddf_ref_val.to_parquet(save_dir / 'logistic_regression_predictions.parquet')
    ddf_ref_val.to_file(save_dir / 'logistic_regression_predictions.fgb', driver='FlatGeobuf')

    all_summary_df = pd.concat(all_summary_reports, names=['Model', 'Metric'])
    all_per_class_df = pd.concat(all_per_class_reports, names=['Model', 'Class'])
    all_summary_df.to_csv(save_dir / 'logistic_regression_summary_reports.csv')
    all_per_class_df.to_csv(save_dir / 'logistic_regression_per_class_reports.csv')
    np.savez(save_dir / 'logistic_regression_confusion_matrices.npz', **all_cms)
    return all_summary_df, all_per_class_df, all_cms

def plot_results(summary_file: str, per_class_file: str, all_cms_file: str, save_dir: str, groups: tuple[str]=None, baseline_name: str='rh98', **kwargs):
    '''
    Plot the results
    Args:
        summary_df_file: path to the summary dataframe
        per_class_df_file: path to the per-class dataframe
        save_dir: path to save the plots
    Returns:
    '''
    all_cms_file = Path(all_cms_file).expanduser()
    summary_file = Path(summary_file).expanduser()
    per_class_file = Path(per_class_file).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    all_cms = np.load(all_cms_file)
    summary_df = pd.read_csv(summary_file, index_col=0)
    per_class_df = pd.read_csv(per_class_file, index_col=0)
    labels = [v['short_name'] for v in LAND_USE_NAMES.values()]
    summary_df, per_class_df = add_accuracy_from_cms(summary_df, per_class_df, all_cms, labels)
    plot_bars(summary_df, per_class_df, groups=groups, metric='Recall', avg='macro', baseline_name=baseline_name, save_dir=save_dir)
    plot_bars(summary_df, per_class_df, groups=groups, metric='Precision', avg='macro', baseline_name=baseline_name, save_dir=save_dir)
    plot_bars(summary_df, per_class_df, groups=groups, metric='F1', avg='macro', baseline_name=baseline_name, save_dir=save_dir)
    plot_bars(summary_df, per_class_df, groups=groups, metric='Accuracy', avg='macro', baseline_name=baseline_name, save_dir=save_dir)
    plot_confusion_matrix(all_cms, save_dir)


def check_distribution(vsm_patch_stats_dir: str, save_dir: str, feature_cols: list, **kwargs):
    '''
    Check the distribution of the VSM patches
    Args:
        vsm_patch_stats_dir: path to the VSM patch statistics directory
        save_dir: path to save the distribution plots
    Returns:
        None
    '''
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    ddf, y = _load_patch_stats(vsm_patch_stats_dir, **kwargs)
    for col in feature_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        groups = ddf.groupby('land_use_id')[col]
        labels = sorted(ddf['land_use_id'].unique())
        data = [groups.get_group(l).dropna().values for l in labels]
        labels = [LAND_USE_NAMES.get(int(l), l)['short_name'] for l in labels]
        ax.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Forest Type')
        ax.set_ylabel(col)
        ax.set_title(f'Distribution of {col} by Forest Type')
        plt.tight_layout()
        plt.savefig(save_dir / f'violin_{col}.png', dpi=150, bbox_inches='tight')
        plt.close()

    return

if __name__ == '__main__':
    # from evaluation.utils import verify_batch_binning
    # verify_batch_binning()
    # split = 'train'
    # vsm_patches_dir = f'~/data/gvs/evaluation/downstream_tasks/naturalness/vsm_patches_ps11_{split}/'
    # vsm_patch_stats_dir = f'/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/vsm_s2_alpha_patch_stats_ps11_{split}/'
    # vsm_patch_stats_dir_val = f'/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/vsm_s2_alpha_patch_stats_ps11_val/'
    save_dir = f'/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/logistic_regression_ps11/'
    # s2_patch_file = '~/data/gvs/evaluation/downstream_tasks/naturalness/s2_2017_ps31.h5'
    # # cal_vsm_patch_stats(vsm_patches_dir, vsm_patch_stats_dir, s2_patch_file=s2_patch_file)
    # # all_summary_df, all_per_class_df, all_cms = run_classification(vsm_patch_stats_dir, vsm_patch_stats_dir_val, save_dir, debug=True)
    save_dir = Path(save_dir).expanduser()
    all_summary_df = pd.read_csv(save_dir / 'logistic_regression_summary_reports.csv', index_col=0)
    all_per_class_df = pd.read_csv(save_dir / 'logistic_regression_per_class_reports.csv', index_col=0)
    all_cms = np.load(save_dir / 'logistic_regression_confusion_matrices.npz')
    plot_bars(all_summary_df, all_per_class_df, metric='Recall', avg='macro', save_dir=save_dir)
    plot_bars(all_summary_df, all_per_class_df, metric='Precision', avg='macro', save_dir=save_dir)
    plot_bars(all_summary_df, all_per_class_df, metric='F1', avg='macro', save_dir=save_dir)
    # plot_improve_heatmap(all_summary_df, all_per_class_df, metric='Recall', avg='macro', save_dir=save_dir)
    # plot_improve_heatmap(all_summary_df, all_per_class_df, metric='Precision', avg='macro', save_dir=save_dir)
    # plot_improve_heatmap(all_summary_df, all_per_class_df, metric='F1', avg='macro', save_dir=save_dir)
    # plot_confusion_matrix(all_cms, save_dir)
    # feature_cols = ['std_vsm_band98', 'avg_vsm_band98'] + [f'std_fhd', 'avg_fhd'] + [f'std_enl1d', 'avg_enl1d'] + [f'std_enl2d', 'avg_enl2d'] + [f'std_cr', 'avg_cr']
    # check_distribution(vsm_patch_stats_dir, save_dir, feature_cols)