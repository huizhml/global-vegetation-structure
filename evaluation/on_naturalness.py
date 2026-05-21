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
from const import VSM_NODATA, KEY_RHS_EVAL, FONT_SIZES, FIGURE_SIZES, set_plot_fonts, fewer_ticks

set_plot_fonts()
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
    'alpha_em':                {'name': 'AlphaEarth',                          'run_id': None},
    'rh98':                    {'name': 'RH98',                                'run_id': 'swdxk7ep'},
    'full_profile':            {'name': 'Full Profile',                        'run_id': 'rjl8zl61'},
    'rh98_s2':                 {'name': 'RH98 + S2',                           'run_id': 'dwl36g52'},
    'rh98_cr':                 {'name': 'RH98 + CR',                           'run_id': '9fgg5og0'},
    'rh98_fhd':                {'name': 'RH98 + FHD',                          'run_id': '0ibhc4s1'},
    'rh98_enl2d':              {'name': 'RH98 + ENL2D',                        'run_id': 'tq0ke6hj'},
    'key_rhs':                 {'name': 'RH98 + RH25, RH50, RH75, RH90, RH95', 'run_id': '5pf9gyse'},
    'rh98_fhd_enl1d_enl2d_cr': {'name': 'RH98 + FHD + ENL1D + ENL2D + CR',     'run_id': '5kk3oo0i'},
    'rh98_center':             {'name': 'RH98 (w/o spatial context)',          'run_id': None},
    'full_profile_center':     {'name': 'Full Profile (w/o spatial context)',  'run_id': None},
    'full_profile_s2':         {'name': 'Full Profile + S2',                   'run_id': 'e1y9vkez'},
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
    if not include_alpha_em and 'alpha_em' in summary_df.columns:
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

    # Manually reorder forest types by Land_use_ID, with ALL placed first
    manual_order_ids = [11, 20, 53, 31, 32, 40, 0]
    land_use_ids = list(LAND_USE_NAMES.keys())
    manual_indices = [land_use_ids.index(_id) for _id in manual_order_ids]
    all_index = len(forest_types)  # ALL is appended after the per-class values
    reorder_indices = [all_index] + manual_indices  # ALL first, then forest types
    forest_types = [forest_types[i] for i in manual_indices]
    for m in values_dict:
        values_dict[m] = [values_dict[m][i] for i in reorder_indices]

    class_names = ['ALL'] + forest_types
    baseline_vals = np.array(values_dict[baseline_name])

    # Plot setup
    bar_width = 0.5
    group_spacing = 0.5
    all_gap = n_models * bar_width  # extra space between ALL and the forest-type groups
    x_pos = np.arange(len(class_names)) * (n_models * bar_width + group_spacing)
    x_pos[1:] = x_pos[1:] + all_gap

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', FIGURE_SIZES['panel']))
    if baseline_name == 'full_profile_center':
        MODEL_NAMES['full_profile']['name'] = 'Full profile (w/ spatial context)'
    # First pass: draw bars, collect annotations
    annotations = {j: [] for j in range(len(class_names))}

    for i, model_name in enumerate(groups):
        values = np.array(values_dict[model_name])
        offsets = x_pos + i * bar_width

        ax.bar(offsets, values, bar_width, label=MODEL_NAMES[model_name]['name'])

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
    if show_improve and baseline_name == 'full_profile_center':
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
            for k, (lx, ly, txt) in enumerate(labels):
                label_y = y_positions[k]
                ax.text(lx, label_y, txt, ha='center', va='bottom', fontsize=FONT_SIZES['annot'])
                if label_y - ly > 0.02:
                    ax.plot([lx, lx], [ly, label_y],
                            color='gray', linewidth=0.5, alpha=0.5)

    # Vertical separator between ALL (first group) and the forest types
    sep_x = (x_pos[0] + (n_models - 1) * bar_width + bar_width / 2 + x_pos[1] - bar_width / 2) / 2
    ax.axvline(sep_x, color='gray', linestyle='--', linewidth=1.5)

    ax.set_xticks(x_pos + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(class_names, fontsize=FONT_SIZES['ticks'], rotation=45, ha='center')
    ax.set_ylabel(f'{metric}', fontsize=FONT_SIZES['label'])
    ax.tick_params(axis='y', labelsize=FONT_SIZES['ticks'])
    ax.set_ylim(0, 1.01 )
    fewer_ticks(ax, axis='y')
    if baseline_name == 'full_profile_center':
        ax.legend(bbox_to_anchor=(0.62, 1), loc='upper left', fontsize=FONT_SIZES['legend'], ncol=1)
    else:
        ax.legend(bbox_to_anchor=(0.09, 1), loc='upper left', fontsize=FONT_SIZES['legend'], ncol=3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'barplot_{metric}_{avg}_baseline_{baseline_name}.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
def plot_dots(summary_df: pd.DataFrame, per_class_df: pd.DataFrame, groups: tuple[str], metric: str='F1', avg: str='macro', baseline_name: str='rh98', save_dir: str=None, include_alpha_em: bool=False, **kwargs):
    '''
    Dot plot of a metric per group.
    x axis: score value of `metric`; y axis: categories ('ALL' on top, then forest types).
    Each group gets a distinct marker shape and a distinct color.
    Args:
        summary_df: dataframe of summary reports
        per_class_df: dataframe of per-class reports
        groups: tuple of group (model) names
        metric: which score to plot (e.g. 'F1', 'Precision', 'Recall', 'Accuracy')
        avg: 'macro' or 'weighted'
        baseline_name: group used as the reference for the improvement coloring
        save_dir: path to save the plot
    '''
    assert avg in ['macro', 'weighted']
    mask = summary_df.Metric == f'{avg} avg'
    summary_df = summary_df.loc[mask, [metric]]
    per_class_df = per_class_df.loc[:, ['Class', metric]]
    if not include_alpha_em and 'alpha_em' in summary_df.columns:
        summary_df = summary_df.drop(index='alpha_em')
        per_class_df = per_class_df.drop(index='alpha_em')

    forest_types = [c['short_name'] for c in LAND_USE_NAMES.values()]
    n_models = len(groups)

    # Build values dict: model_name -> list of values (per class + ALL)
    values_dict = {}
    for model_name in groups:
        values_dict[model_name] = (
            per_class_df.loc[model_name, metric].values.tolist()
            + [summary_df.loc[model_name, metric]]
        )

    # Manually reorder forest types by Land_use_ID, with ALL placed first
    manual_order_ids = [11, 20, 53, 31, 32, 40, 0]
    land_use_ids = list(LAND_USE_NAMES.keys())
    manual_indices = [land_use_ids.index(_id) for _id in manual_order_ids]
    all_index = len(forest_types)  # ALL is appended after the per-class values
    reorder_indices = [all_index] + manual_indices  # ALL first, then forest types
    forest_types = [forest_types[i] for i in manual_indices]
    for m in values_dict:
        values_dict[m] = [values_dict[m][i] for i in reorder_indices]

    class_names = ['ALL'] + forest_types
    y_base = np.arange(len(class_names))

    # Distinct shape per group; per-shape size factors equalize visual weight.
    base_s = 130
    # markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    markers = ['D', '^',  'v', '<', '>', 'o', 's', 'p', 'h', '*']
    marker_size_factor = {
        'o': 1.0, 's': 0.85, 'D': 1.5, 'p': 1.05, 'h': 1,
        '^': 1.15, 'v': 1.15, '<': 1.15, '>': 1.15, '*': 2,
    }
 
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', FIGURE_SIZES['large']))

    # Treat missing/NaN scores as 0 so every group is always drawn
    plot_vals = {m: np.nan_to_num(np.array(values_dict[m], dtype=float), nan=0.0)
                 for m in groups}

    # Distinct color per group (categorical)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Connector line per category spanning the range of group values
    all_vals = np.array([plot_vals[m] for m in groups])  # (n_models, n_categories)
    x_min = all_vals.min(axis=0)
    x_max = all_vals.max(axis=0)
    ax.hlines(y_base, x_min, x_max, color='gray', linewidth=1.2, zorder=1)

    # All groups share the same y position per forest type
    for i, model_name in enumerate(groups):
        mk = markers[i % len(markers)]
        ax.scatter(
            plot_vals[model_name],
            y_base,
            marker=mk,
            facecolors=colors[i % len(colors)],
            edgecolors='white',
            linewidths=0.8,
            s=base_s * marker_size_factor.get(mk, 1.0),
            label=MODEL_NAMES[model_name]['name'],
            zorder=3,
            clip_on=False,
        )

    # Horizontal separator between ALL (top) and the forest types
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5)

    ax.set_yticks(y_base)
    ax.set_yticklabels(class_names, fontsize=FONT_SIZES['ticks'])
    ax.invert_yaxis()  # ALL on top
    ax.set_xlabel(metric, fontsize=FONT_SIZES['label'])
    ax.set_xlim(-0.02, 0.86)
    ax.tick_params(axis='x', labelsize=FONT_SIZES['ticks'])
    fewer_ticks(ax, axis='x')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3, fontsize=FONT_SIZES['legend'])
    plt.tight_layout()
    plt.savefig(save_dir / f'dotplot_{metric}_{avg}.pdf', dpi=150, bbox_inches='tight')
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
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (FIGURE_SIZES['medium'][0], max(3, len(model_names) * 0.6 + 1))))
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
    ax.set_yticklabels([MODEL_NAMES[name]['name'] for name in model_names], rotation=0, ha='right')
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



def build_and_save_reports(y_true_by_group: dict, y_pred_by_group: dict, save_dir: str, prefix: str, labels=None):
    '''
    Build the same summary/per-class/confusion-matrix artifacts that
    run_classification produces, but from arbitrary predictions, so any
    model (e.g. CNN) plugs straight into plot_results.
    Args:
        y_true_by_group: {group_name: ground-truth array (LAND_USE_NAMES codes)}
        y_pred_by_group: {group_name: predicted array, aligned to y_true}
        save_dir: directory to write {prefix}_*.csv / .npz into
        prefix: filename prefix (e.g. 'cnn'); plot_results reads these back
        labels: explicit class label order; defaults to sorted(LAND_USE_NAMES)
    Returns:
        (all_summary_df, all_per_class_df, all_cms)
    '''
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    if labels is None:
        labels = sorted(LAND_USE_NAMES.keys())
    all_summary_reports, all_per_class_reports, all_cms = {}, {}, {}
    for name in y_pred_by_group:
        y_true = np.asarray(y_true_by_group[name])
        y_pred = np.asarray(y_pred_by_group[name])
        all_cms[name] = confusion_matrix(y_true, y_pred, labels=labels)

        per_class_report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        per_class_report = pd.DataFrame(per_class_report).T
        df = per_class_report.rename(columns={'f1-score': 'F1', 'support': 'Support'})
        df.columns = df.columns.str.title()
        df['Support'] = df['Support'].astype(int)

        summary_keys = ['macro avg', 'weighted avg']
        df_summary = df.loc[summary_keys]
        df_summary.loc['macro avg', 'accuracy'] = accuracy
        all_summary_reports[name] = df_summary

        df_per_class = df.drop(index=summary_keys + ['accuracy'], errors='ignore')
        df_per_class.index = [LAND_USE_NAMES[int(idx)]['short_name']
                              if int(idx) in LAND_USE_NAMES else idx
                              for idx in df_per_class.index]
        all_per_class_reports[name] = df_per_class

    all_summary_df = pd.concat(all_summary_reports, names=['Model', 'Metric'])
    all_per_class_df = pd.concat(all_per_class_reports, names=['Model', 'Class'])
    all_summary_df.to_csv(save_dir / f'{prefix}_summary_reports.csv')
    all_per_class_df.to_csv(save_dir / f'{prefix}_per_class_reports.csv')
    np.savez(save_dir / f'{prefix}_confusion_matrices.npz', **all_cms)
    return all_summary_df, all_per_class_df, all_cms


def evaluate_cnn_predictions(pred_dir: str, save_dir: str, groups: tuple[str]=None,
                             prefix: str='cnn',
                             class_codes=None, predictions_are_class_idx: bool=True,
                             truth_col: str='Land_use_ID', pred_col: str='pred',
                             group_files: dict=None, **kwargs):
    '''
    Turn per-group CNN prediction parquets (one file per group, as written by
    callbacks.naturalness_prediction_logger.NaturalnessPredictionLogger) into
    the artifacts plot_results expects, then point plot_results at
    {save_dir}/{prefix}_*.csv / .npz unchanged.

    File resolution: each group's run id comes from MODEL_NAMES[group]['run_id'];
    the file is {pred_dir}/naturalness_predictions_{run_id}{outfile_suffix}.parquet
    (the logger's naming). Pass group_files to override with explicit paths.
    Args:
        pred_dir: directory holding the per-run prediction parquets
        save_dir: where to write {prefix}_*.csv / .npz
        groups: model keys to include; defaults to every MODEL_NAMES entry
            that has a non-None run_id
        prefix: filename prefix consumed by plot_results (default 'cnn')
        outfile_suffix: suffix used when the logger wrote the files
        class_codes: ordered list mapping class index -> Land_use_ID code;
            defaults to sorted(LAND_USE_NAMES). Used only when
            predictions_are_class_idx is True.
        predictions_are_class_idx: the logger stores contiguous class indices
            (0..C-1); when True, remap truth/pred back to Land_use_ID codes so
            they align with LAND_USE_NAMES (mirrors the LR `classes[y_pred]`
            idiom). Set False if the parquet already holds raw codes.
        truth_col / pred_col: column names in the parquet.
        group_files: optional {group_name: parquet path} explicit override.
    Returns:
        (all_summary_df, all_per_class_df, all_cms)
    '''
    pred_dir = Path(pred_dir).expanduser()
    if class_codes is None:
        class_codes = sorted(LAND_USE_NAMES.keys())
    codes = np.asarray(class_codes)

    if group_files is None:
        if groups is None:
            groups = [g for g, v in MODEL_NAMES.items() if v.get('run_id')]
        group_files = {}
        for g in groups:
            run_id = MODEL_NAMES[g]['run_id']
            if run_id is None:
                raise ValueError(f"MODEL_NAMES[{g!r}]['run_id'] is None; set it "
                                 f"or pass group_files explicitly")
            group_files[g] = list(pred_dir.glob(f'naturalness_predictions_{run_id}*.parquet'))[0]

    y_true_by_group, y_pred_by_group = {}, {}
    for name, fp in group_files.items():
        df = pd.read_parquet(Path(fp).expanduser(), columns=[truth_col, pred_col])
        y_true = df[truth_col].to_numpy()
        y_pred = df[pred_col].to_numpy()
        if predictions_are_class_idx:
            y_true = codes[y_true.astype(int)]
            y_pred = codes[y_pred.astype(int)]
        y_true_by_group[name] = y_true
        y_pred_by_group[name] = y_pred
    return build_and_save_reports(y_true_by_group, y_pred_by_group,
                                  save_dir, prefix, labels=list(class_codes))


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
    plot_dots(summary_df, per_class_df, groups=groups, metric='Recall', avg='macro', save_dir=save_dir)
    plot_dots(summary_df, per_class_df, groups=groups, metric='Precision', avg='macro', save_dir=save_dir)
    plot_dots(summary_df, per_class_df, groups=groups, metric='F1', avg='macro', save_dir=save_dir)
    plot_dots(summary_df, per_class_df, groups=groups, metric='Accuracy', avg='macro', save_dir=save_dir)
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
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', FIGURE_SIZES['medium']))
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
        fewer_ticks(ax, axis='y')
        plt.tight_layout()
        plt.savefig(save_dir / f'violin_{col}.png', dpi=150, bbox_inches='tight')
        plt.close()

    return

# ----------------------------------------------------------------------------------------
#  Step 5. Investigate predictions
# ----------------------------------------------------------------------------------------

def agg_preds_from_models(cnn_pred_dir: str, logreg_pred_file: str, save_dir: str=None,
                          out_name: str='aggregated_predictions.fgb',
                          class_codes=None, **kwargs):
    '''
    Collect per-sample predictions from every CNN run + the logistic
    regression models into one spatial table and write a FlatGeobuf.

    Join key is `rowid`; runs are outer-joined and placed side by side, no
    aggregation. Per CNN run (named <group>) the raw per-class
    probabilities (prob_0..prob_{C-1}, written by
    callbacks.naturalness_prediction_logger.NaturalnessPredictionLogger)
    become columns `<group>_prob_<c>`, plus its hard prediction
    `cnn_<group>`. LR models contribute `lr_<group>` hard predictions.
    `Land_use_ID` is the ground truth; geometry comes from the LR file.
    Missing rowids: probs -> NaN, int columns -> -1.

    Args:
        cnn_pred_dir: dir of naturalness_predictions_*.parquet (one per run)
        logreg_pred_file: logistic_regression_predictions.parquet
            (rowid-indexed, geometry + one column per LR model)
        save_dir: output dir (defaults to cnn_pred_dir)
        out_name: FlatGeobuf filename
        class_codes: ordered class-index -> Land_use_ID code map;
            defaults to sorted(LAND_USE_NAMES)
    Returns:
        the aggregated GeoDataFrame
    '''
    cnn_pred_dir = Path(cnn_pred_dir).expanduser()
    logreg_pred_file = Path(logreg_pred_file).expanduser()
    save_dir = Path(save_dir).expanduser() if save_dir else cnn_pred_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    if class_codes is None:
        class_codes = sorted(LAND_USE_NAMES.keys())
    codes = np.asarray(class_codes)

    # run_id -> group name, to label each CNN file
    runid2group = {v['run_id']: g for g, v in MODEL_NAMES.items() if v.get('run_id')}

    cnn_files = sorted(cnn_pred_dir.glob('naturalness_predictions_*.parquet'))
    if not cnn_files:
        raise ValueError(f'no naturalness_predictions_*.parquet in {cnn_pred_dir}')

    # One block of columns per CNN run: raw per-class probs + the hard pred
    parts = []               # list of rowid-indexed DataFrames, one per run
    truth = None
    for fp in cnn_files:
        token = fp.stem.replace('naturalness_predictions_', '')
        group = next((g for rid, g in runid2group.items() if token.startswith(rid)), token)
        df = pd.read_parquet(fp).set_index('rowid')
        prob_cols = sorted([c for c in df.columns if c.startswith('prob_')],
                           key=lambda c: int(c.split('_')[1]))
        part = df[prob_cols].astype(np.float32)
        part.columns = [f'{group}_prob_{c.split("_")[1]}' for c in prob_cols]
        part[f'cnn_{group}'] = codes[df['pred'].to_numpy().astype(int)].astype('int16')
        parts.append(part)
        if truth is None:
            truth = pd.Series(codes[df['Land_use_ID'].to_numpy().astype(int)],
                              index=df.index, name='Land_use_ID')

    # Outer-join every run on rowid (no aggregation, just collected side by side)
    agg = pd.concat(parts, axis=1)
    common = agg.index
    agg.insert(0, 'Land_use_ID', truth.reindex(common).fillna(-1).astype('int16'))
    for col in [c for c in agg.columns if c.startswith('cnn_')]:
        # FlatGeobuf has no nullable ints; -1 = rowid absent from that run
        agg[col] = agg[col].fillna(-1).astype('int16')

    # Logistic-regression predictions + geometry, joined by rowid
    lr = gpd.read_parquet(logreg_pred_file)
    if 'rowid' in lr.columns:
        lr = lr.set_index('rowid')
    lr.index.name = 'rowid'
    meta = {'Land_use_ID', 'slope', 'geometry', 'flag'}
    for col in [c for c in lr.columns if c not in meta]:
        agg[f'lr_{col}'] = lr[col].reindex(common).fillna(-1).astype('int16')

    geom = lr['geometry'].reindex(common)
    out = gpd.GeoDataFrame(agg.reset_index(), geometry=geom.values, crs='EPSG:4326')
    missing = out['geometry'].isna().sum()
    if missing:
        print(f'[agg_preds_from_models] WARNING: {missing} rows have no geometry '
              f'(rowid absent from {logreg_pred_file.name}); dropping them')
        out = out[out['geometry'].notna()]

    out_fp = save_dir / out_name
    out.to_file(out_fp, driver='FlatGeobuf')
    print(f'wrote {len(out)} rows from {len(parts)} CNN runs to {out_fp}')
    return out

def _per_class_metrics_by_cell(joined: pd.DataFrame, pred_col: str, ref_col: str,
                               classes: list) -> pd.DataFrame:
    '''
    Per-cell, per-class recall / precision / F1 (+ macro F1), aggregated from
    a row-per-sample frame already tagged with `cell_id`.
    Args:
        joined: row-per-sample frame with `cell_id`, `ref_col`, `pred_col`
        pred_col: hard-prediction column (Land_use_ID codes)
        ref_col: ground-truth column (Land_use_ID codes)
        classes: ordered Land_use_ID codes to score
    Returns:
        DataFrame indexed by cell_id with columns
        recall_<c>, precision_<c>, f1_<c> for each c, plus f1_macro.
        NaN where a class has no support (true or predicted) in a cell.
    '''
    yt = joined[ref_col].to_numpy()
    yp = joined[pred_col].to_numpy()
    ind = pd.DataFrame({'cell_id': joined['cell_id'].values})
    for c in classes:
        ind[f'_t{c}'] = (yt == c)
        ind[f'_p{c}'] = (yp == c)
        ind[f'_tp{c}'] = (yt == c) & (yp == c)
    sums = ind.groupby('cell_id').sum()

    out = pd.DataFrame(index=sums.index)
    f1_cols = []
    for c in classes:
        tp = sums[f'_tp{c}']
        support = sums[f'_t{c}']
        pred_pos = sums[f'_p{c}']
        recall = tp / support.where(support > 0)
        precision = tp / pred_pos.where(pred_pos > 0)
        f1 = 2 * precision * recall / (precision + recall)
        out[f'recall_{c}'] = recall
        out[f'precision_{c}'] = precision
        out[f'f1_{c}'] = f1
        f1_cols.append(f'f1_{c}')
    out['f1_macro'] = out[f1_cols].mean(axis=1, skipna=True)
    return out


def plot_grid_metric_circles(grid_pred: gpd.GeoDataFrame, size_col: str, color_col: str,
                             save_path: Path, basemap: gpd.GeoDataFrame = None,
                             world_file: str = None,
                             size_range: tuple = (4, 80), cmap: str = 'RdBu_r',
                             color_vlim: tuple = None, **kwargs):
    '''
    Scatter per-cell metrics over a world map: each cell rendered as a circle
    at its centroid, with marker area driven by `size_col` and color by
    `color_col`.
    Args:
        grid_pred: GeoDataFrame with `geometry` (point or polygon, EPSG:4326).
        size_col / color_col: numeric columns to encode as size / color.
        save_path: output figure path.
        world_file: optional basemap polygons (e.g. naturalearth land). If
            None no background is drawn.
        size_range: (min, max) marker area in points^2.
        cmap: colormap; diverging recommended for delta metrics.
        color_vlim: (vmin, vmax); defaults to symmetric around 0.
    '''
    save_path = Path(save_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    g = grid_pred.dropna(subset=[size_col, color_col]).copy()
    if g.geom_type.iloc[0] != 'Point':
        g = g.set_geometry(g.geometry.centroid)

    s_vals = g[size_col].to_numpy(dtype=float)
    c_vals = g[color_col].to_numpy(dtype=float)
    s_lo, s_hi = np.nanmin(s_vals), np.nanmax(s_vals)
    if s_hi > s_lo:
        sizes = size_range[0] + (s_vals - s_lo) / (s_hi - s_lo) * (size_range[1] - size_range[0])
    else:
        sizes = np.full_like(s_vals, sum(size_range) / 2)

    if color_vlim is None:
        vmax = float(np.nanmax(np.abs(c_vals)))
        vmin = -vmax
    else:
        vmin, vmax = color_vlim

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', FIGURE_SIZES['panel']))
    if basemap is None and world_file:
        basemap = gpd.read_file(Path(world_file).expanduser())
    if basemap is not None:
        bm = basemap
        if bm.crs and g.crs and bm.crs != g.crs:
            bm = bm.to_crs(g.crs)
        bm.plot(ax=ax, color='#e8e8e8', edgecolor='white', linewidth=0.2, zorder=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)

    sc = ax.scatter(g.geometry.x, g.geometry.y, s=sizes, c=c_vals, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.3,
                    alpha=0.9)

    # Horizontal colorbar pinned to the lower-left corner of the axes.
    cax = ax.inset_axes([0.02, 0.06, 0.22, 0.025])
    cbar = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cax.xaxis.set_label_position('top')
    cbar.set_label(color_col, fontsize=FONT_SIZES['label'])
    cbar.ax.tick_params(labelsize=FONT_SIZES['ticks'])

    if s_hi > s_lo:
        legend_vals = np.linspace(s_lo, s_hi, 4)
        handles = [
            plt.scatter([], [], c='lightgray', edgecolor='black', linewidth=0.3,
                        s=size_range[0] + (v - s_lo) / (s_hi - s_lo) * (size_range[1] - size_range[0]),
                        label=f'{v:.2f}')
            for v in legend_vals
        ]
        ax.legend(handles=handles, title=size_col, loc='lower left',
                  bbox_to_anchor=(0.02, 0.16), bbox_transform=ax.transAxes,
                  fontsize=FONT_SIZES['legend'], title_fontsize=FONT_SIZES['legend'],
                  ncol=4, columnspacing=1.4, handletextpad=0.4,
                  frameon=False)

    ax.set_xlabel('Longitude', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Latitude', fontsize=FONT_SIZES['label'])
    ax.tick_params(labelsize=FONT_SIZES['ticks'])
    fewer_ticks(ax)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def agg_preds_on_grid(preds_file: str, grid_file: str, save_dir: str = None,
                      out_name: str = 'preds_on_world_grid.parquet',
                      model: str='cnn',
                      baseline_cfg: str = 'rh98',
                      best_cfg: str = 'full_profile_s2',
                      ref_col: str = 'Land_use_ID',
                      metric: str = 'f1',
                      min_points_per_cell: int = 10, **kwargs):
    '''
    Spatially aggregate per-sample naturalness predictions onto a polygon grid
    (e.g. the 1°x1° world land grid from tools/make_world_grid.py).

    For each grid cell, computes:
        - n_points: number of samples falling in the cell
        - <group>_prob_<c>: mean per-class probability across the cell
          (one column per CNN run × class)
        - cnn_<group>, lr_<group>: majority hard prediction (ignoring -1
          sentinel for "rowid absent from that run")
        - Land_use_ID: majority ground-truth class (ignoring -1)
    Any biome / metadata columns already on the grid are preserved.

    Args:
        preds_file: per-sample predictions written by `agg_preds_from_models`
            (FlatGeobuf or GeoParquet, point geometry in EPSG:4326).
        grid_file: polygon grid GeoParquet with a `cell_id` column.
        save_dir: output dir (defaults to dir of preds_file).
        out_name: output filename (.parquet writes GeoParquet, .fgb FlatGeobuf).
        hard_pred_prefixes: column-name prefixes to treat as hard predictions
            (mode-aggregated, -1 treated as missing).
        min_points_per_cell: drop cells with fewer than this many samples.
    Returns:
        the aggregated GeoDataFrame indexed on grid cells.
    '''
    preds_file = Path(preds_file).expanduser()
    grid_file = Path(grid_file).expanduser()
    save_dir = Path(save_dir).expanduser() if save_dir else preds_file.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    if preds_file.suffix == '.parquet':
        preds = gpd.read_parquet(preds_file)
    else:
        preds = gpd.read_file(preds_file)
    grid = gpd.read_parquet(grid_file)
    if preds.crs != grid.crs:
        preds = preds.to_crs(grid.crs)

    joined = gpd.sjoin(preds, grid[['cell_id', 'geometry']], how='inner', predicate='within')
    joined = joined.drop(columns=['geometry', 'index_right'])
    grid = grid.set_index('cell_id')

    joined['acc'] = joined[f'{model}_{best_cfg}'] == joined[ref_col]
    grid_pred = (
            joined
            .groupby('cell_id')
            .agg(n_points=('rowid', 'count'), acc=('acc', 'mean'))
        )
    grid_pred['acc'] = joined.groupby('cell_id')[['acc']].sum()

    per_class_best = _per_class_metrics_by_cell(
        joined, pred_col=f'{model}_{best_cfg}', ref_col=ref_col,
        classes=sorted(LAND_USE_NAMES.keys()),
    )
    per_calss_baseline = _per_class_metrics_by_cell(
        joined, pred_col=f'{model}_{baseline_cfg}', ref_col=ref_col,
        classes=sorted(LAND_USE_NAMES.keys()),
    )
    improve = per_class_best -  per_calss_baseline
    grid_pred = grid_pred.join(per_class_best, rsuffix=f'_{best_cfg}').join(per_calss_baseline, rsuffix=f'_{baseline_cfg}').join(improve, rsuffix='_improve')
    grid_pred = grid_pred[grid_pred['n_points'] >= min_points_per_cell]
    grid_pred['geometry'] = grid.loc[grid_pred.index, 'geometry']
    grid_pred = gpd.GeoDataFrame(grid_pred, geometry='geometry', crs=grid.crs)

    plot_grid_metric_circles(
        grid_pred,
        size_col=f'f1_macro_{baseline_cfg}',
        color_col='f1_macro_improve',
        save_path=save_dir / f'grid_f1_macro_improve_{baseline_cfg}_vs_{best_cfg}.pdf',
        basemap=grid,
        world_file=kwargs.get('world_file'),
    )

    

    # out_fp = save_dir / out_name
    # if out_fp.suffix == '.parquet':
    #     out.to_parquet(out_fp)
    # else:
    #     out.to_file(out_fp, driver='FlatGeobuf')
    # print(f'wrote {len(out)} cells ({len(preds)} samples -> {joined.shape[0]} joined) to {out_fp}')
    # return out


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='on_naturalness',
    default_run='cal_s2_patch_stats',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()