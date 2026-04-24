"""
Compute 2D GLCM contrast and homogeneity per height stratum
from 3D voxel grids (x, y, height_bin) of point density.

Assumes your data is organized as:
- Each sample point has an 11x11 spatial patch
- Each patch has multiple height bins (the vertical profile)
- Voxel values represent point density (or normalized returns)

Adjust STRATA_BINS, N_LEVELS, and DISTANCES to match your data.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, precision_score
from evaluation.utils import load_vsm_naturalness, batch_binning


from evaluation.naturalness import LAND_USE_NAMES
# ── Configuration ────────────────────────────────────────────────────────────
# STRATA = {
#     "understory": (0, 4),      # 0-5m
#     "mid_canopy": (5, 12),     # 5-13m
#     "upper_canopy": (13, 19),  # 13-20m
# }
# Each bin is 5m, so strata indices map 1:1 to bins
STRATA = {f"h_{i*5}_{(i+1)*5}m": (i, i) for i in range(20)}

# GLCM parameters
N_LEVELS = 16          # quantization levels (reduce if patches are small/sparse)
DISTANCES = [1]        # pixel offsets for co-occurrence
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # all 4 directions, then average


# ── Core functions ───────────────────────────────────────────────────────────

def batch_glcm_features(patches, nodata_mask: Optional[np.ndarray] = None, n_levels=16):
    """
    Compute GLCM texture features for a batch of 2D patches,
    averaged over 4 directions. Matches scikit-image output.
 
    Parameters
    ----------
    patches : ndarray, shape (n_points, rows, cols), uint8
        Values must be in [0, n_levels).
 
    n_levels : int
        Number of quantization levels.
 
    Returns
    -------
    dict of ndarray, each shape (n_points,)
        Keys: contrast, homogeneity, dissimilarity, ASM, energy,
              correlation, mean, variance, entropy
    """
    offsets = [
        (0, 1),     # horizontal
        (1, 0),     # vertical
        (1, 1),     # diagonal
        (1, -1),    # anti-diagonal
    ]
 
    # Precompute index matrices
    i_idx, j_idx = np.meshgrid(
        np.arange(n_levels), np.arange(n_levels), indexing='ij'
    )
    i_idx = i_idx.astype(np.float32)
    j_idx = j_idx.astype(np.float32)
    diff = i_idx - j_idx
    diff_sq = diff ** 2
 
    results = {k: [] for k in [
        "contrast", "homogeneity", "dissimilarity", "ASM", "energy",
        "correlation", "mean", "variance", "entropy"
    ]}
    n_points, n_rows, n_cols, n_bins = patches.shape
    patches = patches.transpose(3, 0, 1, 2)
    patches = patches.reshape(n_bins*n_points, n_rows, n_cols)
    if nodata_mask is not None:
        # Repeat mask for each bin: (n_points, 11, 11) → (n_bins * n_points, 11, 11)
        nodata_mask = np.tile(nodata_mask, (n_bins, 1, 1))
 
    for dr, dc in offsets:
        left = patches[:, max(0, -dr):patches.shape[1] - max(0, dr),
                        max(0, -dc):patches.shape[2] - max(0, dc)]
        right = patches[:, max(0, dr):patches.shape[1] - max(0, -dr),
                         max(0, dc):patches.shape[2] - max(0, -dc)]

        if nodata_mask is not None:
            mask_left = nodata_mask[:, max(0, -dr):patches.shape[1] - max(0, dr),
                                      max(0, -dc):patches.shape[2] - max(0, dc)]
            mask_right = nodata_mask[:, max(0, dr):patches.shape[1] - max(0, -dr),
                                       max(0, dc):patches.shape[2] - max(0, -dc)]
            # Both pixels must be valid
            pair_valid = ~mask_left & ~mask_right
        else:
            pair_valid = np.ones_like(left, dtype=bool)

        n = patches.shape[0]
        l_flat = left.reshape(n, -1).astype(np.int32)
        r_flat = right.reshape(n, -1).astype(np.int32)
        v_flat = pair_valid.reshape(n, -1)

        flat_idx = l_flat * n_levels + r_flat
        glcm = np.zeros((n, n_levels * n_levels), dtype=np.float32)

        row_idx = np.broadcast_to(np.arange(n)[:, None], l_flat.shape)  # use l_flat.shape, not flat_idx.shape
        np.add.at(glcm, (row_idx[v_flat], flat_idx[v_flat]), 1.0)
        
        glcm /= glcm.sum(axis=1, keepdims=True).clip(min=1e-10)
        glcm = glcm.reshape(n, n_levels, n_levels)
 
        # ── Standard metrics (match scikit-image) ────────────────────
 
        # contrast: sum(P[i,j] * (i-j)^2)
        results["contrast"].append(
            (glcm * diff_sq[None]).sum(axis=(1, 2))
        )
 
        # homogeneity: sum(P[i,j] / (1 + (i-j)^2))
        results["homogeneity"].append(
            (glcm / (1 + diff_sq[None])).sum(axis=(1, 2))
        )
 
        # dissimilarity: sum(P[i,j] * |i-j|)
        results["dissimilarity"].append(
            (glcm * np.abs(diff)[None]).sum(axis=(1, 2))
        )
 
        # ASM (angular second moment): sum(P[i,j]^2)
        asm = (glcm ** 2).sum(axis=(1, 2))
        results["ASM"].append(asm)
 
        # energy: sqrt(ASM)
        results["energy"].append(np.sqrt(asm))
 
        # correlation: sum(P[i,j] * (i - mu_i)(j - mu_j) / (sigma_i * sigma_j))
        mu_i = (glcm * i_idx[None]).sum(axis=(1, 2))    # (n,)
        mu_j = (glcm * j_idx[None]).sum(axis=(1, 2))    # (n,)
        var_i = (glcm * (i_idx[None] - mu_i[:, None, None]) ** 2).sum(axis=(1, 2))
        var_j = (glcm * (j_idx[None] - mu_j[:, None, None]) ** 2).sum(axis=(1, 2))
        std_ij = np.sqrt(var_i * var_j).clip(min=1e-10)
        corr = (
            glcm * (i_idx[None] - mu_i[:, None, None]) * (j_idx[None] - mu_j[:, None, None])
        ).sum(axis=(1, 2)) / std_ij
        results["correlation"].append(corr)
 
        # ── Additional useful metrics (not in scikit-image) ──────────
 
        # mean: mu_i (marginal mean of rows)
        results["mean"].append(mu_i)
 
        # variance: var_i (marginal variance of rows)
        results["variance"].append(var_i)
 
        # entropy: -sum(P[i,j] * log(P[i,j]))
        with np.errstate(divide='ignore'):
            log_glcm = np.where(glcm > 0, np.log(glcm), 0.0)

        results["entropy"].append(-(glcm * log_glcm).sum(axis=(1, 2)))
    
    for k in results:
        results[k] = np.nanmean(results[k], axis=0).reshape(n_bins, n_points)
    data = np.stack(list(results.values()), axis=0)
    data = data.reshape(-1, n_points).T
    metrics = list(results.keys())
    height_bins = [f'{i*5}_{(i+1)*5}m' for i in range(n_bins)]
    cols = [f'{m}_{h}' for m in metrics for h in height_bins]
    df = pd.DataFrame(data, columns=cols)
    return df
 
# ── Verification ─────────────────────────────────────────────────────────
def skimage_reference(patches_4d, n_levels=16):
    """
    Per-point, per-stratum skimage reference.
 
    Parameters
    ----------
    patches_4d : ndarray, shape (n_points, rows, cols, n_bins), uint8
 
    Returns
    -------
    pd.DataFrame with same columns as batch_glcm_features
    """
    n_points, n_rows, n_cols, n_bins = patches_4d.shape
    sk_metrics = ["contrast", "homogeneity", "dissimilarity", "ASM", "energy", "correlation"]
 
    # Storage: {metric: array of shape (n_points, n_bins)}
    sk_results = {m: np.zeros((n_points, n_bins)) for m in sk_metrics}
 
    for p in range(n_points):
        for z in range(n_bins):
            patch = patches_4d[p, :, :, z]
            glcm = graycomatrix(
                patch, distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=n_levels, symmetric=False, normed=True,
            )
            for m in sk_metrics:
                sk_results[m][p, z] = graycoprops(glcm, m).mean()
 
    # Build DataFrame in same column order
    metrics_order = ["contrast", "homogeneity", "dissimilarity", "ASM", "energy", "correlation"]
    height_bins = [f'{i*5}_{(i+1)*5}m' for i in range(n_bins)]
    cols = [f'{m}_{h}' for m in metrics_order for h in height_bins]
    data = np.concatenate([sk_results[m] for m in metrics_order], axis=1)
    return pd.DataFrame(data, columns=cols)


def skimage_glcm_features_nodata(patch, nodata_mask_single, n_levels=16):
    """
    Use skimage's graycomatrix with a dummy level for nodata pixels.
    Compute GLCM with n_levels+1, then strip the dummy row/column
    and renormalize before computing metrics.
 
    Parameters
    ----------
    patch : ndarray, shape (rows, cols), uint8
    nodata_mask_single : ndarray, shape (rows, cols), bool
    n_levels : int
    """
    dummy_level = n_levels  # use n_levels as the nodata sentinel
    extended_levels = n_levels + 1
 
    # Replace nodata pixels with dummy level
    patch_with_dummy = patch.copy().astype(np.uint8)
    patch_with_dummy[nodata_mask_single] = dummy_level
 
    # Compute GLCM with extended levels using skimage
    glcm_full = graycomatrix(
        patch_with_dummy,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=extended_levels,
        symmetric=False,
        normed=False,  # don't normalize yet
    )
 
    # For each angle, strip dummy row/column and renormalize
    n_angles = glcm_full.shape[3]
    metrics = {k: [] for k in [
        "contrast", "homogeneity", "dissimilarity", "ASM", "energy", "correlation"
    ]}
 
    i_idx, j_idx = np.meshgrid(np.arange(n_levels), np.arange(n_levels), indexing='ij')
    i_idx = i_idx.astype(np.float64)
    j_idx = j_idx.astype(np.float64)
    diff = i_idx - j_idx
    diff_sq = diff ** 2
 
    for a in range(n_angles):
        # Extract and strip dummy level
        g = glcm_full[:n_levels, :n_levels, 0, a].astype(np.float64)
 
        total = g.sum()
        if total < 20:
            for k in metrics:
                metrics[k].append(np.nan)
            continue
 
        g /= total
 
        # Compute metrics using same formulas as graycoprops
        metrics["contrast"].append((g * diff_sq).sum())
        metrics["homogeneity"].append((g / (1 + diff_sq)).sum())
        metrics["dissimilarity"].append((g * np.abs(diff)).sum())
 
        asm_val = (g ** 2).sum()
        metrics["ASM"].append(asm_val)
        metrics["energy"].append(np.sqrt(asm_val))
 
        mu_i = (g * i_idx).sum()
        mu_j = (g * j_idx).sum()
        var_i = (g * (i_idx - mu_i) ** 2).sum()
        var_j = (g * (j_idx - mu_j) ** 2).sum()
        std_ij = np.sqrt(var_i * var_j)
        if std_ij < 1e-10:
            metrics["correlation"].append(0.0)
        else:
            metrics["correlation"].append(
                (g * (i_idx - mu_i) * (j_idx - mu_j)).sum() / std_ij
            )
 
    return {k: np.nanmean(v) for k, v in metrics.items()}
 
# ── Verification ─────────────────────────────────────────────────────────
def verify_batch_glcm_features():
    np.random.seed(42)
 
    N_LEVELS = 16
    n_points = 10
    n_bins = 4  # small for quick verification
    patches_4d = np.random.randint(0, N_LEVELS, size=(n_points, 11, 11, n_bins), dtype=np.uint8)
 
    print(f"Input shape: {patches_4d.shape}")
    print(f"Levels: {N_LEVELS}, Bins: {n_bins}")
    print()
 
    # Run both
    df_custom = batch_glcm_features(patches_4d.copy(), nodata_mask=None, n_levels=N_LEVELS)
    df_skimage = skimage_reference(patches_4d, n_levels=N_LEVELS)
 
    # Compare overlapping metrics (skimage doesn't have mean, variance, entropy)
    sk_metrics = ["contrast", "homogeneity", "dissimilarity", "ASM", "energy", "correlation"]
    height_bins = [f'{i*5}_{(i+1)*5}m' for i in range(n_bins)]
 
    print("=" * 70)
    print("PER-METRIC MAX DIFFERENCE")
    print("=" * 70)
    all_ok = True
    for m in sk_metrics:
        cols = [f'{m}_{h}' for h in height_bins]
        max_diff = np.max(np.abs(df_custom[cols].values - df_skimage[cols].values))
        match = "OK" if max_diff < 1e-4 else "FAIL"
        if match == "FAIL":
            all_ok = False
        print(f"  {m:<16s} max diff: {max_diff:.10f}  [{match}]")
 
    # Per-stratum detail for contrast
    print()
    print("=" * 70)
    print("CONTRAST PER STRATUM (point 0)")
    print("=" * 70)
    print(f"{'Stratum':<12} | {'skimage':>12} | {'custom':>12} | {'diff':>12}")
    print("-" * 55)
    for h in height_bins:
        col = f'contrast_{h}'
        sk_val = df_skimage[col].iloc[0]
        cu_val = df_custom[col].iloc[0]
        print(f"{h:<12} | {sk_val:>12.4f} | {cu_val:>12.4f} | {abs(sk_val - cu_val):>12.8f}")
 
    # Check DataFrame shape and columns
    print()
    print("=" * 70)
    print("DATAFRAME STRUCTURE")
    print("=" * 70)
    print(f"Custom shape:  {df_custom.shape}")
    print(f"Skimage shape: {df_skimage.shape}")
    print(f"Custom cols (first 8):  {list(df_custom.columns[:8])}")
    print(f"Skimage cols (first 8): {list(df_skimage.columns[:8])}")
 
    # Extra metrics only in custom
    print()
    print("=" * 70)
    print("EXTRA METRICS (mean, variance, entropy) - point 0, stratum 0")
    print("=" * 70)
    for m in ["mean", "variance", "entropy"]:
        col = f'{m}_{height_bins[0]}'
        print(f"  {col}: {df_custom[col].iloc[0]:.6f}")
 
    # Benchmark with larger data
    print()
    print("=" * 70)
    print("BENCHMARK")
    print("=" * 70)
    n_bench_points = 1000
    n_bench_bins = 20
    big_patches = np.random.randint(0, N_LEVELS, size=(n_bench_points, 11, 11, n_bench_bins), dtype=np.uint8)
 
    t0 = time.time()
    batch_glcm_features(big_patches.copy(), nodata_mask=None, n_levels=N_LEVELS)
    t_custom = time.time() - t0
 
    # Only benchmark 100 points for skimage, extrapolate
    t0 = time.time()
    skimage_reference(big_patches[:100], n_levels=N_LEVELS)
    t_sk = time.time() - t0
    t_sk_est = t_sk * (n_bench_points / 100)
 
    print(f"  Custom ({n_bench_points} pts x {n_bench_bins} bins):  {t_custom:.3f}s")
    print(f"  Skimage (100 pts, measured):    {t_sk:.3f}s")
    print(f"  Skimage ({n_bench_points} pts, estimated): {t_sk_est:.1f}s")
    print(f"  Speedup (estimated):            {t_sk_est / t_custom:.1f}x")
 
    print()
    print(f"Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
   
 

def quantize_patch(patch_2d: np.ndarray, n_levels: int = N_LEVELS) -> np.ndarray:
    """
    Quantize a continuous 2D array to integer levels for GLCM.
    Handles the case where patch is all zeros or constant.
    """
    patch = patch_2d.astype(np.float64)
    pmin, pmax = patch.min(), patch.max()

    if pmax - pmin < 1e-10:
        # Constant patch → all same level
        return np.zeros_like(patch, dtype=np.uint8)

    # Scale to [0, n_levels - 1]
    quantized = ((patch - pmin) / (pmax - pmin) * (n_levels - 1)).astype(np.uint8)
    return quantized


# ── Batch processing ─────────────────────────────────────────────────────────

def _cal_vsm_patch_texture(h5_file: Path,
                           out_file: Path,
                           n_levels: int = N_LEVELS,
                           bin_width: int = 5, **kwargs) -> np.ndarray:
    """
    Compute GLCM features for a single VSM patch.
    """
    # if out_file.exists():
    #     return
    vsm_patches, naturalness, rowids = load_vsm_naturalness(h5_file)
    n_points, n_bands, n_rows, n_cols = vsm_patches.shape
    if n_points == 0:
        print('n_points is 0, h5_file: %s' % h5_file.stem)
        return
    voxel_data, nodata_mask = batch_binning(vsm_patches, bin_width=bin_width) # (n_points, w, h, 20)
    voxel_data = voxel_data.reshape(n_points, n_rows, n_cols, -1)
    # voxel_data = quantize_patch(voxel_data, n_levels)
    df = batch_glcm_features(voxel_data, nodata_mask, n_levels)
    df['naturalness'] = naturalness
    df['rowid'] = rowids
    df.to_parquet(out_file)
    return df

def cal_vsm_patch_texture(vsm_patches_dir: str, save_dir: str, bin_width: int = 5, n_levels: int = 16, **kwargs):
    """
    Compute GLCM features for all VSM patches in the directory.
    """
    vsm_patches_dir = Path(vsm_patches_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    files = list(vsm_patches_dir.glob('*.h5'))
    tasks = []
    for h5_file in files:
        out_file = save_dir / f'{h5_file.stem}.parquet'
        # _cal_vsm_patch_texture(h5_file, out_file, **kwargs)
        tasks.append(dask.delayed(_cal_vsm_patch_texture)(h5_file, out_file, bin_width=bin_width, n_levels=n_levels, **kwargs))
    with ProgressBar():
        dask.compute(*tasks)

def _load_patch_stats(vsm_patch_stats_dir: str, **kwargs):
    '''
    Load the VSM patch statistics
    Args:
        vsm_patch_stats_dir: path to the VSM patch statistics directory
    Returns:
        None
    '''
    vsm_patch_stats_dir = Path(vsm_patch_stats_dir).expanduser()
    debug = kwargs.get('debug', False)
    if debug:
        files = list(vsm_patch_stats_dir.glob('*.parquet'))[:10]
    else:
        files = list(vsm_patch_stats_dir.glob('*.parquet'))
    ddf = dd.read_parquet(files)
    ddf = ddf.dropna(subset=['contrast_0_5m'])
    ddf = ddf.compute()
    valid = ~ddf['naturalness'].isin([-1, 1])
    ddf = ddf[valid]
    
    y = ddf['naturalness'].values.flatten().astype(int)
    return ddf, y

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
        groups = ddf.groupby('naturalness')[col]
        labels = sorted(ddf['naturalness'].unique())
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


# ----------------------------------------------------------------------------------------
#  Step 3. Run classification
# ----------------------------------------------------------------------------------------

def logistic_regression(x, x_val, y, variance_threshold=0.01, **kwargs):
    # 1. Remove near-zero variance columns (the root cause of your NaNs)
    var_filter = VarianceThreshold(threshold=variance_threshold)
    x = var_filter.fit_transform(x)
    x_val = var_filter.transform(x_val)
    
    # 2. Standardize (helps solver stability)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_val = scaler.transform(x_val)
    
    # 3. Add constant for intercept
    x = sm.add_constant(x)
    x_val = sm.add_constant(x_val, has_constant='add')  # 'add' forces it even if constant exists
    
    # 4. Fit
    model = sm.MNLogit(y, x)
    clf = model.fit(maxiter=200, method='bfgs')  # bfgs more stable than default newton
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

def run_classification(vsm_patch_stats_dir: str, vsm_patch_stats_dir_val: str, save_dir: str, classifier: str = 'logistic_regression', **kwargs):
    '''
    Perform logistic regression on the VSM patches
    Args:
        vsm_patch_stats_dir: path to the VSM patch statistics directory
        save_dir: path to save the logistic regression model
    Returns:
        None
    '''
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    ddf, y = _load_patch_stats(vsm_patch_stats_dir, **kwargs)
    ddf_val, y_val = _load_patch_stats(vsm_patch_stats_dir_val, **kwargs)
    # y_val = y
    # ddf_val = ddf
    classes = np.unique(y)
    height_bins = [f'{i*5}_{(i+1)*5}m' for i in range(20)]
    metrics = ['contrast', 'mean', 'variance', 'entropy', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation']
    groups = {
        'texture_0_5m': [f'{m}_{h}' for m in metrics for h in ['0_5m']],
        'texture_0_10m': [f'{m}_{h}' for m in metrics for h in ['0_5m', '5_10m']],
        'texture_0_15m': [f'{m}_{h}' for m in metrics for h in ['0_5m', '5_10m', '10_15m']],
        'texture_0_20m': [f'{m}_{h}' for m in metrics for h in ['0_5m', '5_10m', '10_15m', '15_20m']],
        'texture_0_25m': [f'{m}_{h}' for m in metrics for h in ['0_5m', '5_10m', '10_15m', '15_20m', '20_25m']],
        'texture_0_30m': [f'{m}_{h}' for m in metrics for h in ['0_5m', '5_10m', '10_15m', '15_20m', '20_25m', '25_30m']],
        'texture_0_35m': [f'{m}_{h}' for m in metrics for h in ['0_5m', '5_10m', '10_15m', '15_20m', '20_25m', '25_30m', '30_35m']],
        'texture_0_40m': [f'{m}_{h}' for m in metrics for h in ['0_5m', '5_10m', '10_15m', '15_20m', '20_25m', '25_30m', '30_35m', '35_40m']],
        'glcm_texture': [f'{m}_{h}' for m in metrics for h in height_bins]
        }
    # Store results for comparison plots
    all_summary_reports = {}
    all_per_class_reports = {}
    all_cms = {}
    for name, cols in groups.items():
        print(f'{name}')
        x = ddf[cols].values
        rank = np.linalg.matrix_rank(x)
        cond = np.linalg.cond(x)
        print(f"{name}: shape={x.shape}, rank={rank}, cond={cond:.2e}")
        nan_cols = ddf[cols].isna().sum()
        nan_cols = nan_cols[nan_cols > 0]
        print(f"{name}: {len(nan_cols)} cols with NaNs")
        if len(nan_cols):
            print(nan_cols)
            

        x_val = ddf_val[cols].values
        rank = np.linalg.matrix_rank(x_val)
        cond = np.linalg.cond(x_val)
        print(f"{name}: shape={x_val.shape}, rank={rank}, cond={cond:.2e}")
        nan_cols = ddf_val[cols].isna().sum()
        nan_cols = nan_cols[nan_cols > 0]
        print(f"{name}: {len(nan_cols)} cols with NaNs")
        if len(nan_cols):
            print(nan_cols)
            
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x_val_scaled = scaler.transform(x_val)
        if classifier == 'logistic_regression':
            clf, x_val = logistic_regression(x, x_val, y)
        elif classifier == 'xgboost':
            clf, x_val = xgboost_classification(x, x_val, y, y_val)
        y_pred = clf.predict(x_val).argmax(axis=1)
        assert np.isfinite(y_pred).all(), f"{name}: predictions are nan"
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


# ── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # verify_batch_glcm_features()
    vsm_patches_dir = '/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/vsm_patches_ps11_train'
    vsm_patch_stats_dir = '/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/glcm_texture_train'
    vsm_patch_stats_dir_val = '/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/glcm_texture_val'
    
    # save_dir = '/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/glcm_texture_train'
    # cal_vsm_patch_texture(vsm_patches_dir, save_dir, n_levels=100)
    height_bins = [f'{i*5}_{(i+1)*5}m' for i in range(20)]
    metrics = ['contrast', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation', 'mean', 'variance', 'entropy']
    feature_cols = [f'{m}_{h}' for m in metrics for h in height_bins]
    # check_distribution(vsm_patch_stats_dir, save_dir, feature_cols)
    
    save_dir = '/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/glcm_texture_classification'
    run_classification(vsm_patch_stats_dir, vsm_patch_stats_dir_val, save_dir)

    # print(f"Computing GLCM features for {n_points} points...")
    # print(f"Strata: {STRATA}")
    # print(f"Voxel shape per point: {voxel_data.shape[1:]}")
    # print()

    # feature_names, features = process_all_points(voxel_data)

    # print(f"\nFeatures computed: {feature_names}")
    # print(f"Feature array shape: {features.shape}")
    # print(f"\nFirst point features:")
    # for name, val in zip(feature_names, features[0]):
    #     print(f"  {name}: {val:.4f}")

    # ─── Combine with your existing features and analyze ───
    # You'd then do something like:
    #
    # import pandas as pd
    # df = pd.DataFrame(features, columns=feature_names)
    # df["land_use"] = fake_labels
    # df["avg_fhd"] = your_avg_fhd
    # df["std_fhd"] = your_std_fhd
    # ... etc
    #
    # Then feed into PCA / violin plots / Random Forest as before