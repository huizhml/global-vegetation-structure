"""
Verify vectorized FHD approaches against the baseline pixel-level function.
Uses synthetic GEDI-like RH profiles (monotonically increasing, in meters).
"""

import numpy as np
import pandas as pd
import time
from evaluation.on_diversity_indices import pixel_diversity_indices, _chunk_diversity
from const import MAX_HEIGHT_METERS, VSM_NODATA
# --- Constants ---
BIN_WIDTH = 5
N_BINS = int(MAX_HEIGHT_METERS / BIN_WIDTH)

np.random.seed(42)


# =============================================================================
# Baseline: per-pixel loop (reference): pixel_diversity_indices
# =============================================================================

def pixel_diversity_indices(rhs, bin_width=5, max_height=MAX_HEIGHT_METERS):
    """
    Compute per-pixel FHD using a simple histogram approach. NOTE!!!: rhs should be in meters!!!
    """
    if isinstance(rhs, pd.Series):
        rhs = rhs.values
    rhs_arr = np.asarray(rhs, dtype=np.float32)
    valid = np.isfinite(rhs_arr) & (rhs_arr > 0)
    rhs_valid = rhs_arr[valid]
    if rhs_valid.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    rhs_valid = np.clip(rhs_valid, None, max_height)
    n_bins = int(max_height / bin_width)
    hist, bins = np.histogram(rhs_valid, bins=n_bins, range=(0, max_height)) # negative values are ignored
    p = hist / hist.sum() # NOTE: normalize the histogram to get the probability
    mask = p > 0
    fhd = -np.sum(p[mask] * np.log(p[mask])).astype(np.float32)
    enl1d = np.exp(fhd)
    enl2d = np.float32(1.0 / np.sum(p[mask] ** 2))
    if np.isinf(enl2d):
        print(f'ENL2D is inf, setting to NaN, rhs: {rhs}, p: {p}')
        enl2d = np.nan
    rh25 = max(rhs_arr[25], 0)
    if rhs_arr[98] <= 0:
        cr = np.nan
    else:
        cr = (rhs_arr[98] - rh25)/rhs_arr[98]
    
    return fhd, enl1d, enl2d, cr

def baseline_loop(data, bin_width=BIN_WIDTH, max_height=MAX_HEIGHT_METERS):
    """
    Loop over all pixels, call baseline function.
    data: shape (n_pixels, 101), in meters, already cleaned of nodata.
    Returns: fhd, enl1d, enl2d, cr arrays each of shape (n_pixels,)
    """
    n_bands, n_rows, n_cols = data.shape
    valid = np.isfinite(data) & (data != VSM_NODATA) & (data > 0)
    nodata_mask = valid.sum(axis=0) == 0
    fhd = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    enl1d = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    enl2d = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    cr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            fhd[i, j], enl1d[i, j], enl2d[i, j], cr[i, j] = pixel_diversity_indices(
                data[:, i, j], bin_width=bin_width, max_height=max_height
            )
    fhd[nodata_mask] = np.nan
    enl1d[nodata_mask] = np.nan
    enl2d[nodata_mask] = np.nan
    cr[nodata_mask] = np.nan
    return fhd, enl1d, enl2d, cr


# =============================================================================
# Vectorized approach 1: np.add.at
# =============================================================================
def vectorized_add_at(data, bin_width=BIN_WIDTH, max_height=MAX_HEIGHT_METERS):
    n_bands, n_rows, n_cols = data.shape
    n_pixels = n_rows * n_cols
    n_bins = int(max_height / bin_width)

    valid = np.isfinite(data) & (data != VSM_NODATA) & (data > 0)
    nodata_mask = valid.sum(axis=0) == 0  # (rows, cols)

    data_clean = np.where(valid, np.minimum(data, max_height), 0.0)

    bin_idx = np.clip((data_clean / bin_width).astype(np.int32), 0, n_bins - 1)
    bin_idx = np.where(valid, bin_idx, -1)

    # Reshape to (n_bands, n_pixels)
    bin_flat = bin_idx.reshape(n_bands, n_pixels)
    pixel_indices = np.broadcast_to(
        np.arange(n_pixels)[np.newaxis, :], (n_bands, n_pixels)
    )

    hist = np.zeros((n_pixels, n_bins), dtype=np.float32)
    flat_valid = bin_flat != -1
    np.add.at(hist, (pixel_indices[flat_valid], bin_flat[flat_valid]), 1.0)

    total = hist.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    p = hist / total

    log_p = np.where(p > 0, np.log(p), 0.0)
    fhd = -np.sum(p * log_p, axis=1).astype(np.float32)
    enl1d = np.exp(fhd).astype(np.float32)
    sum_p2 = np.sum(np.where(p > 0, p ** 2, 0.0), axis=1)
    enl2d = np.where(sum_p2 > 0, 1.0 / sum_p2, np.nan).astype(np.float32)

    # CR: index band axis directly from original data
    rh25 = np.maximum(data[25].ravel(), 0)
    rh98 = data[98].ravel()
    cr = np.where(rh98 > 0, (rh98 - rh25) / rh98, np.nan).astype(np.float32)

    # Reshape back and apply nodata
    fhd = fhd.reshape(n_rows, n_cols)
    enl1d = enl1d.reshape(n_rows, n_cols)
    enl2d = enl2d.reshape(n_rows, n_cols)
    cr = cr.reshape(n_rows, n_cols)
    fhd[nodata_mask] = np.nan
    enl1d[nodata_mask] = np.nan
    enl2d[nodata_mask] = np.nan
    cr[nodata_mask] = np.nan

    return fhd, enl1d, enl2d, cr


# =============================================================================
# Vectorized approach 2: np.searchsorted, _chunk_diversity
# =============================================================================
def _chunk_diversity(data, bin_width=5, max_height=MAX_HEIGHT_METERS):
    """
    Vectorized Shannon entropy for a single spatial chunk.

    Parameters
    ----------
    tile : ndarray, shape (101, rows, cols)

    Returns
    -------
    out : ndarray, shape (rows, cols), float32
    """
    n_bands, n_rows, n_cols = data.shape
    n_pixels = n_rows * n_cols
    valid = np.isfinite(data) & (data != VSM_NODATA) & (data > 0)
    nodata_mask = valid.sum(axis=0) == 0

    # Remove: data = data / 10  (data is already in meters)

    data = data.reshape(n_bands, n_pixels)
    valid = valid.reshape(n_bands, n_pixels)
    data_clean = np.where(valid, np.minimum(data, max_height), -1.0)

    # Sort along axis=0 (bands), not axis=1 (pixels)
    profiles = np.sort(data_clean, axis=0)
    profiles = np.ascontiguousarray(profiles)

    lower_edges = np.arange(0, max_height, bin_width)
    upper_edges = np.arange(bin_width, max_height + bin_width, bin_width)

    idx_low = np.stack(
        [np.searchsorted(profiles[:, i], lower_edges, side='left') for i in range(n_pixels)]
    )
    idx_high = np.stack(
        [np.searchsorted(profiles[:, i], upper_edges, side='left') for i in range(n_pixels)]
    )
    idx_high[:, -1] = np.array(
        [np.searchsorted(profiles[:, i], upper_edges[-1:], side='right')[0] for i in range(n_pixels)]
    )
    hist = (idx_high - idx_low).astype(np.float32)

    # Normalize
    total = hist.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    p = hist / total

    # FHD
    log_p = np.where(p > 0, np.log(p), 0.0)
    fhd = -np.sum(p * log_p, axis=1).astype(np.float32)

    # ENL1D
    enl1d = np.exp(fhd).astype(np.float32)

    # ENL2D
    sum_p2 = np.sum(np.where(p > 0, p ** 2, 0.0), axis=1)
    enl2d = np.where(sum_p2 > 0, 1.0 / sum_p2, np.nan).astype(np.float32)

    # CR
    rh25 = np.maximum(data[25, :], 0)
    rh98 = data[98, :]
    cr = np.where(rh98 > 0, (rh98 - rh25) / rh98, np.nan).astype(np.float32)

    fhd = fhd.reshape(n_rows, n_cols)
    enl1d = enl1d.reshape(n_rows, n_cols)
    enl2d = enl2d.reshape(n_rows, n_cols)
    cr = cr.reshape(n_rows, n_cols)
    # Apply nodata
    fhd[nodata_mask] = np.nan
    enl1d[nodata_mask] = np.nan
    enl2d[nodata_mask] = np.nan
    cr[nodata_mask] = np.nan

    return fhd, enl1d, enl2d, cr


# =============================================================================
# Generate synthetic data
# =============================================================================
def make_rh_profile(max_tree_height):
    """Generate a monotonically increasing RH profile (101 values) in meters."""
    raw = np.sort(np.random.beta(2, 3, size=101))
    raw = (raw - raw[0]) / (raw[-1] - raw[0])
    profile = (raw * max_tree_height).astype(np.float32)
    # Simulate negative RH values at low percentiles (ground return below elev_lowestmode)
    n_neg = np.random.randint(0, 15)
    if n_neg > 0:
        profile[:n_neg] = np.linspace(-0.5, -0.01, n_neg).astype(np.float32)
    return profile


patch_size = 512
data = np.zeros((1, patch_size, patch_size, 101), dtype=np.float32)
data = data.reshape(-1, 101)
n_pixels = data.shape[0]
for i in range(n_pixels):
    max_h = np.random.uniform(5, 48)
    data[i] = make_rh_profile(max_h)

# Inject some nodata pixels
data[0, :] = np.nan
data[1, :] = VSM_NODATA
data[2, :] = -1.0  # all negative
print(data.shape)
data = data.reshape(patch_size, patch_size, 101)
data = data.transpose(2, 0, 1)


# =============================================================================
# Run all methods
# =============================================================================
print("Running baseline loop...")
t0 = time.time()
fhd_base, enl1d_base, enl2d_base, cr_base = baseline_loop(data)
t_base = time.time() - t0

print("Running vectorized add.at...")
t0 = time.time()
fhd_addat, enl1d_addat, enl2d_addat, cr_addat = vectorized_add_at(data)
t_addat = time.time() - t0

print("Running vectorized searchsorted...")
t0 = time.time()
fhd_search, enl1d_search, enl2d_search, cr_search = _chunk_diversity(data)
t_search = time.time() - t0


# =============================================================================
# Compare results
# =============================================================================
def compare(name, baseline, test):
    """Compare two arrays, handling NaNs."""
    both_nan = np.isnan(baseline) & np.isnan(test)
    nan_mismatch = np.isnan(baseline) != np.isnan(test)
    n_nan_mismatch = nan_mismatch.sum()

    valid = np.isfinite(baseline) & np.isfinite(test)
    if valid.sum() == 0:
        max_diff = 0.0
        mean_diff = 0.0
    else:
        diffs = np.abs(baseline[valid] - test[valid])
        max_diff = diffs.max()
        mean_diff = diffs.mean()
    return max_diff, mean_diff, n_nan_mismatch


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

metrics = {
    'FHD': (fhd_base, fhd_addat, fhd_search),
    'ENL1D': (enl1d_base, enl1d_addat, enl1d_search),
    'ENL2D': (enl2d_base, enl2d_addat, enl2d_search),
    'CR': (cr_base, cr_addat, cr_search),
}

all_pass = True
for metric_name, (base, addat, search) in metrics.items():
    print(f"\n--- {metric_name} ---")

    max_d, mean_d, nan_mm = compare(metric_name, base, addat)
    ok = max_d < 1e-5 and nan_mm == 0
    all_pass &= ok
    print(f"  add.at:       max_diff={max_d:.2e}  mean_diff={mean_d:.2e}  "
          f"nan_mismatch={nan_mm}  {'PASS' if ok else 'FAIL'}")

    max_d, mean_d, nan_mm = compare(metric_name, base, search)
    ok = max_d < 1e-5 and nan_mm == 0
    all_pass &= ok
    print(f"  searchsorted: max_diff={max_d:.2e}  mean_diff={mean_d:.2e}  "
          f"nan_mismatch={nan_mm}  {'PASS' if ok else 'FAIL'}")

print(f"\n--- Timing ({n_pixels} pixels) ---")
print(f"  Baseline loop: {t_base:.4f}s")
print(f"  add.at:        {t_addat:.4f}s  ({t_base/t_addat:.1f}x faster)")
print(f"  searchsorted:  {t_search:.4f}s  ({t_base/t_search:.1f}x faster)")

print(f"\n{'ALL METHODS AGREE' if all_pass else 'MISMATCH DETECTED'}")

# Show a few sample values for sanity check
print(f"\n--- Sample values (first 5 valid pixels) ---")
valid_idx = np.where(np.isfinite(fhd_base))[0][:5]
print(f"{'idx':>4s}  {'FHD_base':>9s} {'FHD_add':>9s} {'FHD_ss':>9s}  "
      f"{'ENL1D_b':>8s} {'ENL2D_b':>8s} {'CR_base':>8s}")
for i in valid_idx:
    print(f"{i:4d}  {fhd_base[i]:9.5f} {fhd_addat[i]:9.5f} {fhd_search[i]:9.5f}  "
          f"{enl1d_base[i]:8.4f} {enl2d_base[i]:8.4f} {cr_base[i]:8.4f}")