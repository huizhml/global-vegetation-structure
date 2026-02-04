
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
file = '~/gvsm/waveform_examples/dk_rhs_examples.npz'
file = Path(file).expanduser()
example_rhs = np.load(file)['example_rhs']





#A single RH profile in which double RH values are removed (avoids the division by zero issue, e.g. in example 2)
example_no = 4  # select a profile
min_rh = -50  # fixed lower RH value for resampling (should this be zero?)
max_rh = 250  # fixed upper RH for resampling
step = 1.  # step size for resampling
window = 30  # smoothing window
single_rhs = np.unique(example_rhs[example_no,:,1,1])

# # Make plot
# ones = np.arange(single_rhs.size)
# grad = np.gradient(ones, single_rhs)
# plt.plot(grad, single_rhs, "x", label="values")

# # Interpolate
# x = np.arange(min_rh, max_rh+step, step)
# grad_inter = interp1d(single_rhs, grad, kind='linear', fill_value=0, bounds_error=False)
# grad_resampled = grad_inter(x)
# plt.plot(grad_resampled, x, "-", label="resampled")

# # Smooth
# plt.plot(savgol_filter(grad_resampled, window, 1), x, "-", lw=3, label="smoothed")
# plt.legend()
# plt.show()


def make_single_plot(ax, single_rhs, min_rh, max_rh, step, window):
    ones = np.arange(single_rhs.size)
    grad = np.gradient(ones, single_rhs)
    grad = np.nan_to_num(grad, nan=1)
    ax.plot(grad, single_rhs, "x", label="values")
    x = np.arange(min_rh, max_rh+step, step)
    grad_inter = interp1d(single_rhs, grad, kind='linear', fill_value=0, bounds_error=False)
    grad_resampled = grad_inter(x)
    ax.plot(grad_resampled, x, "-", label="resampled")
    ax.plot(savgol_filter(grad_resampled, window, 1), x, "-", lw=3, label="smoothed")
    ax.legend()

n_examples = example_rhs.shape[0]
fig, axes = plt.subplots(3, n_examples, figsize=(18, 9), gridspec_kw={'width_ratios': [1] * n_examples})
for i, rhs in enumerate(example_rhs):
    for j, ws in enumerate([15, 20, 30]):
        make_single_plot(axes[j, i], rhs[:, 1,1], min_rh, max_rh, 1, ws)
    # x = np.arange(101)
    # # Row 1 orginal RH profile
    # original_rhs = rhs[:, 1,1]
    # derivative = np.gradient(x, original_rhs)
    # derivative = np.nan_to_num(derivative, nan=1)
    # axes[0, i].plot(derivative, original_rhs/10)
    # axes[0, i].set_ylim(-10, max_rh)
    
    # # Row 2 averaged RH profile, over 3x3 pixels
    # rhs = rhs.mean(axis=(1,2))
    # derivate = np.gradient(x, rhs)
    # axes[1, i].plot(derivate, rhs/10)
    # axes[1, i].set_ylim(-10, max_rh)
    
    # # Row 3 averaged RH profile, over 3x3 pixels, & interpolated
    # f_interp = interp1d(rhs, x, kind='linear')
    # rhs_fine = np.arange(rhs.min(), rhs.max(), 1)
    # x_fine = f_interp(rhs_fine)
    # derivative_fine = np.gradient(x_fine, rhs_fine)
    # axes[2, i].plot(derivative_fine, rhs_fine/10)
    # axes[2, i].set_ylim(-10, max_rh)
    # axes[0, i].set_title(f'point {i}')
plt.show()