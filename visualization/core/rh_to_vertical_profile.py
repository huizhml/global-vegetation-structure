import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.signal import savgol_filter

# --- Data ---
rh = np.array([-6.80,-5.41,-4.40,-3.28,-1.86,-0.78,0.14,1.15,2.05,2.69,
    3.17,3.73,4.40,5.04,5.64,6.24,6.83,7.43,8.03,8.70,
    9.60,10.46,11.24,12.03,12.74,13.34,13.82,14.27,14.64,14.98,
    15.28,15.58,15.84,16.10,16.36,16.59,16.81,17.00,17.19,17.37,
    17.52,17.71,17.86,18.01,18.16,18.31,18.42,18.57,18.72,18.87,
    19.02,19.17,19.35,19.50,19.65,19.80,19.99,20.14,20.32,20.51,
    20.66,20.85,21.07,21.26,21.45,21.67,21.86,22.08,22.27,22.45,
    22.68,22.87,23.05,23.28,23.46,23.65,23.84,24.02,24.21,24.36,
    24.55,24.73,24.88,25.07,25.26,25.41,25.59,25.78,25.97,26.15,
    26.38,26.57,26.83,27.09,27.39,27.69,28.02,28.43,28.92,29.52,
    30.68])

percentiles = np.arange(0, 101)

# Derive vertical profile
bin_size = 0.4
h_min, h_max = -8, 32
bin_centers = np.arange(h_min + bin_size/2, h_max, bin_size)

bin_amp = np.zeros(len(bin_centers))
for b, hc in enumerate(bin_centers):
    for i in range(len(rh) - 1):
        if hc >= rh[i] and hc < rh[i + 1]:
            dh = rh[i + 1] - rh[i]
            bin_amp[b] = 1.0 / dh if dh > 0.01 else 0
            break

bin_amp_pct = bin_amp / bin_amp.sum() * 100
smoothed_pct = savgol_filter(bin_amp_pct, window_length=3, polyorder=1)

# --- RH markers ---
rh_markers = [
    (0,   '#854F0B', 'RH0'),
    (25,  '#1D9E75', 'RH25'),
    (50,  '#378ADD', 'RH50'),
    (75,  '#7F77DD', 'RH75'),
    (98,  '#D85A30', 'RH98'),
    (100, '#E24B4A', 'RH100'),
]

# --- Figure: wider first two panels ---
fig = plt.figure(figsize=(12, 4))
gs = fig.add_gridspec(1, 3, width_ratios=[3.5, 3.5, 1.6], wspace=0.12)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

ylim = (h_min, h_max)

# ============ Panel 1: RH Curve ============
ax1.plot(percentiles, rh, color='#378ADD', linewidth=2, zorder=3)

for p, color, label in rh_markers:
    ax1.plot(p, rh[p], 'o', color=color, markersize=5, zorder=4)
    ax1.axhline(y=rh[p], color=color, linewidth=0.5, linestyle='--', alpha=0.4, zorder=1)

ax1.set_xlabel('RH0–100', fontsize=12)
ax1.set_ylabel('Height (m)', fontsize=12)
ax1.set_title('RH curve', fontsize=12, pad=8)
ax1.set_xlim(0, 100)
ax1.set_ylim(ylim)
ax1.set_xticks([0, 25, 50, 75, 100])
ax1.set_yticks(np.arange(-5, 35, 5))
ax1.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=9)

# ============ Panel 2: Vertical Profile ============
ax2.barh(bin_centers, bin_amp_pct, height=bin_size, color='#27500A', alpha=0.5, zorder=2)
ax2.plot(smoothed_pct, bin_centers, color='#D85A30', linewidth=2, alpha=0.85, zorder=3)

for p, color, label in rh_markers:
    ax2.axhline(y=rh[p], color=color, linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)

ax2.set_xlabel('Energy (%)', fontsize=12)
ax2.set_title('Vertical profile', fontsize=12, pad=8)
ax2.set_ylim(ylim)
ax2.set_yticks(np.arange(-5, 35, 5))
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=9)

# Legend — moved up from lower right
legend_elements = [
    Patch(facecolor='#27500A', alpha=0.5, label='Binned'),
    Line2D([0], [0], color='#D85A30', linewidth=2, alpha=0.85, label='Smoothed'),
]
ax2.legend(handles=legend_elements, fontsize=8, loc='lower right',
           framealpha=0.85, edgecolor='#ccc',
           bbox_to_anchor=(0.98, 0.08))

xmax = max(bin_amp_pct) * 1.15
ax2.set_xlim(0, xmax)

# RH labels
label_configs = [
    # (0,   '#854F0B', 'RH0 = {:.1f} m',   -1.5),
    (25,  '#1D9E75', 'RH25 = {:.1f} m',   0.6),
    (50,  '#378ADD', 'RH50 = {:.1f} m',    0.6),
    (75,  '#7F77DD', 'RH75 = {:.1f} m',    0.6),
    (98,  '#D85A30', 'RH98 = {:.1f} m',   -1.2),
    (100, '#E24B4A', 'RH100 = {:.1f} m',   0.6),
]

for p, color, fmt, yoff in label_configs:
    ax2.text(xmax*1.1, rh[p] + yoff, fmt.format(rh[p]),
             fontsize=6.5, fontweight='bold', color=color,
             ha='right', va='center', zorder=6,
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.75))

# ============ Panel 3: Forest Cross-Section ============
ax3.set_xlim(0, 10)
ax3.set_ylim(ylim)
ax3.set_title('Forest structure', fontsize=11, pad=8)
ax3.set_xticks([])
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.tick_params(left=False, bottom=False)
for spine in ax3.spines.values():
    spine.set_visible(False)

# Ground
ax3.axhline(y=0, color='#8B7355', linewidth=3, alpha=0.5, zorder=1)

# RH dashed lines
for p, color, label in rh_markers:
    ax3.axhline(y=rh[p], color=color, linewidth=0.4, linestyle='--', alpha=0.3, zorder=1)

TRUNK_COLOR = '#3D2B1F'

def draw_tree(ax, x, canopy_top, crown_rx=1.2, crown_ry=3.0):
    """Draw tree with trunk connecting to crown bottom."""
    # Crown center and bottom
    crown_cy = canopy_top - crown_ry  # center so top of crown = canopy_top
    crown_bottom = crown_cy - crown_ry

    # Trunk goes from ground (0) to bottom of crown
    lw = max(2.0, canopy_top / 10)
    ax.plot([x, x], [0, crown_bottom + crown_ry * 0.3], color=TRUNK_COLOR, linewidth=lw,
            alpha=0.6, zorder=2, solid_capstyle='round')

    # Branches near crown base
    bh = crown_bottom + crown_ry * 0.4
    ax.plot([x, x - 0.5], [bh - 1.5, bh], color=TRUNK_COLOR, linewidth=0.9, alpha=0.35, zorder=2)
    ax.plot([x, x + 0.5], [bh - 0.8, bh + 0.5], color=TRUNK_COLOR, linewidth=0.9, alpha=0.35, zorder=2)

    # Crown layers
    layers = [('#1a4d0a', 0.55, 0, 0, 1.0),
              ('#27500A', 0.42, -0.2, 0.3, 0.82),
              ('#3B6D11', 0.35, 0.25, -0.25, 0.65)]
    for cc, ca, ox, oy, sc in layers:
        e = Ellipse((x + ox, crown_cy + oy), crown_rx * 2 * sc, crown_ry * 2 * sc,
                    facecolor=cc, edgecolor='none', alpha=ca, zorder=3)
        ax.add_patch(e)

# RH98 = 28.92m — tallest tree touches this
draw_tree(ax3, 3.5, rh[98], crown_rx=1.6, crown_ry=4.0)   # Emergent, top at RH98
draw_tree(ax3, 7.0, 25.0, crown_rx=1.3, crown_ry=3.2)     # Tall
draw_tree(ax3, 1.5, 19.0, crown_rx=1.1, crown_ry=2.3)     # Medium
draw_tree(ax3, 5.5, 13.0, crown_rx=0.9, crown_ry=1.8)     # Small
draw_tree(ax3, 8.5, 14.0, crown_rx=0.8, crown_ry=1.6)     # Small

# Understory
for sx in [2.5, 4.5, 6.5, 8.0]:
    e = Ellipse((sx, 2.5), 1.0, 2.0, facecolor='#97C459', edgecolor='none', alpha=0.22, zorder=2)
    ax3.add_patch(e)

# --- Save ---
fig.savefig('~/gvsm/results/illustrations/gedi_rh_vertical_profile.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig('~/gvsm/results/illustrations/gedi_rh_vertical_profile.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Done!")