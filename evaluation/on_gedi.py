import re
from typing import List
from concurrent.futures import ThreadPoolExecutor
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from const import FONT_SIZES, FIGURE_SIZES, set_plot_fonts, fewer_ticks

set_plot_fonts(label=20, title=20, annot=20)

# -------------------------------------------------------------
#  Plot functions
# -------------------------------------------------------------
def compute_group_stats(ddf, ours_cols, gedi_cols, group_size=5,
                        n_violin_samples=1_000_000, n_workers=4, seed=0):
    """Per group of `group_size` consecutive RH columns, pool the residuals
    and return exact boxplot stats + a random subsample of raw ours/GEDI
    values for the violin plot.

    Memory: each group materializes 2*group_size columns to pandas. With
    50M rows and group_size=5 that's ~4 GB / group; n_workers caps concurrency.
    Violin KDE is O(n*m); n_violin_samples bounds it to something tractable.
    """
    n = len(ours_cols)
    starts = list(range(0, n, group_size))
    # If the final bin would have fewer than group_size cols (e.g. just RH100
    # when n=101 and group_size=5), merge it into the previous group instead.
    if len(starts) > 1 and n - starts[-1] < group_size:
        starts.pop()
    groups = []
    for j, start in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else n
        groups.append((
            ours_cols[start:end],
            gedi_cols[start:end],
            f'{start}-{end - 1}',
        ))

    rng = np.random.default_rng(seed)

    def _one_group(g):
        ours_g, gedi_g, label = g
        ours_arr = ddf[ours_g].compute().to_numpy().ravel()
        gedi_arr = ddf[gedi_g].compute().to_numpy().ravel()
        residuals = ours_arr - gedi_arr
        mask = np.isfinite(residuals)
        residuals = residuals[mask]
        ours_arr = ours_arr[mask]
        gedi_arr = gedi_arr[mask]

        std = residuals.std()
        n_total = len(residuals)
        if n_violin_samples and n_total > n_violin_samples:
            idx = rng.choice(n_total, n_violin_samples, replace=False)
            ours_arr = ours_arr[idx]
            gedi_arr = gedi_arr[idx]
            residuals = residuals[idx]
        return dict(
            label=label,
            std=std,
            ours_sample=ours_arr,
            gedi_sample=gedi_arr,
            residual_sample=residuals,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_one_group, groups))
    return results


def make_residual_plot(group_results, save_path, normalize=False, **kwargs):
    labels = [r['label'] for r in group_results]
    residual_arrs = [r['residual_sample'] for r in group_results]
    if normalize:
        residual_arrs = [a / r['std'] for a, r in zip(residual_arrs, group_results)]
    df = pd.DataFrame({
        'rh_group': np.concatenate(
            [np.repeat(l, len(a)) for l, a in zip(labels, residual_arrs)]),
        'residual': np.concatenate(residual_arrs),
    })
    figsize = kwargs.get('figsize', FIGURE_SIZES['strip'])
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color='red', linewidth=1)

    sns.boxplot(data=df, x='rh_group', y='residual', order=labels,
                showfliers=False, showmeans=True,
                width=0.5, linewidth=2,
                boxprops=dict(facecolor='C0', edgecolor='black'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black'),
                ax=ax)
    ax.set_xlabel("Relative Height (0-100)", fontsize=FONT_SIZES['label'])
    ax.set_ylabel("Residuals / σ" if normalize else "Residuals (m)",
                  fontsize=FONT_SIZES['label'])
    ax.set_title("Residuals of VSM on GEDI", fontsize=FONT_SIZES['title'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    fewer_ticks(ax, axis='y', nbins=5)
    plt.setp(ax.get_xticklabels(), ha='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def make_violin_plot(group_results, save_path, **kwargs):
    # Long-format frame so seaborn can do a true split violin
    # (left = VSM, right = GEDI per RH bin).
    ours_arrs = [r['ours_sample'] for r in group_results]
    gedi_arrs = [r['gedi_sample'] for r in group_results]
    n_ours = sum(map(len, ours_arrs))
    n_gedi = sum(map(len, gedi_arrs))
    df = pd.DataFrame({
        'rh_group': np.concatenate(
            [np.repeat(r['label'], len(a)) for r, a in zip(group_results, ours_arrs)]
            + [np.repeat(r['label'], len(a)) for r, a in zip(group_results, gedi_arrs)]
        ),
        'height': np.concatenate(ours_arrs + gedi_arrs),
        'source': np.concatenate([np.full(n_ours, 'VSM'), np.full(n_gedi, 'GEDI')]),
    })
    order = [r['label'] for r in group_results]
    figsize = kwargs.get('figsize', FIGURE_SIZES['wide'])
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df, x='rh_group', y='height', hue='source',
                   split=True, inner='quartile', order=order,
                   palette={'VSM': 'C0', 'GEDI': 'C1'}, ax=ax)
    # symlog (not log) because heights can dip slightly negative; linthresh=1
    # keeps the near-zero region linear so the bulk of the distribution stays
    # readable.
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel("Relative Height (0-100)", fontsize=FONT_SIZES['label'])
    ax.set_ylabel("Height (m)", fontsize=FONT_SIZES['label'])
    ax.set_title("RH distributions: VSM vs GEDI", fontsize=FONT_SIZES['title'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.get_legend().set_title('')
    plt.setp(ax.get_xticklabels(), ha='center'), #rotation=45,
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------------------------------------------------------------
#  Runable
# -------------------------------------------------------------

def evaluate_vsm_on_gedi(
        ref_and_ours_dir: str = None,
        save_dir: str = None,
        slope_lt20: bool = True,
        group_size: int=5,
        **kwargs):
    '''
    Evaluate the VSM performance on GEDI
    '''
    ref_and_ours_dir = Path(ref_and_ours_dir).expanduser()
    files = list(ref_and_ours_dir.glob('*.parquet'))#[:10]
    ours_rh_cols = [f'RH{i}_Q1' for i in range(101)]
    gedi_rh_cols = [f'rh{i}' for i in range(101)]
    # Processes-based local cluster: numeric reductions over 101 columns hit
    # the GIL hard under the default threaded scheduler.
    with Client() as client:
        print(client.dashboard_link)
        # Read ONLY the columns the residuals need. dgp.read_parquet would
        # decode WKB geometry into shapely objects per row (millions, unused)
        # and load every column. With one 2.8 GB file, split_row_groups gives
        # parallel partitions. Pushing the slope cut into `filters` prunes row
        # groups via parquet stats AND row-filters internally, so 'slope'
        # never lands in the output frame.
        filters = [('slope', '<', 20)] if slope_lt20 else None
        ddf = dd.read_parquet(files, columns=ours_rh_cols + gedi_rh_cols,
                              split_row_groups=True, filters=filters,
                              dataset={"partitioning": None})
        print(f'npartitions: {ddf.npartitions}')
        # Pool RHs into groups of 5 inside compute_group_stats; it also
        # subsamples ours/GEDI for the paired violin plot.
        group_results = compute_group_stats(ddf, ours_rh_cols, gedi_rh_cols,
                                            group_size=group_size)
    suffix = '_slope_lt20' if slope_lt20 else ''

    save_dir = Path(f'{save_dir}').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    make_residual_plot(group_results, save_dir / f'residual_plot_group{group_size}{suffix}.pdf', **kwargs)
    make_residual_plot(group_results, save_dir / f'residual_plot_group{group_size}_normalized{suffix}.pdf',
                       normalize=True, **kwargs)
    make_violin_plot(group_results, save_dir / f'violin_plot_group{group_size}{suffix}.pdf', **kwargs)
    print('plots saved to:', save_dir)
    
    
    
# ============================================================================
# Hydra entrypoint
# ============================================================================
from pathlib import Path
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='on_gedi',
    default_run='evaluate_vsm_on_gedi',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()