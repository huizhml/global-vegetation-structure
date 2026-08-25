from pathlib import Path
import pandas as pd
import numpy as np
import dask.dataframe as dd

from evaluation.significance import (run_test, bootstrap_metric_diff,
                                     holm_correct, format_p)

# Each SOTA product is compared against our prediction of the RH level it
# actually estimates, so the paired difference is between two estimates of the
# same quantity at the same GEDI footprint.
SOTA_PAIRS = {
    'UMD':  {'ref': 'rh95',  'sota': 'RH95_UMD',  'ours': 'RH95_Q1_raw'},
    'ETH':  {'ref': 'rh98',  'sota': 'RH98_ETH',  'ours': 'RH98_Q1_raw'},
    'UM':   {'ref': 'rh100', 'sota': 'RH100_UM',  'ours': 'RH100_Q1_raw'},
    'META': {'ref': 'rh95',  'sota': 'RH95_META', 'ours': 'RH95_Q1_raw'},
}


def wilcoxon_chm_vs_sota(ours_dir: str, save_dir: str, data_name: str = None,
                         split: str = 'test', year: int = 2020,
                         slope_lt20: bool = False, sota_pairs: dict = None,
                         error_metrics: tuple = ('abs_error', 'sq_error'),
                         test: str = 'signed_rank',
                         zero_method: str = 'wilcox', max_n: int = None,
                         seed: int = 0, **kwargs):
    '''
    Two-sided Wilcoxon test of our CHM against each SOTA canopy-height map, on
    the GEDI footprints where both products are valid.

    For every (product, RH level) pair the per-footprint error of our map and of
    the SOTA map are computed against the same GEDI reference RH, then tested
    pairwise. A significantly negative shift in absolute error means our map is
    closer to GEDI at the same locations — the paired statement behind the RMSE
    / MAE table written by `evaluate_chm_with_sota`, which reports aggregates
    only and so carries no significance.

    Only footprints where the SOTA product, our product and the reference RH are
    all present enter a given comparison (pairwise, per product), so a product
    with sparser coverage never penalises the others.

    Args:
        ours_dir: directory of per-footprint parquets holding both products
            (same input as `evaluate_chm_with_sota`)
        save_dir: where the CSV / LaTeX table is written
        slope_lt20: restrict to footprints on slopes < 20 degrees
        sota_pairs: {product: {ref, sota, ours}} column mapping; defaults to
            SOTA_PAIRS
        error_metrics: which per-footprint errors to test — 'abs_error'
            (|pred - ref|, drives MAE), 'sq_error' ((pred - ref)^2, drives
            RMSE), 'signed_error' (pred - ref, i.e. a bias comparison)
        test: 'signed_rank' for the paired Wilcoxon signed-rank test (the two
            products are scored at the same footprints, so the observations are
            matched), or 'rank_sum' for the unpaired Wilcoxon rank-sum /
            Mann-Whitney U test, which treats the two error distributions as
            independent samples
        zero_method: 'wilcox' (drop zero differences) or 'pratt'; signed_rank only
        max_n: optionally subsample this many footprints per comparison
    Returns:
        DataFrame with one row per (product, RH level, error metric)
    '''
    sota_pairs = sota_pairs or SOTA_PAIRS
    ours_dir = Path(ours_dir.format(data_name=data_name, split=split, year=year)).expanduser()
    save_dir = Path(f'{save_dir}').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # The test goes in the filename so a rank_sum run never clobbers a
    # signed_rank one (and the two stay directly comparable side by side).
    out_file = save_dir / f'chm_with_sota_{test}_{split}_{year}.csv'
    if slope_lt20:
        out_file = out_file.with_suffix('.slope_lt20.csv')

    files = list(ours_dir.glob('*.parquet'))
    cols = sorted({c for p in sota_pairs.values() for c in p.values()} | {'slope'})
    ddf = dd.read_parquet(files, columns=cols)
    if slope_lt20:
        ddf = ddf[ddf['slope'] < 20]
    # No global dropna here: unlike evaluate_chm_with_sota (which needs one
    # common footprint set so its RMSE columns stay comparable), each paired
    # test is self-contained and only needs its own three columns.
    df = ddf.compute()
    print(f'loaded {len(df)} footprints from {len(files)} files in {ours_dir}')

    rows = []
    for product, c in sota_pairs.items():
        sub = df[[c['ref'], c['sota'], c['ours']]].dropna()
        if sub.empty:
            print(f'  skipping {product}: no footprint has all of {list(c.values())}')
            continue
        ref = sub[c['ref']].to_numpy()
        err_ours = sub[c['ours']].to_numpy() - ref
        err_sota = sub[c['sota']].to_numpy() - ref
        for metric in error_metrics:
            if metric == 'abs_error':
                a, b = np.abs(err_ours), np.abs(err_sota)
            elif metric == 'sq_error':
                a, b = err_ours ** 2, err_sota ** 2
            elif metric == 'signed_error':
                a, b = err_ours, err_sota
            else:
                raise ValueError(f'unknown error metric {metric!r}')
            rows.append(run_test(
                a, b, test=test, name_a='Ours', name_b=product,
                alternative='two-sided', zero_method=zero_method,
                max_n=max_n, seed=seed,
                # For an error, smaller wins — so `win_rate` / `a_better` must
                # be read in the opposite direction to a score.
                higher_is_better=(metric == 'signed_error'),
                product=product, rh_level=c['ref'].upper(),
                error_metric=metric,
            ))
            r = rows[-1]
            n = r['n_pairs'] if test == 'signed_rank' else f"{r['n_a']}/{r['n_b']}"
            print(f"  {product} {c['ref'].upper()} {metric}: n={n}, "
                  f"median diff={r['median_diff']:.3f}, "
                  f"HL={r['hodges_lehmann']:.3f}, r={r['rank_biserial']:.3f}, "
                  f"p={format_p(r['p_value'])}")

    results = pd.DataFrame(rows)
    if results.empty:
        print('no comparison could be run; nothing written')
        return results
    # One Holm family per error metric: the four products are the comparisons
    # being made simultaneously for a given claim.
    results = holm_correct(results, within=['error_metric'])
    results['p_value_str'] = results['p_value'].map(format_p)
    results['p_holm_str'] = results['p_holm'].map(format_p)
    results.to_csv(out_file, index=False)

    tex_cols = ['product', 'rh_level', 'error_metric', 'n_pairs', 'median_Ours',
                'median_diff', 'hodges_lehmann', 'rank_biserial', 'win_rate',
                'p_value_str', 'p_holm_str']
    results[[c for c in tex_cols if c in results]].to_latex(
        out_file.with_suffix('.tex'), index=False, float_format='%.3f', na_rep='')
    print(f'wrote {len(results)} paired tests to {out_file}')
    return results


def bootstrap_chm_vs_sota(ours_dir: str, save_dir: str, data_name: str = None,
                          split: str = 'test', year: int = 2020,
                          slope_lt20: bool = False, sota_pairs: dict = None,
                          metrics: tuple = ('RMSE', 'MAE', 'ME', 'absME'),
                          block_col: str = None, n_boot: int = 1000,
                          ci: float = 95, seed: int = 0, **kwargs):
    '''
    Bootstrap confidence intervals for the RMSE / MAE / ME / |ME| difference
    between our CHM and each SOTA canopy-height map.

    Complements `wilcoxon_chm_vs_sota`, which is median-based and therefore
    cannot speak to RMSE or ME — both are means and are driven by the tails a
    rank test deliberately ignores. Where a product is tight in the centre but
    heavy-tailed the two disagree, and only this function tests the aggregate
    that the RMSE / ME table actually reports.

    Footprints enter a comparison only where the SOTA product, our product and
    the reference RH are all present (pairwise per product), matching
    `wilcoxon_chm_vs_sota`.

    Args:
        ours_dir: directory of per-footprint parquets (same input as
            `evaluate_chm_with_sota`)
        save_dir: where the CSV / LaTeX table is written
        metrics: which aggregates to bootstrap. 'ME' is signed bias (its sign
            says which way a product leans); 'absME' is its magnitude, which is
            the one to cite for "less biased".
        block_col: column to block the resampling on (an S2 tile id, a region
            code, ...). Leave None for an i.i.d. bootstrap over footprints —
            but note that neighbouring footprints are spatially correlated, so
            an i.i.d. interval is optimistically narrow. Set this if the
            parquets carry a suitable grouping column.
        n_boot: bootstrap resamples (1000 gives a stable 95% interval)
        ci: interval width in percent
    Returns:
        DataFrame with one row per (product, RH level, metric)
    '''
    sota_pairs = sota_pairs or SOTA_PAIRS
    ours_dir = Path(ours_dir.format(data_name=data_name, split=split, year=year)).expanduser()
    save_dir = Path(f'{save_dir}').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    out_file = save_dir / f'chm_with_sota_bootstrap_{split}_{year}.csv'
    if slope_lt20:
        out_file = out_file.with_suffix('.slope_lt20.csv')

    files = list(ours_dir.glob('*.parquet'))
    cols = sorted({c for p in sota_pairs.values() for c in p.values()} | {'slope'}
                  | ({block_col} if block_col else set()))
    ddf = dd.read_parquet(files, columns=cols)
    if slope_lt20:
        ddf = ddf[ddf['slope'] < 20]
    df = ddf.compute()
    print(f'loaded {len(df)} footprints from {len(files)} files in {ours_dir}')
    if block_col:
        print(f'block bootstrap on {df[block_col].nunique()} distinct {block_col} values')

    rows = []
    for product, c in sota_pairs.items():
        keep = [c['ref'], c['sota'], c['ours']] + ([block_col] if block_col else [])
        sub = df[keep].dropna()
        if sub.empty:
            print(f'  skipping {product}: no footprint has all of {list(c.values())}')
            continue
        ref = sub[c['ref']].to_numpy()
        rows += bootstrap_metric_diff(
            sub[c['ours']].to_numpy() - ref, sub[c['sota']].to_numpy() - ref,
            name_a='Ours', name_b=product, metrics=metrics,
            block=sub[block_col].to_numpy() if block_col else None,
            n_boot=n_boot, ci=ci, seed=seed,
            product=product, rh_level=c['ref'].upper(),
        )
        for r in rows[-len(metrics):]:
            print(f"  {product} {c['ref'].upper()} {r['metric']}: "
                  f"diff={r['diff']:+.3f} "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                  f"{'significant' if r['ci_excludes_zero'] else 'n.s.'}")

    results = pd.DataFrame(rows)
    if results.empty:
        print('no comparison could be run; nothing written')
        return results
    results = holm_correct(results, within=['metric'])
    results['p_value_str'] = results['p_value'].map(format_p)
    results.to_csv(out_file, index=False)

    tex_cols = ['product', 'rh_level', 'metric', 'n_obs', 'diff', 'ci_lo',
                'ci_hi', 'ci_excludes_zero', 'p_value_str']
    results[[c for c in tex_cols if c in results]].to_latex(
        out_file.with_suffix('.tex'), index=False, float_format='%.3f', na_rep='')
    print(f'wrote {len(results)} bootstrap comparisons to {out_file}')
    return results


def evaluate_chm_with_sota(ours_dir: str, save_dir: str, data_name: str =None, split: str = 'test', year: int = 2020, slope_lt20: bool = False, **kwargs):
    '''
    Evaluate the CHM performance with SOTA CHM
    '''
    ours_dir = ours_dir.format(data_name=data_name, split=split, year=year)
    ours_dir = Path(f'{ours_dir}').expanduser()
    
    save_dir = Path(f'{save_dir}').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f'chm_with_sota_performance_{split}_{year}.csv'
    files = list(ours_dir.glob('*.parquet'))
    ref = ['rh95', 'rh98', 'rh100']
    products = {
        'UMD': ['RH95_UMD', None, None], 
        'ETH': [None, 'RH98_ETH', None], 
        'UM': [None, None, 'RH100_UM'], 
        'META': ['RH95_META', None, None], 
        'Ours': ['RH95_Q1_raw', 'RH98_Q1_raw', 'RH100_Q1_raw'],
    }
    product_cols = np.unique([col for sublist in products.values() for col in sublist if col is not None])
    cols = product_cols.tolist() + ref + ['slope']
    ddf = dd.read_parquet(files, columns=cols)
    if slope_lt20:
        ddf = ddf[ddf['slope'] < 20]
        out_file = out_file.with_suffix('.slope_lt20.csv')
    
    ddf = ddf.dropna(subset=product_cols)
    ddf = ddf.compute()
    me = {}
    mae = {}
    rmse = {}

    for product, cols in products.items():
        me[product] = np.full(3, np.nan)
        mae[product] = np.full(3, np.nan)
        rmse[product] = np.full(3, np.nan)
        for i, col in enumerate(cols):
            if col is not None:
                residuals = ddf[col] - ddf[ref[i]]
                me[product][i] = residuals.mean()
                mae[product][i] = np.abs(residuals).mean()
                rmse[product][i] = (residuals**2).mean()**0.5
    index = ['RH95', 'RH98', 'RH100']
    df_me = pd.DataFrame(me, index=index).T
    df_mae = pd.DataFrame(mae, index=index).T
    df_rmse = pd.DataFrame(rmse, index=index).T
    df = pd.concat([df_rmse, df_mae, df_me], keys=['RMSE', 'MAE', 'ME'], axis=1)
    df.to_csv(out_file)
    df.to_latex(out_file.with_suffix('.tex'), float_format=f"%.2f", na_rep='')


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='on_sota_chm',
    default_run='evaluate_chm_with_sota',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()
