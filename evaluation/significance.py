'''
Paired significance tests shared by the evaluation ops.

Everything here is *paired*: the two competing products / models are scored on
exactly the same observations (same GEDI footprint, same reference sample, same
grid cell), so the comparison is a one-sample test on the within-pair
differences. That is the Wilcoxon **signed-rank** test — the paired counterpart
of the (unpaired) Wilcoxon rank-sum / Mann-Whitney U test. `paired_wilcoxon`
therefore calls `scipy.stats.wilcoxon`, not `ranksums`; `ranksums` would throw
away the pairing and badly overstate the variance here.

Each test returns a flat dict so callers can `pd.DataFrame(rows)` a tidy CSV.
'''

import numpy as np
import pandas as pd
from scipy import stats


# The p-values below come out astronomically small on millions of paired
# footprints; formatting them as "< 1e-300" beats printing a hard 0.0.
P_FLOOR = 1e-300


def _rank_biserial(d: np.ndarray) -> float:
    '''
    Matched-pairs rank-biserial correlation, the effect size that goes with the
    signed-rank test: (W+ - W-) / (W+ + W-), i.e. the normalised difference
    between the rank mass of positive and negative differences. Ranges from -1
    (every pair favours b) to +1 (every pair favours a); 0 = no shift. Zeros are
    dropped first, matching zero_method='wilcox'.
    '''
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    return (w_plus - w_minus) / (w_plus + w_minus)


def _hodges_lehmann(d: np.ndarray, max_exact: int = 3000,
                    n_pairs: int = 2_000_000, seed: int = 0) -> float:
    '''
    Hodges-Lehmann pseudomedian: the median of all Walsh averages
    (d_i + d_j) / 2, i <= j. This — not the median difference — is the location
    shift the signed-rank test actually tests for, so it is the estimate to
    quote next to the p-value.

    Exact for n <= max_exact (O(n^2) Walsh averages); above that it is estimated
    from `n_pairs` randomly drawn (i, j) pairs, which is an unbiased sample of
    the same population and converges to well under 0.001 of the exact value.
    '''
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return np.nan
    if n <= max_exact:
        i, j = np.triu_indices(n, k=0)
        return float(np.median((d[i] + d[j]) / 2))
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    return float(np.median((d[i] + d[j]) / 2))


def _hodges_lehmann_2samp(a: np.ndarray, b: np.ndarray, max_exact: int = 2000,
                          n_pairs: int = 2_000_000, seed: int = 0) -> float:
    '''
    Two-sample Hodges-Lehmann estimator: the median of all cross-sample
    differences a_i - b_j. This is the location shift the rank-sum test
    estimates, the unpaired counterpart of the Walsh-average pseudomedian.
    Exact for small samples, sampled above that (see `_hodges_lehmann`).
    '''
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    if len(a) <= max_exact and len(b) <= max_exact:
        return float(np.median(a[:, None] - b[None, :]))
    rng = np.random.default_rng(seed)
    return float(np.median(a[rng.integers(0, len(a), n_pairs)]
                           - b[rng.integers(0, len(b), n_pairs)]))


def unpaired_ranksum(a, b, name_a: str = 'a', name_b: str = 'b',
                     alternative: str = 'two-sided',
                     max_n: int = None, seed: int = 0,
                     higher_is_better: bool = True, **extra) -> dict:
    '''
    Two-sided Wilcoxon rank-sum (Mann-Whitney U) test on two INDEPENDENT
    samples.

    Use this when the two score sets are not matched observation-by-observation
    — comparing groups of different size, or deliberately treating the two
    products as independent distributions. When the observations *are* matched
    (same footprint, same reference sample), `paired_wilcoxon` is the more
    powerful choice: rank-sum discards the pairing, so it answers "do the two
    score distributions differ" rather than "does the score improve at a given
    location", and it needs a much larger shift to reach the same p-value.

    Args:
        a, b: score arrays; need not be the same length, and are NOT assumed to
            be aligned. NaNs are dropped within each sample independently.
        higher_is_better: orientation of the score, used for `a_better` and to
            phrase `win_rate` (here the probability that a random draw from `a`
            beats a random draw from `b`, i.e. the common-language effect size).
    Returns:
        dict with the same keys as `paired_wilcoxon` where they mean the same
        thing, so both tests can share one tidy CSV.
    '''
    a = np.asarray(a, dtype='float64')
    b = np.asarray(b, dtype='float64')
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if max_n is not None:
        rng = np.random.default_rng(seed)
        if len(a) > max_n:
            a = a[rng.choice(len(a), max_n, replace=False)]
        if len(b) > max_n:
            b = b[rng.choice(len(b), max_n, replace=False)]

    n_a, n_b = len(a), len(b)
    row = {
        'comparison': f'{name_a} vs {name_b}',
        'model': name_a, 'baseline': name_b,
        'test': 'wilcoxon_rank_sum', 'alternative': alternative,
        'n_a': n_a, 'n_b': n_b, 'n_pairs': np.nan,
        f'median_{name_a}': float(np.median(a)) if n_a else np.nan,
        f'median_{name_b}': float(np.median(b)) if n_b else np.nan,
        **extra,
    }
    if n_a == 0 or n_b == 0:
        row.update({'median_diff': np.nan, 'hodges_lehmann': np.nan,
                    'statistic': np.nan, 'p_value': np.nan,
                    'rank_biserial': np.nan, 'win_rate': np.nan,
                    'a_better': False})
        return row

    res = stats.mannwhitneyu(a, b, alternative=alternative)
    # Common-language effect size: P(a > b) + 0.5 P(a = b) = U / (n_a * n_b).
    # The rank-biserial correlation is its linear rescaling onto [-1, 1], so the
    # column means the same thing as in the paired rows.
    p_sup = res.statistic / (n_a * n_b)
    shift = _hodges_lehmann_2samp(a, b, seed=seed)
    row.update({
        # No within-pair difference exists, so `median_diff` is the difference
        # of the two medians rather than the median of differences.
        'median_diff': row[f'median_{name_a}'] - row[f'median_{name_b}'],
        'hodges_lehmann': shift,
        'statistic': float(res.statistic),
        'p_value': float(res.pvalue),
        'rank_biserial': 2 * p_sup - 1,
        'win_rate': p_sup if higher_is_better else 1 - p_sup,
        'a_better': bool(shift > 0) == bool(higher_is_better),
    })
    return row


def run_test(a, b, test: str = 'signed_rank', **kwargs) -> dict:
    '''
    Dispatch to `paired_wilcoxon` ('signed_rank') or `unpaired_ranksum`
    ('rank_sum'), so an op can expose the choice as one config field.
    '''
    if test in ('signed_rank', 'paired', 'wilcoxon'):
        return paired_wilcoxon(a, b, **kwargs)
    if test in ('rank_sum', 'ranksum', 'mannwhitney', 'unpaired'):
        kwargs.pop('zero_method', None)      # paired-only argument
        return unpaired_ranksum(a, b, **kwargs)
    raise ValueError(f"unknown test {test!r}; use 'signed_rank' or 'rank_sum'")


def paired_wilcoxon(a, b, name_a: str = 'a', name_b: str = 'b',
                    alternative: str = 'two-sided',
                    zero_method: str = 'wilcox',
                    max_n: int = None, seed: int = 0,
                    higher_is_better: bool = True, **extra) -> dict:
    '''
    Two-sided paired Wilcoxon signed-rank test on `a` - `b`.

    Pairs with a NaN on either side are dropped pairwise, so `a` and `b` must be
    aligned element-by-element (same footprint / sample / cell in both).

    Args:
        a, b: paired 1-D score arrays. `a` is the model under test, `b` the
            baseline, so a positive difference means `a` wins when
            `higher_is_better` (e.g. recall) and loses when it is False
            (e.g. absolute error).
        name_a, name_b: labels carried into the output row.
        alternative: passed to scipy; 'two-sided' as required for the paper.
        zero_method: 'wilcox' drops zero differences (the classic definition);
            'pratt' ranks them and then drops them, which is more conservative
            and preferable when exact ties are common (per-cell recall of 1.0
            for both models, for instance). n_zero is reported either way.
        max_n: optionally subsample to this many pairs (seeded) before testing.
            Only for tractability on very large tables; leave None to use all.
        higher_is_better: orientation of the score, used to fill in the
            `a_better` flag and to phrase `win_rate`.
        extra: any extra key/values to merge into the returned row (test name,
            class, metric, ...).
    Returns:
        dict with n, n_zero, medians, mean/median difference, Hodges-Lehmann
        pseudomedian, W statistic, p-value, rank-biserial effect size, the
        fraction of pairs won, and whichever `extra` keys were passed.
    '''
    a = np.asarray(a, dtype='float64')
    b = np.asarray(b, dtype='float64')
    if a.shape != b.shape:
        raise ValueError(f'paired arrays must align: {a.shape} vs {b.shape}')

    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if max_n is not None and len(a) > max_n:
        idx = np.random.default_rng(seed).choice(len(a), max_n, replace=False)
        a, b = a[idx], b[idx]

    d = a - b
    n, n_zero = len(d), int((d == 0).sum())
    row = {
        'comparison': f'{name_a} vs {name_b}',
        'model': name_a, 'baseline': name_b,
        'test': 'wilcoxon_signed_rank', 'alternative': alternative,
        'n_pairs': n, 'n_zero_diff': n_zero,
        f'median_{name_a}': float(np.median(a)) if n else np.nan,
        f'median_{name_b}': float(np.median(b)) if n else np.nan,
        'median_diff': float(np.median(d)) if n else np.nan,
        'mean_diff': float(d.mean()) if n else np.nan,
        **extra,
    }
    # All-zero differences (identical models) leave scipy nothing to rank.
    if n == 0 or n - n_zero == 0:
        row.update({'hodges_lehmann': 0.0 if n else np.nan, 'statistic': np.nan,
                    'p_value': np.nan, 'rank_biserial': np.nan,
                    'win_rate': np.nan, 'a_better': False})
        return row

    res = stats.wilcoxon(a, b, alternative=alternative, zero_method=zero_method)
    wins = (d > 0).sum() if higher_is_better else (d < 0).sum()
    row.update({
        'hodges_lehmann': _hodges_lehmann(d, seed=seed),
        'statistic': float(res.statistic),
        'p_value': float(res.pvalue),
        'rank_biserial': _rank_biserial(d),
        'win_rate': float(wins / (n - n_zero)),
        'a_better': bool(np.median(d) > 0) == bool(higher_is_better),
    })
    return row


def mcnemar_exact(correct_a, correct_b, name_a: str = 'a', name_b: str = 'b',
                  **extra) -> dict:
    '''
    Exact McNemar test on two boolean correctness vectors over the same samples.

    For a paired *binary* outcome (was this sample classified correctly?) the
    signed-rank test degenerates — every non-zero difference has magnitude 1, so
    the ranks carry no information. McNemar is the right paired test there: it
    conditions on the discordant pairs (b right / a wrong vs a right / b wrong)
    and asks whether the split is 50/50 under a two-sided exact binomial.
    Reported next to the Wilcoxon rows so the accuracy-style claims have a test
    whose assumptions actually hold.
    '''
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n01 = int((~a & b).sum())   # baseline right, model wrong
    n10 = int((a & ~b).sum())   # model right, baseline wrong
    disc = n01 + n10
    p = stats.binomtest(n10, disc, 0.5).pvalue if disc else np.nan
    return {
        'comparison': f'{name_a} vs {name_b}',
        'model': name_a, 'baseline': name_b,
        'test': 'mcnemar_exact', 'alternative': 'two-sided',
        'n_pairs': int(len(a)), 'n_discordant': disc,
        'n_model_only_correct': n10, 'n_baseline_only_correct': n01,
        f'accuracy_{name_a}': float(a.mean()) if len(a) else np.nan,
        f'accuracy_{name_b}': float(b.mean()) if len(b) else np.nan,
        'odds_ratio': (n10 / n01) if n01 else np.inf,
        'p_value': float(p),
        **extra,
    }


def _agg_metrics(n, sum_e, sum_abs, sum_sq) -> dict:
    '''RMSE / MAE / ME / |ME| from the four sufficient statistics of a set of
    signed errors. Keeping the statistics rather than the errors is what makes
    the block bootstrap cheap: a resample is just a sum over sampled blocks.'''
    with np.errstate(invalid='ignore', divide='ignore'):
        me = sum_e / n
        return {'RMSE': np.sqrt(sum_sq / n), 'MAE': sum_abs / n,
                'ME': me, 'absME': np.abs(me)}


def bootstrap_metric_diff(err_a, err_b, name_a: str = 'a', name_b: str = 'b',
                          metrics: tuple = ('RMSE', 'MAE', 'ME', 'absME'),
                          block=None, n_boot: int = 1000, ci: float = 95,
                          seed: int = 0, **extra) -> list[dict]:
    '''
    Bootstrap confidence intervals for the difference in aggregate error metrics
    (RMSE, MAE, ME, |ME|) between two products scored on the same observations.

    The Wilcoxon tests in this module are median-based and are therefore blind
    to the tails; RMSE and ME are means and live precisely in the tails. When
    the two disagree — as they do for a product that is tight in the centre but
    heavy-tailed — a signed-rank result cannot be cited in support of an RMSE or
    ME claim. This function tests those aggregates directly: resample the
    observations, recompute both products' metrics on each resample, and report
    the percentile interval of the difference.

    Args:
        err_a, err_b: per-observation SIGNED errors (pred - reference) for the
            two products, aligned element-by-element. Absolute and squared
            errors are derived internally so all metrics come from one pass.
        block: optional per-observation group label (S2 tile, 1-degree cell...).
            When given, whole blocks are resampled instead of individual
            observations, which keeps the interval honest under spatial
            autocorrelation — neighbouring footprints are not independent, so an
            i.i.d. bootstrap over footprints understates the uncertainty.
        n_boot: number of bootstrap resamples.
        ci: central interval width in percent (95 -> 2.5th/97.5th percentiles).
    Returns:
        list of dicts, one per metric: both products' values, the difference,
        its CI, and a two-sided bootstrap p-value (the proportion of resamples
        falling on the other side of zero, doubled). Negative `diff` means
        `name_a` is better for RMSE / MAE / |ME|.
    '''
    a = np.asarray(err_a, dtype='float64')
    b = np.asarray(err_b, dtype='float64')
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n == 0:
        return []

    # Sufficient statistics, per block when blocked and per observation otherwise.
    if block is not None:
        block = np.asarray(block)[ok]
        codes, inv = np.unique(block, return_inverse=True)
        n_g = np.bincount(inv).astype('float64')
        stats_a = np.stack([n_g] + [np.bincount(inv, weights=w)
                                    for w in (a, np.abs(a), a ** 2)])
        stats_b = np.stack([n_g] + [np.bincount(inv, weights=w)
                                    for w in (b, np.abs(b), b ** 2)])
        n_units = len(codes)
    else:
        ones = np.ones(n)
        stats_a = np.stack([ones, a, np.abs(a), a ** 2])
        stats_b = np.stack([ones, b, np.abs(b), b ** 2])
        n_units = n

    point_a = _agg_metrics(*stats_a.sum(axis=1))
    point_b = _agg_metrics(*stats_b.sum(axis=1))

    rng = np.random.default_rng(seed)
    draws = {m: np.empty(n_boot) for m in metrics}
    for i in range(n_boot):
        idx = rng.integers(0, n_units, n_units)
        ra = _agg_metrics(*stats_a[:, idx].sum(axis=1))
        rb = _agg_metrics(*stats_b[:, idx].sum(axis=1))
        for m in metrics:
            draws[m][i] = ra[m] - rb[m]

    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    rows = []
    for m in metrics:
        d = draws[m]
        # Two-sided bootstrap p: how often the resampled difference lands on the
        # other side of zero. Floored at 1/n_boot -- the resolution limit.
        frac = min((d > 0).mean(), (d < 0).mean())
        rows.append({
            'comparison': f'{name_a} vs {name_b}',
            'model': name_a, 'baseline': name_b,
            'test': f'bootstrap_{"block" if block is not None else "iid"}',
            'alternative': 'two-sided', 'metric': m,
            'n_obs': n, 'n_units': n_units, 'n_boot': n_boot,
            f'{m}_{name_a}': point_a[m], f'{m}_{name_b}': point_b[m],
            'diff': point_a[m] - point_b[m],
            'ci_lo': float(np.percentile(d, lo_q)),
            'ci_hi': float(np.percentile(d, hi_q)),
            'p_value': max(2 * frac, 1.0 / n_boot),
            'ci_excludes_zero': bool(np.percentile(d, lo_q) > 0
                                     or np.percentile(d, hi_q) < 0),
            **extra,
        })
    return rows


def holm_correct(df: pd.DataFrame, p_col: str = 'p_value',
                 out_col: str = 'p_holm', within: list = None) -> pd.DataFrame:
    '''
    Holm-Bonferroni adjust `p_col` across the rows of `df` (in place on a copy),
    optionally restarting the correction within each group of `within` columns
    so unrelated families of tests don't inflate each other. NaN p-values are
    passed through untouched and excluded from the family size.
    '''
    df = df.copy()
    df[out_col] = np.nan

    def _adjust(sub: pd.DataFrame) -> None:
        valid = sub[p_col].notna()
        p = sub.loc[valid, p_col].to_numpy()
        if len(p) == 0:
            return
        order = np.argsort(p)
        m = len(p)
        adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
        adj = np.maximum.accumulate(adj)   # enforce monotonicity
        out = np.empty(m)
        out[order] = adj
        df.loc[sub.index[valid], out_col] = out

    if within:
        for _, sub in df.groupby(within, dropna=False):
            _adjust(sub)
    else:
        _adjust(df)
    return df


def format_p(p) -> str:
    '''Render a p-value for a table cell: "< 1e-300" once it underflows.'''
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    return f'< {P_FLOOR:.0e}' if p < P_FLOOR else f'{p:.3e}'
