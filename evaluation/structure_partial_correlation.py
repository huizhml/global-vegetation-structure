"""
Does the VSM encode height-independent vertical structure?
==========================================================
Tests whether VSM-predicted lower RH metrics (RH25, RH50) track the TRUE GEDI
measurements after controlling for canopy top height (RH98).

If optical imagery only sensed canopy surface height, then given RH98 the
predicted RH25 would carry no information about the true RH25. A significant
partial correlation refutes this: the model has learned structural information
beyond canopy height — the claim the SAR critique targets.

Three equivalent methods (results should agree):
  A. Binned correlation: within narrow RH98 bins, corr(pred_RHx, true_RHx)
  B. Partial correlation: corr(pred_RHx, true_RHx | RH98)
  C. Residual-on-residual: regress out RH98 from both, correlate residuals

The control variable is TRUE GEDI RH98 (clean, no prediction-error propagation).

Input is the held-out GEDI test set used elsewhere in evaluation/: per-tile
parquet of footprints carrying GEDI measurements `rh{lvl}` and VSM predictions
`RH{lvl}_Q1` (mirrors `evaluation/on_gedi.py`), optionally a biome column for
the stratified pass.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from const import BIOMES_BY_VALUE

warnings.filterwarnings("ignore")


# Lower RH metrics share the cool ramp; RH98 (the control) sits in the warm
# contrast — same convention as protected_area_analysis so the "structure
# beyond canopy height" story reads at a glance across figures.
_METRIC_COLORS = {"rh25": "#2166ac", "rh50": "#4393c3", "rh98": "#d6604d"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_test_footprints(test_data, cols, slope_lt20=True):
    """Read the held-out GEDI footprints, keeping only `cols` (+ `slope` when
    filtering). `test_data` is either a single parquet/csv file or a directory
    of per-tile parquet (as produced upstream of `on_gedi`). Reading just the
    needed columns avoids decoding the millions of unused WKB geometries /
    101 RH bands per file.
    """
    test_data = Path(test_data).expanduser()
    read_cols = list(cols) + (["slope"] if slope_lt20 else [])

    if test_data.is_dir():
        files = sorted(test_data.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no .parquet files under {test_data}")
        frames = [pd.read_parquet(f, columns=read_cols) for f in files]
        df = pd.concat(frames, ignore_index=True)
    elif test_data.suffix == ".parquet":
        df = pd.read_parquet(test_data, columns=read_cols)
    else:
        df = pd.read_csv(test_data, usecols=read_cols)

    if slope_lt20:
        df = df[df["slope"] < 20]
    return df


# =============================================================================
# METHOD A: BINNED CORRELATION
# =============================================================================
def binned_correlation(df, true_col, pred_col, rh98_col,
                       rh98_min, rh98_max, bin_width, min_bin_samples):
    """Within narrow RH98 bins, correlate predicted vs. true lower RH. A
    positive correlation means the model distinguishes structure among
    footprints of (near) identical canopy height. Bins with fewer than
    `min_bin_samples` valid footprints are skipped."""
    bins = np.arange(rh98_min, rh98_max + bin_width, bin_width)
    df = df.copy()
    df["rh98_bin"] = pd.cut(df[rh98_col], bins=bins)

    results = []
    for bin_label, bin_df in df.groupby("rh98_bin"):
        true_vals = bin_df[true_col].values
        pred_vals = bin_df[pred_col].values
        valid = np.isfinite(true_vals) & np.isfinite(pred_vals)
        if valid.sum() < min_bin_samples:
            continue
        r, p = stats.pearsonr(true_vals[valid], pred_vals[valid])
        results.append({
            "rh98_bin_mid": bin_label.mid,
            "n": int(valid.sum()),
            "pearson_r": r,
            "p_value": p,
        })
    return pd.DataFrame(results)


# =============================================================================
# METHOD B: PARTIAL CORRELATION
# =============================================================================
def partial_correlation(x, y, z):
    """Partial correlation between x and y controlling for z, computed as the
    correlation of residuals after regressing each on z. The design uses z and
    z**2 so the control absorbs a nonlinear height dependence, not just a
    linear trend. Returns (r, p, n)."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n

    Z = np.column_stack([np.ones(n), z, z ** 2])
    beta_x, _, _, _ = np.linalg.lstsq(Z, x, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    res_x = x - Z @ beta_x
    res_y = y - Z @ beta_y

    r, p = stats.pearsonr(res_x, res_y)
    return r, p, n


# =============================================================================
# METHOD C: RESIDUAL-ON-RESIDUAL (returns residuals for plotting)
# =============================================================================
def residual_analysis(df, true_col, pred_col, rh98_col):
    """Regress out RH98 (with quadratic term) from both true and predicted
    lower RH and return the residuals. Their correlation equals the method-B
    result; the residuals are kept so the relationship can be scatter-plotted."""
    true_vals = df[true_col].values
    pred_vals = df[pred_col].values
    rh98_vals = df[rh98_col].values

    valid = np.isfinite(true_vals) & np.isfinite(pred_vals) & np.isfinite(rh98_vals)
    true_vals, pred_vals, rh98_vals = true_vals[valid], pred_vals[valid], rh98_vals[valid]
    n = len(true_vals)

    Z = np.column_stack([np.ones(n), rh98_vals, rh98_vals ** 2])
    beta_true, _, _, _ = np.linalg.lstsq(Z, true_vals, rcond=None)
    beta_pred, _, _, _ = np.linalg.lstsq(Z, pred_vals, rcond=None)

    res_true = true_vals - Z @ beta_true
    res_pred = pred_vals - Z @ beta_pred
    return res_true, res_pred


# =============================================================================
# BASELINE: naive correlation (NOT controlling for RH98)
# =============================================================================
def naive_correlation(df, true_col, pred_col):
    """Plain true-vs-predicted correlation, for reference. Inflated by the
    shared RH98 signal — the partial correlation is the honest number."""
    t = df[true_col].values
    p = df[pred_col].values
    valid = np.isfinite(t) & np.isfinite(p)
    r, pval = stats.pearsonr(t[valid], p[valid])
    return r, pval, int(valid.sum())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_residual_scatter(res_true, res_pred, metric, r, output_path, rng):
    """Scatter of true vs predicted residuals after removing RH98. A positive
    trend is the visual proof of height-independent signal. Subsampled to keep
    the figure light when there are many footprints."""
    fig, ax = plt.subplots(figsize=(5, 5))
    n = len(res_true)
    if n > 5000:
        idx = rng.choice(n, 5000, replace=False)
        rt, rp = res_true[idx], res_pred[idx]
    else:
        rt, rp = res_true, res_pred
    ax.scatter(rt, rp, s=4, alpha=0.2, color=_METRIC_COLORS.get(metric, "#2166ac"))

    coef = np.polyfit(res_true, res_pred, 1)
    xline = np.linspace(res_true.min(), res_true.max(), 100)
    ax.plot(xline, np.polyval(coef, xline), color="#d6604d", linewidth=2)

    ax.set_xlabel(f"True {metric.upper()} residual (RH98 removed)")
    ax.set_ylabel(f"Predicted {metric.upper()} residual (RH98 removed)")
    ax.set_title(f"{metric.upper()}: partial r = {r:.3f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Residual scatter saved: {output_path.name}")


def plot_binned_correlation(binned, metric, bin_width, output_path):
    """Per-RH98-bin correlation, showing the signal holds across height ranges."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(binned["rh98_bin_mid"], binned["pearson_r"],
           width=bin_width * 0.8, color=_METRIC_COLORS.get(metric, "#2166ac"))
    ax.set_xlabel("RH98 bin midpoint (m)")
    ax.set_ylabel(f"corr(pred, true) for {metric.upper()}")
    ax.set_title(f"{metric.upper()}: within-height-bin correlation")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Binned plot saved: {output_path.name}")


def plot_biome_bars(biome_df, metrics, output_path):
    """Per-biome partial correlation (controlling RH98), one bar per metric."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = biome_df.pivot(index="biome", columns="metric", values="partial_r")
    pivot = pivot.reindex(columns=metrics)
    colors = [_METRIC_COLORS.get(m, "#888888") for m in metrics]
    pivot.plot(kind="bar", ax=ax, width=0.7, color=colors)
    ax.set_ylabel("Partial correlation (controlling RH98)")
    ax.set_xlabel("")
    ax.set_title("Height-independent structural signal by biome")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend([m.upper() for m in metrics])
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Biome plot saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main runnable
# ---------------------------------------------------------------------------
def structure_partial_correlation(
    test_data: str,
    save_dir: str,
    test_levels: list = None,
    control_level: int = 98,
    true_pattern: str = "rh{lvl}",
    pred_pattern: str = "RH{lvl}_Q1",
    biome_col: str = "biome",
    exclude_biomes: list = None,
    slope_lt20: bool = True,
    rh98_min: float = 5.0,
    rh98_max: float = 60.0,
    rh98_bin_width: float = 2.0,
    min_bin_samples: int = 50,
    random_seed: int = 0,
    **kwargs,
):
    """Test whether the VSM encodes vertical structure beyond canopy top height
    by partialling RH98 out of the predicted-vs-true correlation for the lower
    RH metrics.

    Three equivalent estimators are reported per metric — naive (uncontrolled)
    correlation for contrast, the partial correlation (the key number), and the
    per-RH98-bin correlation — plus a residual-on-residual scatter. A positive,
    significant partial correlation refutes the claim that optical imagery only
    senses canopy surface height. Stratified by WWF biome when `biome_col` is
    present.

    Parameters
    ----------
    test_data : Held-out GEDI test set — a single parquet/csv file, or a
        directory of per-tile parquet (globbed and concatenated). Must carry the
        true (`true_pattern`) and predicted (`pred_pattern`) RH columns for every
        tested level and the control level; `slope` is needed only when
        `slope_lt20` is set.
    save_dir : Output directory for the summary/by-biome CSVs and the plots.
    test_levels : RH levels (integers) to test — the lower-canopy metrics the
        SAR critique targets. Defaults to [25, 50] (i.e. RH25, RH50).
    control_level : RH level used as the canopy-height control. Default 98.
    true_pattern, pred_pattern : `str.format` patterns mapping an RH level to its
        GEDI-measured and VSM-predicted column names. Defaults `rh{lvl}` /
        `RH{lvl}_Q1` follow the convention in `evaluation/on_gedi.py`.
    biome_col : Column holding the WWF biome id for the stratified pass; set to
        None or omit the column to skip stratification.
    exclude_biomes : WWF biome ids to drop from the whole analysis (both the
        pooled correlation and the stratified pass), e.g. non-vegetated /
        non-forest classes. Defaults to [11, 98, 99] — Tundra and the
        Lake / Rock-and-ice placeholders. Set to [] to keep every biome.
        Ignored when `biome_col` is absent.
    slope_lt20 : Keep only footprints on slopes < 20 deg (needs a `slope`
        column), matching the GEDI evaluation filter.
    rh98_min, rh98_max : Forest height window (metres) the analysis is run over;
        caps the sparse tall tail.
    rh98_bin_width : RH98 bin width (metres) for the binned correlation.
    min_bin_samples : Skip RH98 bins with fewer valid footprints.
    random_seed : Seed for subsampling the residual scatter plots.
    """
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    test_levels = test_levels or [25, 50]
    exclude_biomes = [11, 98, 99] if exclude_biomes is None else exclude_biomes
    rng = np.random.default_rng(random_seed)

    # Resolve the column names once: one (true, pred) pair per tested level plus
    # the RH98 control. We control with the TRUE RH98 so no prediction error
    # leaks into the conditioning variable.
    rh98_col = true_pattern.format(lvl=control_level)
    metrics = {f"rh{lvl}": (true_pattern.format(lvl=lvl), pred_pattern.format(lvl=lvl))
               for lvl in test_levels}

    needed = {rh98_col}
    for true_col, pred_col in metrics.values():
        needed |= {true_col, pred_col}

    # --- 1. Load + forest-height filter ----------------------------------
    print("Loading held-out GEDI test footprints...")
    df = _load_test_footprints(test_data, needed | {biome_col}
                               if biome_col else needed, slope_lt20=slope_lt20)
    print(f"  Loaded {len(df):,} footprints; columns: {list(df.columns)}")
    df = df[(df[rh98_col] >= rh98_min) & (df[rh98_col] <= rh98_max)].copy()
    print(f"  After RH98 in [{rh98_min}, {rh98_max}] m filter: {len(df):,}")
    # Drop non-vegetated / non-forest biomes (Tundra, Lake/Rock-ice placeholders)
    # so they pollute neither the pooled correlation nor the stratified pass.
    if biome_col and biome_col in df.columns and exclude_biomes:
        df = df[~df[biome_col].isin(exclude_biomes)].copy()
        print(f"  After excluding biomes {list(exclude_biomes)}: {len(df):,}")
    print()

    # --- 2. Per-metric analysis (the three methods) ----------------------
    summary_rows = []
    for metric, (true_col, pred_col) in metrics.items():
        print("=" * 70)
        print(f"METRIC: {metric.upper()}")
        print("=" * 70)

        # Naive (uncontrolled) correlation — for contrast only.
        r_naive, p_naive, n_naive = naive_correlation(df, true_col, pred_col)
        print(f"\n  Naive corr (NOT controlling RH98): r={r_naive:.3f} "
              f"(p={p_naive:.2e}, n={n_naive:,})")
        print(f"    ^ inflated by the shared canopy-height signal")

        # Method B: partial correlation (the key result).
        r_partial, p_partial, n_partial = partial_correlation(
            df[pred_col].values, df[true_col].values, df[rh98_col].values,
        )
        print(f"\n  PARTIAL corr (controlling RH98): r={r_partial:.3f} "
              f"(p={p_partial:.2e}, n={n_partial:,})")
        print(f"    ^ THE KEY NUMBER: height-independent structural signal")
        if r_partial > 0 and p_partial < 0.05:
            print(f"    => VSM predicts {metric.upper()} variation beyond canopy height.")
        else:
            print(f"    => No significant height-independent signal for {metric.upper()}.")

        # Method A: binned correlation.
        binned = binned_correlation(df, true_col, pred_col, rh98_col,
                                    rh98_min, rh98_max, rh98_bin_width,
                                    min_bin_samples)
        mean_binned_r = binned["pearson_r"].mean() if len(binned) else np.nan
        if len(binned):
            print(f"\n  Binned corr (mean across {len(binned)} RH98 bins): "
                  f"r={mean_binned_r:.3f}")
            binned.to_csv(save_dir / f"binned_{metric}.csv", index=False)
            plot_binned_correlation(binned, metric, rh98_bin_width,
                                    save_dir / f"binned_{metric}.png")

        # Method C: residuals for the scatter (correlation == method B).
        res_true, res_pred = residual_analysis(df, true_col, pred_col, rh98_col)
        plot_residual_scatter(res_true, res_pred, metric, r_partial,
                              save_dir / f"residuals_{metric}.png", rng)

        summary_rows.append({
            "metric": metric,
            "naive_r": r_naive,
            "partial_r": r_partial,
            "partial_p": p_partial,
            "mean_binned_r": mean_binned_r,
            "n": n_partial,
        })
        print()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(save_dir / "summary.csv", index=False)

    # --- 3. Stratified by biome ------------------------------------------
    if biome_col and biome_col in df.columns:
        print("=" * 70)
        print("STRATIFIED BY BIOME (partial correlation)")
        print("=" * 70)
        rows = []
        for biome_id, bdf in df.groupby(biome_col):
            biome_name = BIOMES_BY_VALUE.get(biome_id, {}).get("name", f"Biome {biome_id}")
            for metric, (true_col, pred_col) in metrics.items():
                r, p, n = partial_correlation(
                    bdf[pred_col].values, bdf[true_col].values, bdf[rh98_col].values,
                )
                rows.append({"biome": biome_name, "metric": metric,
                             "partial_r": r, "p_value": p, "n": n})
        biome_df = pd.DataFrame(rows)
        biome_df.to_csv(save_dir / "by_biome.csv", index=False)

        pivot = biome_df.pivot(index="biome", columns="metric",
                               values="partial_r").round(3)
        print("\nPartial correlation (controlling RH98) by biome:")
        print(pivot.to_string())
        plot_biome_bars(biome_df, list(metrics), save_dir / "by_biome.png")
    else:
        print("No biome column; skipping stratified analysis.")

    # --- 4. Headline summary ---------------------------------------------
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("\nKey message: a positive, significant partial_r means the VSM "
          "encodes vertical\nstructure beyond canopy top height — refuting the "
          "claim that optical only\nsenses surface height.")
    print("\nDone. Results saved to:", save_dir)
    return summary_df


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='structure_partial_correlation',
    default_run='structure_partial_correlation',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()
