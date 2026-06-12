import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cp.config import GVSCPConfig
from cp.constants import BIOME_MAPPING, BIOME_SHORT_MAPPING
from cp.cp import ConformalPredictor
from cp.dl import load_data
from cp.plot import (
    plot_rh_coverage_per_biome,
    plot_rh_width_changes_per_biome,
    plot_global_coverage,
)


def run_cp(config: GVSCPConfig, save_root):
    data = load_data(
        config.data,
        config.cp.cal_data_root,
        config.cp.cal_data_path,
        config.cp.preprocess_data,
    )
    cp_res_rows = []
    for cp_method in config.cp.cqr_methods:
        for biome in BIOME_MAPPING.keys():
            for rh_cols in config.data:
                biome_data = data[data[config.data.biome_col] == biome]
                # q_lo, q_med, q_hi, y
                biome_rh_data = biome_data[rh_cols.all_cols()].values
                logging.info(
                    "Running %s for biome %s and RH%d (n=%d)",
                    cp_method,
                    biome,
                    rh_cols.rh_val,
                    len(biome_rh_data),
                )
                cp_predictor = ConformalPredictor(
                    cp_method,
                    config.cp.alpha,
                    biome_rh_data[:, 0],
                    biome_rh_data[:, 2],
                    biome_rh_data[:, 1],
                    biome_rh_data[:, 3],
                )
                cp_res_rows.append(
                    {
                        "method": cp_method,
                        "biome": biome,
                        "rh": rh_cols.rh_val,
                        "q_lo": cp_predictor.q_lo,
                        "q_hi": cp_predictor.q_hi,
                        "n_cal": len(biome_rh_data),
                    }
                )
    cp_res_df = pd.DataFrame(cp_res_rows).set_index(["method", "biome", "rh"])
    cp_res_df.to_parquet(save_root / "cp_results.parquet", index=True)
    return cp_res_df


def coverage(q_lo, q_hi, y):
    return np.mean((q_lo <= y) & (y <= q_hi))


def avg_width(q_lo, q_hi):
    return np.mean(q_hi - q_lo) / 10


def compute_baseline_stats(config, data):
    baseline_stats_rows = []
    for biome in BIOME_MAPPING.keys():
        for rh_cols in config.data:
            # q_lo, q_med, q_hi, y
            biome_rh_data = data.loc[
                data[config.data.biome_col] == biome, rh_cols.all_cols()
            ].values
            q_lo, q_hi = biome_rh_data[:, 0], biome_rh_data[:, 2]
            baseline_stats_rows.append(
                {
                    "method": "Baseline",
                    "biome": biome,
                    "rh": rh_cols.rh_val,
                    "n_eval": len(biome_rh_data),
                    "coverage": coverage(q_lo, q_hi, biome_rh_data[:, 3]),
                    "avg_width_m": avg_width(q_lo, q_hi),
                }
            )
    return baseline_stats_rows


def eval_cp(config: GVSCPConfig, cp_res_df: pd.DataFrame, save_root):
    logging.info("Computing evaluation metrics...")
    data = load_data(
        config.data,
        config.eval.data_root,
        config.eval.data_path,
        config.eval.preprocess_data,
    )
    eval_stats_rows = compute_baseline_stats(config, data)
    for (cp_method, biome, rh_val), row in cp_res_df.iterrows():
        rh_cols = config.data.get_rh_cols(rh_val)
        # q_lo, q_med, q_hi, y
        biome_rh_data = data.loc[data["BIOME"] == biome, rh_cols].values
        q_lo, q_hi = biome_rh_data[:, 0], biome_rh_data[:, 2]
        q_med, y = biome_rh_data[:, 1], biome_rh_data[:, 3]
        cp_predictor = ConformalPredictor.from_precomputed(
            cp_method,
            config.cp.alpha,
            row["q_lo"],
            row["q_hi"],
        )
        q_lo_corr, q_hi_corr = cp_predictor.calibrate(q_lo, q_hi, q_med)
        eval_stats_rows.append(
            {
                "method": cp_method,
                "biome": biome,
                "rh": rh_val,
                "n_eval": len(biome_rh_data),
                "coverage": coverage(q_lo_corr, q_hi_corr, y),
                "avg_width_m": avg_width(q_lo_corr, q_hi_corr),
            }
        )
    eval_stats_df = pd.DataFrame(eval_stats_rows).set_index(
        ["method", "biome", "rh"]
    )
    eval_stats_df.to_parquet(save_root / "eval_stats.parquet", index=True)
    return eval_stats_df


def get_cp_method_eval_stat(eval_stats_df: pd.DataFrame, method: str, stat):
    method_stat_df = eval_stats_df.loc[pd.IndexSlice[method, :, :], stat]
    # reset index so we can pivot
    method_stat_df = method_stat_df.reset_index()
    # pivot: rows=rh, columns=biome, values=coverage
    method_stat_df = method_stat_df.pivot(
        index="rh", columns="biome", values=stat
    )
    method_stat_df.rename(columns=BIOME_MAPPING, inplace=True)
    method_stat_df.rename(columns=BIOME_SHORT_MAPPING, inplace=True)
    method_stat_df.index = ["RH" + str(rh) for rh in method_stat_df.index]
    return method_stat_df


def get_global_cp_method_eval_stat(
    eval_stats_df: pd.DataFrame, method: str, stat: str
):
    """Weighted average across biomes (weighted by n_eval) for each RH level."""
    method_df = eval_stats_df.loc[
        pd.IndexSlice[method, :, :], [stat, "n_eval"]
    ].reset_index()
    global_stat = method_df.groupby("rh").apply(
        lambda g: np.average(g[stat], weights=g["n_eval"])
    )
    global_stat.index = ["RH" + str(rh) for rh in global_stat.index]
    return global_stat  # Series indexed by RH label


def plot(config: GVSCPConfig, eval_stats_df: pd.DataFrame, save_root: Path):
    logging.info(
        "Generating the figures...",
    )
    cp_methods = eval_stats_df.index.get_level_values("method").unique()
    baseline_cov_df = get_cp_method_eval_stat(
        eval_stats_df, "Baseline", "coverage"
    )
    baseline_width_df = get_cp_method_eval_stat(
        eval_stats_df, "Baseline", "avg_width_m"
    )
    rh_order = sorted(baseline_cov_df.index.tolist(), key=lambda x: int(x[2:]))
    for cp_method in cp_methods:
        if cp_method == "Baseline":
            continue
        fig_save_root = save_root / "figures" / cp_method
        os.makedirs(fig_save_root, exist_ok=True)

        method_cov_df = get_cp_method_eval_stat(
            eval_stats_df, method=cp_method, stat="coverage"
        )
        config.plot.set_coverage_plot_style()
        n_rows, n_cols = config.plot.get_coverage_n_rows_cols()
        cov_figsize = config.plot.get_coverage_figsize()
        xlims, xticks = plot_rh_coverage_per_biome(
            method_cov_df,
            baseline_cov_df,
            f"{fig_save_root}/coverage.pdf",
            config.plot.get_coverage_biome_order(),
            rh_order,
            config.cp.alpha,
            skip_groups=config.plot.skip_biomes,
            figsize=cov_figsize,
            n_rows=n_rows,
            n_cols=n_cols,
            legend_loc=config.plot.get_coverage_legend_loc(),
        )

        baseline_global = get_global_cp_method_eval_stat(
            eval_stats_df, "Baseline", "coverage"
        )
        method_global = get_global_cp_method_eval_stat(
            eval_stats_df, cp_method, "coverage"
        )
        config.plot.set_coverage_plot_style()
        plot_global_coverage(
            method_global,
            baseline_global,
            f"{fig_save_root}/coverage_global.pdf",
            rh_order,
            config.cp.alpha,
            figsize=(cov_figsize[0] * 1 / 5, cov_figsize[1]),
            xticks=xticks,
            xlim=xlims,
            # color=OKABE_ITO_PALETTE["yellow"],
        )

        method_width_df = get_cp_method_eval_stat(
            eval_stats_df, method=cp_method, stat="avg_width_m"
        )
        config.plot.set_avg_width_plot_style()
        n_rows, n_cols = config.plot.get_avg_width_n_rows_cols()
        plot_rh_width_changes_per_biome(
            baseline_width_df,
            method_width_df,
            f"{fig_save_root}/width.pdf",
            config.plot.get_avg_width_biome_order(),
            rh_order,
            figsize=config.plot.get_avg_width_figsize(),
            skip_groups=config.plot.skip_biomes,
            n_rows=n_rows,
            n_cols=n_cols,
            legend_loc=config.plot.get_avg_width_legend_loc(),
        )


def main(args: argparse.Namespace):
    config = GVSCPConfig(args.config_path)
    save_root = Path(args.config_path).parent
    print(save_root)
    # Skip CP and evaluation if only plotting
    if args.plot_only:
        logging.info(
            "Loading evaluation statistics from %s",
            save_root / "eval_stats.parquet",
        )
        eval_res_df = pd.read_parquet(
            save_root / "eval_stats.parquet",
        )
        plot(config, eval_res_df, save_root)
        return 0
    # Otherwise run everything
    if args.load_corrections:
        logging.info(
            "Loading CP corrections from %s",
            save_root / "cp_results.parquet",
        )
        cp_res_df = pd.read_parquet(
            save_root / "cp_results.parquet",
        )
    else:
        cp_res_df = run_cp(config, save_root)
    eval_res_df = eval_cp(config, cp_res_df, save_root)
    plot(config, eval_res_df, save_root)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the CP and data columns config file (YAML)",
        required=True,
    )
    parser.add_argument(
        "--load_corrections",
        action="store_true",
        help="Use precomputed quantile corrections instead of running CP calibration",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only generate plots using precomputed CP results and evaluation stats",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main(parse_args())
