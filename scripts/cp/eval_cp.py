import argparse
import logging
import sys
from cp.dl import load_config_and_data
from cp.cp import ConformalPredictor
import json
import numpy as np
from cp.constants import BIOME_MAPPING
from pathlib import Path


def coverage(q_lo, q_hi, y):
    return np.mean((q_lo <= y) & (y <= q_hi))


def avg_width(q_lo, q_hi):
    return np.mean(q_hi - q_lo)


def main(args: argparse.Namespace):
    config, data = load_config_and_data(
        args.config_path, args.data_root, args.data_path
    )
    with open(args.cp_res_path, "r") as f:
        cp_res = json.load(f)
    baseline_stats = {
        "BiomeIDs": {str(biome): {"RH": {}} for biome in BIOME_MAPPING.keys()}
    }
    res = {
        "ConfigPath": str(Path(args.config_path).resolve()),
        "CPSource": str(Path(args.cp_res_path).resolve()),
        "DataRoot": (
            str(Path(args.data_root).resolve()) if args.data_root else None
        ),
        "DataPath": (
            str(Path(args.data_path).resolve()) if args.data_path else None
        ),
        "Methods": {},
    }
    for cp_method, cp_method_qs in cp_res.items():
        cp_method_res = {"BiomeIDs": {}}
        for biome, cp_biome_qs in cp_method_qs["BiomeIDs"].items():
            biome_data = data[data["BIOME"] == int(biome)]
            biome_res = {"RH": {}}
            for rh_val, cp_biome_rh_qs in cp_biome_qs["RH"].items():
                rh_val = int(rh_val)
                logging.info(
                    "Evaluating %s for biome %s and RH%d (n=%d)",
                    cp_method,
                    biome,
                    rh_val,
                    len(biome_data),
                )
                # q_lo, q_med, q_hi, y
                rh_cols = config.get_rh_cols(rh_val)
                biome_rh_data = biome_data[rh_cols].values
                q_lo, q_med = biome_rh_data[:, 0], biome_rh_data[:, 1]
                q_hi, y = biome_rh_data[:, 2], biome_rh_data[:, 3]
                baseline_rh_stats_all = baseline_stats["BiomeIDs"][biome]["RH"]
                if rh_val not in baseline_rh_stats_all:
                    baseline_rh_stats = {
                        "coverage": coverage(q_lo, q_hi, y),
                        "avg_width": avg_width(q_lo, q_hi),
                    }
                    baseline_rh_stats_all[rh_val] = baseline_rh_stats

                cp_predictor = ConformalPredictor.from_precomputed(
                    cp_method,
                    config.alpha,
                    cp_biome_rh_qs["cp_q_lo"],
                    cp_biome_rh_qs["cp_q_hi"],
                )
                q_lo_corr, q_hi_corr = cp_predictor.calibrate(q_lo, q_hi, q_med)
                biome_res["RH"][rh_val] = {
                    "coverage": coverage(q_lo_corr, q_hi_corr, y),
                    "avg_width": avg_width(q_lo_corr, q_hi_corr),
                }
            cp_method_res["BiomeIDs"][biome] = biome_res
        res["Methods"][cp_method] = cp_method_res
    res["Methods"]["Baseline"] = baseline_stats
    logging.info("Saving results to %s", args.save_path)
    json.dump(res, open(args.save_path, "w"), indent=4)
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
        "--save_path",
        type=str,
        help="Path for saving CP evaluation results (JSON)",
        required=True,
    )
    parser.add_argument(
        "--cp_res_path",
        type=str,
        help="Path to the root containing data parquet files (JSON)",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the parquet file containing data",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to the parquet file containing data",
        default=None,
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main(parse_args())
