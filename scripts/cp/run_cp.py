import argparse
import logging
import sys

import json
from cp.constants import BIOME_MAPPING
from cp.cp import ConformalPredictor
from cp.dl import load_config_and_data


def main(args: argparse.Namespace):
    config, data = load_config_and_data(
        args.config_path, args.data_root, args.data_path
    )
    res = {}
    for cp_method in config.cqr_methods:
        cp_method_res = {"BiomeIDs": {}}
        for biome in BIOME_MAPPING.keys():
            biome_res = {"RH": {}}
            for rh_cols in config:
                biome_data = data[data["BIOME"] == biome]
                # q_lo, q_med, q_hi, y
                rh_data = biome_data[rh_cols.all_cols()].values
                logging.info(
                    "Running %s for biome %s and RH%d (n=%d)",
                    cp_method,
                    biome,
                    rh_cols.rh_val,
                    len(rh_data),
                )
                cp_predictor = ConformalPredictor(
                    cp_method,
                    config.alpha,
                    rh_data[:, 0],
                    rh_data[:, 2],
                    rh_data[:, 1],
                    rh_data[:, 3],
                )
                biome_res["RH"][rh_cols.rh_val] = {
                    "cp_q_lo": float(cp_predictor.q_lo),
                    "cp_q_hi": float(cp_predictor.q_hi),
                }
            cp_method_res["BiomeIDs"][biome] = biome_res
        res[cp_method] = cp_method_res
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
        help="Path for saving CP results (YAML)",
        required=True,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to the root containing data parquet files",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--data_path",
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
