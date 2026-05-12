import argparse
import logging
import sys

import geopandas as gpd
import yaml
from cp.constants import BIOME_MAPPING
from cp.cp import ConformalPredictor
from cp.dl import RHDataCPConfig, collect_data, preprocess_rh_data


def main(args):
    if args.data_root is None and args.data_path is None:
        raise ValueError("Must provide either --data_root or --data_path")
    if args.data_root is not None and args.data_path is not None:
        raise ValueError("Must provide only one of --data_root or --data_path")

    config = RHDataCPConfig(args.config_path)
    if args.data_root is not None:
        logging.info(
            "Reading parquet files from %s",
            args.data_root,
        )
        data = collect_data(args.data_root, config)
        logging.info(
            "Preprocessing the data",
        )
        data = preprocess_rh_data(data, config)
    else:
        logging.info(
            "Reading single parquet file from %s",
            args.data_path,
        )
        data = gpd.read_parquet(args.data_path).to_crs("epsg:4326")
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
                    "q_lo_corr": float(cp_predictor.q_lo),
                    "q_hi_corr": float(cp_predictor.q_hi),
                }
            cp_method_res["BiomeIDs"][biome] = biome_res
        res[cp_method] = cp_method_res
    logging.info("Saving results to %s", args.save_path)
    yaml.dump(res, open(args.save_path, "w"))
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
