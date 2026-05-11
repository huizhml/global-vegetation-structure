import argparse

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

    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    config = RHDataCPConfig(config)
    if args.data_root is not None:
        data = collect_data(args.data_root, config)
    else:
        data = gpd.read_parquet(args.data_path).to_crs("epsg:4326")
    data = preprocess_rh_data(data, config)
    res = {}
    for cp_method in config.cqr_methods:
        cp_method_res = {"BiomeIDs": {}}
        for biome in BIOME_MAPPING.keys():
            biome_res = {"RH": {}}
            for rh_cols in config:
                cp_predictor = ConformalPredictor(cp_method, config.alpha)
                rh_data = data[rh_cols].values
                cp_predictor.fit(rh_data[0], rh_data[1], rh_data[2], rh_data[3])
                biome_res["RH"][rh_cols.rh_val] = {
                    "q_lo_corr": cp_predictor.q_lo,
                    "q_hi_corr": cp_predictor.q_hi,
                }
            cp_method_res["BiomeIDs"][biome] = biome_res
        res[cp_method] = cp_method_res
    yaml.dump(res, open(args.output_dir, "w"))
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
        "--output_dir",
        type=str,
        help="Directory to save CP results (YAML)",
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
    main(parse_args())
