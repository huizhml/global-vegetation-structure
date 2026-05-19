import argparse

import yaml

from cp.dl import RHDataCPConfig, collect_data, preprocess_rh_data


def main(args):
    with open(args.config_path, "r") as f:
        config = RHDataCPConfig(yaml.safe_load(f)["data"])
    data = collect_data(args.data_root, config)
    data = preprocess_rh_data(data, config)
    data.to_parquet(args.save_path)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Directory with all the input parquet files",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Path to the output combined parquet file",
        type=str,
    )
    parser.add_argument(
        "--config_path",
        required=True,
        help="Path to the CP and data columns config file (YAML)",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
