from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .constants import BIOME_MAPPING


class RHDataCPConfig:
    class RHColumns:
        def __init__(self, rh_val, rh_cols: dict[str, str]):
            self.rh_val: float = rh_val
            self.ground_truth_col = rh_cols["ground_truth"]
            self.q_lo_col = rh_cols["q_lo"]
            self.q_med_col = rh_cols["q_med"]
            self.q_hi_col = rh_cols["q_hi"]

        def q_cols(self):
            return [self.q_lo_col, self.q_med_col, self.q_hi_col]

        def all_cols(self):
            return list(self.q_cols()) + [self.ground_truth_col]

    rh_cols: list[RHColumns]

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "RH" not in config:
            raise ValueError("Config file must contain 'RH' key")
        self.rh_cols = [
            self.RHColumns(rh_val, rh_cols)
            for rh_val, rh_cols in config["RH"].items()
        ]
        self.rh_cols.sort(key=lambda x: x.rh_val)
        self.cqr_methods = config.get("cqr_methods")
        self.other_cols = config.get("other_cols", [])
        self.alpha = float(config["alpha"])

    def __iter__(self):
        return iter(self.rh_cols)

    def all_rh_cols(self):
        all_rh_cols = [
            self.rh_cols[i].all_cols() for i in range(len(self.rh_cols))
        ]
        return [cols for rh_cols in all_rh_cols for cols in rh_cols]

    def all_cols(self):
        return self.other_cols + self.all_rh_cols()


def collect_data(data_root, config: RHDataCPConfig):
    geo_dfs = []
    all_files = glob(f"{data_root}/*.parquet")
    columns = config.all_cols()
    with tqdm(all_files, total=len(all_files)) as pbar:
        for path in pbar:
            gdf = gpd.read_parquet(path).to_crs("epsg:4326")
            geo_dfs.append(gdf[columns])
    full_gdf = gpd.GeoDataFrame(
        pd.concat(geo_dfs, ignore_index=True), crs="epsg:4326"
    )
    if "BIO_REALM" in full_gdf.columns:
        full_gdf["BIO_REALM"] = full_gdf["ECO_ID"] // 100
    return full_gdf


def correct_quantile_crossing(
    data: gpd.GeoDataFrame, config: RHDataCPConfig, min_err: float = 0.0
):
    for rh_cols in config:
        # 1. Sort row-wise to fix crossing
        sorted_vals = np.sort(data[rh_cols.q_cols()].values, axis=1)

        # 2. Enforce minimum distance (clip values)
        # The lower quantile can be AT MOST (median - min_dist)
        q_lo_fixed = np.minimum(sorted_vals[:, 0], sorted_vals[:, 1] - min_err)

        # The upper quantile must be AT LEAST (median + min_dist)
        q_hi_fixed = np.maximum(sorted_vals[:, 2], sorted_vals[:, 1] + min_err)

        # 3. Assign back to a new dataframe to prevent SettingWithCopyWarnings
        data[rh_cols.q_lo_col] = q_lo_fixed
        data[rh_cols.q_med_col] = sorted_vals[:, 1]
        data[rh_cols.q_hi_col] = q_hi_fixed
    return data


def convert_to_decimeter(data: gpd.GeoDataFrame, config: RHDataCPConfig):
    for rh_cols in config:
        cols = rh_cols.all_cols()
        data[cols] = (data[cols] * 10).round().astype(int)
    return data


def preprocess_rh_data(
    data: gpd.GeoDataFrame, config: RHDataCPConfig, min_err=0.0
):
    # 1. Drop nan predictions and BIOME values
    required_cols = config.all_rh_cols() + ["BIOME"]
    data.dropna(subset=required_cols, inplace=True)

    # 2. Keep only standard biomes (1-14)
    data = data[data["BIOME"].isin(BIOME_MAPPING.keys())]

    # 4. Drop duplicates
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 5. Correct quantile crossing and enforce minimum distance
    data_corrected = correct_quantile_crossing(data, config, min_err)

    # 6. Convert to decimeters and integers
    data_corrected = convert_to_decimeter(data_corrected, config)
    return data_corrected


"""
if __name__ == "__main__":
    main(
        "./data/full_2020.parquet",
        "./data/full_2020_corrected.parquet",
        min_err=0.0,
    )
"""

# FULL_GDF = collect_data("./2020")
# FULL_GDF.to_parquet("./data/full_2020.parquet")
