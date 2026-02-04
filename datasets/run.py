import os
import time
from typing import Any, Callable
from osgeo import gdal
from osgeo import osr
from glob import glob
from pathlib import Path
from typing import List, Union, Optional
import geopandas as gpd
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dataclasses import dataclass, field
import hydra
import dask
import subprocess
import numpy as np
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import pystac
from tqdm import tqdm
from omegaconf import OmegaConf, MISSING



    
@dataclass
class RepartitionDataConfig:
    _target_: str = "datasets.repartition_data.repartition_index_table"
    parquet_dir: str = '~/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables/val'
    save_dir: str = '~/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables_by_splitted_tile/val'
    based_on_col: str = 'assigned_tile'


@dataclass
class ExtractGEDIFromH5Config:
    _target_: str = "datasets._h5_dataset.extract_gedi_from_h5"
    h5_file: str = '~/data/gvs/datasets/splits/data_val.h5'
    index_table_dir: str = '~/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables_by_splitted_tile/val'
    save_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_val/original/2020/'
    keep_columns: List[str] = field(default_factory=lambda: [])
    year: int = 2020


defaults = [
    {'run': 'repartition_data'},
    "_self_"
]

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
cs = ConfigStore.instance()
cs.store(group='run', name='repartition_data', node=RepartitionDataConfig)
cs.store(group='run', name='extract_gedi_from_h5', node=ExtractGEDIFromH5Config)
cs.store(name='config', node=RunConfig)

@hydra.main(config_name='config', version_base='1.2')
def main(cfg):
    t0 = time.time()
    print(OmegaConf.to_yaml(cfg))
    instantiate(cfg.run)
    t1 = time.time()
    print(f'Time taken: {t1 - t0} seconds')


if __name__ == "__main__":
    main()