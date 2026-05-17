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
from config.base_config_class import FunctionConfig, ClassConfig




defaults = [
    {'run': 'repartition_data'},
    "_self_"
]

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
cs = ConfigStore.instance()


# -----------------------------------------------------------------------
# Calculate mean and std for naturalness classfication
# -----------------------------------------------------------------------
@dataclass
class CalNaturalnessStasConfig(FunctionConfig):
    data_file: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2017/vsm_patches_ps15_single_h5_72xl3wma_ps31.zarr'
    out_fp: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2017/vsm_patches_ps15_single_h5_72xl3wma_ps31.npz'
    _target_: str = 'preprocessing.pipeline.step6_calculate_naturalness_stats.calculate_naturalness_mean_std'

cs.store(group='run', name='cal_naturalness_input_stats', node=CalNaturalnessStasConfig)


# -----------------------------------------------------------------------
# Convert the chunk-by-1 zarr to a single HDF5 (network-FS friendly)
# -----------------------------------------------------------------------
@dataclass
class ZarrToH5Config(FunctionConfig):
    data_file: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2017/vsm_patches_ps15_single_h5_72xl3wma_ps31.zarr'
    out_fp: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2017/vsm_patches_ps15_single_h5_72xl3wma_ps31.h5'
    _target_: str = 'preprocessing.pipeline.step7_zarr_to_h5.convert_zarr_to_h5'

cs.store(group='run', name='zarr_to_h5', node=ZarrToH5Config)


cs.store(name='base_config', node=RunConfig) # NOTE: name here should match the default in ../config/base/no_log.yaml

@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    t0 = time.time()
    print(OmegaConf.to_yaml(cfg))
    if cfg.run.target_type == 'function':
        instantiate(cfg.run)
    elif cfg.run.target_type == 'class':
        obj = instantiate(cfg.run)
        excute_method = getattr(obj, cfg.run.target_method)
        excute_method()
    else:
        raise ValueError(f"Invalid target: {cfg.run.target_type}")
    t1 = time.time()
    print(f'Time taken: {t1 - t0} seconds')

if __name__ == "__main__":
    main()