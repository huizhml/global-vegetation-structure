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
class KeyRHsGlobalConfig:
    _target_: str = "visualization.run.run_mosaic_for_key_rhs_global"
    year: int = 2020
    rh_idx: str = '98'
    q_idx: int = 1
    s2_grid_file: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'
    

def run_mosaic_for_key_rhs_global(year=2020, rh_idx: str = '98', q_idx=1, s2_grid_file: str = None,
                           stac_collection_dir: str = None):

    rh_idxs = [int(rh) for rh in rh_idx.split(' ')]
    for rh_idx in rh_idxs:
        print(rh_idx)
        # resample_and_mosaic(year, rh_idx, q_idx, s2_grid_file, stac_collection_dir) 


        
@dataclass
class FullRHsGlobalConfig:
    year: int = 2020
    q_idx: int = 1
    _target_: str = "visualization.run.run_mosaic_for_full_rhs_global"

def run_mosaic_for_full_rhs_global(year=2020, q_idx: int=1):
    for rh_idx in range(0, 101):
        print(rh_idx)
        # resample_and_mosaic(year, rh_idx, 1, s2_grid_file, stac_collection_dir)


@dataclass
class CheckfterBiasCorrectionConfig:
    year: int = 2020
    bias_dir: str = '~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/2020/stats_by_tile'
    save_dir: str = '~/data/gvs/predictions/2020/bias_corrected_slope_lt20_minpoints2000/mosaic'
    bias_cutoff: Optional[float] = None
    average_across_rhs: bool = False
    _target_: str = "visualization.run.check_mosaic_after_bias_correction"
    
def check_mosaic_after_bias_correction(year=2020, rh_idx: int = 98, q_idx: int = 1, bias_dir: str = None, save_dir: str = None, bias_cutoff: Optional[float] = None, average_across_rhs: bool = False):
    pass


defaults = [
    {'mosaic': 'full_global'},
    "_self_"
]

@dataclass
class MosaicConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    mosaic: Any = MISSING
    
cs = ConfigStore.instance()
cs.store(group='mosaic', name='full_global', node=FullRHsGlobalConfig)
cs.store(group='mosaic', name='key_rhs_global', node=KeyRHsGlobalConfig)
cs.store(group='mosaic', name='check_after_bias_correction', node=CheckfterBiasCorrectionConfig)
cs.store(name='config', node=MosaicConfig)

@hydra.main(config_name='config', version_base='1.2')
def main(cfg):
    t0 = time.time()
    print(OmegaConf.to_yaml(cfg))
    instantiate(cfg.mosaic)
    t1 = time.time()
    print(f'Time taken: {t1 - t0} seconds')


if __name__ == "__main__":
    main()