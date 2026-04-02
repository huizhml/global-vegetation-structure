
import os
import time
from typing import List, Optional, Any
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dataclasses import dataclass, field, fields
import hydra
from omegaconf import OmegaConf, MISSING
from config.base_config_class import ClassConfig, FunctionConfig
from const import KEY_RHS, CHM_COLS
from postprocessing.core.utils import generate_run_log


@dataclass
class ComputeEntropyConfig(FunctionConfig):
    output_dir: str = '~/data/gvs/products/profile_entropy/2020/tiles/geotiff'
    tile_id: str = '36NTF'
    year: int = 2020
    chunk_size: int = 512
    max_workers: int = 8
    bin_width: int = 50
    _target_: str = "evaluation.diversity_indices.compute_entropy"

@dataclass
class ComputeDiversityIndicesConfig(FunctionConfig):
    save_dir: str = '~/data/gvs/evaluation/with_gedi_on_diversity_indices/indices_by_tile/'
    gedi_ours_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_biome_and_ours_full/2020'
    bin_width: int = 5
    year: int = 2020
    _target_: str = "evaluation.diversity_indices.cal_diversity_indices"

@dataclass
class EvaluateDiversityIndicesConfig(FunctionConfig):
    indices_dir: str = '~/data/gvs/evaluation/with_gedi_on_diversity_indices/indices_by_tile/bin_width_5m/2020'
    group_by: Optional[str] = 'BIOME'
    filter_steep_slope: bool = True
    year: int = 2020
    save_dir: str = '~/data/gvs/evaluation/with_gedi_on_diversity_indices/results/'
    _target_: str = "evaluation.diversity_indices.eval_diversity_indices"
    

@dataclass
class ExtractPixelsAndSaveConfig(FunctionConfig):
    ref_dir: str = '~/data/gvs/evaluation/with_airborne_lidar/ALS_MaxGEDIFootprint_GSD10m'
    ours_root_dir: str = '~/data/gvs/predictions'
    save_dir: str = '~/data/gvs/evaluation/with_airborne_lidar/lidar_and_ours_year_matching'
    _target_: str = "evaluation.eval_als.extract_pixels_and_save"

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
defaults = [
    {'run': 'compute_entropy'},
    "_self_"
]


cs = ConfigStore.instance()
cs.store(group='run', name='compute_entropy', node=ComputeEntropyConfig)
cs.store(group='run', name='compute_diversity_indices', node=ComputeDiversityIndicesConfig)
cs.store(group='run', name='evaluate_diversity_indices', node=EvaluateDiversityIndicesConfig)
cs.store(group='run', name='extract_pixels_and_save', node=ExtractPixelsAndSaveConfig)
# ================================ Main Config ================================
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
    runtime = t1 - t0
    print(f'Time taken: {runtime} seconds')
    if cfg.run.get('save_dir', None) is not None:
        generate_run_log(os.path.join(cfg.run.save_dir, 'run.log'), cfg, runtime)
    else:
        print(f'No save_dir provided, skipping run log')

if __name__ == "__main__":
    main()
