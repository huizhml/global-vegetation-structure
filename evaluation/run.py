
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
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
defaults = [
    {'run': 'compute_entropy'},
    "_self_"
]


cs = ConfigStore.instance()
cs.store(group='run', name='compute_entropy', node=ComputeEntropyConfig)


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
