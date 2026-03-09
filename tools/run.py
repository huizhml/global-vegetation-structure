
import time
from typing import List, Optional, Any
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dataclasses import dataclass, field, fields
import hydra
from omegaconf import OmegaConf, MISSING
from config.base_config_class import ClassConfig, FunctionConfig


@dataclass
class UpdateReadmeConfig(FunctionConfig):
    data_dir: str = '~/data/gvs/assets'
    dry_run: bool = False
    _target_: str = "tools.update_readme.update_all_readmes"


defaults = [
    {'run': 'update_readme'}, # default group
    "_self_"
]

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
cs = ConfigStore.instance()
cs.store(group='run', name='update_readme', node=UpdateReadmeConfig)
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
    print(f'Time taken: {t1 - t0} seconds')

if __name__ == "__main__":
    main()
