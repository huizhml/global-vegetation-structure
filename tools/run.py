
import time
from typing import List, Optional, Any
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dataclasses import dataclass, field, fields
import hydra
from omegaconf import OmegaConf, MISSING
from config.base_config_class import ClassConfig, FunctionConfig
from tools.utils import resolve_args


defaults = [
    {'run': 'update_readme'}, # default group
    "_self_"
]

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
cs = ConfigStore.instance()

@dataclass
class UpdateReadmeConfig(FunctionConfig):
    data_dir: str = '~/data/gvs/assets'
    dry_run: bool = False
    _target_: str = "tools.update_readme.update_all_readmes"
cs.store(group='run', name='update_readme', node=UpdateReadmeConfig)


@dataclass
class AddColsFromDirConfig(FunctionConfig):
    target_dir: str = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_ours_biome/2020'
    source_dir: str = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chms/2020'
    validate_cols: List = field(default_factory=lambda: ['shot_number', 'geometry'])
    _target_: str = 'tools.parq_ops.add_columns_from_dir'
cs.store(group='run', name='add_cols_from_dir', node=AddColsFromDirConfig)


@dataclass
class MergeColsFromDirsConfig(FunctionConfig):
    year: int = 2020
    target_dir: str = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/version=masked/{year}'
    source_dir: str = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_biome/{year}'
    save_fp: str = '/projects/dereeco/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/version=masked/original_with_ours_sota_biome_2020.parquet'
    validate_cols: List = field(default_factory=lambda: ['shot_number', 'geometry'])
    _target_: str = 'tools.parq_ops.merge_columns_from_dirs'
cs.store(group='run', name='merge_cols_from_dirs', node=MergeColsFromDirsConfig)



# ================================ Main Config ================================
cs.store(name='base_config', node=RunConfig) # NOTE: name here should match the default in ../config/base/no_log.yaml

@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    t0 = time.time()
    print(OmegaConf.to_yaml(cfg))
    resolve_args(cfg)
    if cfg.run.target_type == 'function':
        instantiate(cfg.run)
    elif cfg.run.target_type == 'class':
        obj = instantiate(cfg.run)
        excute_method = getattr(obj, cfg.run.target_method)
        excute_method(**cfg.run.func_args)
    else:
        raise ValueError(f"Invalid target: {cfg.run.target_type}")
    t1 = time.time()
    print(f'Time taken: {t1 - t0} seconds')


if __name__ == "__main__":
    main()
