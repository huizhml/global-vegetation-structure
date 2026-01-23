
import time
from typing import List, Optional, Any
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dataclasses import dataclass, field
import hydra
from omegaconf import OmegaConf, MISSING

    
@dataclass
class CheckTwoDatasetsConfig:
    source_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original/2020/'
    target_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chm/2020/'
    _target_: str = "download.sanity_check.check_npoints_for_two_datasets"


defaults = [
    {'run': 'check_two_datasets'},
    "_self_"
]

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
cs = ConfigStore.instance()
cs.store(group='run', name='check_two_datasets', node=CheckTwoDatasetsConfig)
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