
import time
from typing import List, Optional, Any
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dataclasses import dataclass, field
import hydra
from omegaconf import OmegaConf, MISSING

@dataclass
class DownloadSOTAChmConfig:
    location_files: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original/2020/*.parquet'
    save_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chms/2020'
    n_parallel: int = 60
    rewrite: bool = False
    debug: bool = False
    _target_: str = "download._7_download_sota_chm.download_sota_chms"

@dataclass
class CheckTwoDatasetsConfig:
    source_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original/2020/'
    target_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chms/2020/'
    _target_: str = "download.sanity_check.check_npoints_for_two_datasets"
    

@dataclass
class CheckTwoPartitionedDatasetsConfig:
    source_dir: str = '~/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables/cal'
    target_dir: str = '~/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables_by_splitted_tile/cal'
    _target_: str = "download.sanity_check.check_total_points_for_two_partitioned_datasets"

@dataclass
class MakeManifestConfig:
    data_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original/2020/'
    dataset_name: str = 'gedi_cal_2020'
    root_note: str = ''
    _target_: str = "download.make_manifest.make_manifest"

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
cs.store(group='run', name='check_two_partitioned_datasets', node=CheckTwoPartitionedDatasetsConfig)
cs.store(group='run', name='make_manifest', node=MakeManifestConfig)
cs.store(group='run', name='download_sota_chms', node=DownloadSOTAChmConfig)
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