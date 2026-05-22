
from typing import List, Any
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
import hydra
from omegaconf import MISSING
from config.base_config_class import ClassConfig, FunctionConfig
from config.runner import run_cli
from tools.utils import resolve_args

defaults = [
    {'run': 'download_sota_chms'},
    "_self_"
]

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
cs = ConfigStore.instance()


@dataclass
class DownloadForestTempConfig(FunctionConfig):
    url: str = 'https://figshare.com/ndownloader/files/39528400'
    out_dir: str = '~/data/gvs/evaluation/downstream_tasks/forest_temp'
    _target_: str = "download.products.forest_temp.download_forest_temp"

# ================================ Download MGRS Configs ================================
@dataclass
class DownloadMGRSConfig(ClassConfig):
    gee_asset: str = 'projects/gisproject-1/assets/mgrs_with_landmass_and_gedi_counts_high_sens'
    missing_file: str = '~/data/GEDI/missing.csv'
    mgrs_file: str = '~/data/GEDI/mgrs_with_nbest_v2.parquet'
    _target_: str = "download.products.mgrs.MGRS"
    target_method: str = 'get_mgrs'
    
@dataclass
class AddGediCountToMGRSConfig(ClassConfig):
    version: int = 3
    mgrs_file: str = '~/data/GEDI/mgrs_with_nbest_v2.parquet'
    zone_count_tmp: str = '~/data/GEDI/zone_count_tmp'
    _target_: str = "download.products.mgrs.MGRS"
    target_method: str = 'add_gedi_count'

# ================================ Download GEDI Configs ================================
@dataclass
class DownloadAllValidGEDIConfig(ClassConfig):
    year: int = 2020
    n_parallel: int = 100
    gedi_table_index_file: str = f'~/data/gvs/gedi/l2a_orbit_table/l2a_orbit_table_{year}.parquet'
    save_dir: str = f'~/data/gvs/gedi/veg_sensitivity_gt0p95/all_valid/{year}'
    s2_table_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    _target_: str = "download.products.gedi.GEDI"
    target_method: str = 'download_all_valid'
@dataclass
class AddSlopeToGEDIConfig(ClassConfig):
    year: int = 2020
    location_dir: str = f'~/data/gvs/gedi/veg_sensitivity_gt0p95/all_valid/{year}'
    dem_meta_file: str = '~/data/gvs/assets/dem/cop-dem-glo-30_items.parquet'
    save_dir: str = f'~/data/gvs/gedi/veg_sensitivity_gt0p95/all_valid_with_slope/{year}'
    _target_: str = "download.products.gedi.GEDI"
    target_method: str = 'add_slope'
@dataclass
class CheckSlopeDistributionGEDIConfig(ClassConfig):
    year: int = 2020
    deploy_status_dir: str = f'~/data/gvs/state'
    input_dir: str = f'~/data/gvs/gedi/veg_sensitivity_gt0p95/all_valid_with_slope/{year}'
    _target_: str = "download.products.gedi.GEDI"
    target_method: str = 'check_slope_distribution'
@dataclass
class GetUsedGEDIPointsConfig(ClassConfig):
    year: int = 2020
    s2_grid_file: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    used_parq_dir: str = '~/data/gvs/train_subsets'
    save_dir: str = '~/data/gvs/fitting_data_coord_partitions'
    _target_: str = "download.products.gedi.GEDI"
    target_method: str = 'get_used_gedi_points'
@dataclass
class GetTilesCoveredByGEDIConfig(ClassConfig):
    year: int = 2020
    s2_grid_file: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    gedi_table_index_file: str = f'~/data/gvs/gedi/l2a_orbit_table/l2a_orbit_table_{year}.parquet'
    save_dir: str = f'~/data/gvs/deploy/correction_{year}'
    _target_: str = "download.products.gedi.GEDI"
    target_method: str = 'get_tiles_covered_by_gedi'

@dataclass
class DedupShotsConfig(FunctionConfig):
    root_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95'
    split: str = 'test'
    year: int = 2020
    data_name: str = 'original_with_sota_chms_biome'
    parq_dir: str = '{root_dir}/subset_{split}/{data_name}/{year}'
    _target_: str = 'download.products.gedi.dedup_shots'
cs.store(group='run', name='dedup_shots', node=DedupShotsConfig)

# ================================ Download S2 Configs ================================
@dataclass
class S2MetaGatherConfig(ClassConfig):
    year: int = 2020
    zone: str = '23J'
    rewrite: bool = False
    debug: bool = False
    merge_zones: str = ''
    n_parallel: int = 100
    query_days: int = 90
    max_cloud_cover: int = 50
    max_water_percentage: int = 99
    max_retries: int = 3
    gedi_dir: str = '~/data/gvs/GEDI_extra'
    S2_meta_dir: str = '~/data/GEDI/S2_geoparquet_items'
    temp_dir: str = '~/data/gvs/S2_temp'
    s2_grid_file: str = '~/data/GEDI/Sentinel-2_tilling_shp/sentinel_2_index_shapefile.shp'
    save_dir: str = '~/data/gvs/GEDI_extra_with_s2_candidates'
    _target_: str = "download.products.sentinel2.train_meta.S2MetaGather"
    target_method: str = 'process_zone'
@dataclass
class S2BestCandidatesAPIConfig(ClassConfig):
    year: int = 2020
    zone: str = '23J'
    rewrite: bool = False
    debug: bool = False
    patch_size: int = 15
    out_res: int = 10
    maxCloudCover: int = 50
    maxWaterPercentage: int = 100
    queryDaysRange: int = 90
    extendDays: int = 30
    sem_max_release: int = 60
    n_parallel: int = 100
    gedi_dir: str = '~/data/gvs/GEDI_extra_with_s2_candidates'
    S2_meta_dir: str = '~/data/GEDI/S2_geoparquet_items'
    save_dir: str = '~/data/gvs/GEDI_extra_with_s2_candidates_and_best'
    flag_dir: str = '~/data/gvs/flags_find_best/'
    _target_: str = "download.products.sentinel2.train_best_candidates_api.BestS2FinderAPI"
    target_method: str = 'find_best_s2'
@dataclass
class S2BestCandidatesConfig(ClassConfig):
    year: int = 2020
    zone: str = '23J'
    rewrite: bool = False
    debug: bool = False
    patch_size: int = 15
    out_res: int = 10
    n_parallel: int = 100
    gedi_dir: str = '~/data/gvs/GEDI_extra_with_s2_candidates'
    save_dir: str = '~/data/gvs/GEDI_extra_with_s2_candidates_and_best'
    S2_meta_dir: str = '~/data/GEDI/S2_geoparquet_items'
    flag_dir: str = '~/data/gvs/flags_find_best/'
    _target_: str = "download.products.sentinel2.train_best_candidates.BestS2Finder"
    target_method: str = 'find_best_s2'
@dataclass
class S2DownloadConfig(ClassConfig):
    year: int = 2020
    zone: str = '23J'
    rewrite: bool = False
    debug: bool = False
    patch_size: int = 15
    out_res: int = 10
    n_parallel: int = 8
    gedi_dir: str = '~/data/gvs/GEDI_extra_with_s2_candidates_and_best'
    save_dir: str = '~/data/gvs/GEDI_S2_h5_extra'
    flag_dir: str = '~/data/gvs/Correct_order_flags/'
    S2_meta_dir: str = '~/data/GEDI/S2_geoparquet_items'
    wc_dem_meta_dir: str = '~/data/GEDI'
    _target_: str = "download.products.sentinel2.train_data.S2Downloader"
    target_method: str = 'download_zone'

# ================================ Download Inference Configs ================================

@dataclass
class AggGrowingSeasonConfig(ClassConfig):
    year: int = 2020
    version: str = 'v1'
    asset_id: str = 'projects/gisproject-1/assets/sentinel_2_shapefile_with_growing_month_counts'
    save_dir: str = '~/data/gvs/state'
    _target_: str = "download.products.growing_season.GrowingSeason"
    target_method: str = 'get_growing_months_per_tile'
@dataclass
class DownloadInferenceConfig(ClassConfig):
    year: int = 2020
    zone: str = '23J'
    rewrite: bool = False
    debug: bool = False
    s2_grid_file: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    wc_parq_file:str = '~/data/gvs/deploy/esa_wc.parquet'
    store_name: str = 'inference'
    comp_name: str = 'lz4'
    comp_level: int = 7
    max_cloud_cover: int = 90
    max_water_percentage: int = 99
    total_splits: int = 21
    job_id: int = 5
    _target_: str = "download.products.sentinel2.inference.WorldS2"
    target_method: str = 'download' # TODO: needs further refactoring
    
# ================================ Download Downstream Task Data Configs ================================
@dataclass
class DownloadDownstreamTaskDataConfig(ClassConfig):
    task_name: str = ''
    crowd_source_data_file: str = '~/data/gvs/downstream_task_data/naturalness/reference_data_set_updated.csv'
    output_dir: str = '~/data/gvs/downstream_task_data'
    s2_parquet: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    wc_dem_meta_dir: str = '~/data/GEDI'
    n_parallel: int = 100
    maxCloudCover: int = 50
    year: int = 2017
    patch_size: int = 31
    out_res: int = 10
    job_id: int = 0
    _target_: str = "download.products.sentinel2.train_data.S2DownloaderDownstream"
    target_method: str = 'download' # TODO: needs further refactoring

# ================================ Download DEM Configs ================================
@dataclass
class DownloadDEMConfig(ClassConfig):
    year: int = 2020
    loc_dir: str = f'~/data/gvs/gedi/veg_sensitivity_gt0p95/{year}/subset_4k/original/'
    dem_meta_file: str = '~/data/gvs/assets/dem/cop-dem-glo-30_items.parquet'
    save_dir: str = f'~/data/gvs/gedi/veg_sensitivity_gt0p95/{year}/subset_4k/original_with_slope'
    _target_: str = "download.products.dem.DEMDownloader"
    target_method: str = 'add_slope'


# ================================ Download SOTA CHM Configs ================================

@dataclass
class DownloadSOTAChmConfig(ClassConfig):
    location_files: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original/2020/*.parquet'
    save_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chms/2020'
    n_parallel: int = 60
    rewrite: bool = False
    debug: bool = False
    _target_: str = "download.products.sota_chm.SOTAChmDownloader"
    target_method: str = 'download'



cs.store(group='run', name='download_sota_chms', node=DownloadSOTAChmConfig)
cs.store(group='run', name='download_forest_temp', node=DownloadForestTempConfig)
# ================================ MGRS Configs ================================
cs.store(group='run', name='get_mgrs', node=DownloadMGRSConfig)
cs.store(group='run', name='add_gedi_count', node=AddGediCountToMGRSConfig)

# ================================ GEDI Configs ================================
cs.store(group='run', name='download_all_valid_gedi', node=DownloadAllValidGEDIConfig)
cs.store(group='run', name='add_slope', node=AddSlopeToGEDIConfig)
cs.store(group='run', name='check_slope_distribution', node=CheckSlopeDistributionGEDIConfig)
cs.store(group='run', name='get_used_gedi_points', node=GetUsedGEDIPointsConfig)
cs.store(group='run', name='get_tiles_covered_by_gedi', node=GetTilesCoveredByGEDIConfig)

# ================================ S2 Configs ================================
cs.store(group='run', name='s2_meta_gather', node=S2MetaGatherConfig)
cs.store(group='run', name='s2_best_candidates_api', node=S2BestCandidatesAPIConfig)
cs.store(group='run', name='s2_best_candidates', node=S2BestCandidatesConfig)
cs.store(group='run', name='s2_download', node=S2DownloadConfig)

# ================================ Download Inference Configs ================================
cs.store(group='run', name='agg_growing_season', node=AggGrowingSeasonConfig)
cs.store(group='run', name='download_inference', node=DownloadInferenceConfig)

# ================================ Download Downstream Task Data Configs ================================
cs.store(group='run', name='download_downstream_task_data', node=DownloadDownstreamTaskDataConfig)

# ================================ Download SOTA CHM Configs ================================
cs.store(group='run', name='download_sota_chms', node=DownloadSOTAChmConfig)

# ================================ Main Config ================================
cs.store(name='base_config', node=RunConfig) # NOTE: name here should match the default in ../config/base/no_log.yaml

@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    resolve_args(cfg)
    run_cli(cfg)


if __name__ == "__main__":
    main()