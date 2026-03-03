
import time
from typing import List, Optional, Any
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dataclasses import dataclass, field, fields
import hydra
from omegaconf import OmegaConf, MISSING
from config.base_config_class import ClassConfig, FunctionConfig
from const import KEY_RHS, CHM_COLS


@dataclass
class SampleForestTempConfig(FunctionConfig):
    tif_dir: str = '~/data/gvs/downstream_tasks/forest_temp/'
    p: float = 1e-4
    seed: int = 42
    _target_: str = "postprocessing.core.sample_raster.sample_forest_temp"

@dataclass
class MakeParqSubcolumnsConfig(FunctionConfig):
    parq_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_ours_blended/2020'
    subcolumns: list[str] = field(default_factory=lambda: CHM_COLS)
    save_fp: str = '~/data/gvs/evaluation/sota_chm_gedi_ours_test.parquet'
    _target_: str = "tools.make_parq_subcolumns.make_parq_subcolumns"

@dataclass
class GetTilesNodataConfig(FunctionConfig):
    cog_dir: str = '~/data/gvs/predictions/2020/blended/tiles/cog'
    _target_: str = "postprocessing.core.s2_tiling.get_tiles_nodata"

@dataclass
class GetTilesRedundantConfig(FunctionConfig):
    s2_grid_file: str = '~/data/gvs/state/deploy_status.fgb'
    cog_dir: str = '~/data/gvs/predictions/2020/blended/tiles/cog'
    save_dir: str = '~/data/gvs/assets/worklists/'
    _target_: str = "postprocessing.core.s2_tiling.get_tiles_redundant"


@dataclass
class GetTilesReblendConfig(FunctionConfig):
    tiles_list_file: str = '~/data/gvs/assets/worklists/tiles_reblend_2020.txt'
    s2_grid_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    _target_: str = "postprocessing.core.s2_tiling.get_tiles_reblend"

@dataclass
class TranslatePredictionsConfig(FunctionConfig):
    src_dir: str = '~/data/gvs/predictions/2020/original/tiles/geotiff'
    dst_dir: str = '~/data/gvs/predictions/2020/original/tiles/cog'
    _target_: str = "postprocessing.core.translate.translate_tile"

@dataclass
class CreateDistanceMapsConfig(FunctionConfig):
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'
    save_dir: str = '~/data/gvs/assets/blending/distance_maps'
    _target_: str = "postprocessing.corrections.blending.create_distance_map_for_all_tiles"
    

@dataclass
class RepartitionDataConfig(FunctionConfig):
    based_on_col: str = 'assigned_tile'
    parquet_dir: str = '~/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables/val'
    save_dir: str = '~/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables_by_splitted_tile/val'
    _target_: str = "postprocessing.core.repartition_data.repartition_index_table"

@dataclass
class ExtractGEDIFromH5Config(FunctionConfig):
    keep_columns: List[str] = field(default_factory=lambda: [])
    year: int = 2020
    h5_file: str = '~/data/gvs/datasets/splits/data_val.h5'
    index_table_dir: str = '~/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables_by_splitted_tile/val'
    save_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_val/original/2020/'
    _target_: str = "postprocessing.core.extract_sparse_points.extract_gedi_from_h5"

@dataclass
class CheckTwoPartitionedDatasetsConfig(FunctionConfig):
    source_dir: str = '~/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables/val'
    target_dir: str = '~/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables_by_splitted_tile/val'
    _target_: str = "tools.sanity_check.check_two_partitioned_datasets"
    
@dataclass
class MakeManifestConfig(FunctionConfig):
    data_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_val/original/2020/'
    dataset_name: str = 'gedi_val_2020'
    root_note: str = ''
    _target_: str = "tools.make_manifest.make_manifest"

@dataclass
class RunBlendingConfig(ClassConfig):
    year: int = 2020
    tile_id: str = '20MRS'
    flag_dir: str = '~/data/gvs/state/2020/blended'
    output_dir: str = '~/data/gvs/predictions/2020/blended/tiles'
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'
    distance_map_dir: str = '~/data/gvs/assets/blending/distance_maps'
    s2_grid_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    costal_tiles_file: str = '~/data/gvs/assets/worklists/tiles_coastal_snow_regions.txt'
    chunksize: int = 1024
    use_flash: bool = False
    rhs_idx: str = 'key_rhs'
    _target_: str = "postprocessing.corrections.handle_border_artifacts.VSMCorrection"
    target_method: str = 'run_blending_and_mask'

@dataclass
class GetTilesWooEnoughGEDIConfig(FunctionConfig):
    tiles_list_file: str = '~/data/gvs/assets/worklists/total_tiles_2020.txt'
    bias_stats_dir: str = '~/data/gvs/assets/bias_correction_stats/slope_all_minpoints200/2020/stats_by_tile'
    s2_grid_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    _target_: str = "postprocessing.core.s2_tiling.get_tiles_wo_enough_gedi_gt"


@dataclass
class CheckfterBiasCorrectionConfig(FunctionConfig):
    year: int = 2020
    bias_dir: str = '~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/2020/stats_by_tile'
    save_dir: str = '~/data/gvs/predictions/2020/bias_corrected_slope_lt20_minpoints2000/mosaic'
    bias_cutoff: Optional[float] = None
    average_across_rhs: bool = False
    _target_: str = "visualization.create_global_view.check_mosaic_after_bias_correction"

@dataclass
class AddOursToSOTAGEDIConfig(FunctionConfig):
    year: int = 2020
    gedi_chm_reference_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chms/2020'
    save_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chms_ours/2020'
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'
    rh_idxs: list[int] = field(default_factory=lambda: list(range(101)))
    _target_: str = "postprocessing.core.extract_sparse_points.pair_predictions_with_gedi_ref_data"
    
@dataclass
class AddOursBlendedToSOTAGEDIConfig(FunctionConfig):
    year: int = 2020
    gedi_chm_reference_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms/2020'
    save_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_ours_blended/2020'
    pred_parent_dir: str = '~/data/gvs/predictions/2020/blended/tiles/cog'
    rh_idxs: list[int] = field(default_factory=lambda: KEY_RHS)
    _target_: str = "postprocessing.core.extract_sparse_points.pair_predictions_with_gedi_ref_data"


@dataclass
class ExtractPredBiomeConfig(FunctionConfig):
    gedi_ref_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original/2020'
    vsm_dir: str = '~/data/gvs/predictions/2020/blended/tiles/geotiff/'
    save_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_ours_biome/2020'
    biome_file: str = '~/data/GEDI/ecoregions/wwf_terr_ecos.shp'
    _target_: str = "postprocessing.core.extract_sparse_points.extract_pred_add_biome"

@dataclass
class EvaluateBiasCorrectionConfig(FunctionConfig):
    year: int = 2020
    slope_lt20: bool = False
    gedi_chm_ours_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chms_ours/2020'
    correction_stats_dir: str = '~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/2020/stats_with_median_and_trimmed_5_95_by_tile'
    save_dir: str = '~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/2020/figures/cal_slope_lt20'
    _target_: str = "postprocessing.corrections.bias_correction.evaluate_bias_correction_against_sota_chm"

defaults = [
    {'run': 'add_ours_to_sota_gedi'}, # default group
    "_self_"
]

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
cs = ConfigStore.instance()
cs.store(group='run', name='make_parq_subcolumns', node=MakeParqSubcolumnsConfig)
cs.store(group='run', name='get_tiles_reblend', node=GetTilesReblendConfig)
cs.store(group='run', name='repartition_data', node=RepartitionDataConfig)
cs.store(group='run', name='extract_gedi_from_h5', node=ExtractGEDIFromH5Config)
cs.store(group='run', name='check_after_bias_correction', node=CheckfterBiasCorrectionConfig)
cs.store(group='run', name='add_ours_to_sota_gedi', node=AddOursToSOTAGEDIConfig)
cs.store(group='run', name='add_ours_blended_to_sota_gedi', node=AddOursBlendedToSOTAGEDIConfig)
cs.store(group='run', name='evaluate_bias_correction', node=EvaluateBiasCorrectionConfig)
cs.store(group='run', name='get_tiles_wo_enough_gedi_gt', node=GetTilesWooEnoughGEDIConfig)
cs.store(group='run', name='run_blending', node=RunBlendingConfig)
cs.store(group='run', name='extract_pred', node=ExtractPredBiomeConfig)
cs.store(group='run', name='create_distance_maps', node=CreateDistanceMapsConfig)
cs.store(group='run', name='translate_predictions', node=TranslatePredictionsConfig)
cs.store(group='run', name='get_tiles_redundant', node=GetTilesRedundantConfig)
cs.store(group='run', name='get_tiles_nodata', node=GetTilesNodataConfig)
cs.store(group='run', name='sample_forest_temp', node=SampleForestTempConfig)
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
