
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
from tools.utils import resolve_args


cs = ConfigStore.instance()
    
defaults = [
    {'run': 'compute_entropy'},
    "_self_"
]

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING

# ================================ Main Config ================================
cs.store(name='base_config', node=RunConfig) # NOTE: name here should match the default in ../config/base/no_log.yaml


# =======================================
#   Evaluate VSM on GEDI
# =======================================
@dataclass
class EvaluateVSMOnGEDIConfig(FunctionConfig):
    ref_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original/'
    ours_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_biome_and_ours_full/2020'
    save_dir: str = '~/data/gvs/evaluation/with_gedi_on_diversity_indices/results/'
    _target_: str = "evaluation.on_gedi.evaluate_vsm_on_gedi"
    
cs.store(group='run', name='evaluate_vsm_on_gedi', node=EvaluateVSMOnGEDIConfig)
    
# =======================================
#   Evaluate VSM top height with SOTA CHMs
# =======================================
@dataclass
class EvaluateVSMTopHeightWithSOTAChmsConfig(FunctionConfig):
    year: int = 2020
    split: str = 'test'
    data_name: str = 'original_with_sota_chms_biome_and_ours_full'
    ours_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_{split}/{data_name}/{year}'
    save_dir: str = '~/data/gvs/evaluation/with_sota_chm/results/'
    _target_: str = "evaluation.on_sota_chm.evaluate_chm_with_sota"
    
cs.store(group='run', name='evaluate_chm_with_sota', node=EvaluateVSMTopHeightWithSOTAChmsConfig)


# =======================================
#   Evaluate CHM with ALS and LVIS
# =======================================
@dataclass
class EvaluateCHMWithALSAndLVISConfig(FunctionConfig):
    df_dir: str = '/projects/dereeco/data/gvs/evaluation/with_airborne_lidar/lidar_and_ours_year_matching'
    save_dir: str = '/projects/dereeco/data/gvs/evaluation/with_airborne_lidar/figures/evaluated_on_ours_year_matching'
    ref_col: str = 'als'
    _target_: str = "evaluation.on_als.evaluate"

cs.store(group='run', name='evaluate_chm_with_als_and_lvis', node=EvaluateCHMWithALSAndLVISConfig)

# =======================================
#   Diversity indices
# =======================================

@dataclass
class DiversityIndicesMapConfig(FunctionConfig):
    year: int = 2020
    bin_width: int = 5
    max_height: int = 150
    product: str = 'diversity_indices'
    product_version: str = 'masked'
    product_format: str = 'cog'
    tif_dir: str = '~/data/gvs/products/vsm/2020/{product_version}/mosaic/{product_format}'
    save_dir: str = '~/data/gvs/products/{product}/{year}/{product_version}/mosaic/'
    _target_: str = "evaluation.diversity_maps.create_global_diversity_maps"
cs.store(group='run', name='generate_diversity_indices_map', node=DiversityIndicesMapConfig)

# =======================================
#   Biome analysis - diversity indices
# =======================================
@dataclass
class SamplePointsByBiomeConfig(FunctionConfig):
    biome_file: str = '~/data/GEDI/ecoregions/wwf_terr_ecos.shp'
    n_samples: int = 100000
    save_dir: str = '~/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome_veg'
    plot_points: bool = True
    _target_: str = "evaluation.utils.sample_points_by_biome"

cs.store(group='run', name='sample_points_by_biome', node=SamplePointsByBiomeConfig)
@dataclass
class PartitionPointsByTileConfig(FunctionConfig):
    gdf_file: str = '~/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome_veg/random_sample_100000_points_per_biome.parquet'
    s2_tile_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    save_dir: str = '~/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome_veg_by_tile'
    _target_: str = "evaluation.utils.partition_points_by_tile"

cs.store(group='run', name='partition_points_by_tile', node=PartitionPointsByTileConfig)
@dataclass
class ComputeEntropyConfig(FunctionConfig):
    output_dir: str = '~/data/gvs/products/profile_entropy/2020/tiles/geotiff'
    tile_id: str = '36NTF'
    year: int = 2020
    chunk_size: int = 512
    max_workers: int = 8
    bin_width: int = 50
    _target_: str = "evaluation.on_diversity_indices.compute_entropy"

cs.store(group='run', name='compute_entropy', node=ComputeEntropyConfig)
@dataclass
class ComputeDiversityIndicesConfig(FunctionConfig):
    bin_width: int = 5
    max_height: int = 150
    save_dir: str = '~/data/gvs/evaluation/with_gedi_on_diversity_indices/indices_by_tile/max_height_{max_height}m_bin_{bin_width}m'
    gedi_ours_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_biome_and_ours_full/2020'
    year: int = 2020
    _target_: str = "evaluation.on_diversity_indices.cal_diversity_indices"
    
cs.store(group='run', name='compute_diversity_indices', node=ComputeDiversityIndicesConfig)
@dataclass
class EvaluateDiversityIndicesConfig(FunctionConfig):
    max_height: int = 150
    bin_width: int = 5
    indices_dir: str = '~/data/gvs/evaluation/with_gedi_on_diversity_indices/indices_by_tile/max_height_{max_height}m_bin_{bin_width}m'
    group_by: Optional[str] = 'BIOME'
    filter_steep_slope: bool = True
    year: int = 2020
    plot_scatter: bool = False
    plot_boxplot: bool = True
    save_dir: str = '~/data/gvs/evaluation/with_gedi_on_diversity_indices/results/'
    _target_: str = "evaluation.on_diversity_indices.eval_diversity_indices"
    
cs.store(group='run', name='evaluate_diversity_indices', node=EvaluateDiversityIndicesConfig)

@dataclass
class PlotBiomeCombinedBoxplotConfig(FunctionConfig):
    max_height: int = 150
    bin_width: int = 5
    indices_dir: str = '~/data/gvs/evaluation/with_gedi_on_diversity_indices/indices_by_tile/max_height_{max_height}m_bin_{bin_width}m'
    group_by: Optional[str] = None
    filter_steep_slope: bool = True
    year: int = 2020
    plot_biome_combined_boxplot: bool = True
    save_dir: str = '~/data/gvs/evaluation/with_gedi_on_diversity_indices/results/'
    _target_: str = "evaluation.on_diversity_indices.eval_diversity_indices"
   
cs.store(group='run', name='plot_biome_combined_boxplot', node=PlotBiomeCombinedBoxplotConfig) 


@dataclass
class PlotResidualsRh98BinedConfig(FunctionConfig):
    bin_width: int = 5
    max_height: int = 150
    group_by: Optional[str] = None
    filter_steep_slope: bool = True
    year: int = 2020
    plot_biome_combined_boxplot: bool = False
    root_dir: str = '~/data/gvs/evaluation/with_gedi_on_diversity_indices'
    indices_dir: str = '{root_dir}/indices_by_tile/max_height_{max_height}m_bin_{bin_width}m'
    save_dir: str = '{root_dir}/results/steep_slope_filtered_bin_{bin_width}m_max_height_{max_height}m'
    _target_: str = "evaluation.on_diversity_indices.eval_diversity_indices"
   
cs.store(group='run', name='plot_residuals_rh98_bined', node=PlotResidualsRh98BinedConfig) 
@dataclass
class CalS2PatchStatsConfig(FunctionConfig):
    year: int = 2017
    ps: int = 15
    product: str = 's2'
    root_dir: str = '~/data/gvs/evaluation/downstream_tasks/naturalness'
    ref_csv_train: str = '{root_dir}/reference_data_set_updated_train.csv'
    patch_file: str = '{root_dir}/results_from_vsm_{year}/s2_gedi_patches_ps31/s2_{year}_ps31.h5'
    out_file: str = '{root_dir}/results_from_vsm_{year}/intermediates/s2_patch_stats_ps{ps}_train.parquet'
    _target_: str = "evaluation.on_naturalness.cal_patch_stats"
    
cs.store(group='run', name='cal_s2_patch_stats', node=CalS2PatchStatsConfig)

@dataclass
class CalAlphaEMPatchStatsConfig(FunctionConfig):
    year: int = 2017
    ps: int = 15
    product: str = 'alpha_em'
    root_dir: str = '~/data/gvs/evaluation/downstream_tasks/naturalness'
    ref_csv_train: str = '{root_dir}/reference_data_set_updated_train.csv'
    patch_file: str = '{root_dir}/alphaearth_embeddings/alphaearth_embeddings.h5'
    out_file: str = '{root_dir}/results_from_vsm_{year}/intermediates/alpha_em_patch_stats_ps{ps}_train.parquet'
    _target_: str = "evaluation.on_naturalness.cal_patch_stats"
    
cs.store(group='run', name='cal_alpha_em_patch_stats', node=CalAlphaEMPatchStatsConfig)

@dataclass
class CalVSM17PatchStatsConfig(FunctionConfig): # extract both train and val patches at once
    year: int = 2017
    ps: int = 15
    product: str = 'vsm'
    root_dir: str = '~/data/gvs/evaluation/downstream_tasks/naturalness'
    ref_csv_train: str = '{root_dir}/reference_data_set_updated_train.csv'
    patch_file: str = '{root_dir}/results_from_vsm_{year}/vsm_patches_ps15_single_h5_72xl3wma_ps31.zarr'
    # patch_file: str = '{root_dir}/results_from_vsm_{year}/vsm_patches_ps15_single_h5/rhs_predictions_2017_cg11fpjr_old.h5'
    out_file: str = '{root_dir}/results_from_vsm_{year}/intermediates/vsm_patch_stats_ps{ps}_train.parquet'
    _target_: str = "evaluation.on_naturalness.cal_patch_stats"
    
cs.store(group='run', name='cal_vsm_17_patch_stats', node=CalVSM17PatchStatsConfig)
# For 2020
@dataclass
class CalVSMPatchStatsConfig(FunctionConfig):
    split: str = 'val'
    year: int = 2020
    root_dir: str = '~/data/gvs/evaluation/downstream_tasks/naturalness'
    ref_by_tile_dir: str = '{root_dir}/loc_by_tile_{split}'
    vsm_patches_dir: str = '{root_dir}/results_from_vsm_{year}/vsm_patches_ps15_single_h5'
    save_dir: str = '{root_dir}/results_from_vsm_{year}/vsm_s2_alpha_patch_stats_ps15_{split}'
    s2_patch_file: str = '{root_dir}/results_from_vsm_{year}/s2_gedi_patches_ps31/s2_{year}_ps31.h5'
    alpha_em_patch_file: str = '{root_dir}/alphaearth_embeddings/alphaearth_embeddings.h5'
    _target_: str = "evaluation.on_naturalness.cal_vsm_patch_stats"
    
cs.store(group='run', name='cal_vsm_patch_stats', node=CalVSMPatchStatsConfig)
@dataclass
class MergePatchStatsConfig(FunctionConfig):
    parq_dir: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2017/intermediates'
    ps: int = 15
    filename_pattern: str = '*train.parquet'
    save_fp: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2017/vsm_patch_stats_ps{ps}_train.parquet'
    _target_: str = "tools.parq_ops.merge_columns_from_files"
    
cs.store(group='run', name='merge_patch_stats', node=MergePatchStatsConfig)

@dataclass
class RunNaturalnessClassificationConfig(FunctionConfig):
    year: int = 2017
    ps: int = 15
    root_dir: str = '~/data/gvs/evaluation/downstream_tasks/naturalness'
    classifier: str = 'logistic_regression'
    patch_stats_dir: str = '{root_dir}/results_from_vsm_{year}'
    save_dir: str = '{root_dir}/results_from_vsm_{year}/naturalness_classification_ps{ps}'
    _target_: str = "evaluation.on_naturalness.run_classification"
    
cs.store(group='run', name='run_naturalness_classification', node=RunNaturalnessClassificationConfig)

@dataclass
class PlotBarsConfig(FunctionConfig):
    root_dir: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2017/naturalness_classification_ps15'
    summary_file: str = '{root_dir}/logistic_regression_summary_reports.csv'
    per_class_file: str = '{root_dir}/logistic_regression_per_class_reports.csv'
    all_cms_file: str = '{root_dir}/logistic_regression_confusion_matrices.npz'
    save_dir: str = '{root_dir}'
    baseline_name: str = 'rh98'
    groups: tuple[str] = field(default_factory=lambda: ('rh98', 'full_profile', 'rh98_s2', 'key_rhs', 'rh98_cr', 'rh98_fhd', 'rh98_enl2d', 'rh98_fhd_enl1d_enl2d_cr', 'full_profile_s2'))
    _target_: str = "evaluation.on_naturalness.plot_results"
    
cs.store(group='run', name='plot_bars_spatial_context', node=PlotBarsConfig)

@dataclass
class PlotBarsCenterPixelConfig(FunctionConfig):
    root_dir: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2017/naturalness_classification_ps15'
    summary_file: str = '{root_dir}/logistic_regression_summary_reports.csv'
    per_class_file: str = '{root_dir}/logistic_regression_per_class_reports.csv'
    all_cms_file: str = '{root_dir}/logistic_regression_confusion_matrices.npz'
    save_dir: str = '{root_dir}'
    baseline_name: str = 'full_profile_center'
    groups: tuple[str] = field(default_factory=lambda: ('full_profile_center', 'full_profile'))
    _target_: str = "evaluation.on_naturalness.plot_results"
    
cs.store(group='run', name='plot_bars_center_pixel', node=PlotBarsCenterPixelConfig)


# =======================================
#   Compute GLCM texture
# =======================================

@dataclass
class ComputeGLCMTextureConfig(FunctionConfig):
    vsm_patches_dir: str = '~/data/gvs/evaluation/downstream_tasks/naturalness/results_from_vsm_2020/vsm_patches_ps11_train'
    save_dir: str = '/projects/dereeco/data/gvs/downstream_tasks/naturalness/results_from_vsm_2020/glcm_texture_train'
    bin_width: int = 5
    n_levels: int = 100
    _target_: str = "evaluation.on_glcm_texture.cal_vsm_patch_texture"
    
cs.store(group='run', name='compute_glcm_texture', node=ComputeGLCMTextureConfig)




@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    t0 = time.time()
    print(OmegaConf.to_yaml(cfg))
    resolve_args(cfg.run)
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
