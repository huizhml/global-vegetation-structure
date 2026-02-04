
import time
from typing import List, Optional, Any
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dataclasses import dataclass, field
import hydra
from omegaconf import OmegaConf, MISSING


@dataclass
class ClassConfig:
    _target_: str = MISSING
    target_type: str = 'class'
    target_method: str = MISSING

@dataclass
class FunctionConfig:
    _target_: str = MISSING
    target_type: str = 'function'

@dataclass
class RunBlendingConfig(ClassConfig):
    year: int = 2020
    tile_id: str = '20MRS'
    flag_dir: str = '~/data/gvs/state/2020/blended'
    output_dir: str = '~/data/gvs/predictions/2020/blended/tiles'
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'
    distance_map_dir: str = '~/data/gvs/assets/blending/distance_maps'
    s2_grid_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    total_tiles_file: str = '~/data/gvs/assets/worklists/total_tiles_2020.txt'
    costal_tiles_file: str = '~/data/gvs/assets/worklists/tiles_coastal_regions.txt'
    chunksize: int = 1024
    use_flash: bool = False
    rhs_idx: str = 'key_rhs'
    _target_: str = "postprocess.handle_border_artifacts.VSMCorrection"
    target_method: str = 'run_blending'

@dataclass
class GetTilesWooEnoughGEDIConfig(FunctionConfig):
    tiles_list_file: str = '~/data/gvs/assets/worklists/total_tiles_2020.txt'
    bias_stats_dir: str = '~/data/gvs/assets/bias_correction_stats/slope_all_minpoints200/2020/stats_by_tile'
    s2_grid_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    _target_: str = "postprocess.handle_border_artifacts.get_tiles_wo_enough_gedi_gt"


@dataclass
class CheckfterBiasCorrectionConfig(FunctionConfig):
    year: int = 2020
    bias_dir: str = '~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/2020/stats_by_tile'
    save_dir: str = '~/data/gvs/predictions/2020/bias_corrected_slope_lt20_minpoints2000/mosaic'
    bias_cutoff: Optional[float] = None
    average_across_rhs: bool = False
    _target_: str = "visualization.run.check_mosaic_after_bias_correction"

@dataclass
class PairOursSotaGEDIConfig(FunctionConfig):
    year: int = 2020
    gedi_chm_reference_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chms/2020'
    save_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chms_ours/2020'
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'
    _target_: str = "postprocess.bias_correction.pair_predictions_with_gedi_ref_data"
    


@dataclass
@dataclass
class ExtractPredBiomeConfig(FunctionConfig):
    gedi_ref_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original/2020'
    vsm_dir: str = '~/data/gvs/predictions/2020/blended/tiles/geotiff/'
    save_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_ours_biome/2020'
    biome_file: str = '~/data/GEDI/ecoregions/wwf_terr_ecos.shp'
    _target_: str = "postprocess.extract_sparse_pred.extract_pred_add_biome"


class EvaluateBiasCorrectionConfig(FunctionConfig):
    year: int = 2020
    slope_lt20: bool = False
    gedi_chm_ours_dir: str = '~/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/original_with_sota_chms_ours/2020'
    correction_stats_dir: str = '~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/2020/stats_with_median_and_trimmed_5_95_by_tile'
    save_dir: str = '~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/2020/figures/cal_slope_lt20'
    _target_: str = "postprocess.bias_correction.evaluate_bias_correction_against_sota_chm"

defaults = [
    {'run': 'pair_ours_sota_gedi'}, # default group
    "_self_"
]

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
cs = ConfigStore.instance()
cs.store(group='run', name='check_after_bias_correction', node=CheckfterBiasCorrectionConfig)
cs.store(group='run', name='pair_ours_sota_gedi', node=PairOursSotaGEDIConfig)
cs.store(group='run', name='evaluate_bias_correction', node=EvaluateBiasCorrectionConfig)
cs.store(group='run', name='get_tiles_wo_enough_gedi_gt', node=GetTilesWooEnoughGEDIConfig)
cs.store(group='run', name='run_blending', node=RunBlendingConfig)
cs.store(group='run', name='extract_pred', node=ExtractPredBiomeConfig)
cs.store(name='config', node=RunConfig)

@hydra.main(config_name='config', version_base='1.2')
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
    return obj

if __name__ == "__main__":
    main()
