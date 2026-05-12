import time
from typing import List, Optional, Any
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dataclasses import dataclass, field
import hydra
from omegaconf import OmegaConf, MISSING
from config.base_config_class import FunctionConfig, ClassConfig
from tools.utils import resolve_args


defaults = [
    {'run': 'resample_and_mosaic'},
    "_self_"
]

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    run: Any = MISSING
    
cs = ConfigStore.instance()

cs.store(name='base_config', node=RunConfig) # NOTE: name here should match the default in ../config/base/no_log.yaml


# -----------------------------------------------------------------
#  Create global mosaics
# -----------------------------------------------------------------
@dataclass
class CreateGlobalMosaicConfig(ClassConfig):
    year: int = 2020
    version: str = 'masked'
    rh_idx: int = 99
    q_idx: int = 1
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local_masked'
    total_tiles_file: str = '~/data/gvs/assets/worklists/total_tiles_2020.txt'
    save_dir: str = '~/data/gvs/products/vsm/{year}/{version}/mosaic/'
    _target_: str = "visualization.core.create_global_view.GlobalMosaicker"
    target_method: str = 'create_global_mosaic'
    
cs.store(group='run', name='create_global_mosaic', node=CreateGlobalMosaicConfig)


@dataclass
class CreateGlobalDiffMosaicConfig(ClassConfig):
    year: int = 2020
    version: str = 'masked'
    rh_idx: int = 99
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local_masked'
    total_tiles_file: str = '~/data/gvs/assets/worklists/total_tiles_2020.txt'
    save_dir: str = '~/data/gvs/products/prediction_intervals/{year}/{version}/mosaic/'
    _target_: str = "visualization.core.create_global_view.GlobalMosaicker"
    target_method: str = 'create_global_diff_mosaic'
    
cs.store(group='run', name='create_global_diff_mosaic', node=CreateGlobalDiffMosaicConfig)

@dataclass
class CreateGlobalRelativeDiffMosaicConfig(ClassConfig):
    year: int = 2020
    version: str = 'masked'
    rh_idx: int = 99
    q_idx: int = 1
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local_masked'
    total_tiles_file: str = '~/data/gvs/assets/worklists/total_tiles_2020.txt'
    save_dir: str = '~/data/gvs/products/relative_prediction_intervals/{year}/{version}/mosaic/'
    _target_: str = "visualization.core.create_global_view.GlobalMosaicker"
    target_method: str = 'create_global_relative_diff_mosaic'
    
cs.store(group='run', name='create_global_relative_diff_mosaic', node=CreateGlobalRelativeDiffMosaicConfig)

@dataclass
class CreateGlobalMosaicPdfConfig(FunctionConfig):
    year: int = 2020
    version: str = 'masked'
    data_format: str = 'cog'
    product: str = 'vsm'
    mosaic_dir: str = '~/data/gvs/products/{product}/{year}/{version}/mosaic/{data_format}'
    tif_filename_pattern: str = '*.tif'
    pdf_file: str = '~/data/gvs/results/{product}/global_mosaic_{year}_{version}.pdf'
    multi_pages: bool = False
    cmin: Optional[int] = None
    cmax: Optional[int] = None
    cmap: Optional[str] = None
    _target_: str = "visualization.core.create_pdf_thumb.make_global_mosaic_pdf"
cs.store(group='run', name='create_global_mosaic_pdf', node=CreateGlobalMosaicPdfConfig)


# -----------------------------------------------------------------
#  Visualize datacube
# -----------------------------------------------------------------
@dataclass
class VisualizeDatacubeConfig(FunctionConfig):
    product: str = 'prediction_intervals'
    product_format: str = 'cog'
    product_version: str = 'masked'
    filename_pattern: str = '*.tif'
    year: int = 2020
    rh_step: int = 2
    version: str = '2'
    data_dir: str = '~/data/gvs/products/{product}/{year}/{product_version}/mosaic/{product_format}/'
    save_path: str = '~/data/gvs/results/vsm_datacube/{product}_every{rh_step}rhs_black_bg_v{version}.png'
    cmap: str = 'cividis'
    _target_: str = "visualization.core.datacube.visualize_datacube"
cs.store(group='run', name='visualize_datacube', node=VisualizeDatacubeConfig)

# -----------------------------------------------------------------
#  Create cloud cover boxplot
# -----------------------------------------------------------------
@dataclass
class CreateCloudCoverBoxplotConfig(FunctionConfig):
    h5_dir: str = '~/data/gvs/inputs/inference_2020/'
    tile_id_file: str = '~/data/gvs/assets/worklists/tiles_system_biased.txt'
    pdf_file: str = '~/data/gvs/diagnostics/pred_thumbs/system_biased_tiles_cloud_cover.pdf'
    s2_grid_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    year: int = 2020
    max_cloud_cover: int = 90
    _target_: str = "visualization.core.create_boxplot.make_cloud_cover_boxplot"
    
    
@dataclass
class CreateRhPairPdfConfig(FunctionConfig):
    tif_dir: str = '~/data/gvs/products/vsm/2020/blended/tiles/cog/'
    tile_id_file: str = '~/data/gvs/assets/worklists/tiles_intile_abnormal.txt'
    pdf_file: str = '~/data/gvs/diagnostics/pred_thumbs/issue_tiles.pdf'
    overview_level: int = 2
    top_rh: int = 98
    low_rh: int = 25
    _target_: str = "visualization.core.create_pdf_thumb.make_rh_pair_pdf"
@dataclass
class CreatePredNeighborPdfConfig(FunctionConfig):
    tif_dir: str = '~/data/gvs/products/vsm/2020/blended/tiles/cog/'
    tile_id_file: str = '~/data/gvs/assets/worklists/tiles_system_biased.txt'
    pdf_file: str = '~/data/gvs/diagnostics/pred_thumbs/system_biased_tiles.pdf'
    s2_grid_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'
    year: int = 2020
    resolution: int = 100
    _target_: str = "visualization.core.create_pdf_thumb.make_pred_neighbor_pdf"
@dataclass
class ResampleAndMosaicConfig(FunctionConfig):
    year: int = 2020
    rh_idx: int = 98
    q_idx: int = 1
    pred_dir: str = '~/data/gvs/products/vsm/2020/blended/tiles/geotiff/'
    s2_grid_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    save_dir: str = '~/data/gvs/products/vsm/2020/blended/mosaic/'
    _target_: str = "visualization.core.create_global_view.resample_and_mosaic"

@dataclass
class CheckfterBiasCorrectionConfig:
    year: int = 2020
    bias_dir: str = '~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/2020/stats_by_tile'
    save_dir: str = '~/data/gvs/products/vsm/2020/bias_corrected_slope_lt20_minpoints2000/mosaic'
    bias_cutoff: Optional[float] = None
    average_across_rhs: bool = False
    _target_: str = "visualization.core.create_global_view.check_mosaic_after_bias_correction"
    

cs.store(group='run', name='resample_and_mosaic', node=ResampleAndMosaicConfig)
cs.store(group='run', name='check_after_bias_correction', node=CheckfterBiasCorrectionConfig)
cs.store(group='run', name='create_pdf_thumb', node=CreateRhPairPdfConfig)
cs.store(group='run', name='create_pred_neighbor_pdf', node=CreatePredNeighborPdfConfig)
cs.store(group='run', name='create_cloud_cover_boxplot', node=CreateCloudCoverBoxplotConfig)

# ================================ Main Config ================================


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