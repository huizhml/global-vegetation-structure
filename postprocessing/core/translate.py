import hydra
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from dataclasses import dataclass
from pathlib import Path
import time
import dask

@dask.delayed
def _translate(src_path, dst_path, profile="ZSTD", profile_options={}, **options):
    """Convert image to COG."""
    # Format creation option (see gdalwarp `-co` option)
    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(
        BIGTIFF="IF_SAFER",
        ZSTD_LEVEL=1,
        PREDICTOR=2,
        BLOCKYSIZE=1024,
        BLOCKXSIZE=1024,
        MAX_Z_ERROR=0
    ))
    output_profile.update(profile_options)

    # Dataset Open option (see gdalwarp `-oo` option)
    config = dict(
        GDAL_NUM_THREADS=2,
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )

    cog_translate(
        src_path,
        dst_path,
        output_profile,
        config=config,
        in_memory=False,
        quiet=True,
        **options,
    )
    return True

def translate_tile(src_dir, dst_dir, profile="ZSTD", **kwargs):
    src_dir = Path(src_dir).expanduser()
    dst_dir = Path(dst_dir).expanduser()
    dst_dir.mkdir(exist_ok=True)
    tiff_files = src_dir.glob('RH*_Q*.tif')
    futures = []
    for tiff_file in tiff_files:
        dst_path = dst_dir / tiff_file.name #.replace('_uncompressed.tif', '.cog.tif')
        futures.append(_translate(tiff_file, dst_path, profile))
    results = dask.compute(*futures)
    if all(results):
        print(f"Successfully translated {len(results)} files")
    else:
        print(f"Failed to translate {len(results)} files")
        

@dataclass
class TranslateConfig:
    src_dir: str='~/data/gvs/deploy/predictions_2020/09WWQ_fp32_infer'
    dst_dir: str='~/data/gvs/deploy/predictions_2020/09WWQ_fp32_infer_cog'
    profile: str = "LERC_ZSTD"
    
cs = ConfigStore.instance()
cs.store(name="config", node=TranslateConfig)
    
@hydra.main(config_path=None, config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    
    t0 = time.time()
    translate_tile(cfg.src_dir, cfg.dst_dir, cfg.profile)
    print(f"Time taken: {time.time() - t0} seconds")


if __name__ == "__main__":
    main() 