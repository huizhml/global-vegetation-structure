import glob
import os
from osgeo import gdal
from pathlib import Path
import dask
import pandas as pd
import geopandas as gpd
from datetime import datetime
from omegaconf import OmegaConf


def init_gtiff(prediction_fp: Path, tile_info: dict, options: list):
    prediction_fp = prediction_fp.with_suffix('.tif')
    if prediction_fp.exists():
        os.remove(prediction_fp)
    driver = gdal.GetDriverByName('GTiff')
    tiff_output = driver.Create(
        str(prediction_fp),
        xsize=tile_info['width'],
        ysize=tile_info['height'],
        bands=1,
        eType=gdal.GDT_Int16,
        options=options
    )
    tiff_output.SetGeoTransform(tile_info['transform'])
    tiff_output.SetProjection(tile_info['crs'])
    tiff_output.SetMetadataItem('Year', str(tile_info['year']))
    tiff_output.SetMetadataItem('Sentinel-2 tile', tile_info['tile_id'])
    return tiff_output



def subsample_parquet_files(parquet_dir: str, save_dir: str, n_samples: int, random_state: int = 42):
    parquet_dir = Path(parquet_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = list(parquet_dir.glob('*.parquet'))
    
    @dask.delayed
    def _subsample(parquet_file: str, n_samples: int, random_state: int):
        df = pd.read_parquet(parquet_file)
        if len(df) > n_samples:
            df = df.sample(n_samples, random_state=random_state)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        df.to_parquet(save_dir / parquet_file.name)
    
    tasks = [_subsample(parquet_file, n_samples, random_state) for parquet_file in parquet_files]
    dask.compute(*tasks)
    
    
def generate_run_log(log_file: str, run_config: OmegaConf, runtime: float):
    log_file = Path(log_file).expanduser()
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_entry = f"Run on {date}\n{OmegaConf.to_yaml(run_config)}\nRuntime: {runtime} seconds\n"

    existing = log_file.read_text(encoding='utf-8') if log_file.exists() else ""

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(new_entry + existing)

    return log_file
    
    
def create_vrt_for_tile(tile_dir, vrt_path, q_idx="1"):
    tile_dir = Path(tile_dir).expanduser()
    vrt_path = Path(vrt_path).expanduser()
    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    tile_id = tile_dir.stem
    files = sorted(
        glob.glob(f"{tile_dir}/RH*_Q{q_idx}.tif"),
        key=lambda x: int(x.split("RH")[-1].split("_")[0]),
    )
    files = files[:100]
    print(f"Creating VRT for {tile_id} with {len(files)} files")
    assert len(files) == 100, f"Expected 100 files, found {len(files)}"

    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(str(vrt_path), files, options=vrt_options)

    for i in range(1, len(files) + 1):
        band = vrt.GetRasterBand(i)
        band.SetDescription(f"RH{i - 1}")

    vrt.FlushCache()
    vrt = None
    return vrt_path