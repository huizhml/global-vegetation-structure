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


def _abspath(p) -> Path:
    """Expand ~ and make absolute without resolving symlinks (path may not exist yet)."""
    return Path(os.path.abspath(os.path.expanduser(str(p))))


def link_results_dir(save_dir, root_results_dir, index_name) -> Path:
    """Index an op's output dir at ``root_results_dir/<index_name>`` via a symlink.

    The results stay physically in ``save_dir`` (the data tree), exactly where
    the op writes them — intermediate data included. A symlink at
    ``root_results_dir / index_name`` points back to ``save_dir``, so
    ``root_results_dir`` becomes a clean, browsable catalog of every op's
    results, named logically (e.g. ``<section>/<op>``) and decoupled from where
    the data physically lives.

    Args:
        save_dir: directory the op writes to (the real, data-side path).
        root_results_dir: central directory the result symlinks are collected under.
        index_name: relative name for the symlink under ``root_results_dir``,
            e.g. ``"on_gedi/evaluate_vsm_on_gedi"``.

    Returns:
        Path to ``save_dir`` (where writes actually land).
    """
    save_dir = _abspath(save_dir)
    results_root = _abspath(root_results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # save_dir already lives under the results root -> it's its own index entry.
    if save_dir == results_root or results_root in save_dir.parents:
        return save_dir

    link = results_root / index_name

    if link.is_symlink():
        if _abspath(os.readlink(link)) == save_dir:
            return save_dir  # already pointing where we want it
        print(f"  [link_results_dir] re-pointing {link} -> {save_dir}")
        # missing_ok=True: under SLURM array submission many tasks race
        # here, and a peer may have unlinked between our check and call.
        link.unlink(missing_ok=True)
    elif link.exists():
        # A real dir/file already sits at the index location; don't clobber it.
        print(f"  [link_results_dir] {link} exists and is not a symlink; "
              f"leaving it. Results stay at {save_dir}")
        return save_dir

    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(save_dir, link, target_is_directory=True)
        print(f"  [link_results_dir] {link} -> {save_dir}")
    except FileExistsError:
        # Another concurrent task created the same symlink first. Since they
        # all share the same save_dir for this op, the winner's link is
        # equivalent — accept and move on.
        pass
    return save_dir


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