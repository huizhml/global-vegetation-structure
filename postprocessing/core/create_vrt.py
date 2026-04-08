import glob
from osgeo import gdal
from pathlib import Path
import pystac
import numpy as np
import dask
from dask.diagnostics import ProgressBar


def create_vrt_for_tile(tile_dir: str, vrt_path: str, q_idx: str = "1"):
    '''
    Create a VRT file for the tile
    Parameters:
        tile_dir: path to the tile directory
        vrt_path: path to save the VRT file
        q_idx: index of the quality level
    Returns:
        None
    '''
    tile_dir = Path(tile_dir).expanduser()
    vrt_path = Path(vrt_path).expanduser()
    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    tile_id = tile_dir.stem
    files = sorted(
        glob.glob(f"{tile_dir}/RH*_Q{q_idx}.tif"),
        key=lambda x: int(x.split("RH")[-1].split("_")[0]),
    )
    print(f"Creating VRT for {tile_id} with {len(files)} files")
    assert len(files) == 101, f"Expected 101 files, found {len(files)}"

    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(str(vrt_path), files, options=vrt_options)

    for i in range(1, len(files) + 1):
        band = vrt.GetRasterBand(i)
        band.SetDescription(f"RH{i - 1}")

    vrt.FlushCache()
    vrt = None
    return vrt_path

def solve_vsm_path(stac_collection_dir: Path, tile_id: str, year: int):
    '''
    Solve the VSM path for the tile
    Parameters:
        tile_id: id of the tile
        year: year of the tile
    Returns:
        vsm_path: path to the VSM file
    '''
    if not (stac_collection_dir / f'{tile_id}_{year}').exists():
        print(f'{tile_id} not in stac collection')
        return    
    stac_item = pystac.Item.from_file(str(stac_collection_dir / f'{tile_id}_{year}/{tile_id}_{year}.json'))
    vsm_path = Path(stac_item.assets[f'RH98_Q1'].href.replace('file://', '')).parent
    return vsm_path


def create_vrt(tile_list_file: str, year: int, vrt_dir: str=None, q_idx: str = "1", **kwargs):
    '''
    Create a VRT file for the tile
    Parameters:
        tile_dir: path to the tile directory
        vrt_dir: path to save the VRT files
        q_idx: index of the quality level
    Returns:
        None
    '''
    stac_collection_dir = Path(f'~/data/gvs/products/gvsm_stac_catalog/vsm_local').expanduser()
    tile_list_file = Path(tile_list_file).expanduser()
    vrt_dir = Path(vrt_dir).expanduser()
    vrt_dir.mkdir(parents=True, exist_ok=True)
    tiles = np.loadtxt(tile_list_file, dtype=str)
    tasks = []
    for tile_id in tiles:
        vsm_path = dask.delayed(solve_vsm_path)(stac_collection_dir, tile_id, year)
        tasks.append(dask.delayed(create_vrt_for_tile)(vsm_path, vrt_dir / f'{tile_id}.vrt', q_idx))
    
    
    with ProgressBar():
        dask.compute(tasks)