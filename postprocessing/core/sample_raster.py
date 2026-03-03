from osgeo import gdal
import rasterio
import numpy as np
from pathlib import Path
from rasterio.transform import xy
from rasterio.windows import Window
from rasterio.warp import transform
import pandas as pd


def create_multiband_vrt(tif_fp: str):
    '''
    Create a multiband VRT from a list of TIF files.
    Args:
        tif_fps: List of TIF files.
    Returns:
        vrt_fp: Path to the VRT file.
    '''
    vrt_options = gdal.BuildVRTOptions(
        resolution='highest',  # or 'average'
        separate=True,
    )
    out_dir = tif_fp.parent
    month = tif_fp.stem.split("_")[-1]
    out_name = f'offset_{month}.vrt'
    vrt_fp = out_dir / out_name
    tif_fps = [str(p) for p in out_dir.glob(f'*{month}.tif')]
    tif_fps = sorted(tif_fps)
    gdal.BuildVRT(str(vrt_fp), tif_fps, options=vrt_options)
    return vrt_fp

def sample_block(col_off: int, row_off: int, block_h: int, block_w: int, vrt_fp: str, rng: np.random.Generator, p: float):
    '''
    Sample a block of pixels from a raster.
    Args:
        col_off: Column offset.
        row_off: Row offset.
        block_h: Block height.
        block_w: Block width.
    '''
    with rasterio.open(vrt_fp) as src:
        height, width = src.height, src.width
        h = min(block_h, height - row_off)
        w = min(block_w, width - col_off)
        win = Window(col_off, row_off, w, h)
        data = src.read(window=win, masked=True).data
        valid = ~np.isnan(data)
        valid_2d = valid.all(axis=0)  # (H, W): pixel valid only if all bands valid

        if not valid_2d.any():
            return
        # Bernoulli draw only on valid pixels
        draw = (rng.random(valid_2d.shape) < p) & valid_2d

        if not draw.any():
            return
        rr, cc = np.nonzero(draw)
        rows = (win.row_off + rr).astype(np.int64)
        cols = (win.col_off + cc).astype(np.int64)
        xs, ys = xy(src.transform, rows, cols, offset="center")
        lons, lats = transform(src.crs, "EPSG:4326", xs, ys)
        sampled_data = data[:, draw]
        return np.column_stack([lons, lats, sampled_data.T])

def sample_forest_temp(tif_dir: str, p: float = 1e-4, *, seed: int = None, **kwargs):
    '''
    Sample forest temp points from a directory of TIF files.
    Args:
        tif_dir: Directory containing TIF files.
        p: Probability of sampling a point.
        seed: Random seed.
    Returns:
        coords: Sampled coordinates.
        crs: Coordinate reference system.
    '''
    block_h, block_w = 1024, 1024
    tif_dir = Path(tif_dir).expanduser()
    tif_fps = list(tif_dir.glob('min*.tif'))

    rng = np.random.default_rng(seed)
    coords_chunks = []
    for tif_fp in tif_fps:
        vrt_fp = create_multiband_vrt(tif_fp)
        with rasterio.open(vrt_fp) as src:
            height, width = src.height, src.width
            for row_off in range(0, height, block_h):
                for col_off in range(0, width, block_w):
                    h = min(block_h, height - row_off)
                    w = min(block_w, width - col_off)
                    win = Window(col_off, row_off, w, h)
                    data = src.read(window=win, masked=True).data
                    valid = ~np.isnan(data)
                    valid_2d = valid.all(axis=0)  # (H, W): pixel valid only if all bands valid

                    if not valid_2d.any():
                        continue
                    # Bernoulli draw only on valid pixels
                    draw = (rng.random(valid_2d.shape) < p) & valid_2d

                    if not draw.any():
                        continue
                    rr, cc = np.nonzero(draw)
                    rows = (win.row_off + rr).astype(np.int64)
                    cols = (win.col_off + cc).astype(np.int64)
                    xs, ys = xy(src.transform, rows, cols, offset="center")
                    lons, lats = transform(src.crs, "EPSG:4326", xs, ys)
                    sampled_data = data[:, draw]
                    coords_chunks.append(np.column_stack([lons, lats, sampled_data.T]))

        coords = np.vstack(coords_chunks)
        month = tif_fp.stem.split("_")[-1]
        df = pd.DataFrame(coords, columns=['lon', 'lat', f'max_temp_{month}', f'min_temp_{month}', f'mean_temp_{month}'])
        df.to_parquet(f'{tif_dir}/sampled_forest_temp_{month}.parquet')
    
    