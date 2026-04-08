from pathlib import Path
import pandas as pd
import geopandas as gpd
import dask

def prepare_loc_parqs(naturalness_csv: str, s2_grid_file: str, save_dir: str, **kwargs):
    '''
    Partition the naturalness csv file into Sentinel-2 tile based parquets. 
    Making it easier to sample VSM patches/points for each tile.
    Args:
        naturalness_csv: path to the naturalness csv file
        s2_grid_file: path to the S2 grid file
        save_dir: path to save the location parquets
        year: year of the naturalness dataset
    Returns:
        None
    '''
    naturalness_csv = Path(naturalness_csv).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(naturalness_csv)
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
    s2_grid = gpd.read_parquet(s2_grid_file, columns=['Name', 'geometry', 'covered_by_gedi'])
    s2_grid = s2_grid.to_crs(epsg=4326)
    df = gpd.sjoin(df, s2_grid, how='left', predicate='intersects')
    df = df.drop_duplicates(subset=['geometry'])
    df = df.drop(columns=['index_right'])
    tile_ids = df['Name'].unique()
    
    @dask.delayed
    def _save_tile(df_tile: gpd.GeoDataFrame, tile_id: str):
        df_tile.to_parquet(save_dir / f'{tile_id}.parquet')
        return
    
    tasks = []
    for tile_id in tile_ids:
        tasks.append(dask.delayed(_save_tile)(df[df['Name'] == tile_id], tile_id))
    dask.compute(*tasks)
    return