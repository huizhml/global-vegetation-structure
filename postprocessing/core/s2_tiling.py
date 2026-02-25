import os
import re
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from shapely.ops import unary_union
from shapely.geometry import shape, box
import dask
import geopandas as gpd
import subprocess
import numpy as np


def convert_parquet_to_fgb(s2_grid_file):
    s2_grid_file = Path(s2_grid_file).expanduser()
    gdf = gpd.read_parquet(s2_grid_file)
    gdf.to_file(s2_grid_file.with_suffix('.fgb'), driver='FlatGeobuf')
    

def upload_to_erda(fgb_file: str):
    subprocess.run(['rclone', 'sync', str(fgb_file), 'ucph-erda:GVS/deploy_status/'], check=True)
    if fgb_file.stem != 'deploy_status':
        print(f'{fgb_file.stem} is not deploy_status, deleting and moving to deploy_status.fgb')
        subprocess.run(['rclone', 'delete', 'ucph-erda:GVS/deploy_status/deploy_status.fgb'])
        subprocess.run(['rclone', 'moveto', f'ucph-erda:GVS/deploy_status/{fgb_file.stem}.fgb', 'ucph-erda:GVS/deploy_status/deploy_status.fgb'], check=True)


def remove_redundant_tiles(s2_grid_file: str, save_dir: str):
    s2_grid_file = Path(s2_grid_file).expanduser()
    s2_grid = gpd.read_parquet(s2_grid_file, columns=['geometry', 'Name', 'growing_months', 'covered_by_gedi', 'gedi_correction_count_2020', 'gedi_correction_count_2024'])
    s2_grid['redundant'] = False
    for idx, row in s2_grid.iterrows():
        intersecting_tiles = s2_grid[s2_grid.intersects(row['geometry'])]
        intersecting_tiles = intersecting_tiles[intersecting_tiles['Name'] != row['Name']]
        if len(intersecting_tiles) == 0:
            continue
        union_geometry = unary_union(intersecting_tiles['geometry'])
        diff = row['geometry'].difference(union_geometry) # unique area covered by current tile
        if diff.is_empty or diff.area < 1e-6:
            s2_grid.loc[idx, 'redundant'] = True
    s2_grid.to_parquet(s2_grid_file)
    convert_parquet_to_fgb(s2_grid_file)
    upload_to_erda(s2_grid_file.with_suffix('.fgb'))
  
  
  
def find_intersecting_s2_tiles(s2_grid: gpd.GeoDataFrame=None, current_tile: str=None) -> list[str]:
    '''
    Find intersecting S2 tiles
    '''
    tile_geom = s2_grid[s2_grid['Name'] == current_tile]['geometry'].iloc[0]
    geom = shape(tile_geom)
    idx = list(s2_grid.sindex.query(geom, predicate="intersects"))
    tiles = s2_grid.iloc[idx]['Name'].unique()
    return tiles


    
    
def get_tiles_wo_enough_gedi_gt(tiles_list_file: str=None, bias_stats_dir: str=None, s2_grid_file: str=None):
    '''
    Get tiles not affected by bias correction, which means the bias correction is not applied to these tiles and their surrounding tiles
    '''
    bias_stats_dir = Path(bias_stats_dir).expanduser()
    tiles_list_file = Path(tiles_list_file).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    year = tiles_list_file.stem.split('_')[-1]
    df = gpd.read_parquet(s2_grid_file)
    with open(tiles_list_file, 'r') as f:
        tiles = f.read().splitlines()
        
    @dask.delayed
    def check_if_affected(tile: str):
        '''
        Check if the tile is affected by bias correction
        Any intersecting tile has bias correction stats, then the tile is affected
        '''
        intersecting_tiles = find_intersecting_s2_tiles(df, tile)
        for intersecting_tile in intersecting_tiles:
            bias_stats_file = bias_stats_dir / f'{intersecting_tile}.npz'
            if bias_stats_file.exists():
                return None
        return tile
    
    tiles_wo_enough_gedi_gt = dask.compute(*[check_if_affected(tile) for tile in tiles])
    tiles_wo_enough_gedi_gt = [tile for tile in tiles_wo_enough_gedi_gt if tile is not None]
    with open(tiles_list_file.with_name(f'tiles_wo_enough_gedi_gt_{year}.txt'), 'w') as f:
        f.write('\n'.join(tiles_wo_enough_gedi_gt))  



def get_tiles_in_arctic_regions(arctic_regions_file: str, s2_grid_file: str):
    s2_grid_file = Path(s2_grid_file).expanduser()
    s2_grid = gpd.read_parquet(s2_grid_file)
    save_file = Path('~/data/gvs/deploy/arctic_regions_tiles.txt').expanduser()
    if save_file.exists():
        with open(save_file, 'r') as f:
            tiles = f.read().splitlines()
        tiles = s2_grid[s2_grid['Name'].isin(tiles)]
        return tiles
    arctic_regions_file = Path(arctic_regions_file).expanduser()
    arctic_regions = gpd.read_file(arctic_regions_file)
    tiles = s2_grid[s2_grid.intersects(arctic_regions.union_all())]
    
    with open(save_file, 'w') as f:
        for tile in tiles['Name'].unique():
            f.write(tile + '\n')
    return tiles

def get_land_sea_boundary_tiles(s2_grid_file: str, countries_file: str, ocean_file: str):
    # Load datasets
    s2_grid_file = Path(s2_grid_file).expanduser()
    s2_grid = gpd.read_parquet(s2_grid_file)
    tiles = s2_grid.to_crs(4326)
    countries = gpd.read_file(countries_file).to_crs(4326)
    land = gpd.read_file(ocean_file).to_crs(4326)

    # Create a global ocean polygon
    world_bbox = box(-180, -90, 180, 90)
    land_union = unary_union(land.geometry)
    ocean = gpd.GeoDataFrame(geometry=[world_bbox.difference(land_union)], crs=4326)

    # Intersect tiles with land and ocean
    tiles_land = tiles[tiles.intersects(land_union)]
    tiles_ocean = tiles[tiles.intersects(ocean.iloc[0].geometry)]

    # Tiles that intersect both land and ocean
    coastal_tiles = tiles_land[tiles_land["Name"].isin(tiles_ocean["Name"])]
    # Optionally join with countries to get which country they belong to
    coastal_tiles = gpd.sjoin(coastal_tiles, countries, predicate="intersects")[["Name", "ADMIN", "geometry"]]
    return coastal_tiles



def reorder_tiles_by_circle(tiles_dir: str):
    '''
    Reorder tiles by circle (from center to outer)
    Expecting {zone}.txt files in tiles_dir
    '''
    tiles_dir = Path(tiles_dir).expanduser()
    tiles_files = tiles_dir.glob('*.txt')
    for tiles_file in tiles_files:
        tiles = tiles_file.read_text().splitlines()

        zone = tiles_file.stem
        idx3_list = sorted(set([tile[3] for tile in tiles]))
        idx4_list = sorted(set([tile[4] for tile in tiles]))
        min_l = min(len(idx3_list), len(idx4_list))
        ordered = []
        for i in range(min_l):
            for loc3 in idx3_list[:i+1]:
                tile_name = f'{zone}{loc3}{idx4_list[i]}'
                if tile_name in tiles:
                    ordered.append(tile_name)
            for loc4 in idx4_list[:i]:
                tile_name = f'{zone}{idx3_list[i]}{loc4}'
                if tile_name in tiles:
                    ordered.append(tile_name)
        if len(idx3_list) > len(idx4_list):
            for letter in idx3_list[min_l:]:
                for loc4 in idx4_list:
                    tile_name = f'{zone}{letter}{loc4}'
                    if tile_name in tiles:
                        ordered.append(tile_name)
        else:
            for letter in idx4_list[min_l:]:
                for loc3 in idx3_list:
                    tile_name = f'{zone}{loc3}{letter}'
                    if tile_name in tiles:
                        ordered.append(tile_name)
                        
                        
        assert len(ordered) == len(tiles)
        assert set(ordered) == set(tiles)
        with open(tiles_file, 'w') as f:
            for tile in ordered:
                f.write(tile + '\n')

def get_tiles_reblend(tiles_list_file: str, s2_grid_file: str, **kwargs):
    tiles_list_file = Path(tiles_list_file).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    s2_grid = gpd.read_parquet(s2_grid_file)
    with open(tiles_list_file, 'r') as f:
        tiles = f.read().splitlines()
        
    reblend_tiles = []
    for tile in tiles:
        intersecting_tiles = find_intersecting_s2_tiles(s2_grid, tile)
        reblend_tiles.append(intersecting_tiles)
    reblend_tiles = np.hstack(reblend_tiles)
    reblend_tiles = np.unique(reblend_tiles)
    with open(tiles_list_file.with_name(f"{tiles_list_file.stem}_with_neighbors.txt"), 'w') as f:
        for tile in reblend_tiles:
            f.write(tile + '\n')
    
    
def get_tiles_redundant(s2_grid_file: str, cog_dir: str, save_dir: str, **kwargs):
    '''
    Get redundant tiles
    1. tiles that are completely covered by other tiles
    2. tiles that are covered by non-vegetated areas, nodata percentage == 100%
    Save the tiles to a text file
    Args:
        s2_grid_file: path to the s2 grid file, in fgb format
        cog_dir: path to the cog directory
        save_dir: path to the save directory
        kwargs: keyword arguments
    Returns:
        None
    '''
    cog_dir = Path(cog_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    s2_grid = gpd.read_file(s2_grid_file)
    redundant_tiles = s2_grid[s2_grid['redundant']]
    redundant_pairs = []
    for idx, tile in redundant_tiles.iterrows():
        geom = tile['geometry']
        intersecting_tiles = redundant_tiles[redundant_tiles.intersects(tile['geometry'])]
        intersecting_tiles = intersecting_tiles[intersecting_tiles['Name'] != tile['Name']]
        intersecting_tiles['intersection_area'] = intersecting_tiles.geometry.intersection(geom).area
        if len(intersecting_tiles) == 0:
            continue
        redundant_tile = intersecting_tiles.sort_values('intersection_area', ascending=False).iloc[0]
        redundant_pairs.append(set([tile['Name'], redundant_tile['Name']]))
    remove = [min(pair) for pair in redundant_pairs]
    remove = set(remove)
    
    with open(save_dir / 'tiles_redundant.txt', 'w') as f:
        for tile in remove:
            f.write(tile + '\n')
            
def get_tiles_nodata(cog_dir: str, save_dir: str, s2_grid_file: str, **kwargs):
    '''
    Get tiles that are completely covered by nodata, e.g, covered by snow or ice
    simply by checking the file size, which seems to be 124K in size for a cog file.
    So if all files are smaller than 124K, then the tile is likely to be nodata.
    Args:
        s2_grid_file: path to the s2 grid file, in fgb format
        cog_dir: path to the cog directory
    Returns:
        None
    '''
    cog_dir = Path(cog_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    s2_grid_file = Path(s2_grid_file).expanduser()
    s2_grid = gpd.read_file(s2_grid_file)
    match = re.search(r'(20\d{2})', str(cog_dir))
    year = match.group(1) if match else ''
    tiles = os.listdir(cog_dir)
    nodata_tiles = []
    for tile in tiles:
        files = list((cog_dir / tile).glob('*.tif'))
        nodata_files = 0
        for file in files:
            if file.stat().st_size > 126976:
                break
            nodata_files += 1
        if nodata_files == len(files):
            nodata_tiles.append(tile)

    with open(save_dir / f'tiles_nodata_{year}.txt', 'w') as f:
        for tile in nodata_tiles:
            f.write(tile + '\n')   


@dataclass
class RemoveRedundantTiles:
    s2_grid_file: str = '~/data/gvs/deploy/deploy_status.parquet'
    save_dir: str = '~/data/gvs/deploy/'
    task: str = 'remove_redundant_tiles'


disable_outputs = {
    "hydra": {
        "run": {"dir": "."},
        "output_subdir": None,
        "job_logging": {"enabled": False},
        "hydra_logging": {"enabled": False},
    }
}

cs = ConfigStore.instance()
cs.store(name='remove_redundant_tiles', node=RemoveRedundantTiles)
cs.store(group="hydra", name="disable_logging", node=disable_outputs)

@hydra.main(config_path=None,config_name='remove_redundant_tiles', version_base="1.2")
def main(cfg):
    print(cfg)
    if cfg.task == 'remove_redundant_tiles':
        remove_redundant_tiles(s2_grid_file=cfg.s2_grid_file, save_dir=cfg.save_dir)
    elif cfg.task == 'convert_and_upload_to_erda':
        convert_parquet_to_fgb(s2_grid_file=cfg.s2_grid_file)
        fgb_file = Path(cfg.s2_grid_file).expanduser().with_suffix('.fgb')
        upload_to_erda(fgb_file=fgb_file)
        
if __name__ == '__main__':
    main()