import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from shapely.ops import unary_union


import geopandas as gpd
import subprocess

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