import pandas as pd
import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import dask_geopandas as dgpd

def save_tile_ids(parquet_dir):
    for parquet_file in parquet_dir.glob('deploy*.parquet'):
        df = pd.read_parquet(parquet_file)
        unique_tiles = df['s2:mgrs_tile'].unique()
        print(f'{parquet_file} has {len(unique_tiles)} unique tiles')
        np.savetxt(parquet_file.with_suffix('.txt'), unique_tiles, fmt='%s')
        
def resplite_parquet_files(parquet_dir, save_dir, partition_size=72):
    files = list(parquet_dir.glob('deploy*.parquet'))
    gdf = dgpd.read_parquet(files, gather_spatial_partitions=False)
    gdf = gdf.compute()
    gdf = gdf.sort_values(by='s2:mgrs_tile')
    year = files[0].stem.split('_')[3]
    unique_tiles = gdf['s2:mgrs_tile'].unique()
    for i, k in enumerate(range(0, len(unique_tiles), partition_size)):
        tile_ids = unique_tiles[k:k+partition_size]
        gdf_subset = gdf[gdf['s2:mgrs_tile'].isin(tile_ids)]
        gdf_subset.to_parquet(save_dir / f'deploy_s2_items_{year}_part{i}.parquet')



@dataclass
class DeployConfig:
    parquet_dir: str='~/data/GVS/Deploy'
    save_dir: str='~/data/GVS/Deploy/lumi_job_files'
    
cs = ConfigStore.instance()
cs.store(name='deploy', node=DeployConfig)

@hydra.main(config_name='deploy', version_base='1.2')
def main(cfg):
    parquet_dir = Path(cfg.parquet_dir).expanduser()
    print(f'parquet_dir: {parquet_dir}')
    save_dir = Path(cfg.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    resplite_parquet_files(parquet_dir, save_dir)
    # save_tile_ids(save_dir)

if __name__ == '__main__':
    main()