import pandas as pd
import hydra
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import dask_geopandas as dgpd
import geopandas as gpd
import geodatasets


def configure_slurm_jobs_with_priority(
        prioritized_countries=None, old_slurm_job_dir=None, new_slurm_job_dir: str = None, s2_grid_file: str = None,
        n_tiles_per_job: int = 72, year: int = 2024, parallel_files: bool = False, **kwargs):
    """

    Args:
        prioritized_countries (_type_): _description_
        old_slurm_job_dir (_type_): _description_
        new_slurm_job_dir (_type_): _description_
        s2_grid_file (_type_): _description_
        n_tiles_per_job (int, optional): _description_. Defaults to 72.
        year (int, optional): _description_. Defaults to 2024.
        parallel_files (bool, optional): if True, tiles in the same file will be predicted in parallel, else in sequence. Defaults to False.
    """
    print(f'Split S2 tiles and images for parallel inference in {year}. Prioritize tiles in {prioritized_countries}')
    countries_url = "~/data/GVS/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    countries = gpd.read_file(countries_url)
    regions = countries[countries['ADMIN'].isin(prioritized_countries)]
    s2_grid = gpd.read_parquet(s2_grid_file)
    prioritized_tiles = s2_grid[s2_grid.intersects(regions.union_all())]['Name'].unique().tolist()
    all_tiles = s2_grid['Name'].unique()
    old_slurm_job_dir = Path(old_slurm_job_dir).expanduser()
    new_slurm_job_dir = Path(new_slurm_job_dir).expanduser()
    new_slurm_job_dir.mkdir(parents=True, exist_ok=True)
    all_images = dgpd.read_parquet(old_slurm_job_dir / f'*_items_{year}_part*.parquet', gather_spatial_partitions=False)
    all_images = all_images.compute()
    tiles_with_images = all_images['s2:mgrs_tile'].unique()
    tiles_without_images = np.setdiff1d(all_tiles, tiles_with_images)
    np.savetxt(new_slurm_job_dir / f'deploy_s2_items_{year}_without_images.txt', tiles_without_images, fmt='%s')
    rest_tile_with_images = sorted(np.setdiff1d(tiles_with_images, prioritized_tiles))
    prioritized_tiles_with_images = sorted(np.setdiff1d(prioritized_tiles, tiles_without_images))
    print(f'There are {len(all_tiles)} tiles, {len(tiles_without_images)} withouth images...')
    print(f'Allocating {len(prioritized_tiles_with_images)} prioritized tiles + {len(rest_tile_with_images)} rest tiles with images...')
    if parallel_files:
        n_allocated_tiles = 0
        job_id = 0
        while n_allocated_tiles < len(rest_tile_with_images):
            # prioritized tiles are all allocated, get all n_tiles_per_job from rest tiles
            if job_id >= len(prioritized_tiles_with_images):
                tile_ids = rest_tile_with_images[n_allocated_tiles: n_allocated_tiles+n_tiles_per_job]
            # prioritized tiles are not all allocated, get n_tiles_per_job - 1  from rest tiles and one from prioritized tiles
            else:
                tile_ids = prioritized_tiles_with_images[job_id:job_id+1] + rest_tile_with_images[job_id: job_id+n_tiles_per_job-1]
            job_images = all_images[all_images['s2:mgrs_tile'].isin(tile_ids)]
            job_images.to_parquet(new_slurm_job_dir / f'deploy_s2_items_{year}_part{job_id}.parquet')
            np.savetxt(new_slurm_job_dir / f'deploy_s2_items_{year}_part{job_id}.txt', tile_ids, fmt='%s')
            n_allocated_tiles += n_tiles_per_job
            job_id += 1
    else:
        if len(prioritized_tiles_with_images) < n_tiles_per_job:
            n_from_rest_tiles = n_tiles_per_job - len(prioritized_tiles_with_images)
            prioritized_tiles_with_images.extend(rest_tile_with_images[:n_from_rest_tiles])
            rest_tile_with_images = rest_tile_with_images[n_from_rest_tiles:]
        prioritized_images = all_images[all_images['s2:mgrs_tile'].isin(prioritized_tiles_with_images)]
        prioritized_images.to_parquet(new_slurm_job_dir / f'deploy_s2_items_{year}_part0.parquet')
        assert len(prioritized_tiles_with_images) == n_tiles_per_job, 'prioritized_tiles_with_images should be equal to n_tiles_per_job'
        assert len(prioritized_tiles_with_images) == prioritized_images['s2:mgrs_tile'].nunique(), 'prioritized_tiles_with_images should be equal to unique tiles in prioritized_images'
        np.savetxt(new_slurm_job_dir / f'deploy_s2_items_{year}_part0.txt', prioritized_tiles_with_images, fmt='%s')
        actual_tiles_with_images = len(prioritized_tiles_with_images)
        n_allocated_tiles = 0
        job_id = 1
        while n_allocated_tiles < len(rest_tile_with_images):
            tile_ids = rest_tile_with_images[n_allocated_tiles: n_allocated_tiles+n_tiles_per_job]
            job_images = all_images[all_images['s2:mgrs_tile'].isin(tile_ids)]
            job_images.to_parquet(new_slurm_job_dir / f'deploy_s2_items_{year}_part{job_id}.parquet')
            assert len(tile_ids) == job_images['s2:mgrs_tile'].nunique(), 'tile_ids should be equal to unique tiles in prioritized_images'
            np.savetxt(new_slurm_job_dir / f'deploy_s2_items_{year}_part{job_id}.txt', tile_ids, fmt='%s')
            n_allocated_tiles += n_tiles_per_job
            job_id += 1
            actual_tiles_with_images += len(tile_ids)
        print(f'Allocated {actual_tiles_with_images} tiles, {len(tiles_with_images)} tiles in total')


def save_tile_ids(parquet_dir):
    for parquet_file in parquet_dir.glob('s2*.parquet'):
        df = pd.read_parquet(parquet_file)
        unique_tiles = df['s2:mgrs_tile'].unique()
        print(f'{parquet_file} has {len(unique_tiles)} unique tiles')
        np.savetxt(parquet_file.with_suffix('.txt'), unique_tiles, fmt='%s')


def resplite_parquet_files(parquet_dir, save_dir, n_tiles_per_job=72):
    files = list(parquet_dir.glob('deploy*.parquet'))
    gdf = dgpd.read_parquet(files, gather_spatial_partitions=False)
    gdf = gdf.compute()
    gdf = gdf.sort_values(by='s2:mgrs_tile')
    year = files[0].stem.split('_')[3]
    unique_tiles = gdf['s2:mgrs_tile'].unique()
    for i, k in enumerate(range(0, len(unique_tiles), n_tiles_per_job)):
        tile_ids = unique_tiles[k:k+n_tiles_per_job]
        gdf_subset = gdf[gdf['s2:mgrs_tile'].isin(tile_ids)]
        gdf_subset.to_parquet(save_dir / f'deploy_s2_items_{year}_part{i}.parquet')


@dataclass
class DeployConfig:
    parquet_dir: str = '~/data/GVS/Deploy/'
    save_dir: str = '~/data/GVS/Deploy/slurm_job_files_2020'
    s2_grid_file: str = '~/data/GVS/S2_tiles_with_growing_months.parquet'
    year: int = 2020
    n_tiles_per_job: int=200
    prioritized_countries: list = field(default_factory=lambda: ['Gabon', 'Switzerland'])


cs = ConfigStore.instance()
cs.store(name='deploy', node=DeployConfig)


@hydra.main(config_name='deploy', version_base='1.2')
def main(cfg):
    configure_slurm_jobs_with_priority(cfg.prioritized_countries, cfg.parquet_dir,
                                       cfg.save_dir, cfg.s2_grid_file, year=cfg.year, parallel_files=False, n_tiles_per_job=cfg.n_tiles_per_job)
    # resplite_parquet_files(parquet_dir, save_dir)
    # save_tile_ids(save_dir)


if __name__ == '__main__':
    main()
