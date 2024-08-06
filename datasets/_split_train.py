import os
from typing import List, Iterable
from pathlib import Path
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField, FloatField
from datasets._h5_dataset import S2Dataset
import random
import geopandas as gpd
import numpy as np
import dask_geopandas as dgp
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class MyConfig:
    index_table: str = '~/data/split_test0.1_cal0.1_val0.1_seed42/index_table_train'
    h5_file: str = '~/scratch/data/GVS.h5'
    out_dir: str = '~/flash/data'
    nsplit: int= 10
    seed: int = 42
    shuffle_indices: bool = False

cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)

def split_index_table(nsplit, out_dir, index_table_fp):
    index_table = dgp.read_parquet(index_table_fp, gather_spatial_partitions=False).sample(frac=1).compute()
    N = len(index_table)
    n = N // nsplit
    for split in range(nsplit):
        index_ = index_table.iloc[n*split:n*(split+1)] # not shuffled!
        if split == nsplit - 1:
            index_ = index_table.iloc[n*split:]
        index_.to_parquet(out_dir / f'train{split}.parquet')

def write_beton(out_file, dataset, shuffle_indices=False):
    
    # Pass a type for each data field
    print("Writing dataset to", out_file)
    writer = DatasetWriter(out_file, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': NDArrayField(dtype=np.dtype("int16"), shape=(12, 15, 15)),
        'rhs': NDArrayField(dtype=np.dtype("float32"), shape=(101,)),
        'wc': IntField(),
        'slope': FloatField(),
        'latlon': NDArrayField(dtype=np.dtype("float64"), shape=(2,)),
    })

    # Write dataset
    writer.from_indexed_dataset(dataset, shuffle_indices=shuffle_indices)

@hydra.main(config_name='config')
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    h5_file = Path(cfg.h5_file).expanduser()
    index_dir = Path(cfg.index_table).expanduser()
    out_dir = Path(cfg.out_dir).expanduser()
    splited_idx_dir = out_dir / 'index_table_train_subsets'
    splited_idx_dir.mkdir(exist_ok=True, parents=True)

    if len(list(splited_idx_dir.glob('*.parquet'))) != cfg.nsplit:
        print('Re-splitting index table')
        split_index_table(cfg.nsplit, splited_idx_dir, index_dir/f'*.parquet')
    
    for split in range(cfg.nsplit):
        out_file = out_dir / f'train{split}.beton'
        if out_file.exists():
            print("Skipping", out_file)
            continue

        index_ = gpd.read_parquet(splited_idx_dir / f'train{split}.parquet')
        dataset = S2Dataset(h5_file, index_)
        write_beton(out_file, dataset, shuffle_indices=cfg.shuffle_indices)
    

if __name__ == '__main__':
    main()