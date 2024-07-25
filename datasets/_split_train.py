import os
from typing import List, Iterable
from pathlib import Path
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField, FloatField
from datasets.s2 import S2Dataset
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
    index_table: str = '~/data/geo_index_table'
    train_index_dir: str = '~/data/index_table_train_subsets'
    h5_file: str = '~/scratch/data/GVS.h5'
    out_dir: str = '~/flash/data'
    nsplit: int= 10
    seed: int = 42

cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)

def split_index_table(nsplit, index_dir, index_table_fp):
    index_table = dgp.read_parquet(index_table_fp, gather_spatial_partitions=False).sample(frac=1).compute()
    N = len(index_table)
    n = N // nsplit
    for split in range(nsplit):
        index_ = index_table.iloc[n*split:n*(split+1)]
        if split == nsplit - 1:
            index_ = index_table.iloc[n*split:]
        index_.to_parquet(index_dir / f'train{split}.parquet')

@hydra.main(config_name='config')
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    h5_file = Path(cfg.h5_file).expanduser()
    split_dir = Path(cfg.index_table).expanduser()
    index_dir = Path(cfg.train_index_dir).expanduser()
    index_dir.mkdir(exist_ok=True, parents=True)
    input_shape = (12, 15, 15)
    if not index_dir.exists() or not len(list(index_dir.glob('*.parquet'))) == cfg.nsplit:
        split_index_table(cfg.nsplit, index_dir, split_dir/f'*.parquet')
    
    for split in range(cfg.nsplit):
        out_file = Path(cfg.out_dir).expanduser() / f'train{split}.beton'
        if out_file.exists():
            print("Skipping", out_file)
            continue
        index_ = gpd.read_parquet(index_dir / f'train{split}.parquet')
        dataset = S2Dataset(h5_file, index_)
        
        # Pass a type for each data field
        print("Writing dataset to", out_file)
        writer = DatasetWriter(out_file, {
            # Tune options to optimize dataset size, throughput at train-time
            'image': NDArrayField(dtype=np.dtype("int16"), shape=input_shape),
            'rhs': NDArrayField(dtype=np.dtype("float32"), shape=(101,)),
            'wc': IntField(),
            'slope': FloatField(),
            'latlon': NDArrayField(dtype=np.dtype("float64"), shape=(2,)),
        })

        # Write dataset
        writer.from_indexed_dataset(dataset)
    

if __name__ == '__main__':
    main()