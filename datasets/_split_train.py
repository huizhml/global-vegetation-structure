from typing import List, Iterable
from pathlib import Path
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from datasets.s2 import S2Dataset
import random
import pandas as pd
import numpy as np
import dask_geopandas as dgp
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class MyConfig:
    index_table: str = '~/data/geo_index_table'
    h5_file: str = '~/scratch/data/GVS.h5'
    out_dir: str = '~/flash/data'
    splits: int= 10
    seed: int = 42

cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)

@hydra.main(config_name='config')
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    h5_file = Path(cfg.h5_file).expanduser()
    split_dir = Path(cfg.index_table).expanduser()
    input_shape = (12, 15, 15)
    index_table = split_dir/f'*.parquet'
    index_table = dgp.read_parquet(index_table, gather_spatial_partitions=False).sample(frac=1).compute()
    N = len(index_table)
    n = N // cfg.splits
    for split in range(cfg.splits):
        out_file = Path(cfg.out_dir).expanduser() / f'train{split}.beton'
        index_ = index_table.iloc[n*split:n*(split+1)]
        if split == cfg.splits - 1:
            index_ = index_table.iloc[n*split:]
        dataset = S2Dataset(h5_file, index_)
        
        # Pass a type for each data field
        print("Writing dataset to", out_file)
        writer = DatasetWriter(out_file, {
            # Tune options to optimize dataset size, throughput at train-time
            'image': NDArrayField(dtype=np.dtype("int16"), shape=input_shape),
            'rhs': NDArrayField(dtype=np.dtype("float32"), shape=(101,)),
            'wc': NDArrayField(dtype=np.dtype("int16"), shape=(15,15)),
            'slope': NDArrayField(dtype=np.dtype("float32"), shape=(15,15)),
            'latlon': NDArrayField(dtype=np.dtype("float64"), shape=(2,)),
        })

        # Write dataset
        writer.from_indexed_dataset(dataset)
    

if __name__ == '__main__':
    main()