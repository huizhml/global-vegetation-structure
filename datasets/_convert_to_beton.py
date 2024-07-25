from typing import List, Iterable
from pathlib import Path
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField, FloatField
from datasets.s2 import S2Dataset
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class MyConfig:
    split_dir: str = '~/data/split_test0.1_cal0.1_val0.1_seed42'
    h5_file: str = '~/scratch/data/GVS.h5'
    out_dir: str = '~/flash/data'
    splits: list = field(default_factory=lambda: ['train'])

cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)

@hydra.main(config_name='config')
def main(cfg: DictConfig):
    h5_file = Path(cfg.h5_file).expanduser()
    split_dir = Path(cfg.split_dir).expanduser()
    input_shape = (12, 15, 15)
    for split in cfg.splits:
        index_table = split_dir/f'index_table_{split}/*.parquet'
        out_file = Path(cfg.out_dir).expanduser() / f'{split}.beton'
        index_table = dd.read_parquet(index_table).compute()
        dataset = S2Dataset(h5_file, index_table)
        
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