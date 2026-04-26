from multiprocessing import Process, Queue, Value
import ctypes
import os
from time import sleep
from typing import List, Iterable
from pathlib import Path
from ffcv.memory_allocator import MemoryAllocator
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader, OrderOption
from ffcv.fields import NDArrayField, IntField, FloatField
from tqdm import tqdm
from datasets.h5_dataset import S2Dataset
import random
import pandas as pd
import geopandas as gpd
import numpy as np
import dask_geopandas as dgp
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import h5py
import dask
import dask.bag as db
import dask.dataframe as dd

from datasets._3_merge_h5s import dst_conf


class MyDatasetWriter(DatasetWriter):

    def _write_common(self, num_samples, queue_content, work_fn, extra_worker_args):
        self.num_samples = num_samples

        self.prepare()
        allocation_list = []

        # Makes a memmap to the metadata for the samples

        # We publish all the work that has to be done into a queue
        workqueue: Queue = Queue()
        for todo in queue_content:
            workqueue.put(todo)

        # This will contain all the memory allocations each worker
        # produced. This will go at the end of the file
        allocations_queue: Queue = Queue()

        # We add a token for each worker to warn them that there
        # is no more work to be done
        for _ in range(self.num_workers):
            workqueue.put(None)

        # Define counters we need to orchestrate the workers
        done_number = Value(ctypes.c_uint64, 0)
        allocator = MemoryAllocator(self.fname,
                                    self.data_region_start,
                                    self.page_size)

        # Arguments that have to be passed to the workers
        worker_args = (workqueue, self.metadata_sm,
                       self.metadata_type, self.fields,
                       allocator, done_number,
                       allocations_queue, *extra_worker_args)

        # Create the workers
        processes = [Process(target=work_fn, args=worker_args)
                     for _ in range(self.num_workers)]
        # start the workers
        for p in processes: p.start()
        # Wait for all the workers to be done

        # Display progress
        progress = tqdm(total=self.num_samples)
        previous = 0
        while previous != self.num_samples:
            val = done_number.value
            diff = val - previous
            if diff > 0:
                progress.update(diff)
            previous = val
            sleep(0.1)
        progress.close()

        # Wait for all the workers to be done and get their allocations
        for p in processes:
            content = allocations_queue.get()
            allocation_list.extend(content)

        self.finalize(allocation_list)
        self.metadata_sm.close()
        try:
            self.metadata_sm.unlink()
        except FileNotFoundError:
            print("file not found")


def split_index_table(nsplit, out_dir, index_table_fp,version):
    index_table = dgp.read_parquet(index_table_fp, gather_spatial_partitions=False).compute()
    # Reindex in_partition_idx, it's not continuous, use it with train.h5 will cause the index out of range error
    # we use train.h5, the orignal whole data is too large
    index_table = index_table.sort_values(['path', 'in_partition_idx'])
    index_table['in_partition_idx'] = index_table.groupby('path').cumcount()    
    # index_table = index_table[index_table['sensitivity'] >= 0.95]
    print('number of training samples (high sensitivity): ', len(index_table))
    index_table = index_table.sample(frac=1)
    N = len(index_table)
    n = N // nsplit
    index_table = index_table.astype({'shot_number': int})
    for split in range(nsplit):
        index_ = index_table.iloc[n*split:n*(split+1)] # not shuffled!
        if split == nsplit - 1:
            index_ = index_table.iloc[n*split:]
        index_.to_parquet(out_dir / f'train{split}_v{version}.parquet')

def split_h5(h5_dir, index_table_fps, save_dir): 
    import glob
    from dask.distributed import Client, LocalCluster
    from dask import config
    cluster = LocalCluster()
    client = Client(cluster)
    print(client)

    index_table_fps = glob.glob(index_table_fps)
    h5_dir = Path(h5_dir).expanduser()
    save_dir = Path(save_dir).expanduser()

    bg = db.from_sequence(index_table_fps, npartitions=10)
    bg.map(_split_h5_per_subsest, save_dir, h5_dir).compute()

def _split_h5_per_subsest(index_table_fp, save_dir, h5_dir):
    print(index_table_fp)
    index_table_fp = Path(index_table_fp).expanduser()
    index_table = gpd.read_parquet(index_table_fp)
    index_table = index_table.sort_values(['path', 'in_partition_idx'])
    # index_table['in_partition_idx'] = index_table.groupby('path').cumcount()
    with h5py.File(h5_dir, 'r') as f:
        with h5py.File(save_dir / f'{index_table_fp.stem}.h5', 'w') as out:
            for path in index_table.path.unique():
                # path = path[4:]
                idx = index_table[index_table['path']==path]['in_partition_idx'].unique()
                for name, config in dst_conf.items():
                    # if f'{path}/{name}' in f:
                    #     continue
                    data = f[f'{path}/{name}'][idx]
                    out.create_dataset(f'{path}/{name}', 
                                            shape=data.shape, 
                                            chunks=(1,) +config['shape'], 
                                            dtype=config['dtype'],
                                            compression="gzip", #lzf
                                            compression_opts=7, 
                                            data=data)
    # index_table['zone'] = index_table['path'].str.split('/').str[0]
    # index_table = dd.from_pandas(index_table, npartitions=16)
    # index_table.groupby('zone').apply(_split_h5_per_zone, index_table_fp.stem, save_dir, h5_dir, meta=('x', 'i8')).compute()


                    
def _split_h5_per_zone(index_table, subset, h5_dir, save_dir):
    zone = index_table['zone'].iloc[0]
    (save_dir/subset).mkdir(exist_ok=True, parents=True)
    h5_file = h5_dir / f'{zone}.h5'
    with h5py.File(h5_file, 'r') as f:
        with h5py.File(save_dir / subset / f'{zone}.h5', 'w') as out:
            for path in index_table.path.unique():
                path = path[4:]
                idx = index_table[index_table['path']==path]['in_partition_idx'].unique()
                for name, config in dst_conf.items():
                    data = f[f'{path}/{name}'][idx]
                    out.create_dataset(f'{path}/{name}', 
                                            shape=data.shape, 
                                            chunks=(1,) +config['shape'], 
                                            dtype=config['dtype'],
                                            compression="gzip", #lzf
                                            compression_opts=7, 
                                            data=data)


def write_beton(out_file, dataset, shuffle_indices=False):
    
    # Pass a type for each data field
    print("Writing dataset to", out_file)
    writer = MyDatasetWriter(out_file, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': NDArrayField(dtype=np.dtype("uint16"), shape=(12, 15, 15)),
        'rhs': NDArrayField(dtype=np.dtype("float32"), shape=(101,)),
        'wc': NDArrayField(dtype=np.dtype("uint16"), shape=(15, 15)), #IntField(),
        'slope':  NDArrayField(dtype=np.dtype("float32"), shape=(15, 15)),
        'lat': NDArrayField(dtype=np.dtype("float64"), shape=(15,)),
        'lon': NDArrayField(dtype=np.dtype("float64"), shape=(15,)),
        # 'latlon': NDArrayField(dtype=np.dtype("float64"), shape=(2,)),
        # 'sensitivity': FloatField(),
        # 'shot_number': IntField()
    })

    # Write dataset
    writer.from_indexed_dataset(dataset, shuffle_indices=shuffle_indices)

def visualize_subset_distribution(index_table_fp):
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.utils import lnglat_to_meters
    import matplotlib.pyplot as plt
    import fsspec

    print('Visualizing subset distribution, ', index_table_fp)
    index_table = gpd.read_parquet(index_table_fp)
    # Convert to a format suitable for Datashader
    index_table = index_table.to_crs(epsg=3857)  # Convert to Web Mercator for better alignment
    index_table['x'], index_table['y'] = index_table.geometry.x, index_table.geometry.y
    # Create a Canvas object for rasterization
    x_range = (-20037508.342789244, 20037508.342789244)
    y_range = (-20037508.342789244, 20037508.342789244)
    canvas = ds.Canvas(plot_width=1000, plot_height=1000, x_range=x_range, y_range=y_range)
    # Rasterize the points
    print('Rasterizing')
    agg = canvas.points(index_table, 'x', 'y')
    # Convert to an image
    print('Converting to image')
    img = tf.shade(agg, cmap=['lightblue', 'darkblue'])
    # Display the image
    print('Displaying')
    pil_img = img.to_pil()
    print('Saving')
    # plt.savefig(f'/users/zhanghui/data/distribution_{index_table_fp.stem}.png', dpi=300, bbox_inches='tight')
    pil_img.save(f'output/distribution_{index_table_fp.stem}.png')


def check_s2_value(train_fp: Path):
    print('Checking values for ', train_fp)
    loader = Loader(train_fp, batch_size=4196, num_workers=4, order=OrderOption.SEQUENTIAL, os_cache=False)
    for i, batch in tqdm(enumerate(loader)):
        if batch[0].min() < 0:
            print('negative values found')
            print(i)
            print(batch[0].min())
            return


def run_convert_to_beton(cfg: DictConfig):
    h5_file = Path(cfg.h5_file).expanduser()
    index_dir = Path(cfg.index_table).expanduser()
    out_idx_dir = Path(cfg.out_idx_dir).expanduser()
    out_idx_dir.mkdir(exist_ok=True, parents=True)

    if 'train' in index_dir.stem:
        print('Generating train subsets in beton...')
        if len(list(out_idx_dir.glob(f'*v{cfg.version}.parquet'))) != cfg.nsplit:
            print('Re-splitting index table')
            split_index_table(cfg.nsplit, out_idx_dir, index_dir/f'*.parquet', cfg.version)
            visualize_subset_distribution(out_idx_dir / f'train{cfg.split_idx}_v{cfg.version}.parquet')
        
        # split_h5(h5_file, str(splited_idx_dir / f'train*.parquet'), out_dir)
        out_beton_name = 'debug' if cfg.get('debug', False) else 'train'
        out_file = out_idx_dir.parent / f'train_subsets/{out_beton_name}{cfg.split_idx}_filtered_v{cfg.version}.beton'
        if not out_file.exists():
            index_ = pd.read_parquet(out_idx_dir / f'train{cfg.split_idx}_v{cfg.version}.parquet')
            if cfg.get('debug', False):
                size = 4096 * 28 # 1% of one subset
                index_ = index_.iloc[:size]
            print(f'size of training subset {cfg.version}:', len(index_))
            dataset = S2Dataset(h5_file, index_)
            write_beton(out_file, dataset, shuffle_indices=cfg.shuffle_indices)
    else:
        split = index_dir.stem.split('_')[2]
        assert h5_file.stem == split, f'cannot generate {split}.beton from {h5_file}, check index_dir and h5_file'
        
        if cfg.get('create_subset'):
            subset_sizes = [1000000, 5000000, 10000000]
            out_files = [h5_file.parent / f'train_subsets/{split}_filtered_v{cfg.version}_{str(size//1000000)}m.beton' for size in subset_sizes]
            out_files = [(Path(file), subset_sizes[i]) for i, file in enumerate(out_files) if not Path(file).exists()]
            if len(out_files) > 0:
                index_table = dgp.read_parquet(index_dir/'*.parquet', gather_spatial_partitions=False).compute()
                #NOTE: Reindex in_partition_idx, it's not continuous, use it with val.h5 will cause the index out of range error
                # we use val.h5, the orignal whole data is too large
                index_table = index_table.sort_values(['path', 'in_partition_idx'])
                index_table['in_partition_idx'] = index_table.groupby('path').cumcount()

            for out_file, size in out_files:
                print(f'Generating {out_file} from {h5_file}...')
                index_table = index_table.sample(frac=1).iloc[:size]
                print(f'number of {split} subset {size} samples (high sensitivity): ', len(index_table))
                dataset = S2Dataset(h5_file, index_table)
                write_beton(out_file, dataset, shuffle_indices=False)

        else:
            out_file = h5_file.parent / f'train_subsets/{split}_filtered_v{cfg.version}.beton'
            if not Path(out_file).exists():
                print(f'Generating {out_file} from {h5_file}...')
                index_table = dgp.read_parquet(index_dir/'*.parquet', gather_spatial_partitions=False).compute()
                #NOTE: Reindex in_partition_idx, it's not continuous, use it with val.h5 will cause the index out of range error
                # we use val.h5, the orignal whole data is too large
                index_table = index_table.sort_values(['path', 'in_partition_idx'])
                index_table['in_partition_idx'] = index_table.groupby('path').cumcount()    
                print(f'number of {split} samples (high sensitivity): ', len(index_table))
                dataset = S2Dataset(h5_file, index_table)
                write_beton(out_file, dataset, shuffle_indices=False)

def check_total_size(index_table: Path):
    import pyarrow.parquet as pq
    index_table = Path(index_table).expanduser()
    n = 0
    for file in index_table.glob('*.parquet'):
        n += pq.ParquetFile(file).metadata.num_rows
    print('Total number of samples: ', n)
 
@dataclass
class MyConfig:
    index_table: str = '~/data/gvs/split_test0.1_cal0.1_val0.1_seed42_v1/index_table_train'
    h5_file: str = '~/data/gvs/data_train.h5'
    out_idx_dir: str = '~/data/gvs/index_table_train_subsets'
    nsplit: int= 5
    split_idx: int = 0
    seed: int = 42
    shuffle_indices: bool = False
    version: str = '1' # version of the beton file
    task: str = 'convert_to_beton'

cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)

@hydra.main(config_name='config', version_base="1.2")
def main(cfg: DictConfig):
    print(cfg)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.task == 'convert_to_beton':
        run_convert_to_beton(cfg)
    elif cfg.task == 'check_s2_value':
        check_s2_value(cfg.out_file)
    elif cfg.task == 'check_total_size':
        check_total_size(cfg.index_table)
    

if __name__ == '__main__':
    print('Running main')
    main()