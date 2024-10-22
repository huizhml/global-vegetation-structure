import os
import time
import logging
import hydra
import h5py
import dask
import dask.bag as db
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from dask.utils import natural_sort_key
from datatree.io import _iter_nc_groups
from h5netcdf.legacyapi import Dataset as h5Dataset

logger = logging.getLogger(__name__)

class IndexTableGenerater:
    """
    Using dask to process zones in parallel.
    Generate an index table for each zone.
    """

    def __init__(self, ref_index_table_dir:str=None, old_index_table_dir:str='~/data/index_table') -> None:
        """
        * h5_dir: folder where {zone}.h5 is
        * index_table_dir: folder where {zone}.parquet is saved to
        """
        self.ref_index_table_dir = Path(ref_index_table_dir).expanduser()
        self.old_index_table_dir = Path(old_index_table_dir).expanduser()
        parent_dir = self.old_index_table_dir.parent
        self.new_index_table_dir = parent_dir/f'{self.old_index_table_dir.stem}_with_sensitivity'
        self.new_index_table_dir.mkdir(parents=True, exist_ok=True)

        old_index_files = self.old_index_table_dir.glob('*.parquet')
        zones = [f.stem for f in old_index_files]
        self.target_files = [str(self.new_index_table_dir/f'{z}.parquet') for z in zones]
        self.target_files = sorted(self.target_files, key=natural_sort_key)
    
    def __call__(self):
        print('generating index tables for ', self.unfinished_files)
        if len(self.unfinished_files) == 0:
            logger.info('All zones have index tables generated.')
            return
        files_db = db.from_sequence(self.unfinished_files)
        files_db.map(self.update_index_table_for_zone).compute()


    @property
    def unfinished_files(self):
        files = []
        for fp in self.target_files:
            if not os.path.exists(fp):
                files.append(fp)
        return files

    def update_index_table_for_zone(self, index_table_file, save=True):
        """
        Go through all groups in the HDF5 file and generate index table.

        Args:
            index_table_file (str): The overall index table for zones used.
            save (bool, optional): Whether to save the index tables as CSV files. Defaults to True.

        Returns:
            None
        """
        index_table_file = Path(index_table_file)
        df_old = pd.read_parquet(self.old_index_table_dir/index_table_file.name)
        df_ref = pd.read_parquet(self.ref_index_table_dir/index_table_file.name)
        df_new = pd.merge(df_old, df_ref[['path', 'in_partition_idx', 'sensitivity', 'shot_number']], on=['path', 'in_partition_idx'], how='left')
        df_new.to_parquet(index_table_file)

def add_sensitivity_for_train_subsets(index_table_fp='~/data/GEDI/split_test0.1_cal0.1_val0.1_seed42/index_table_train_with_sensitivity/*.parquet', subset_index_dir='~/data/GEDI/index_table_train_subsets/'):
    import dask.dataframe as dd
    import glob
    index_table_fp = Path(index_table_fp).expanduser()
    index_table_fp = glob.glob(str(index_table_fp))
    index_table = dd.read_parquet(index_table_fp, gather_spatial_partitions=False).compute()
    # Reindex in_partition_idx, it's not continuous, use it with train.h5 will cause the index out of range error
    # we use train.h5, the orignal whole data is too large
    index_table = index_table.sort_values(['path', 'in_partition_idx'])
    index_table['in_partition_idx'] = index_table.groupby('path').cumcount()    
    subset_index_dir = Path(subset_index_dir).expanduser()
    index_table_subset_fps = glob.glob(str(subset_index_dir/'*.parquet'))
    for subset_fp in index_table_subset_fps:
        print(subset_fp)
        if 'train1' in subset_fp:
            continue
        subset = pd.read_parquet(subset_fp)
        new_idx_table = pd.merge(subset, index_table, on=['path', 'in_partition_idx'], how='left')
        new_idx_table['shot_number'] = new_idx_table['shot_number'].astype('str')
        new_idx_table.to_parquet(subset_fp)


@dataclass
class MyConfig:
    ref_index_table_dir: str = '~/data/GEDI/geo_index_table_with_sensitivity'
    old_index_table_dir: str = '~/data/GEDI/split_test0.1_cal0.1_val0.1_seed42/'


cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg):
    # check if the zone has Sentinel-2 candidates gathered
    # from dask.distributed import Client, LocalCluster
    # # might fix the communication error caused by I/O. ref: https://github.com/dask/distributed/issues/3129#issuecomment-1684858307
    # dask.config.set({"distributed.comm.retry.count": 10})
    # dask.config.set({"distributed.comm.timeouts.connect": 30})
    # dask.config.set({"distributed.scheduler.active-memory-manager.MALLOC_TRIM_THRESHOLD_": 0})
    # cluster = LocalCluster()
    # client = Client(cluster)
    # print(client)

    t0 = time.time()
    # split_index_dir = Path(cfg.old_index_table_dir)
    # for name in ['train', 'cal', 'val', 'test']:
    #     old_index_table_dir = split_index_dir/f'index_table_{name}'
    #     bestS2Finder = IndexTableGenerater(cfg.ref_index_table_dir, old_index_table_dir)()
    add_sensitivity_for_train_subsets()
    logger.info(f'time taken: {time.time() - t0}')
    # client.close()

if __name__ == '__main__':
    main()