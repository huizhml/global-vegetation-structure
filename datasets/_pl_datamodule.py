import os
import logging
from pathlib import Path
from typing import Callable, List, Union
import pandas as pd
import torch
import lightning.pytorch as pl
import tables
import h5py
from datatree.io import _iter_nc_groups
from h5netcdf.legacyapi import Dataset as h5Dataset

from datasets.s2 import S2Dataset

logger = logging.getLogger(__name__)

class PLDataModle(pl.LightningDataModule):
    TEST_ZONES = ['32M', '01G']
    TROPIC_SIX = ['32M', '32N', '32P', '33M', '33N', '33P']
    
    def __init__(self, 
        cache_dir: str,
        data_dir: str,
        split_files_dir: str,
        h5_file: str,
        use_zones:Union[str, List[str]]='TEST_ZONES',
        batch_size: int=64,
        num_workers: int=8,
        pin_memory: bool=True,
        shuffle: bool=True,
        drop_last: bool=False,
        **kwargs
        ):
        super().__init__()
        data_dir = Path.home() / data_dir
        h5_file = data_dir / h5_file if isinstance(h5_file, str) else h5_file
        if not h5_file.exists():
            logger.info('Merging h5 files...')
            self._merge_h5_files(use_zones, h5_file)

        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if not (self.cache_dir / 'train_index_table.csv').exists():
                logger.info('Generating index tables for train,val,cal and test dataset...')
                self._gen_index_table(h5_file, data_dir / split_files_dir)   

        self.train_dataset = S2Dataset(h5_file, index_table=self.cache_dir / 'train_index_table.csv')
        self.val_dataset = S2Dataset(h5_file, index_table=self.cache_dir / 'val_index_table.csv')
        self.test_dataset = S2Dataset(h5_file, index_table=self.cache_dir / 'cal_index_table.csv')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last

    def _gen_index_table(self, h5_file:Path, split_files_dir, save=True):
        """
        Generate index tables for train, validation, calibration, and test datasets.

        Args:
            split_files_dir (str): The directory containing the split files.
            save (bool, optional): Whether to save the index tables as CSV files. Defaults to True.

        Returns:
            None
        """

        val_tiles_df = pd.read_csv(split_files_dir / "val_tiles.txt", sep=" ", names=['s2_tile'])
        cal_tiles_df = pd.read_csv(split_files_dir / "cal_tiles.txt", sep=" ", names=['s2_tile'])
        test_tiles_df = pd.read_csv(split_files_dir / "test_tiles.txt", sep=" ", names=['s2_tile'])
        train_index_table = []
        val_index_table = []
        cal_index_table = []
        test_index_table = []

        with h5Dataset(h5_file, mode='r') as ncds:
            with h5py.File(h5_file) as data:
                for group in _iter_nc_groups(ncds):
                    if len(group.split('/')) == 4: #TODO: not eligant
                        _, zone, year, partition = group.split('/')
                        partition = int(partition)
                        val_tiles = val_tiles_df[val_tiles_df['s2_tile'].str.contains(zone)].values
                        cal_tiles = cal_tiles_df[cal_tiles_df['s2_tile'].str.contains(zone)].values
                        test_tiles = test_tiles_df[test_tiles_df['s2_tile'].str.contains(zone)].values
                        
                        s2_ids = data[f'{zone}/{year}/{partition}/id'][:].astype('U')
                        in_partition_idx_val = []
                        in_partition_idx_cal = []
                        in_partition_idx_test = []
                        in_partition_idx_train = []
                        for i, s2_id in enumerate(s2_ids):
                            s2_id = s2_id[33:38]
                            if s2_id in val_tiles:
                                in_partition_idx_val.append(i)
                            elif s2_id in cal_tiles:
                                in_partition_idx_cal.append(i)
                            elif s2_id in test_tiles:
                                in_partition_idx_test.append(i)
                            else:
                                in_partition_idx_train.append(i)
                        val_index_table.extend([(zone, year, partition, idx) for idx in in_partition_idx_val])
                        cal_index_table.extend([(zone, year, partition, idx) for idx in in_partition_idx_cal])
                        test_index_table.extend([(zone, year, partition, idx) for idx in in_partition_idx_test])
                        train_index_table.extend([(zone, year, partition, idx) for idx in in_partition_idx_train])

        for index_table, name in zip([train_index_table, val_index_table, cal_index_table, test_index_table], ['train', 'val', 'cal', 'test']):
            index_table = pd.DataFrame(index_table, columns=['zone', 'year', 'partition_idx', 'in_partition_idx'])
            if save:
                index_table.to_csv(self.cache_dir / f'{name}_index_table.csv')
                print(f'{name} index table saved to: ', self.cache_dir / f'{name}_index_table.csv')

    def _merge_h5_files(self, use_zones, output_file):
            """
            Merges the HDF5 files for the specified zones into a single HDF5 file.

            Parameters:
            use_zones (str or list): The zones to merge. If 'all', merges all zones in the directory.
                                    If a list, merges the specified zones.
            output_file (str): The path to the output HDF5 file.

            Returns:
            tables.File: The merged HDF5 file opened in read mode.
            """
            
            if isinstance(use_zones, list):
                zones = use_zones
            elif use_zones == 'all':
                zones = [f for f in os.listdir(output_file.parent) if f.is_dir()]
            else:
                zones = self.__getattribute__(use_zones.upper())
            
            with h5py.File(output_file, 'w') as h5_out:
                for zone in zones:
                    h5_out[zone] = h5py.ExternalLink(output_file.parent/f'{zone}.h5', '/')


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
    

if __name__ == '__main__':
    import hydra
    @hydra.main(config_name='train', config_path='../config', version_base='1.2')
    def main(cfg):
        datamodel = PLDataModle(**cfg.data.init_args)
        for img, label in datamodel.train_dataloader():
            print(img.shape, label.shape)

    main()
    