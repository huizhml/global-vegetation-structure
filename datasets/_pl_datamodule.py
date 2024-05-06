import os
import logging
from pathlib import Path
from typing import List, Union
import pandas as pd
import torch
import lightning as L
import h5py
from datatree.io import _iter_nc_groups
from h5netcdf.legacyapi import Dataset as h5Dataset

from datasets.s2 import S2Dataset

logger = logging.getLogger(__name__)

class PLDataModel(L.LightningDataModule):
    TEST_ZONES = ['23L', '23K']
    TROPIC_SIX = ['32M', '32N', '32P', '33M', '33N', '33P']
    
    def __init__(self, 
        cache_dir: str=None,
        data_dir: str=None,
        h5_dir: str=None,
        split_files_dir: str=None,
        index_table_file: str=None,
        merged_h5_file: str=None,
        test_data_name: str='cal',
        use_zones:Union[str, List[str]]='TEST_ZONES',
        val_ratio: float=0.2,
        random_state: int=42,
        batch_size: int=64,
        num_workers: int=8,
        pin_memory: bool=True,
        shuffle: bool=True,
        drop_last: bool=False,
        **kwargs
        ):
        super().__init__()
        data_dir = Path(data_dir).expanduser()
        h5_dir = Path(h5_dir).expanduser()
        index_table_file = Path(index_table_file).expanduser()
        split_files_dir = Path(split_files_dir).expanduser()
        self.merged_h5_file = data_dir / merged_h5_file
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.test_data_name = test_data_name
        self.val_ratio = val_ratio
        self.random_state = random_state

        self._merge_h5_files(use_zones, h5_dir)
        self._gen_index_table(index_table_file)
        self.train_cal_test_split(index_table_file, split_files_dir)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last


    def _gen_index_table(self, index_table_file, save=True):
        """
        Go through all groups in the HDF5 file and generate index table.

        Args:
            index_table_file (str): The overall index table for zones used.
            save (bool, optional): Whether to save the index tables as CSV files. Defaults to True.

        Returns:
            None
        """
        #TODO: use dask for larger data?
        if index_table_file.exists():
            logger.info('index table exists, skipping...')
            return
        logger.info('Generating index tables for all zones used...')
        index_df = []
        with h5Dataset(self.merged_h5_file, mode='r') as ncds:
            index_df = []
            with h5py.File(self.merged_h5_file) as data:
                for group in _iter_nc_groups(ncds):
                    if len(group.split('/')) == 4:
                        s2_ids = data[f'{group}/id'][:].astype('U')
                        index_df.append([group, s2_ids])
        index_df = pd.DataFrame(index_df, columns=['path','s2_tile'])
        index_df = index_df.explode('s2_tile')
        index_df['s2_tile'] = index_df['s2_tile'].str[33:38]
        index_df['in_partition_idx'] = index_df.groupby('path').cumcount()

        if save:
            index_df.to_csv(index_table_file)
            print(f'index table saved to: ', index_table_file)

    def train_cal_test_split(self, index_table_file, split_files_dir):
        """
        Split the index table into train, cal, and test datasets.

        Args:
            index_table_file (str): The path to the index table.
            split_files_dir (str): The path to the file containing the test tiles.

        Returns:
            None
        """
        # Make sure the train dataset doesn't include any images from the cal and test splits
        if  (self.cache_dir / 'cal_index_table.csv').exists() and (self.cache_dir / 'test_index_table.csv').exists() and (self.cache_dir / 'train_index_table.csv').exists():
            logger.info('index tables for train, cal, test datasets exist, skipping...')
            return
        logger.info('Splitting index tables for train, cal, test datasets...')
        cal_tiles_df = pd.read_csv(split_files_dir / "cal_tiles.txt", sep=" ", names=['s2_tile'])
        test_tiles_df = pd.read_csv(split_files_dir / "test_tiles.txt", sep=" ", names=['s2_tile'])
        index_df = pd.read_csv(index_table_file)

        cal_index_df = index_df[index_df['s2_tile'].isin(cal_tiles_df['s2_tile'])]
        test_index_df = index_df[index_df['s2_tile'].isin(test_tiles_df['s2_tile'])]
        train_index_df = index_df[~index_df['s2_tile'].isin(cal_tiles_df['s2_tile']) & ~index_df['s2_tile'].isin(test_tiles_df['s2_tile'])]
        val_index_df = train_index_df.sample(frac=self.val_ratio, random_state=self.random_state)
        train_index_df = train_index_df.drop(val_index_df.index)
        train_index_df.to_csv(self.cache_dir / 'train_index_table.csv')
        val_index_df.to_csv(self.cache_dir / 'val_index_table.csv')
        cal_index_df.to_csv(self.cache_dir / 'cal_index_table.csv')
        test_index_df.to_csv(self.cache_dir / 'test_index_table.csv')
    
    def _merge_h5_files(self, use_zones, h5_dir):
        """
        Merges the HDF5 files for the specified zones into a single HDF5 file.

        Parameters:
        use_zones (str or list): The zones to merge. If 'all', merges all zones in the directory.
                                If a list, merges the specified zones.
        h5_dir (Path): The directory containing the HDF5 files.

        """
        
        if self.merged_h5_file.exists():
            # TODO: check if existing merged file contains exactly the same zones
            logger.info('h5 files have been merged, skipping...')
            return
        
        logger.info('Merging h5 files...')
        if isinstance(use_zones, list):
            zones = use_zones
        elif use_zones == 'all':
            zones = list(h5_dir.glob('*.h5'))
            zones = [zone.stem for zone in zones]
        else:
            zones = self.__getattribute__(use_zones.upper())
        
        with h5py.File(self.merged_h5_file, 'w') as h5_out:
            for zone in zones:
                h5_out[zone] = h5py.ExternalLink(h5_dir/f'{zone}.h5', '/')
        # with h5py.File(self.merged_h5_file, 'w') as h5_out:
        #     for zone in zones:
        #         if zone in h5_out:
        #             print(f"Group '{zone}' already exists in the destination file.")
        #         else:
        #             # Copy the entire source file under a new group in the destination file
        #             with h5py.File(h5_dir/f'{zone}.h5', 'r') as h5_in:
        #                 h5_in.copy('/', h5_out, name=zone)
                

                    

    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'):
            self.train_dataset = S2Dataset(self.merged_h5_file, index_table=self.cache_dir / 'train_index_table.csv')
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'):
            self.val_dataset = S2Dataset(self.merged_h5_file, index_table=self.cache_dir / 'val_index_table.csv')
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            self.test_dataset = S2Dataset(self.merged_h5_file, index_table=self.cache_dir / f'{self.test_data_name}_index_table.csv')
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
        datamodel = PLDataModel(**cfg.data.init_args)
        import ipdb; ipdb.set_trace()
        for img, label, wc, slope in datamodel.train_dataloader():
            continue
        print('----------------------------------------------------------')
        for img, label, wc, slope in datamodel.train_dataloader():
            continue

    main()
    