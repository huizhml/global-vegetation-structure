import os
import logging
from pathlib import Path
from typing import List, Union
import pandas as pd
import torch
import lightning as L
import h5py
import dask.dataframe as dd


from datasets.s2 import S2Dataset

logger = logging.getLogger(__name__)

def read_index_table(index_table_fp: Path):
    if os.path.isdir(index_table_fp):
        files = list(Path(index_table_fp).glob('*.parquet'))
        index_table = dd.read_parquet(files, columns=['path', 'in_partition_idx']).compute()
    else:
        index_table = pd.read_parquet(index_table_fp)
    index_table = index_table.reset_index(drop=True)
    return index_table

class PLDataModel(L.LightningDataModule):
    
    def __init__(self, 
        h5_dir: str=None,
        index_dir: str=None,
        merged_h5_file: str=None,
        use_subset: bool=False,
        test_ratio: float=0.1,
        cal_ratio: float=0.1,
        val_ratio: float=0.1,
        random_state: int=42,
        batch_size: int=64,
        num_workers: int=8,
        pin_memory: bool=True,
        shuffle: bool=True,
        drop_last: bool=False,
        **kwargs
        ):
        super().__init__()
        h5_dir = Path(h5_dir).expanduser()
        index_dir = Path(index_dir).expanduser()
        self.split_dir = h5_dir.parent/f'split_test{test_ratio}_cal{cal_ratio}_val{val_ratio}_seed{random_state}'
        self.merged_h5_file = Path(merged_h5_file).expanduser()

        self.use_subset = use_subset
        self.test_ratio = test_ratio
        self.cal_ratio = cal_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state

        self._merge_h5_files(h5_dir)
        self.train_cal_test_split(index_dir)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            name = 'subset' if self.use_subset else 'train'
            index_table_train = read_index_table(self.split_dir / f'index_table_{name}') #index_table_{name} geo_index_table
            
            self.train_dataset = S2Dataset(self.merged_h5_file, index_table=index_table_train)
            
            index_table_val = read_index_table(self.split_dir / 'index_table_val')
            self.val_dataset = S2Dataset(self.merged_h5_file, index_table=index_table_val)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            index_table_test = read_index_table(self.split_dir / 'index_table_test')
            self.test_dataset = S2Dataset(self.merged_h5_file, index_table=index_table_test)

    def _gen_index_table(self, index_dir):
        """
        Go through all groups in the HDF5 file and generate index table for the whole dataset.

        Args:
            index_dir (str): The overall index table for all zones.

        Returns:
            None
        """
        if len(os.listdir(index_dir)) >= 435:
            logger.info('index table exists, skipping...')
            return
        logger.info('Generating index tables for all zones...')
        os.system(f'python -m datasets._generate_index_table')

    def train_cal_test_split(self, index_dir):
        """
        Split the index table into train, cal, and test datasets.

        Args:
            index_dir (str): The path to the index table.

        Returns:
            None
        """
        # Make sure the train dataset doesn't include any images from the cal and test splits
        index_table_exists = True
        for split in ['train', 'val', 'cal', 'test']:
            if not len(os.listdir(self.split_dir/f'index_table_{split}')) > 1:
                index_table_exists = False
                break
        if index_table_exists:
            logger.info('index tables for train, val, cal, test sets exist, skipping...')
            return

        logger.info('Check if the full dataset index table exists...')
        self._gen_index_table(index_dir)
        logger.info('Splitting index tables for train, cal, test datasets...')
        os.system(f'python -m datasets._data_split test_ratio={self.test_ratio} cal_ratio={self.cal_ratio} val_ratio={self.val_ratio} random_state={self.random_state}')

    
    def _merge_h5_files(self, h5_dir):
        """
        Merges the HDF5 files for all zones into a single HDF5 file.

        Parameters:
        h5_dir (Path): The directory containing the HDF5 files.

        """
        
        if self.merged_h5_file.exists():
            logger.info('h5 files have been merged, skipping...')
            return
        
        logger.info('Merging h5 files...')
        from datasets._merge_h5s import merge_all_zones
        merge_all_zones(h5_dir, self.merged_h5_file)
                    

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
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
        datamodel = PLDataModel(**cfg.data.init_args)
        datamodel.setup('fit')
        # import ipdb; ipdb.set_trace()
        dataloader = datamodel.train_dataloader()
        n = len(datamodel.train_dataset)
        print('starting')
        import time
        t0 = time.time()
        # for i in range(n):
        #     data = datamodel.train_dataset[i]
        #     print(i)
        # print('----------------------------------------------------------')
        
        for i, (img, label, wc, slope) in enumerate(datamodel.train_dataloader()):
            print(i)
        print('----------------------------------------------------------')
        print('time taken: ', time.time()-t0)
        print(len(datamodel.train_dataset))
        # for img, label, wc, slope in datamodel.train_dataloader():
        #     continue

    main()
    