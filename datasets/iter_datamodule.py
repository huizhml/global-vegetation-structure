import os
import logging
from pathlib import Path
from typing import List, Union
import random
from torch.utils.data import DataLoader
import lightning as L
from datasets._mp_iter_dataset import IterH5Dataset

logger = logging.getLogger(__name__)


class MPIterDataModel(L.LightningDataModule):

    def __init__(self,
                 train_fp: str = None,
                 train_index_folder: str = None,
                 val_fp: str = None,
                 val_index_folder: str = None,
                 num_jobs: int = 112,
                 num_samples: Union[int, float] = 2e10,
                 batches_ahead: int = 3,
                 batch_size: int = 64,
                 **kwargs
                 ):
        super().__init__()
        self.train_fp = Path(train_fp).expanduser()
        self.train_index_folder = Path(train_index_folder).expanduser()
        self.val_fp = Path(val_fp).expanduser()
        self.val_index_folder = Path(val_index_folder).expanduser()
        

        # if not self.train_fp.exists() or not self.val_fp.exists():
        #     raise FileNotFoundError(f'{self.train_fp} does not exist.')
        if self.train_fp.is_dir():
            self.train_fp = sorted(list(self.train_fp.glob('*.h5')))
        else:
            self.train_fp = [self.train_fp]

        self.num_jobs = num_jobs
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.batches_ahead = batches_ahead

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        pid = os.getpid()
        seed = 42 + random.randint(0, pid)
        if stage == "fit":
            self.train_dataset = IterH5Dataset(
                self.train_fp, num_samples=self.num_samples, cache_size=self.batch_size * self.batches_ahead,
                index_folder=self.train_index_folder, num_jobs=self.num_jobs, seed=seed)
            # self.val_dataset = IterH5Dataset(
            #     self.val_fp, num_samples=self.num_samples, cache_size=self.batch_size * self.batches_ahead,
            #     index_folder=self.val_index_folder_val, num_jobs=self.num_jobs, seed=seed) # QUESTION: do we need the index tree for val & test?

        # # Assign test dataset for use in dataloader(s)
        # if stage == "test":
            # self.test_dataset = IterH5Dataset(
            #     self.test_fp, num_samples=self.num_samples, cache_size=self.batch_size * self.batches_ahead,
            #     index_folder=self.val_index_folder_test, num_jobs=self.num_jobs, seed=seed) 

        # if stage == "predict":
            # self.cal_dataset = IterH5Dataset(
            #     self.cal_fp, num_samples=self.num_samples, cache_size=self.batch_size * self.batches_ahead,
            #     index_folder=self.val_index_folder_cal, num_jobs=self.num_jobs, seed=seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False, drop_last=True)

if __name__ == '__main__':
    import hydra

    @hydra.main(config_name='train', config_path='../config', version_base='1.2')
    def main(cfg):
        datamodel = MPIterDataModel(**cfg.data.init_args)
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
