import time
import logging
from typing import List
from pathlib import Path
import lightning as L
from ffcv.loader import Loader, OrderOption
from tqdm import tqdm
import numpy as np
import torch
import gc
import psutil
from torch.utils.data import DataLoader
from utils import get_deep_size

logger = logging.getLogger(__name__)

import random

class SubsetCycler:
    def __init__(self, subsets, seed):
        self.subsets = subsets
        self.seed = seed
        self.current_epoch = 0
        self.current_index = 0
        self._generate_new_order()
        
    def _generate_new_order(self):
        self.subset_order = self.subsets.copy()
        random.seed(self.seed + self.current_epoch)
        random.shuffle(self.subset_order)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.subset_order):
            # All subsets have been used; reshuffle and start over
            self._generate_new_order()
        subset = self.subset_order[self.current_index]
        self.current_index += 1
        self.current_epoch += 1
        return subset

    def __len__(self):
        return len(self.subsets)
        
    
    # Function to break references
def break_references(target_obj):
    # Get all referrers
    referrers = gc.get_referrers(target_obj)
    for ref in referrers:
        # Skip the current frame and gc internals
        if isinstance(ref, dict):  # If the referrer is a dictionary (e.g., locals, globals)
            for key, value in list(ref.items()):
                if value is target_obj:
                    ref[key] = None  # Remove reference
        elif isinstance(ref, list):  # If the referrer is a list
            for i, value in enumerate(ref):
                if value is target_obj:
                    ref[i] = None  # Remove reference


class FFCVDataIter(Loader):

    def __init__(self, fname, batch_size, order=None, **kwargs):
        super().__init__(fname, batch_size, order=getattr(OrderOption, order), **kwargs)
        
        self.it = None

    def __iter__(self):
        if self.it is None:
            self.it = super().__iter__()
        return self.it
    
    def close(self):
        self.it.close()
        self.it = None
        # torch.cuda.empty_cache()


class FFCVDataModel(L.LightningDataModule):
    
    def __init__(self,
        train_fp: str=None,
        val_fp: str=None,
        test_fp: str=None,
        cal_fp: str=None,
        pred_fp: str=None,
        distributed: bool=False,
        batches_ahead: int=3,
        batch_size: int=64,
        num_workers: int=8,
        order: str='RANDOM',
        os_cache: bool=False,
        return_iter: bool=True,
        # test params
        tile_id: str=None,
        patch_size=512, border=8,
        bands:List[int]=None, input_lat_lon=False,
        img_idx:int=None,
        seed:int=42,
        **kwargs
        ):
        super().__init__()
        self.train_fp = Path(train_fp).expanduser()
        self.val_fp = Path(val_fp).expanduser()
        if cal_fp is not None:
            self.cal_fp = Path(cal_fp).expanduser()
        if pred_fp is not None:
            self.pred_fp = Path(pred_fp).expanduser()

        if test_fp is not None:
            self.test_fp = Path(test_fp).expanduser()
        if '*' in self.train_fp.stem:
            train_fp = sorted(list(self.train_fp.parent.glob(self.train_fp.name)))
            self.train_fp = SubsetCycler(train_fp, seed)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.order = order # getattr(OrderOption, order)
        self.distributed = distributed
        self.batches_ahead = batches_ahead
        self.os_cache = os_cache
        self.return_iter = return_iter
                    

    def train_dataloader(self):
        if isinstance(self.train_fp, Path) and hasattr(self, 'train_loader'): # using one subset
            print('closing ', self.train_loader.fname)
            self.train_loader.close()
            return self.train_loader
        
        if isinstance(self.train_fp, SubsetCycler):
            train_fp = next(self.train_fp)
            if hasattr(self, 'train_loader'):
                print('closing ', self.train_loader.fname)
                self.train_loader.close()
        else:
            train_fp = self.train_fp
        
        seed = 42 + self.train_fp.current_epoch if isinstance(self.train_fp, SubsetCycler) else 42
        print('loading from ', train_fp, 'seed', seed)
        t0 = time.time()
        self.train_loader = FFCVDataIter(train_fp, self.batch_size, num_workers=self.num_workers,
                distributed=self.distributed, batches_ahead=self.batches_ahead,
                order=self.order, os_cache=self.os_cache, drop_last=True, seed=seed)
        print('time taken: ', time.time()-t0)
        return self.train_loader
        # Return the iterator and close it after epoch end would solve the memory leak issue.
        # ref: https://github.com/libffcv/ffcv/issues/393
        # but using iterator with pytorch lightning results in duplicated batches(few).
        # some model perhaps overfitted to the duplicated batches. (QuantileLoss)

    def val_dataloader(self):
        t0 = time.time()
        if hasattr(self, 'val_loader'):
            self.val_loader.close()
            return self.val_loader
        print('loading from: ', self.val_fp)
        self.val_loader = FFCVDataIter(self.val_fp, self.batch_size, num_workers=self.num_workers,
                distributed=self.distributed, batches_ahead=self.batches_ahead,
                order='SEQUENTIAL', os_cache=self.os_cache, drop_last=False)

        print('time taken for val dataloader: ', time.time()-t0)
        return self.val_loader

    def test_dataloader(self):
        t0 = time.time()
        if hasattr(self, 'test_loader'):
            return self.test_loader
        print('loading from: ', self.test_fp)
        self.test_loader = FFCVDataIter(self.test_fp, self.batch_size, num_workers=self.num_workers,
                distributed=self.distributed, batches_ahead=self.batches_ahead,
                order='SEQUENTIAL', os_cache=self.os_cache, drop_last=False)

        print('time taken for val dataloader: ', time.time()-t0)
        return self.test_loader
    
    def cal_dataloader(self):
        t0 = time.time()
        if hasattr(self, 'cal_loader'):
            self.cal_loader.close()
            return self.cal_loader
        print('loading from: ', self.cal_fp)
        self.cal_loader = FFCVDataIter(self.cal_fp, self.batch_size, num_workers=self.num_workers,
                distributed=self.distributed, batches_ahead=self.batches_ahead,
                order='SEQUENTIAL', os_cache=self.os_cache, drop_last=False)
        print('time taken for cal dataloader: ', time.time()-t0)
        return self.cal_loader

    # def predict_dataloader(self):
    #     t0 = time.time()
    #     if hasattr(self, 'pred_loader'):
    #         return self.pred_loader
    #     print('loading from: ', self.pred_fp)
    #     self.pred_loader = FFCVDataIter(self.pred_fp, self.batch_size, num_workers=self.num_workers,
    #             distributed=self.distributed, batches_ahead=self.batches_ahead,
    #             order='SEQUENTIAL', os_cache=self.os_cache, drop_last=False)

    #     print('time taken for val dataloader: ', time.time()-t0)
    #     return self.pred_loader

def collate_batch(batch):
    return batch

def plot_boxplots(fp, boxplot_dir:str='~/data/GEDI/boxplots'):
    import matplotlib.cbook as cbook
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from const import ESA_WC
    import json
    import os
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    boxplot_dir = Path(boxplot_dir).expanduser()
    dataloader = Loader(fp, batch_size=4098, num_workers=4,
                distributed=False, batches_ahead=3,
                order='SEQUENTIAL', os_cache=False)
    for idx in range(101):
        file = boxplot_dir / f'boxplot_stats_rh{idx}.json'
        # if file.exists():
        #     continue
        rhs = []
        wc = []
        for i, batch in tqdm(enumerate(dataloader)):
            rhs.append(batch[1][:, idx])
            wc.append(batch[2])
        rhs = torch.cat(rhs).numpy()
        wc = torch.cat(wc).numpy()
        data = np.concatenate([rhs[:, None], wc], axis=1)
        df = pd.DataFrame(data, columns=[f'rh{idx}', 'wc'])
        grouped_data = df.groupby('wc')[f'rh{idx}'].apply(list)
        stats = cbook.boxplot_stats(grouped_data.tolist(), labels=grouped_data.index, whis=[5, 95])

        
    # plot boxplots for each ESA_WC
    for name, wc in ESA_WC.items():
        stats = []
        for idx in range(101):
            file = boxplot_dir / f'boxplot_stats_rh{idx}.json'
            with open(file, 'r') as f:
                data = json.load(f)

            import ipdb; ipdb.set_trace()
            data = [d for d in data if d['label'] == wc]
            data = data[0]
            data['label'] = f'{idx}'
            stats.append(data)
        fig, ax = plt.subplots(figsize=(20,6))
        # import ipdb; ipdb.set_trace()
        ax.bxp(stats, patch_artist=True, boxprops={'facecolor': 'bisque'}, showfliers=False)
        ax.set_xticks(np.arange(1, 102, 10))
        ax.set_xticklabels(np.arange(0, 101, 10))
        plt.xlabel('Relative Heights')
        name = name.replace(' ', '_').replace('/', '_or_')
        plt.savefig(f'{boxplot_dir}/RHs_boxplot_{name}.png', dpi=300)
        print('plotting boxplots')

def get_number_of_open_files():
    total_open_files = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Get the list of open files for the process
            open_files = proc.open_files()
            total_open_files += len(open_files)
            print(f"PID: {proc.pid}, Name: {proc.name()}, Open Files: {len(open_files)}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Skip processes that no longer exist or are inaccessible
            continue
    return total_open_files

if __name__ == '__main__':
    import hydra
    @hydra.main(config_name='train', config_path='../config', version_base='1.2')
    def main(cfg):
        cfg.data.init_args.val_fp = '~/data/gvs/train_subsets/train0_filtered_v1.beton'
        cfg.data.init_args.train_fp = '~/data/gvs/train_subsets/train*_filtered_v1.beton'
        cfg.data.init_args.batch_size = 4096
        cfg.data.init_args.distributed = False
        cfg.data.init_args.order = 'SEQUENTIAL'

        # ## USING DATALOADER
        # datamodel = FFCVDataModel(**cfg.data.init_args)
        # val_dataloader = datamodel.val_dataloader()
        # for epoch in range(10):
        #     batch_from_loader = []
        #     for i, batch in enumerate(val_dataloader):
        #         batch_from_loader.append(batch[-1].numpy().copy())
        #     batch_from_loader = np.concatenate(batch_from_loader)
        #     print(batch_from_loader.sum(), np.unique(batch_from_loader).shape)

        # ## USING ITERATOR
        # # cfg.data.init_args.order = 'RANDOM'
        # datamodel = FFCVDataModel(return_iter = True, **cfg.data.init_args)
        # for epoch in range(10):
        #     val_dataiter = datamodel.val_dataloader()
        #     batch_from_loader = []
        #     for i, batch in enumerate(val_dataiter):
        #         batch_from_loader.append(batch[-1].numpy().copy())
        #     batch_from_loader = np.concatenate(batch_from_loader)
        #     val_dataiter.close()
        #     print(batch_from_loader.sum(), np.unique(batch_from_loader).shape)
        
        


        # datamodel = FFCVDataModel(return_iter=False,**cfg.data.init_args)
        # train_dataloader1 = datamodel.train_dataloader()
        # t0 = time.time()
        # iter1 = iter(train_dataloader1)
        # print('time taken to get iterator: ', time.time()-t0)
        # for epoch in range(2):

        #     print('length of train_dataloader: ', len(train_dataloader1))
        #     batch_from_loader = []
        #     for i, batch in enumerate(train_dataloader1):
        #         print(batch[1].mean())
        #         batch_from_loader.append(batch[-1].numpy().copy())
        #         if i >= 9:
        #             break
        #     batch_from_loader = np.concatenate(batch_from_loader)
        #     print(batch_from_loader.sum(), np.unique(batch_from_loader).shape)
        # print('finished loading from loader1')
        
        # print()
        # print()
        datamodel = FFCVDataModel(**cfg.data.init_args)
        rh1_above_100_idxs = []
        rh1_above_20_idxs = []
        for epoch in range(20):
            # print('length of train_dataloader2: ', len(train_dataloader2))
            # train_dataiter = iter(train_dataloader2)
            train_dataiter = datamodel.train_dataloader()
            for batch in tqdm(train_dataiter):
                import ipdb; ipdb.set_trace()
                rh1_above_100_idx, = torch.where(batch[1][:, 1] > 100)
                if rh1_above_100_idx.shape[0] > 0:
                    print('abnormal batch found')
                    rh1_above_100_idx
                    rh1_above_100_idxs.append(rh1_above_100_idx)
                rh1_above_20_idx, = torch.where(batch[1][:, 1] > 20)
                if rh1_above_20_idx.shape[0] > 0:
                    rh1_above_20_idxs.append(rh1_above_20_idx)
        print('finished loading from iterator')
        print(len(rh1_above_100_idxs), len(rh1_above_20_idxs))
        print(rh1_above_20_idxs)
        print(rh1_above_100_idxs)
        rh1_above_20_idxs = torch.cat(rh1_above_20_idxs)
        rh1_above_100_idxs = torch.cat(rh1_above_100_idxs)
        torch.save(rh1_above_20_idxs, 'output/rh1_above_20_idxs.pt')
        torch.save(rh1_above_100_idxs, 'output/rh1_above_100_idxs.pt')
        

    main()
    
