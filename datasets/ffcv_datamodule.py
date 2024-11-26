import time
import logging
from pathlib import Path
import lightning as L
from ffcv.loader import Loader, OrderOption
from tqdm import tqdm
import numpy as np
import torch

from datasets.stats import AverageMeter

logger = logging.getLogger(__name__)


class FFCVDataModel(L.LightningDataModule):
    
    def __init__(self,
        train_fp: str=None,
        val_fp: str=None,
        test_fp: str=None,
        distributed: bool=False,
        batches_ahead: int=3,
        batch_size: int=64,
        num_workers: int=8,
        order: str='RANDOM',
        os_cache: bool=False,
        **kwargs
        ):
        super().__init__()
        self.train_fp = Path(train_fp).expanduser()
        self.val_fp = Path(val_fp).expanduser()
        if test_fp is not None:
            self.test_fp = Path(test_fp).expanduser()
        if not self.train_fp.exists():
            raise FileNotFoundError(f'{self.train_fp} does not exist. Please run python -m datasets._convert_to_beton.')
        if self.train_fp.is_dir():
            self.train_fp = sorted(list(self.train_fp.glob('*.beton')))
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.order = getattr(OrderOption, order)
        self.distributed = distributed
        self.batches_ahead = batches_ahead
        self.os_cache = os_cache
                    

    def train_dataloader(self):
        print('loading from ', self.train_fp)

        t0 = time.time()
        loader = Loader(self.train_fp, batch_size=self.batch_size, num_workers=self.num_workers,
                distributed=self.distributed, batches_ahead=self.batches_ahead,
                order=self.order, os_cache=self.os_cache)
        print('time taken: ', time.time()-t0)
        return loader


    def val_dataloader(self):
        t0 = time.time()
        loader = Loader(self.val_fp, batch_size=self.batch_size, num_workers=self.num_workers,
                distributed=self.distributed, batches_ahead=self.batches_ahead,
                order=OrderOption.SEQUENTIAL, os_cache=self.os_cache)
        print('time taken for val dataloader: ', time.time()-t0)
        return loader

    def test_dataloader(self):
        return Loader(self.test_fp, batch_size=self.batch_size, num_workers=self.num_workers,
                distributed=self.distributed, batches_ahead=self.batches_ahead,
                order=OrderOption.SEQUENTIAL, os_cache=self.os_cache)



def plot_boxplots(fp, boxplot_dir:str='~/data/GEDI/boxplots'):
    import matplotlib.cbook as cbook
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from const import ESA_WC
    import json

    boxplot_dir = Path(boxplot_dir).expanduser()
    dataloader = Loader(fp, batch_size=4098, num_workers=4,
                distributed=False, batches_ahead=3,
                order=OrderOption.SEQUENTIAL, os_cache=False)
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
        import ipdb; ipdb.set_trace()
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



if __name__ == '__main__':
    import hydra
    @hydra.main(config_name='train', config_path='../config', version_base='1.2')
    def main(cfg):
        datamodel = FFCVDataModel(**cfg.data.init_args)
        dataloader = datamodel.train_dataloader()
        for i, batch in enumerate(dataloader):
            if batch[2].min() ==0:
                print(batch[2].min())
                import ipdb; ipdb.set_trace()
        task = cfg.task
        print(task)
        if task in globals():
            globals()[task](cfg.data.init_args.train_fp)
        # get_mean_std(cfg.data.init_args.train_fp)

    main()
    
