import time
import logging
from pathlib import Path
import lightning as L
from ffcv.loader import Loader, OrderOption
from tqdm import tqdm

from datasets.stats import AverageMeter

logger = logging.getLogger(__name__)


class FFCVDataModel(L.LightningDataModule):
    
    def __init__(self,
        train_fp: str=None,
        val_fp: str=None,
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
        if not self.train_fp.exists() or not self.val_fp.exists():
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

    # def test_dataloader(self):
    #     return Loader(self.test_fp, batch_size=self.batch_size, num_workers=self.num_workers,
    #             distributed=self.distributed, batches_ahead=self.batches_ahead,
    #             order=OrderOption.SEQUENTIAL, os_cache=self.os_cache)
    
class RunningStats:
    def __init__(self):
        self.n = 0
        self.sum = 0.0

    def update(self, x):
        self.n += x.shape[0]
        self.sum += x.sum(axis=(0, 2, 3))

    def mean(self):
        return self.sum / self.n / 225
    
    def std(self):
        return (self.sum  / (self.n*225 - 1)).sqrt()

    def count(self):
        return self.n
    
    def clear(self):
        self.n = 0
        self.sum = 0.0

    def __repr__(self):
        return f'mean: {self.mean()}, std: {self.std()}, count: {self.count()}'

def get_mean_std(dataloader):
    stats = RunningStats()
    for i, batch in tqdm(enumerate(dataloader)):
        if (batch[0]<0).any():
            print('negative values found')
            print(i)
            print(batch[0].min())
            print(batch[-1])
            import ipdb; ipdb.set_trace()
        stats.update(batch[0]/1e4)
    mean = stats.mean() 
    print('mean: ', mean * 1e4)
    stats.clear()

    for batch in tqdm(dataloader):
        mse = (batch[0]/1e4 - mean[None, :, None, None]) ** 2
        stats.update(mse)
    print('std: ', stats.std()*1e4)

if __name__ == '__main__':
    import hydra
    @hydra.main(config_name='train', config_path='../config', version_base='1.2')
    def main(cfg):
        datamodel = FFCVDataModel(**cfg.data.init_args)
        # import ipdb; ipdb.set_trace()
        dataloader = datamodel.train_dataloader()
        get_mean_std(dataloader)

    main()
    