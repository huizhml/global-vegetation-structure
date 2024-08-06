import time
import logging
from pathlib import Path
import lightning as L
from ffcv.loader import Loader, OrderOption

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
        if isinstance(self.train_fp, list):
            idx = self.trainer.current_epoch // self.trainer.reload_dataloaders_every_n_epochs # train each subset 5 epochs and then switch
            train_fp = self.train_fp[idx] if isinstance(self.train_fp, list) else self.train_fp
        else:
            train_fp = self.train_fp
        print('loading from ', train_fp)

        t0 = time.time()
        loader = Loader(train_fp, batch_size=self.batch_size, num_workers=self.num_workers,
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
    

if __name__ == '__main__':
    import hydra
    @hydra.main(config_name='train', config_path='../config', version_base='1.2')
    def main(cfg):
        datamodel = FFCVDataModel(**cfg.data.init_args)
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
    