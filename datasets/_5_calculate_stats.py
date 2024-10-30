from typing import List
from pathlib import Path
from glob import glob
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm
import torch
import hydra
import numpy as np
from ffcv.loader import Loader, OrderOption


def calculate_s2_mean_std(beton_fp:str):
    """
    Calculate mean and std for the beton file.
    """
    batch_size = 100 if 'debug' in beton_fp else 4096
    beton_fp = Path(beton_fp).expanduser()
    beton_fp = glob(str(beton_fp))
    n = 0
    avg = torch.zeros(12)
    avg_x_square = torch.zeros(12)
    for fp in beton_fp:
        loader = Loader(fp, batch_size=batch_size, num_workers=4,
                    distributed=False, batches_ahead=3,
                    order=OrderOption.SEQUENTIAL, os_cache=False)

        for batch in tqdm(loader):
            img = batch[0]
            img = img.double()/1e4
            n_old = n
            n += img.shape[0]
            avg = (avg * n_old + img.sum(axis=(0, 2, 3)))/ n
            avg_x_square = (avg_x_square * n_old + (img**2).sum(axis=(0, 2, 3))) / n          
            import ipdb; ipdb.set_trace()  
            
    avg *= n
    variance = avg_x_square * n / (n*225 -1) - (avg**2/(n*225)/(n*225-1))
    print('n: ', n)
    print('mean: ', avg)
    print('std: ', np.sqrt(variance))
    np.savetxt('output/s2_mean_filtered.txt', avg)
    np.savetxt('output/s2_std_filtered.txt', np.sqrt(variance))

@dataclass
class Config:
    beton_fp: str = '~/data/GEDI/train_subsets/train*_filtered.beton'

cs = ConfigStore.instance()
cs.store(name='config', node=Config)

@hydra.main(config_name='config', version_base='1.2')
def main(cfg: DictConfig):
    calculate_s2_mean_std(cfg.beton_fp)

if __name__ == '__main__':
    main()