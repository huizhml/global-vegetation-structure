from typing import List
from pathlib import Path
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import hydra
import numpy as np
import torch
from tqdm import tqdm
import glob
import pandas as pd
import geopandas as gpd
from ffcv.loader import Loader, OrderOption
from download._const import gedi_attr_dtype
from const import BIOMES


def filter_by_sensitivity(beton_fps:List[str]=None, slope_th:float=None, sens_th:float=None):
    """
    Filter the data by sensitivity, and save RHs + slope + wc + latlon + attrs as parquets
    """
    beton_fps = glob.glob(str(Path(beton_fps).expanduser()))
    print(beton_fps)
    cols =  [f'rh{i}' for i in range(101)] + ['wc', 'slope', 'lat', 'lon'] + list(gedi_attr_dtype.keys())
    data_dir = Path(beton_fps[0]).parent.parent
    ecoregions = gpd.read_file(data_dir / 'ecoregions/wwf_terr_ecos.shp')
    batch_size = 100 if 'debug' in beton_fps[0] else 4096
    sens_th = sens_th/100 if sens_th is not None else None
    for data_fp in beton_fps:
        file = Path(data_fp).with_suffix('.parquet')
        if file.exists():
            print('file exists', file)
            continue
        loader = Loader(data_fp, batch_size=batch_size, num_workers=4,
                        distributed=False, batches_ahead=3,
                        order=OrderOption.SEQUENTIAL, os_cache=False)

        data = []
        if slope_th:
            for batch in tqdm(loader):
                zero_idx, _ = torch.where(torch.isin(batch[2], torch.tensor([50, 70, 80])))
                excl = torch.where(torch.isin(batch[2], torch.tensor([30, 60, 100])), 1, 0)  # exluded because steep terrain
                steep = torch.where(batch[3] > slope_th, 1, 0)
                exclude = excl * steep
                exclude_idx, _ = torch.where(exclude)
                exclude_idx = torch.cat([zero_idx, exclude_idx])
                fs = [np.delete(f, exclude_idx, axis=0) for f in batch[1:]]
                data.append(np.concatenate(fs, axis=1))

        elif sens_th is not None:
            for batch in tqdm(loader):
                sens = batch[-1][:, 24:25]
                exclude_idx, _ = torch.where(sens < sens_th)
                fs = [np.delete(f, exclude_idx, axis=0) for f in batch[1:]]
                data.append(np.concatenate(fs, axis=1))

        data = np.concatenate(data, axis=0)
        df = pd.DataFrame(data, columns=cols)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        df = df.set_crs(epsg=4326)
        df = df.to_crs(epsg=3857)
        rhs_with_biomes = gpd.sjoin(df, ecoregions, how='left', predicate='within')
        rhs_with_biomes = rhs_with_biomes[cols + ['BIOME']]
        rhs_with_biomes.to_parquet(Path(data_fp).with_suffix('.parquet'))
        print('saved to', Path(data_fp).with_suffix('.parquet'))
    group_df_by_biome(beton_fps)

def group_df_by_biome(df_fps:List[str]=None):
    """
    Group the dataframe by biome, and save parquets for each biome
    """
    import dask.dataframe as dd
    ddf = dd.read_parquet(df_fps)
    data_dir = Path(df_fps[0]).parent
    df = ddf.compute()
    print(df.BIOME.unique())
    def save_parquet(x):
        print(x)
        if x.name is not None and int(x.name) < 15:
            name = f'rhs_attrs_{BIOMES[int(x.name)-1].replace(" ", "_").replace("&", 'and')}.parquet'
            x.to_parquet(data_dir / name)
        else:
            x.to_parquet(data_dir / f'rhs_attrs_biome_{x.name}.parquet')
    df.groupby('BIOME').apply(save_parquet)


    # ddf.groupby('BIOME').apply(save_parquet, meta=(object, None)).compute()


@dataclass
class MyConfig:
    data_fps: str = '~/data/GEDI/train_subsets/debug0_attrs.beton'
    sens_th: float = 95

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(cfg)
    import time
    t0 = time.time()
    filter_by_sensitivity(data_fps=cfg.data_fps, sens_th=cfg.sens_th)
    data_fps = glob.glob(str(Path(cfg.data_fps).with_suffix('.parquet').expanduser()))
    group_df_by_biome(df_fps=data_fps)
    print(f"Time taken: {time.time()-t0}")

if __name__ == '__main__':
    main()