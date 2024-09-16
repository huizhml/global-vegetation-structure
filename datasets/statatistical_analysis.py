from typing import List
from pathlib import Path
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import hydra
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import time
import glob
import joblib
from ffcv.loader import Loader, OrderOption


BIOMES = [
# Don't change the order, index is the BIOME number
'Tropical & Subtropical Moist Broadleaf Forests',
'Tropical & Subtropical Dry Broadleaf Forests',
'Tropical & Subtropical Coniferous Forests',
'Temperate Broadleaf & Mixed Forests',
'Temperate Conifer Forests',
'Boreal Forests/Taiga',
'Tropical & Subtropical Grasslands, Savannas & Shrublands',
'Temperate Grasslands, Savannas & Shrublands',
'Flooded Grasslands & Savannas',
'Montane Grasslands & Shrublands',
'Tundra',
'Mediterranean Forests, Woodlands & Scrub',
'Deserts & Xeric Shrublands',
'Mangroves'
]

def prepare_data(fp, slope_th: int = None):
    data_name = Path(fp).stem
    batch_size = 4096 if 'train' in data_name else 100
    dataloader = Loader(fp, batch_size=batch_size, num_workers=4,
                        distributed=False, batches_ahead=3,
                        order=OrderOption.SEQUENTIAL, os_cache=False)
    rhs = []
    latlon = []
    if slope_th:
        for batch in tqdm(dataloader):
            zero_idx, _ = torch.where(torch.isin(batch[2], torch.tensor([50, 70, 80])))
            excl = torch.where(torch.isin(batch[2], torch.tensor([30, 60, 100])), 1, 0)  # exluded because steep terrain
            steep = torch.where(batch[3] > slope_th, 1, 0)
            exclude = excl * steep
            exclude_idx, _ = torch.where(exclude)
            exclude_idx = torch.cat([zero_idx, exclude_idx])
            rhs.append(np.delete(batch[1], exclude_idx, axis=0))
            latlon.append(np.delete(batch[-1], exclude_idx, axis=0))
    else:
        for batch in tqdm(dataloader):
            rhs.append(batch[1])
            latlon.append(batch[-1])

    return np.concatenate(rhs), latlon


def pca_sanity_check(pca, rh_examples, idx: List[int], data_name: str = None):
    '''
    Use first few components to reconstruct the data and compare with the original data
    @param pca: PCA model
    @param rh_examples: rhs examples to check (4984928 # North Saharan steppe and woodlands,5817 # rainforest)
    @param idx: index of components to use
    @param data_name: construct the file path to save the figure
    '''
    print('run sanity check')
    n_features = rh_examples.shape[1]
    rhs_projected = pca.transform(rh_examples)  # (n,101)
    rhs_inversed = rhs_projected[:, idx] @ pca.components_[idx] + pca.mean_.reshape(1, n_features)

    fig, axs = plt.subplots(len(idx), figsize=(6, 6), tight_layout=True)
    for i in range(len(idx)):
        axs[i].plot(rhs_inversed[i], 'o', markersize=5,
                    label=f'Inversed RHs explained by PC {str(idx)}', color='orange')
        axs[i].plot(rh_examples[i], label='Original RHs', color='blue')
        axs[i].plot(pca.mean_, label='Mean RHs', color='green')
        axs[i].legend()
    plt.savefig(f'output/pca_sanity_check_{data_name}.png')
    # f'output/pca_sanity_check_{data_name}_slope{slope_th}_PC{"-".join(idx)}.png'


def plot_projected_data(
        pca: PCA = None, rhs: 'np.ndarray' = None, latlon: List = None, pc_idxs: List[int] = None, biome_list: List[int] = None,
        data_name: str = None):
    '''
    Plot projected data on main components
    @param pca: PCA model
    @param rhs: Relative height data
    @param latlon: latlon data
    @param pc_idxs: index of components to use
    @param biome_list: list of biomes, we expect the RH curves are different from different biomes
    @param data_name: name of the data
    '''
    print('Plot projected data')
    import pandas as pd
    import geopandas as gpd
    ecoregions = gpd.read_file('~/data/GEDI/ecoregions/wwf_terr_ecos.shp')

    rhs_projected = pca.transform(rhs)
    rhs_projected = rhs_projected[:, pc_idxs]
    latlon = np.concatenate(latlon)
    data = np.concatenate([rhs_projected, latlon], axis=1)
    df = pd.DataFrame(data, columns=[f'PC{i+1}' for i in pc_idxs] + ['lat', 'lon'])
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    df = df.set_crs(epsg=4326)
    df = df.to_crs(epsg=3857)
    rhs_with_biomes = gpd.sjoin(df, ecoregions, how='left', predicate='within')
    rhs_with_2_biomes = rhs_with_biomes[rhs_with_biomes['BIOME'].isin(biome_list)]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.scatter(rhs_with_2_biomes['PC1'], rhs_with_2_biomes['PC2'], c=rhs_with_2_biomes['BIOME'])
    plt.colorbar()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(f'output/pca_projected_{data_name}.png')


def run_pearson_corr(pca: PCA, rhs: 'np.ndarray', data_name: str = None):
    '''
    Run Pearson correlation between RH100/RH98 with the first component to check if PC1 is highly correlated with the top height
    '''
    print('Run Pearson correlation')
    from scipy import stats
    rhs_projected = pca.transform(rhs)  # (n,101)
    projected_pc1 = rhs_projected[:, 0]  # (n,1)
    corr = []
    for i in range(101):
        c = stats.pearsonr(projected_pc1, rhs[:, i])
        corr.append(c)
        print(f'corr between PC1 and RH{i}: {corr[i]}')
    plt.figure()
    plt.plot(corr, '-o', markersize=4)
    plt.legend()
    plt.grid()
    plt.xlabel('Relative Height')
    plt.ylabel('Pearson correlation between PC1 and RH')
    plt.savefig(f'output/pearson_corr_PC1_RHs_{data_name}.png')


def run_biome_effect_size_analysis(
        pca: PCA, rhs: 'np.ndarray', latlon: List, pc_idxs: List[int],
        biome_list: List[int],
        data_name: str = None):
    print('Run biome effect size analysis')
    import pandas as pd
    import geopandas as gpd
    ecoregions = gpd.read_file('~/data/GEDI/ecoregions/wwf_terr_ecos.shp')
    ecoregions = ecoregions[ecoregions['BIOME']<15]

    rhs_projected = pca.transform(rhs)
    rhs_projected = rhs_projected[:, pc_idxs]
    latlon = np.concatenate(latlon)
    data = np.concatenate([rhs_projected, rhs[:, [98, 100]], latlon], axis=1)
    df = pd.DataFrame(data, columns=[f'PC{i+1}' for i in pc_idxs] + ['RH98', 'RH100', 'lat', 'lon'])
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    df = df.set_crs(epsg=4326)
    df = df.to_crs(epsg=3857)
    rhs_with_biomes = gpd.sjoin(df, ecoregions, how='left', predicate='within')
    rhs_with_biomes = rhs_with_biomes[['PC1', 'PC2', 'PC3', 'RH98', 'RH100', 'BIOME']]
    std = np.std(rhs_projected, axis=0)  # (len(pc_idxs),)
    pc_mean = rhs_with_biomes.groupby('BIOME').mean()
    pc_mean = pc_mean.sort_index()
    labels = [BIOMES[int(i-1)] for i in pc_mean.index]
    for i, idx in enumerate(pc_idxs):
        means = pc_mean[f'PC{str(idx+1)}'].values
        cohend_matrix = (means.reshape(-1, 1) - means) / std[i]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cohend_matrix, cmap='bwr', vmin=-2, vmax=2)
        plt.xticks(np.arange(len(labels)), labels=labels, rotation=90)
        plt.yticks(np.arange(len(labels)), labels=labels)
        plt.colorbar()
        plt.title(f'Biome effect size matrix for PC{str(idx+1)}')
        plt.tight_layout()
        plt.savefig(f'output/biome_effect_size_matrix_{data_name}_PC{str(idx+1)}.png')
    
    std = np.std(rhs[:, [98, 100]], axis=0)
    for idx, name in enumerate(['RH98', 'RH100']):
        means = pc_mean[name].values
        cohend_matrix = (means.reshape(-1, 1) - means) / std[idx]
        plt.figure(figsize=(10, 10))
        plt.imshow(cohend_matrix, cmap='bwr', vmin=-2, vmax=2)
        plt.xticks(np.arange(len(labels)), labels=labels, rotation=90)
        plt.yticks(np.arange(len(labels)), labels=labels)
        plt.colorbar()
        plt.title(f'Biome effect size matrix for {name}')
        plt.tight_layout()
        plt.savefig(f'output/biome_effect_size_matrix_{data_name}_{name}.png')


def cohend(pci, pcj, s):
    '''
    Calculate Cohen's d
    @param pci: component k from biome i
    @param pcj: component k from biome j
    @param s: standard deviation
    '''
    u1, u2 = np.mean(pci), np.mean(pcj)
    return (u1 - u2) / s


def run_pca_on_rhs(fps):
    pca_model_fp = Path('output/pca_model.pkl')
    fps = glob.glob(fps)
    slope_th = 5
    support_num = 101
    support = np.arange(support_num)
    if not pca_model_fp.exists():
        ipca = IncrementalPCA(n_components=support_num)
        for fp in fps:
            print('loading data from ', fp)
            rhs, _ = prepare_data(fp, slope_th)
            print(rhs.shape)
            print('Partially fit PCA')
            ipca.partial_fit(rhs)

        print('Saving pca model')
        joblib.dump(ipca, f'output/pca_model_slope{slope_th}.pkl')
    else:
        ipca = joblib.load(pca_model_fp)

    # explained variance
    print('Plotting explained variance')
    fig = plt.figure(figsize=(20, 6), tight_layout=True)
    plt.plot(ipca.explained_variance_ratio_, '-o')
    plt.xticks(ticks=np.arange(0, support_num-1, 10), labels=np.arange(0, support_num-1, 10))
    plt.ylabel('explained variance (logscale)')
    plt.xlabel('component')
    plt.yscale('log')
    plt.grid()
    plt.savefig(f'output/pca_rhs_slope{slope_th}.png')

    # plot mean and components
    print('Plotting mean and components')
    n_components = 10
    fig, axs = plt.subplots(n_components + 1, figsize=(6, 16), tight_layout=True)
    axs[0].grid()
    axs[0].plot(support, ipca.mean_)
    axs[0].set_ylabel('mean')

    for i in range(n_components):
        axs[i+1].grid()
        axs[i+1].set_ylabel('PC %i' % (i+1))
        axs[i+1].plot(support, ipca.components_[i])
    plt.savefig(f'output/pca_components_{slope_th}.png')

    rhs, latlon = prepare_data(fps[0], slope_th)
    data_name = Path(fps[0]).stem
    # pca_sanity_check(ipca, rhs[:2], [0, 1, 2], f'{data_name}_slope{slope_th}_PC0-1-2.png')
    # plot_projected_data(ipca, rhs, latlon, [1, 13], f'{data_name}_slope{slope_th}.png')
    # run_pearson_corr(ipca, rhs, data_name)
    run_biome_effect_size_analysis(ipca, rhs, latlon, [0, 1, 2], [1, 13], data_name)


@dataclass
class MyConfig:
    fp: str = '~/data/GEDI/train_subsets/debug1.beton'
    task: str = 'run_pca_on_rhs'


cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(cfg)
    import time
    t0 = time.time()
    task = cfg.task
    print(task)
    if task in globals():
        globals()[task](cfg.fp)

    print(f'time taken for running {cfg.task}: {time.time() - t0}')


if __name__ == '__main__':
    main()
