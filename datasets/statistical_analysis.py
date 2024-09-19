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
# Don't change the order, index is the corresponding BIOME number
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


def _get_slope(filepath):
    '''
    Get slope from the file name
    '''
    import re
    match = re.search(r'slope(\d+)', filepath)
    if match:
        return int(match.group(1))
    else:
        return 90

class PCAAnalysis:
    def __init__(self, model_path, data_fps=None,output_dir:str='output', **kwargs) -> None:
        self.output_dir = Path(output_dir).expanduser()
        data_fps = Path(data_fps).expanduser()
        self.data_fps = glob.glob(str(data_fps))
        self.slope = _get_slope(model_path) # if no slope info provided in the model's file name, use 90 as default, meaning no slope filtering
        model_path = Path(model_path).expanduser()
        if model_path.exists():
            print('Loading pca model from ', model_path)
            self.pca = joblib.load(model_path)
        else:
            self.pca = self.fit_pca(model_path)

    def fit_pca(self, save_path):
        
        ipca = IncrementalPCA(n_components=101)
        for fp in self.data_fps:
            print('loading data from ', fp)
            rhs, _ = prepare_data(fp, self.slope)
            print(rhs.shape)
            print('Partially fit PCA')
            ipca.partial_fit(rhs)

        print('Saving pca model')
        joblib.dump(ipca, save_path)
        return ipca
    
    def plot_explained_variance(self):
        '''
        Plot the explained variance
        '''
        support_num = 101
        support = np.arange(support_num)
        print('Plotting explained variance')
        fig = plt.figure(figsize=(20, 6), tight_layout=True)
        plt.plot(self.pca.explained_variance_ratio_, '-o')
        plt.xticks(ticks=np.arange(0, support_num-1, 10), labels=np.arange(0, support_num-1, 10))
        plt.ylabel('explained variance (logscale)')
        plt.xlabel('component')
        plt.yscale('log')
        plt.grid()
        plt.savefig(self.output_dir / f'pca_explained_variance_slope{self.slope}.png')

    def plot_components(self, n_components:int=10):
        '''
        Plot the mean and first n components
        Parameters
        ----------
        * n_components: number of components to plot
        '''
        support_num = 101
        support = np.arange(support_num)

        # plot mean and components
        print('Plotting mean and components')
        fig, axs = plt.subplots(n_components + 1, figsize=(6, 16), tight_layout=True)
        axs[0].grid()
        axs[0].plot(support, self.pca.mean_)
        axs[0].set_ylabel('mean')

        for i in range(n_components):
            axs[i+1].grid()
            axs[i+1].set_ylabel('PC %i' % (i+1))
            axs[i+1].plot(support, self.pca.components_[i])
        plt.savefig(self.output_dir / f'pca_rhs_slope{self.slope_th}.png')

    def run_sanity_check(self, rh_examples, idx: List[int], data_name: str = None):
        '''
        Use first few components to reconstruct the data and compare with the original data
        @param rh_examples: rhs examples to check (4984928 # North Saharan steppe and woodlands,5817 # rainforest)
        @param idx: index of components to use
        @param data_name: construct the file path to save the figure
        '''
        print('run sanity check')
        n_features = rh_examples.shape[1]
        rhs_projected = self.pca.transform(rh_examples)  # (n,101)
        rhs_inversed = rhs_projected[:, idx] @ self.pca.components_[idx] + self.pca.mean_.reshape(1, n_features)

        fig, axs = plt.subplots(len(idx), figsize=(6, 6), tight_layout=True)
        for i in range(len(idx)):
            axs[i].plot(rhs_inversed[i], 'o', markersize=5,
                        label=f'Inversed RHs explained by PC {str(idx)}', color='orange')
            axs[i].plot(rh_examples[i], label='Original RHs', color='blue')
            axs[i].plot(self.pca.mean_, label='Mean RHs', color='green')
            axs[i].legend()
        plt.savefig(self.output_dir / f'pca_sanity_check_{data_name}.png')
        # f'output/pca_sanity_check_{data_name}_slope{slope_th}_PC{"-".join(idx)}.png'

    def plot_projected_data(self, rhs: 'np.ndarray' = None, latlon: List = None, pc_idxs: List[int] = None, biome_list: List[int] = None,
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

        rhs_projected = self.pca.transform(rhs)
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


    def run_pearson_corr(self, rhs: 'np.ndarray', data_name: str = None):
        '''
        Run Pearson correlation between RH100/RH98 with the first component to check if PC1 is highly correlated with the top height
        '''
        print('Run Pearson correlation')
        from scipy import stats
        rhs_projected = self.pca.transform(rhs)  # (n,101)
        projected_pc1 = rhs_projected[:, 0]  # (n,1)
        projected_pc1 = np.concatenate([projected_pc1] * 10)
        rhs = np.concatenate([rhs] * 10, axis=0)
        print(projected_pc1.shape)
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
        plt.savefig(self.output_dir / f'pearson_corr_PC1_RHs_{data_name}.png')


    def run_biome_effect_size_analysis(self, pc_idxs: List[int], biome_list: List[int]):
        print('Run biome effect size analysis')
        import pandas as pd
        import geopandas as gpd
        ecoregions = gpd.read_file('~/data/GEDI/ecoregions/wwf_terr_ecos.shp')
        ecoregions = ecoregions[ecoregions['BIOME']<15]
        pc_cols = [f'PC{i+1}' for i in pc_idxs]
        rh_cols = [f"RH{i}" for i in range(101)]
        cols = pc_cols + rh_cols
        mean_df = pd.DataFrame(columns=cols, index=range(1, len(BIOMES)+1))
        mean_df = mean_df.replace(np.nan, 0)
        var_df = pd.DataFrame(columns=cols, index=range(1, len(BIOMES)+1))
        var_df = var_df.replace(np.nan, 0)
        mean_rh_x_pc1 = np.zeros(101)

        biome_n = pd.Series(index=range(1, len(BIOMES)+1), data=0)
        for fp in self.data_fps:
            data_name = Path(fp).stem
            rhs, latlon = prepare_data(fp, self.slope)
            rhs_projected = self.pca.transform(rhs)
            rhs_projected = rhs_projected[:, pc_idxs]
            latlon = np.concatenate(latlon)
            data = np.concatenate([rhs_projected, rhs, latlon], axis=1)
            df = pd.DataFrame(data, columns=cols + ['lat', 'lon'])
            df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
            df = df.set_crs(epsg=4326)
            df = df.to_crs(epsg=3857)
            rhs_with_biomes = gpd.sjoin(df, ecoregions, how='left', predicate='within')
            rhs_with_biomes = rhs_with_biomes[cols + ['BIOME']]
            rhs_with_biomes = rhs_with_biomes.dropna(subset=['BIOME'])
            biome_n = biome_n.add(rhs_with_biomes['BIOME'].value_counts(), fill_value=0)
            pc_mean = rhs_with_biomes.groupby('BIOME').sum()
            mean_df = mean_df.add(pc_mean, fill_value=0)
            x_square = rhs_with_biomes.groupby('BIOME').apply(lambda x: x[cols]**2)
            x_square = x_square.groupby('BIOME').sum()
            var_df = var_df.add(x_square, fill_value=0)

            # For Pearson correlation between PC1 and RH0-100
            rh_x_pc1 = rhs_with_biomes[rh_cols].mul(rhs_with_biomes['PC1'], axis=0)
            mean_rh_x_pc1 += rh_x_pc1.sum(axis=0)
            
        
        N = biome_n.sum()
        cohend_matrix_dict = {}
        data_name = 'train' if len(self.data_fps) > 1 else data_name
        for col in pc_cols + ['RH98', 'RH100']:
            mean_per_biome = mean_df[col] / biome_n
            var_all_biomes = var_df[col].sum() / N
            std_all_biomes = np.sqrt(var_all_biomes - (mean_df[col].sum()/N)**2)
            cohend_matrix = np.abs(mean_per_biome.values.reshape(-1, 1) - mean_per_biome.values) / std_all_biomes
            cohend_matrix_dict[col] = cohend_matrix
            plt.figure(figsize=(10, 10))
            plt.imshow(cohend_matrix, cmap='Blues', vmin=0, vmax=2)
            plt.xticks(np.arange(len(BIOMES)), labels=BIOMES, rotation=90)
            plt.yticks(np.arange(len(BIOMES)), labels=BIOMES)
            plt.colorbar()
            plt.title(f'Biome effect size matrix for {col}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'biome_effect_size_matrix_{data_name}_{col}.png')


        # For Pearson correlation between PC1 and RH0-100
        mean_pc1 = mean_df['PC1'].sum()/N
        Nvar_pc1 = var_df['PC1'].sum()- mean_pc1**2 * N
        mean_rh = mean_df[rh_cols].sum()/N
        Nvar_rh = var_df[rh_cols].sum() - mean_rh**2 * N
        r_square = (mean_rh_x_pc1 - mean_pc1 * mean_rh * N ) / np.sqrt(Nvar_rh * Nvar_pc1)

        plt.figure()
        plt.plot(r_square, '-o', markersize=4)
        plt.legend()
        plt.grid()
        plt.xlabel('Relative Height')
        plt.ylabel('R square between PC1 and RH')
        plt.savefig(self.output_dir / f'pearson_corr_PC1_RHs_{data_name}.png')

        # relative Cohen's d betweem RH98 and PC1-3
        for col in pc_cols:
            cohend_matrix = np.maximum(cohend_matrix_dict[col] - cohend_matrix_dict['RH98'], 0)
            plt.figure(figsize=(10, 10))
            plt.imshow(cohend_matrix, cmap='Blues', vmin=0, vmax=1)
            plt.xticks(np.arange(len(BIOMES)), labels=BIOMES, rotation=90)
            plt.yticks(np.arange(len(BIOMES)), labels=BIOMES)
            plt.colorbar()
            plt.title(f'Biome effect size matrix for max(0, |d({col}| - |d(RH98)|)')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'biome_effect_size_matrix_{data_name}_{col}-RH98.png')
    
        # relative Cohen's d betweem PC1 and PC2-3
        for col in pc_cols[1:]:
            cohend_matrix = np.maximum(cohend_matrix_dict[col] - cohend_matrix_dict['PC1'], 0)
            plt.figure(figsize=(10, 10))
            plt.imshow(cohend_matrix, cmap='Blues', vmin=0, vmax=1)
            plt.xticks(np.arange(len(BIOMES)), labels=BIOMES, rotation=90)
            plt.yticks(np.arange(len(BIOMES)), labels=BIOMES)
            plt.colorbar()
            plt.title(f'Biome effect size matrix for max(0, |d({col}| - d(PC1))')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'biome_effect_size_matrix_{data_name}_{col}-PC1.png')


@dataclass
class MyConfig:
    model_path: str = 'output/pca_model.pkl'
    data_fps: str = '~/data/GEDI/train_subsets/train1.beton'
    output_dir: str = 'output'
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
    model = PCAAnalysis(**cfg)
    model.run_biome_effect_size_analysis([0, 1, 2], [1, 13])


    print(f'time taken for running {cfg.task}: {time.time() - t0}')


if __name__ == '__main__':
    main()
