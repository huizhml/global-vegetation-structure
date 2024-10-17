from typing import List
from pathlib import Path
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import hydra
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
import time
import glob
import joblib
from const import ESA_WC, BIOMES


class PCAAnalysis:
    def __init__(self, data_fp=None, output_dir: str = 'output', dtype='np.float64', **kwargs) -> None:
        self.dtype = eval(dtype)
        self.output_dir = Path(output_dir).expanduser()
        data_fp = Path(data_fp).expanduser()
        model_path = Path(f'output/pca_model_{data_fp.stem}.pkl')
        self.biome_name = data_fp.stem
        self.df = pd.read_parquet(data_fp)
        if model_path.exists():
            print('Loading pca model from ', model_path)
            self.pca = joblib.load(model_path)
        else:
            self.pca = self.fit_pca(model_path)

    def fit_pca(self, save_path, verify: bool = False):
        '''
        Fit PCA model iteratively.
        Incremental PCA provides estimates of the eigenvectors and eigenvalues of a data matrix, not exact values.
        '''
        print('Fitting PCA')
        # pca = PCA(n_components=101, svd_solver='full')
        cols = [f'rh{i}' for i in range(101)]
        rhs = self.df[cols].values.astype(self.dtype)
        cov_matrix = np.cov(rhs, rowvar=False, ddof=1)
        egnvalues, egnvectors = np.linalg.eigh(cov_matrix)
        egnvalues = egnvalues[::-1]
        egnvectors = egnvectors.T[::-1]
        pca = PCA(n_components=101, svd_solver='full')
        pca.explained_variance_ = egnvalues
        pca.components_ = egnvectors
        pca.mean_ = np.mean(rhs, axis=0)
        pca.explained_variance_ratio_ = egnvalues / egnvalues.sum()
        # pca.fit(rhs)

        joblib.dump(pca, save_path)
        print('PCA model saved to ', save_path)
        return pca

    def plot_explained_variance(self):
        '''
        Plot the explained variance
        '''
        support_num = 101
        support = np.arange(support_num)
        print('Plotting explained variance')
        fig = plt.figure(figsize=(20, 6), tight_layout=True)
        plt.plot(self.pca.explained_variance_, '-o', markersize=3)
        plt.xticks(ticks=np.arange(0, support_num-1, 10), labels=np.arange(0, support_num-1, 10))
        plt.ylabel('explained variance (logscale)')
        plt.xlabel('component')
        plt.yscale('log')
        plt.grid()
        plt.savefig(self.output_dir / f'pca_explained_variance_biome{self.biome_name}.png')

        fig = plt.figure(figsize=(20, 6), tight_layout=True)
        plt.plot(self.pca.explained_variance_ratio_.cumsum(), '-o', markersize=3)
        plt.xticks(ticks=np.arange(0, support_num-1, 10), labels=np.arange(0, support_num-1, 10))
        plt.ylabel('cumulative explained variance ratio')
        plt.xlabel('component')
        plt.grid()
        plt.savefig(self.output_dir / f'pca_cum_explained_variance_biome{self.biome_name}.png')

    def plot_components(self, n_components: int = 10):
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
        plt.savefig(self.output_dir / f'pca_components_biome{self.biome_name}.png')

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

    def plot_projected_data(
            self, rhs: 'np.ndarray' = None, latlon: List = None, pc_idxs: List[int] = None, biome_list: List[int] = None,
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
        data_dir = Path(self.data_fps[0]).parent.parent
        ecoregions = gpd.read_file(data_dir / 'ecoregions/wwf_terr_ecos.shp')

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
        np.savetxt(self.output_dir / f'pearson_corr_PC1_RHs_{data_name}_sens{self.sens}.csv', corr, delimiter=',')
        plt.figure()
        plt.plot(corr, '-o', markersize=4)
        plt.xticks(ticks=np.arange(0, 101, 10), labels=np.arange(0, 101, 10))
        plt.legend()
        plt.grid()
        plt.xlabel('Relative Height')
        plt.ylabel('Pearson correlation between PC1 and RH')
        plt.savefig(self.output_dir / f'pearson_corr_PC1_RHs_{data_name}_sens{self.sens}.png')

    def plot_cohend_matrix(self, cohend_matrix, data_name: str = None, sens: str = None, group_method: str = 'Biome'):
        cohend_matrix_dict = {}
        cohend_matrix_dict['max_idx_all'] = cohend_matrix.argmax(axis=0)
        cohend_matrix_dict['max_value_all'] = cohend_matrix.max(axis=0)
        cohend_matrix_dict['max_idx_pcs'] = cohend_matrix[:101].argmax(axis=0)
        cohend_matrix_dict['max_value_pcs'] = cohend_matrix[:101].max(axis=0)
        cohend_matrix_dict['max_idx_rhs'] = cohend_matrix[101:].argmax(axis=0)
        cohend_matrix_dict['max_value_rhs'] = cohend_matrix[101:].max(axis=0)
        print('Unique max(d(PC)): ', np.unique(cohend_matrix_dict['max_idx_pcs']))
        print('Unique max(d(RH)): ', np.unique(cohend_matrix_dict['max_idx_rhs']))

        labels = BIOMES if group_method == 'Biome' else ESA_WC.keys()
        for name in ['all', 'pcs', 'rhs']:
            plt.figure(figsize=(12, 12))
            value = cohend_matrix_dict[f'max_value_{name}']
            annot = cohend_matrix_dict[f'max_idx_{name}']
            cmap = 'bwr' if 'all' in name else 'Greens'
            vmin = 0
            vmax = 2
            if 'all' in name:
                vmin = value.min()
                vmax = value.max()
                annot[annot >= 101] = annot[annot >= 101] - 101
            elif 'pcs' in name:
                annot = annot + 1
            sns.heatmap(value, cmap=cmap, annot=annot, annot_kws={'fontsize': 10}, fmt='d', vmin=vmin, vmax=vmax,
                        xticklabels=labels, yticklabels=labels)
            title = f'{group_method} effect size: most evident values of {name.upper()} across all {group_method.lower()}s'
            plt.title(title)
            plt.tight_layout()
            plt.show()
            plt.savefig(self.output_dir / f'{group_method.lower}_effect_size_matrix_{data_name}{sens}_max_{name}.png')

        # Biome effect size matrix for PC1, PC2, PC3, RH98, RH100
        names = ['PC1', 'PC2', 'PC3', 'RH98', 'RH100']
        for i, idx in enumerate([0, 1, 2, -3, -1]):  # PC1, PC2, PC3, RH98, RH100
            plt.figure(figsize=(10, 10))
            plt.imshow(cohend_matrix[idx], cmap='Blues', vmin=0, vmax=2)
            plt.xticks(np.arange(len(labels)), labels=labels, rotation=90)
            plt.yticks(np.arange(len(labels)), labels=labels)
            plt.colorbar()
            plt.title(f'{group_method} effect size matrix for {names[i]}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{group_method.lower()}_effect_size_matrix_{data_name}{sens}_{names[i]}.png')

        # relative Cohen's d betweem RH98 and PC1-3
        for col in range(3):
            d_mat = np.maximum(cohend_matrix[col] - cohend_matrix[-3], 0)
            plt.figure(figsize=(10, 10))
            plt.imshow(d_mat, cmap='Blues', vmin=0, vmax=1)
            plt.xticks(np.arange(len(labels)), labels=labels, rotation=90)
            plt.yticks(np.arange(len(labels)), labels=labels)
            plt.colorbar()
            plt.title(f'{group_method} effect size matrix for max(0, |d(PC{col+1}| - |d(RH98)|)')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{group_method.lower()}_effect_size_matrix_{data_name}{sens}_PC{col+1}-RH98.png')

        # relative Cohen's d betweem PC1 and PC2-3
        for col in range(1, 3):
            d_mat = np.maximum(cohend_matrix[col] - cohend_matrix[0], 0)
            plt.figure(figsize=(10, 10))
            plt.imshow(d_mat, cmap='Blues', vmin=0, vmax=1)
            plt.xticks(np.arange(len(labels)), labels=labels, rotation=90)
            plt.yticks(np.arange(len(labels)), labels=labels)
            plt.colorbar()
            plt.title(f'{group_method} effect size matrix for max(0, |d(PC{col+1}| - d(PC1))')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{group_method.lower()}_effect_size_matrix_{data_name}{sens}_PC{col+1}-PC1.png')


@dataclass
class MyConfig:
    data_fp: str = '~/data/GEDI/train_subsets/train1_attrs.beton'
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
    # print('Find the correlation between sensitivity and beam coverage')
    # find_sensitivity_beam_cor(glob.glob(cfg.data_fps))
    # find_sensitivity_biome_cor(glob.glob(cfg.data_fps))
    print(task)
    model = PCAAnalysis(**cfg)
    model.plot_explained_variance()
    model.plot_components()
    # model.run_land_cover_effect_size_analysis()

    print(f'time taken for running {cfg.task}: {time.time() - t0}')


if __name__ == '__main__':
    main()
