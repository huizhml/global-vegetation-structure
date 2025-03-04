from typing import List
from pathlib import Path
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import hydra
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import geopandas as gpd
import glob
import joblib
from const import ESA_WC, BIOMES



def _get_param_from_filepath(filepath, param_name):
    '''
    Get parameter from the file name
    '''
    import re
    match = re.search(rf'{param_name}(\d+)', filepath)
    if match:
        return int(match.group(1))
    else:
        return None

class PCAAnalysis:
    '''
    Run PCA analysis on the relative height data using {biome}.parquet file, one PCA model for each biome
    '''

    def __init__(self, parquet_fp=None, model_path:str=None, output_dir: str = 'output/pca_analysis', dtype='np.float64', method:str='single', **kwargs) -> None:
        self.method = method
        self.dtype = eval(dtype)
        self.output_dir = Path(output_dir).expanduser()
        self.slope = _get_param_from_filepath(model_path, 'slope')
        self.parquet_fps = Path(parquet_fp).expanduser()
        if method == 'multi':
            model_path = self.output_dir / f'pca_model_{self.parquet_fps.stem}.pkl'
            self.biome_name = self.parquet_fps.stem
            if not self.parquet_fps.exists():
                print('parquet file not found')
                # raise here, generate the parquet file here will cause race condition if multiple processes of PCA analysis are running
                raise f'{parquet_fp} not found, please run python -m datasets._6_visual_check task=aggregate_gedi_by_biome first'
        else:
            model_path = Path(model_path).expanduser()
            self.parquet_fps = glob.glob(str(self.parquet_fps))
            self.biome_name = ''
        if model_path.exists():
            print('Loading pca model from ', model_path)
            self.pca = joblib.load(model_path)
        else:
            self.pca = self.fit_pca(model_path)
        

    @property
    def mean(self):
        if (self.output_dir / f'RHs_mean_slope{self.slope}.npy').exists():
            mean = np.load(self.output_dir / f'RHs_mean_slope{self.slope}.npy')
        else:
            mean = self.aggregate_mean()
        mean = mean.astype(self.dtype)
        return mean

    def aggregate_mean(self):
        '''
        Aggregate mean of the data
        '''
        avg = np.zeros(101, dtype=self.dtype)
        N = np.zeros(1)
        for fp in self.parquet_fps:
            print('loading data from ', fp)
            rhs = pd.read_parquet(fp)
            rhs = rhs[[f'rh{i}' for i in range(101)]].values.astype(self.dtype)
            m = np.mean(rhs, axis=0, dtype=self.dtype)
            n = rhs.shape[0]
            N_old = N
            N = N + n
            avg = (N_old * avg + n * m) / N
        print('mean of the data')
        print(avg, avg.dtype)
        np.save(self.output_dir / f'RHs_mean_slope{self.slope}.npy', avg)
        return avg

    def fit_pca(self, save_path, verify: bool = False):
        '''
        Fit PCA model iteratively.
        Incremental PCA provides estimates of the eigenvectors and eigenvalues of a data matrix, not exact values.
        '''
        cols = [f'rh{i}' for i in range(101)]
        pca = PCA(n_components=101, svd_solver='full')
        if self.method == 'multi':
            print('Fitting PCA for biome ', self.biome_name)
            df = pd.read_parquet(self.parquet_fps)
            rhs = df[cols].values.astype(self.dtype)
            cov_matrix = np.cov(rhs, rowvar=False, ddof=1)
            pca.mean_ = np.mean(rhs, axis=0)
            # pca.fit(rhs)
        else:
            print('Fitting PCA for all biomes')
            cov_matrix = np.zeros((101, 101), dtype=self.dtype)
            N = 0
            for fp in self.parquet_fps:
                print('loading data from ', fp)
                df = pd.read_parquet(fp)
                rhs = df[cols].values.astype(self.dtype)
                if verify:
                    self.mean = np.mean(rhs, axis=0, dtype=self.dtype)
                rhs_ = rhs - self.mean #self.mean
                print('Calculating covariance matrix')
                cov_matrix += rhs_.T @ rhs_
                N += rhs.shape[0]
            cov_matrix = cov_matrix / (N-1)
            pca.mean_ = self.mean
        egnvalues, egnvectors = np.linalg.eigh(cov_matrix)
        egnvalues = egnvalues[::-1]
        egnvectors = egnvectors.T[::-1]
        # verify the result
        if verify:
            F = rhs_ @ egnvectors.T
            
            print('Fitting PCA')
            pca.fit(rhs)
            isclose = np.allclose(pca.explained_variance_, egnvalues)
            F_ = pca.transform(rhs)
            isclose_F = np.allclose(F, F_)
            isclose_pc = np.allclose(abs(pca.components_), abs(egnvectors)) # the sign of the components can be different
            print('Is components close? ', isclose_pc)
            print('Is explained variance close? ', isclose)
            print('Is F close? ', isclose_F)
    
        pca.explained_variance_ = egnvalues
        pca.components_ = egnvectors
        pca.explained_variance_ratio_ = egnvalues / egnvalues.sum()
        joblib.dump(pca, save_path)
        print('PCA model saved to ', save_path)
        return pca

    def plot_explained_variance(self):
        '''
        Plot the explained variance
        '''
        fig_name = self.biome_name if self.method == 'multi' else 'all_biomes'
        support_num = 101
        print('Plotting explained variance')
        fig = plt.figure(figsize=(20, 6), tight_layout=True)
        plt.plot(self.pca.explained_variance_, '-o', markersize=3)
        plt.xticks(ticks=np.arange(0, support_num-1, 10), labels=np.arange(0, support_num-1, 10))
        plt.ylabel('explained variance (logscale)')
        plt.xlabel('component')
        plt.yscale('log')
        plt.grid()
        plt.savefig(self.output_dir / f'pca_explained_variance_biome{fig_name}.png')

        fig = plt.figure(figsize=(20, 6), tight_layout=True)
        plt.plot(self.pca.explained_variance_ratio_.cumsum(), '-o', markersize=3)
        plt.xticks(ticks=np.arange(0, support_num-1, 10), labels=np.arange(0, support_num-1, 10))
        plt.ylabel('cumulative explained variance ratio')
        plt.xlabel('component')
        plt.grid()
        plt.savefig(self.output_dir / f'pca_cum_explained_variance_biome{fig_name}.png')

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
        data_dir = Path(self.parquet_fp[0]).parent.parent
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
        np.savetxt(self.output_dir / f'pearson_corr_PC1_RHs_{data_name}.csv', corr, delimiter=',')
        plt.figure()
        plt.plot(corr, '-o', markersize=4)
        plt.xticks(ticks=np.arange(0, 101, 10), labels=np.arange(0, 101, 10))
        plt.legend()
        plt.grid()
        plt.xlabel('Relative Height')
        plt.ylabel('Pearson correlation between PC1 and RH')
        plt.savefig(self.output_dir / f'pearson_corr_PC1_RHs_{data_name}.png')

    def plot_cohend_matrix(self, cohend_matrix, data_name: str = None, group_method: str = 'Biome'):
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
            plt.savefig(self.output_dir / f'{group_method.lower}_effect_size_matrix_{data_name}_max_{name}.png')

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
            plt.savefig(self.output_dir / f'{group_method.lower()}_effect_size_matrix_{data_name}_{names[i]}.png')

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
            plt.savefig(self.output_dir / f'{group_method.lower()}_effect_size_matrix_{data_name}_PC{col+1}-RH98.png')

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
            plt.savefig(self.output_dir / f'{group_method.lower()}_effect_size_matrix_{data_name}_PC{col+1}-PC1.png')

    def run_biome_effect_size_analysis(self):
        print('Run biome effect size analysis')
        import pandas as pd
        import geopandas as gpd
        data_dir = Path(self.parquet_fps[0]).parent.parent
        ecoregions = gpd.read_file(data_dir / 'ecoregions/wwf_terr_ecos.shp')
        ecoregions = ecoregions[ecoregions['BIOME'] < 15]
        pc_cols = [f'PC{i+1}' for i in range(101)]
        rh_cols = [f"rh{i}" for i in range(101)]
        cols = pc_cols + rh_cols
        mean_df = pd.DataFrame(columns=cols, index=range(1, len(BIOMES)+1))
        mean_df = mean_df.replace(np.nan, 0)
        var_df = pd.DataFrame(columns=cols, index=range(1, len(BIOMES)+1))
        var_df = var_df.replace(np.nan, 0)
        mean_rh_x_pc1 = np.zeros(101)
        biome_n = pd.Series(index=range(1, len(BIOMES)+1), data=0)
        for fp in self.parquet_fps:
            data_name = Path(fp).stem
            print('loading data from ', fp)
            df = pd.read_parquet(fp)
            rhs = df[rh_cols].values.astype(self.dtype)
            latlon = df[['lat', 'lon']].values
            print(rhs.shape)
            rhs_projected = self.pca.transform(rhs)
            data = np.concatenate([rhs_projected, rhs, latlon], axis=1)
            
            df = pd.DataFrame(data, columns=cols + ['lat', 'lon'])
            df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
            df = df.set_crs(epsg=4326)
            df = df.to_crs(epsg=3857)
            rhs_with_biomes = gpd.sjoin(df, ecoregions, how='left', predicate='within')
            rhs_with_biomes = rhs_with_biomes[cols + ['BIOME']]
            rhs_with_biomes = rhs_with_biomes.dropna(subset=['BIOME'])
            # rhs_with_biomes['BIOME'] = rhs_with_biomes['BIOME'].fillna(65536) # for sanity check
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
        data_name = 'train' if len(self.parquet_fps) > 1 else data_name
        mean_per_biome = mean_df.div(biome_n, axis=0)
        var_all_biomes = var_df.sum() / (N-1)
        std_all_biomes = np.sqrt(var_all_biomes - mean_df.sum()**2/N/(N-1)) # per col mean
        cohend_matrix = np.abs(
            mean_per_biome.values.T[:, :, None] - mean_per_biome.values.T[:, None, :]) / std_all_biomes.values[:, None, None]
        cohend_matrix_dict['max_idx_all'] = cohend_matrix.argmax(axis=0)
        cohend_matrix_dict['max_value_all'] = cohend_matrix.max(axis=0)
        cohend_matrix_dict['max_idx_pcs'] = cohend_matrix[:101].argmax(axis=0)
        cohend_matrix_dict['max_value_pcs'] = cohend_matrix[:101].max(axis=0)
        cohend_matrix_dict['max_idx_rhs'] = cohend_matrix[101:].argmax(axis=0)
        cohend_matrix_dict['max_value_rhs'] = cohend_matrix[101:].max(axis=0)
        print('Unique max(d(PC)): ', np.unique(cohend_matrix_dict['max_idx_pcs']))
        print('Unique max(d(RH)): ', np.unique(cohend_matrix_dict['max_idx_rhs']))
        biome_n.to_csv(self.output_dir / f'biome_n_{data_name}.csv')
        mean_per_biome.to_csv(self.output_dir / f'biome_mean_{data_name}.csv')
        std_all_biomes.to_csv(self.output_dir / f'biome_effect_size_std_{data_name}.csv')
        np.save(self.output_dir / f'biome_effect_size_matrix_{data_name}.npy', cohend_matrix)

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
                        xticklabels=BIOMES, yticklabels=BIOMES)
            title = f'Biome effect size: most evident values of {name.upper()} across all biomes'
            plt.title(title)
            plt.tight_layout()
            plt.show()
            plt.savefig(self.output_dir / f'biome_effect_size_matrix_{data_name}_max_{name}.png')

        # Biome effect size matrix for PC1, PC2, PC3, RH98, RH100
        names = ['PC1', 'PC2', 'PC3', 'RH98', 'RH100']
        for i, idx in enumerate([0, 1, 2, -3, -1]):  # PC1, PC2, PC3, RH98, RH100
            plt.figure(figsize=(10, 10))
            plt.imshow(cohend_matrix[idx], cmap='Blues', vmin=0, vmax=2)
            plt.xticks(np.arange(len(BIOMES)), labels=BIOMES, rotation=90)
            plt.yticks(np.arange(len(BIOMES)), labels=BIOMES)
            plt.colorbar()
            plt.title(f'Biome effect size matrix for {names[i]}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'biome_effect_size_matrix_{data_name}_{names[i]}.png')

        # For Pearson correlation between PC1 and RH0-100
        mean_pc1 = mean_df['PC1'].sum()/N
        Nvar_pc1 = var_df['PC1'].sum() - mean_pc1**2 * N
        mean_rh = mean_df[rh_cols].sum()/N
        Nvar_rh = var_df[rh_cols].sum() - mean_rh**2 * N
        r_square = (mean_rh_x_pc1 - mean_pc1 * mean_rh * N) / np.sqrt(Nvar_rh * Nvar_pc1)

        plt.figure()
        plt.plot(r_square, '-o', markersize=4)
        plt.legend()
        plt.grid()
        plt.xlabel('Relative Height')
        plt.ylabel('R square between PC1 and RH')
        plt.savefig(self.output_dir / f'pearson_corr_PC1_RHs_{data_name}.png')

        # relative Cohen's d betweem RH98 and PC1-3
        for col in range(3):
            d_mat = np.maximum(cohend_matrix[col] - cohend_matrix[-3], 0)
            plt.figure(figsize=(10, 10))
            plt.imshow(d_mat, cmap='Blues', vmin=0, vmax=1)
            plt.xticks(np.arange(len(BIOMES)), labels=BIOMES, rotation=90)
            plt.yticks(np.arange(len(BIOMES)), labels=BIOMES)
            plt.colorbar()
            plt.title(f'Biome effect size matrix for max(0, |d(PC{col+1}| - |d(RH98)|)')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'biome_effect_size_matrix_{data_name}_PC{col+1}-RH98.png')

        # relative Cohen's d betweem PC1 and PC2-3
        for col in range(1, 3):
            d_mat = np.maximum(cohend_matrix[col] - cohend_matrix[0], 0)
            plt.figure(figsize=(10, 10))
            plt.imshow(d_mat, cmap='Blues', vmin=0, vmax=1)
            plt.xticks(np.arange(len(BIOMES)), labels=BIOMES, rotation=90)
            plt.yticks(np.arange(len(BIOMES)), labels=BIOMES)
            plt.colorbar()
            plt.title(f'Biome effect size matrix for max(0, |d(PC{col+1}| - d(PC1))')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'biome_effect_size_matrix_{data_name}_PC{col+1}-PC1.png')



@dataclass
class MyConfig:
    parquet_fp: str = '~/data/GEDI/train_subsets/train*_filtered_v3.parquet'
    model_path: str = 'output/pca_model.pkl'
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
    # find_sensitivity_beam_cor(glob.glob(cfg.parquet_fp))
    # find_sensitivity_biome_cor(glob.glob(cfg.parquet_fp))
    print(task)

    model = PCAAnalysis(**cfg)
    model.plot_explained_variance()
    model.plot_components()
    model.run_biome_effect_size_analysis()

    # model.run_land_cover_effect_size_analysis()

    print(f'time taken for running {cfg.task}: {time.time() - t0}')


if __name__ == '__main__':
    main()
