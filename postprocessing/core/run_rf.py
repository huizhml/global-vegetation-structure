from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from hydra.core.config_store import ConfigStore
import hydra
from dataclasses import field

def get_data(h5_file: str, naturalness_fp: str, use_full_profile: bool=False, s2_only: bool=False, rhs_only: bool=False, to_meter: bool=False):
    h5_file = Path(h5_file).expanduser()
    naturalness_fp = Path(naturalness_fp).expanduser()
    target_df = pd.read_csv(naturalness_fp, index_col='rowid')
    target_df['class_idx'] = target_df['Land_use_ID'].dropna()
    uncertain_idx = target_df[target_df['class_idx'].isin([1, -1])].index
    
    with h5py.File(h5_file, 'r') as f:
        rowid = f['rowid'][:]
        idx = np.where(~np.isin(rowid, uncertain_idx))[0]
        rowid = rowid[idx]
        
        y = target_df.loc[rowid, 'class_idx']
        if rhs_only:
            x = f['rhs_median'][idx]
            if to_meter:
                x = x / 100
                return x, y
            avg = x.mean(axis=(2,3))
            var = x.var(axis=(2,3))
            return np.concatenate([avg, var], axis=1), y
        elif s2_only:
            x = f['s2'][idx]
            if to_meter:
                return x, y
            avg = x.mean(axis=(2,3))
            var = x.var(axis=(2,3))
            return np.concatenate([avg, var], axis=1), y
        elif use_full_profile:
            rhs = f['rhs_median'][idx]
            s2 = f['s2'][idx]
            if to_meter:
                rhs = rhs / 100
            avg_rhs = rhs.mean(axis=(2,3))
            var_rhs = rhs.var(axis=(2,3))
            avg_s2 = s2.mean(axis=(2,3))
            var_s2 = s2.var(axis=(2,3))
            x = np.concatenate([avg_s2, var_s2, avg_rhs, var_rhs], axis=1)
            return x, y
        else: # use top height
            rhs = f['rhs_median'][idx, 98:99]
            if to_meter:
                rhs = rhs / 100
            avg_rhs = rhs.mean(axis=(2,3))
            var_rhs = rhs.var(axis=(2,3))
            s2 = f['s2'][idx]
            avg_s2 = s2.mean(axis=(2,3))
            var_s2 = s2.var(axis=(2,3))
            x = np.concatenate([avg_s2, var_s2, avg_rhs, var_rhs], axis=1)
            return x, y
        
def run_rf(h5_file: str, naturalness_fp: str, use_full_profile: bool, s2_only: bool, rhs_only: bool, train_val_split: float, n_estimators: int, max_depth: int=None, random_state: int=42):
    assert 0 < train_val_split < 1
    x, y = get_data(h5_file, naturalness_fp, use_full_profile, s2_only, rhs_only)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=train_val_split, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)#, oob_score=True, bootstrap=True)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_val)
    score = classification_report(y_val, y_pred)
    # score = rf.oob_score_
    # print(f'OOB score: {score}')
    postfix = x.shape[1]
    with open(f'output/downstream_task/random_forest_score_fs_{postfix}.txt', 'w') as f:
        f.write(score)
    return score
    
    
def plot_mean_std(h5_file: str, naturalness_fp: str, rhs_only: bool=False, s2_only: bool=False, **kwargs):
    import matplotlib.pyplot as plt
    from const import set_plot_fonts
    set_plot_fonts()
    print('plotting mean and std of RH profile for each naturalness class')
    x, y = get_data(h5_file, naturalness_fp, rhs_only=rhs_only, s2_only=s2_only, to_meter=True)
    feature_size = x.shape[1]
    x = x.transpose(0, 2, 3, 1).reshape(-1, feature_size)
    rowid = y.index
    y = y.values
    y = np.repeat(y, x.shape[0] // y.shape[0])
    rowid = np.repeat(rowid, x.shape[0] // rowid.shape[0])
    name = 'RH' if rhs_only else 'S2'
    columns = [f'{name}_{i}' for i in range(feature_size)]
    df = pd.DataFrame(x, columns=columns)
    df['rowid'] = rowid
    df['class_idx'] = y
    df = df.dropna(subset=['class_idx'])
    df_avg = df.groupby('class_idx').mean()
    df_std = df.groupby('class_idx').std()
    class_names = {
        0: 'No forest',
        11: 'Naturally regenerating forest without any signs of human activities',
        20: 'Naturally regenerating forest with signs of human activities',
        31: 'Planted forest.',
        32: 'Short rotation plantations for timber.',
        40: 'Oil palm plantations.',
        53: 'Agroforestry.',
    }
    
    for class_idx in df['class_idx'].unique():
        plt.figure(figsize=(10, 5))
        plt.plot(range(feature_size), df_avg.loc[class_idx, columns], label='Mean ' + name, linestyle='-', marker='o')
        plt.fill_between(range(feature_size), 
                         df_avg.loc[class_idx, columns] - df_std.loc[class_idx, columns], 
                         df_avg.loc[class_idx, columns] + df_std.loc[class_idx, columns], 
                         alpha=0.2, label='Std ' + name + ' Range')
        plt.title(f'{name} Profile for Class {class_names[class_idx]}')
        plt.xlabel(name + ' Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output/downstream_task/{name}_profile_class_{class_idx}.png')
        plt.close()


    

@dataclass
class RFConfig:
    h5_file: str = '~/data/gvs/downstream_task_data/rhs_predictions_2017_0crmfaia_ps31.h5'
    naturalness_fp: str = '~/data/gvs/downstream_task_data/naturalness/reference_data_set_updated.with_images.csv'
    use_full_profile: bool = True
    rhs_only: bool = False
    s2_only: bool = False
    train_val_split: float = 0.8

    n_estimators: int = 1000
    random_state: int = 42

cs = ConfigStore.instance()
cs.store(name="rf_config", node=RFConfig)

@hydra.main(config_name="rf_config", version_base='1.2')
def main(cfg):
    # run_rf(**cfg)
    plot_mean_std(**cfg)


if __name__ == "__main__":
    main()
