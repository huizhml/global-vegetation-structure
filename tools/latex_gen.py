import pandas as pd
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import hydra
from pathlib import Path

def csv_to_latex_rh(csv_file: str, precision:int=2):
    csv_file = Path(csv_file).expanduser()
    output_file = csv_file.with_suffix('.tex')
    df = pd.read_csv(csv_file, index_col=0)
    df.index = df.index.str.split('_', expand=True, n=1)
    df = df.unstack(level=0)
    ordered_cols = df.columns.get_level_values(1).unique()
    ordered_cols = sorted(ordered_cols, key=lambda x: int(x.split('RH')[1]))
    df = df.reindex(columns=ordered_cols, level=1)
    n_cols = df.shape[1]
    df.to_latex(str(output_file), float_format=f"%.{precision}f", na_rep='', column_format=f'l *{{{n_cols}}}{{S}}')
    return df

def csv_to_latex(csv_file: str, precision:int=2):
    csv_file = Path(csv_file).expanduser()
    output_file = csv_file.with_suffix('.tex')
    df = pd.read_csv(csv_file, index_col=0)
    n_cols = df.shape[1]
    df.to_latex(str(output_file), float_format=f"%.{precision}f", na_rep='', column_format=f'l *{{{n_cols}}}{{S}}')
    return df

@dataclass
class LatexGenConfig:
    csv_file: str = '~/data/gvs/deploy/correction/correction_performance_2020_compared_with_sota_chm_eu.csv'
    task: str = 'csv_to_latex'
    precision: int = 2

cs = ConfigStore.instance()
cs.store(name='latex_gen', node=LatexGenConfig)


@hydra.main(config_name='latex_gen', version_base='1.2')
def main(cfg):
    print(cfg)
    if cfg.task == 'csv_to_latex':
        csv_to_latex(cfg.csv_file, cfg.precision)
        
        
if __name__ == '__main__':
    main()