from pathlib import Path
import pandas as pd
import numpy as np
import dask.dataframe as dd

def evaluate_chm_with_sota(ours_dir: str, save_dir: str, data_name: str =None, split: str = 'test', year: int = 2020, slope_lt20: bool = False, **kwargs):
    '''
    Evaluate the CHM performance with SOTA CHM
    '''
    ours_dir = ours_dir.format(data_name=data_name, split=split, year=year)
    ours_dir = Path(f'{ours_dir}').expanduser()
    
    save_dir = Path(f'{save_dir}').expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f'chm_with_sota_performance_{split}_{year}.csv'
    files = list(ours_dir.glob('*.parquet'))
    ref = ['rh95', 'rh98', 'rh100']
    products = {
        'UMD': ['RH95_UMD', None, None], 
        'ETH': [None, 'RH98_ETH', None], 
        'UM': [None, None, 'RH100_UM'], 
        'META': ['RH95_META', None, None], 
        'Ours': ['RH95_Q1_raw', 'RH98_Q1_raw', 'RH100_Q1_raw'],
    }
    product_cols = np.unique([col for sublist in products.values() for col in sublist if col is not None])
    cols = product_cols.tolist() + ref + ['slope']
    ddf = dd.read_parquet(files, columns=cols)
    if slope_lt20:
        ddf = ddf[ddf['slope'] < 20]
        out_file = out_file.with_suffix('.slope_lt20.csv')
    
    ddf = ddf.dropna(subset=product_cols)
    ddf = ddf.compute()
    me = {}
    mae = {}
    rmse = {}

    for product, cols in products.items():
        me[product] = np.full(3, np.nan)
        mae[product] = np.full(3, np.nan)
        rmse[product] = np.full(3, np.nan)
        for i, col in enumerate(cols):
            if col is not None:
                residuals = ddf[col] - ddf[ref[i]]
                me[product][i] = residuals.mean()
                mae[product][i] = np.abs(residuals).mean()
                rmse[product][i] = (residuals**2).mean()**0.5
    index = ['RH95', 'RH98', 'RH100']
    df_me = pd.DataFrame(me, index=index).T
    df_mae = pd.DataFrame(mae, index=index).T
    df_rmse = pd.DataFrame(rmse, index=index).T
    df = pd.concat([df_rmse, df_mae, df_me], keys=['RMSE', 'MAE', 'ME'], axis=1)
    df.to_csv(out_file)
    df.to_latex(out_file.with_suffix('.tex'), float_format=f"%.2f", na_rep='')


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='on_sota_chm',
    default_run='evaluate_chm_with_sota',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()
