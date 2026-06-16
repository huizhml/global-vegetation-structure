from pathlib import Path
from typing import List
import pandas as pd


def csv_to_latex_rh(csv_file: str, precision: int = 2, **kwargs):
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


def csv_to_latex(csv_file: str, precision: int = 2, drop_cols: List[str] = None, **kwargs):
    csv_file = Path(csv_file).expanduser()
    output_file = csv_file.with_suffix('.tex')
    df = pd.read_csv(csv_file, index_col=0)
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)
    n_cols = df.shape[1]
    df.to_latex(str(output_file), float_format=f"%.{precision}f", na_rep='', column_format=f'l *{{{n_cols}}}{{S}}')
    return df
