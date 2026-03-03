import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from omegaconf import OmegaConf
import omegaconf.listconfig


def make_parq_subcolumns(parq_dir: str, subcolumns: list[str], save_fp: str = None, **kwargs):
    '''
    Make subcolumns from a large geodataframe into a smaller one.
    For analysis on local machine.
    The large geodataframe is saved in multiple parquet files.
    Args:
        parq_dir: directory of the parquet file
        subcolumns: list of columns to make subcolumns of
        save_fp: path to save the smaller dataframe
    Returns:
        df: dataframe with subcolumns
    '''
    parq_dir = Path(parq_dir).expanduser()
    save_fp = Path(save_fp).expanduser()
    save_fp.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(subcolumns, omegaconf.listconfig.ListConfig):
        subcolumns = OmegaConf.to_container(subcolumns, resolve=True)
    df = dd.read_parquet(parq_dir / '*.parquet', columns=subcolumns)
    df.compute().to_parquet(save_fp)
    return df
