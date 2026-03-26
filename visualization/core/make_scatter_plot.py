import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import dask.dataframe as dd
from pathlib import Path


def load_ch_predictions(parq_dir: str):
    parq_dir = Path(parq_dir).expanduser()
    df = dd.read_parquet(parq_dir / '*.parquet', columns=[])
    return df


def make_parity_plot(parq_dir: str):
    parq_dir = Path(parq_dir).expanduser()

