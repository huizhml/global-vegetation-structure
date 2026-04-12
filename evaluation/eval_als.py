import rasterio
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


def extract_valid_pixels(ref_dir: Path, ours_dir: Path, tile_id: str) -> np.ndarray:
    '''
    Extract valid pixels from a list of tif files that cover the same area.
    Args:
        tif_files: list of tif files
    Returns:
        valid_pixels: numpy array of shape (n_pixels, n_bands)
    '''
    with rasterio.open(ref_dir / f'{tile_id}.cog.tif') as src:
        ref = src.read(masked=True)
        ref_data = ref.filled(np.nan)
        ref_data[ref_data > 100] = np.nan # there are very high values (> 1000) for some tiles
        ref_data[ref_data < 0] = 0 # there are negative values, looks like nonvegetation
    with rasterio.open(ours_dir / f'{tile_id}/RH98_Q1.tif') as src:
        ours = src.read(masked=True).astype(np.float32)
        ours_data = ours.filled(np.nan)
    mask = ~np.isnan(ref_data) & ~np.isnan(ours_data)
    ref_data = ref_data[mask]
    ours_data = ours_data[mask]/10
    return ref_data, ours_data


def extract_pixels_and_save(ref_dir: str, ours_root_dir: str, save_dir: str = None, **kwargs) -> dict:
    ref_dir = Path(ref_dir).expanduser()
    ours_root_dir = Path(ours_root_dir).expanduser()
    save_dir = save_dir or ref_dir.parent / 'results'
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    ref_col = 'als' if 'ALS' in ref_dir.name else 'lvis'
    ref_meta = pd.read_csv(ref_dir.parent / f'meta_{ref_col}.csv')

    tiles = [t for t in ref_dir.glob('*.tif') if t.name.endswith('.cog.tif')]
    
    for tile_id in tiles:
        tile_id = tile_id.stem.split('.')[0]
        out_file = save_dir / f'eval_{ref_col}_{tile_id}.parquet'
        if out_file.exists():
            continue
        ref_year = ref_meta[ref_meta['Tile name'] == tile_id]['Year'].values[0]
        if type(ref_year) == str:
            if '-' in ref_year:
                ref_year = ref_year.split('-')[-1]
            ref_year = int(ref_year)

        if ref_year < 2016:
            continue
        if ref_year == 2016:
            ref_year = 2017
        ours_dir = ours_root_dir / f'{ref_year}/original/tiles/geotiff/'
        ref, ours = extract_valid_pixels(ref_dir, ours_dir, tile_id)
        df = pd.DataFrame({'tile_id': np.full(len(ref), tile_id), ref_col: ref, 'ours_rh98': ours})
        df.to_parquet(out_file, index=False)

    
def scatter_plot(tile_id: str, df: pd.DataFrame, ref_col: str, stats: dict, save_dir: Path = None, max_height: int = 80) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(df, x=ref_col, y = 'ours_rh98', bins=50, cbar=True, cmap='viridis', ax=ax)
    ax.collections[0].set_norm(LogNorm(vmin=1, vmax=10000))
    ax.set_xlim(0, max_height)
    ax.set_ylim(0, max_height)
    ax.plot(np.arange(max_height), np.arange(max_height), color='black', linestyle='dashed')
    plt.title(f'{ref_col} vs Ours (RH98) - {tile_id}, R^2 = {stats['r2']:.2f}')
    plt.text(0.05, 0.95, f'R^2 = {stats['r2']:.2f}', ha='left', va='top', transform=ax.transAxes, fontsize=14)
    plt.text(0.05, 0.90, f'RMSE = {stats['rmse']:.2f}', ha='left', va='top', transform=ax.transAxes, fontsize=14)
    plt.text(0.05, 0.85, f'ME = {stats['me']:.2f}', ha='left', va='top', transform=ax.transAxes, fontsize=14)
    plt.text(0.05, 0.80, f'N = {stats['n']}', ha='left', va='top', transform=ax.transAxes, fontsize=14)
    plt.text(0.05, 0.75, f'Avg Height = {stats['avg_height']:.2f}', ha='left', va='top', transform=ax.transAxes, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f'scatter_plot_{ref_col}_{tile_id}.pdf')
    plt.close()
    
def tile_level_evaluate(df: pd.DataFrame, ref_col: str, ours_col: str='ours_rh98') -> dict:
    n = len(df)
    residual = df[ours_col] - df[ref_col]
    rss = (residual**2).sum()
    avg_height = df[ref_col].mean()
    rmse = np.sqrt(rss/n)
    me = residual.mean()
    tss = ((df[ref_col] - avg_height)**2).sum()
    r2 = 1- rss/tss
    
    return {
        'n': n,
        'rss': rss,
        'ref': df[ref_col],
        'rmse': rmse,
        'me': me,
        'r2': r2,
        'avg_height': avg_height
    }

def evaluate(df_dir: str, save_dir: str = None, ref_col: str = 'als') -> dict:
    '''
    Correlation analysis between ALS and our predictions
    Args:
        df_dir: str, the directory of the evaluation results
        save_dir: str, the directory to save the results
        ref_col: str, the column name of the reference data
    Returns:
        stats: dict, the statistics of the evaluation
    '''
    df_dir = Path(df_dir).expanduser()
    save_dir = save_dir or df_dir.parent / 'figures'
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for f in df_dir.glob(f'*{ref_col}*.parquet'):
        df = pd.read_parquet(f)
        tile_id = f.stem.split('_')[-1]
        stats[tile_id] = tile_level_evaluate(df, ref_col)
        scatter_plot(tile_id, df, ref_col, stats[tile_id], save_dir)
    
    rss =0
    me =0
    n =0
    sum_me =0
    ref = []
    for tile_id in stats.keys():
        rss += stats[tile_id]['rss']
        n += stats[tile_id]['n']
        sum_me += stats[tile_id]['me'] * stats[tile_id]['n']
        ref.append(stats[tile_id]['ref'])
    rmse = np.sqrt(rss / n)
    me = sum_me / n
    ref = np.concatenate(ref)
    avg_height = ref.mean()
    r2 = 1 - rss / ((ref - avg_height)**2).sum()
    df = pd.DataFrame([{'rmse': rmse, 'me': me, 'n': n, 'r2': r2, 'avg_height': avg_height}])
    df.to_csv(save_dir / f'overall_stats_{ref_col}.csv', index=False)
    tile_level_stats = pd.DataFrame(stats).T
    tile_level_stats = tile_level_stats.drop(columns='ref')
    tile_level_stats.to_csv(save_dir / f'tile_level_stats_{ref_col}.csv', index=True) 
    return stats
