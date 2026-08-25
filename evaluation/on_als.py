import rasterio
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error

from const import FIGURE_SIZES, set_plot_fonts
from evaluation.plots import hexbin_density_plot, hexbin_density_grid, sci_notation

set_plot_fonts()


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

        if ref_year <= 2016:
            ref_year = 2017
        ours_dir = ours_root_dir / f'{ref_year}/original/tiles/geotiff/'
        ref, ours = extract_valid_pixels(ref_dir, ours_dir, tile_id)
        df = pd.DataFrame({'tile_id': np.full(len(ref), tile_id), ref_col: ref, 'ours_rh98': ours})
        df.to_parquet(out_file, index=False)

    
def _stats_annotation(stats: dict) -> str:
    '''Multi-line boxed-annotation text, shared by the single plot and grid.'''
    return '\n'.join([
        f'$\\rho$ = {stats['rho']:.2f}',
        f'$R^2$ = {stats['r2']:.2f}',
        f'RMSE = {stats['rmse']:.2f}',
        f'ME = {stats['me']:.2f}',
        f'N = ${sci_notation(stats['n'])}$',
        f'Avg Height = {stats['avg_height']:.2f}m',
    ])


def _panel_title(tile_id: str, ref_col: str, ref_year: str = None, our_year: str = None) -> str:
    '''Panel title: tile id, with the acquisition years on a second line when
    available (e.g. "32MPC\\nALS 2019 / Ours 2019").'''
    if ref_year and our_year:
        return f'{tile_id}\n{ref_col.upper()} ({ref_year}) vs Ours ({our_year})'
    return tile_id


def scatter_plot(tile_id: str, df: pd.DataFrame, ref_col: str, stats: dict, save_dir: Path = None,
                 max_height: int = 80, ref_year: str = None, our_year: str = None, **kwargs) -> None:
    hexbin_density_plot(
        df[ref_col].to_numpy(), df['ours_rh98'].to_numpy(),
        save_path=save_dir / f'scatter_plot_{ref_col}_{tile_id}.pdf',
        figsize=kwargs.get('figsize', FIGURE_SIZES['medium']),
        gridsize=80, vmax=100_000,
        extent=(0, max_height, 0, max_height),
        refline='identity', equal_aspect=True,
        annotation=_stats_annotation(stats), annot_corner='upper left', annot_fontsize=14,
        title=_panel_title(tile_id, ref_col, ref_year, our_year),
        x_label=ref_col.upper(),
        y_label='Ours (RH98)',
    )


def scatter_plot_grid(items: list, ref_col: str, save_dir: Path = None,
                      max_height: int = 80, nrows: int = 3, ncols: int = 4, **kwargs) -> None:
    '''Combined nrows x ncols panel of per-tile hexbin scatters with shared x/y
    axes and a single shared colorbar. `items` is a list of
    (tile_id, parquet_path, stats, ref_year, our_year); the per-tile acquisition
    years go on a second line of each panel title.

    Panel data is passed as callables so each tile's pixels are read only while
    that panel is drawn — holding all of them at once OOMs on the full set.'''
    def loader(path, col):
        return lambda: pd.read_parquet(path, columns=[col])[col].to_numpy()

    panels = [
        {'x': loader(path, ref_col), 'y': loader(path, 'ours_rh98'),
         'annotation': _stats_annotation(stats),
         'title': _panel_title(tile_id, ref_col, ref_year, our_year)}
        for tile_id, path, stats, ref_year, our_year in items
    ]
    hexbin_density_grid(
        panels,
        save_path=save_dir / f'scatter_grid_{ref_col}.pdf',
        nrows=nrows, ncols=ncols,
        gridsize=80, vmax=100_000,
        extent=(0, max_height, 0, max_height),
        refline='identity', equal_aspect=True,
        annot_corner='upper left', annot_fontsize=10,
        panel_title_fontsize=14,
        x_label=ref_col.upper(), y_label='Ours (RH98)',
    )


# Spearman needs a full ranking, which scipy does in float64 + int64 argsort:
# ~32 bytes/pixel on top of the arrays themselves. Above this many pixels rho
# is computed on a fixed random subsample instead — at 1e7 points the estimate
# is stable well past the 2 decimals we print, and memory stays bounded.
RHO_MAX_N = 10_000_000


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) > RHO_MAX_N:
        idx = np.random.default_rng(0).choice(len(y_true), RHO_MAX_N, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]
    return float(spearmanr(y_true, y_pred).statistic)


def tile_level_evaluate(df: pd.DataFrame, ref_col: str, ours_col: str='ours_rh98') -> dict:
    y_true = df[ref_col].to_numpy()
    y_pred = df[ours_col].to_numpy()
    residual = y_pred - y_true
    # rss + the two ref moments are enough for evaluate() to pool a global
    # RMSE/R^2 across tiles; keeping the raw arrays instead would hold every
    # tile's pixels in memory at once (OOM on the full ALS set). Sums are
    # accumulated in float64 — float32 loses precision over ~1e8 pixels.
    ref64 = y_true.astype(np.float64, copy=False)
    return {
        'n': len(df),
        'rss': float((residual ** 2).sum()),
        'sum_ref': float(ref64.sum()),
        'sum_ref_sq': float((ref64 ** 2).sum()),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'me': float(residual.mean()),  # signed bias; no sklearn/scipy equivalent
        'r2': float(r2_score(y_true, y_pred)),
        'rho': _spearman(y_true, y_pred),
        'avg_height': float(y_true.mean()),
    }

def evaluate(df_dir: str, save_dir: str = None, ref_col: str = 'als', **kwargs) -> dict:
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

    # Per-tile acquisition years for the axis labels: 'Year' is the reference
    # (ALS/LVIS) year, 'Ours' our prediction year. Optional — fall back to no
    # year if the meta CSV is missing or a tile isn't listed.
    meta_path = df_dir.parent / f'meta_{ref_col}.csv'
    year_map = {}
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        year_map = {str(r['Tile name']): (str(r['Year']), str(r['Ours']))
                    for _, r in meta.iterrows()}
    else:
        print(f'  evaluate: no meta file at {meta_path}; skipping year labels')

    stats = {}
    grid_items = []
    for f in sorted(df_dir.glob(f'*{ref_col}*.parquet')):
        df = pd.read_parquet(f)
        tile_id = f.stem.split('_')[-1]
        ref_year, our_year = year_map.get(tile_id, (None, None))
        stats[tile_id] = tile_level_evaluate(df, ref_col)
        print(f'  {tile_id}: {len(df):,} px', flush=True)
        scatter_plot(tile_id, df, ref_col, stats[tile_id], save_dir,
                     ref_year=ref_year, our_year=our_year)
        grid_items.append((tile_id, f, stats[tile_id], ref_year, our_year))
        del df
    if grid_items:
        # Order panels by reference avg height (ascending: shortest -> tallest).
        grid_items.sort(key=lambda it: it[2]['avg_height'])
        scatter_plot_grid(grid_items, ref_col, save_dir)

    rss =0
    me =0
    n =0
    sum_me =0
    sum_ref =0
    sum_ref_sq =0
    for tile_id in stats.keys():
        # if stats[tile_id]['r2'] < 0:
        #     print(f"Warning: tile {tile_id} has negative R^2 ({stats[tile_id]['r2']:.2f}), skipping in global stats")
        #     continue
        rss += stats[tile_id]['rss']
        n += stats[tile_id]['n']
        sum_me += stats[tile_id]['me'] * stats[tile_id]['n']
        sum_ref += stats[tile_id]['sum_ref']
        sum_ref_sq += stats[tile_id]['sum_ref_sq']
    rmse = np.sqrt(rss / n)
    me = sum_me / n
    avg_height = sum_ref / n
    # SS_tot = sum((ref - mean)^2) = sum(ref^2) - n * mean^2, so the pooled R^2
    # needs only the two moments above, not the pixels themselves.
    r2 = 1 - rss / (sum_ref_sq - n * avg_height ** 2)
    # No pooled rho: Spearman is not decomposable into per-tile summaries, and
    # ranking every pixel at once is what blew up memory. Use the per-tile rho
    # in tile_level_stats_*.csv instead.
    df = pd.DataFrame([{'rmse': rmse, 'me': me, 'n': n, 'r2': r2, 'avg_height': avg_height}])
    df.to_csv(save_dir / f'overall_stats_{ref_col}.csv', index=False)
    tile_level_stats = pd.DataFrame(stats).T
    tile_level_stats = tile_level_stats.drop(columns=['sum_ref', 'sum_ref_sq'])
    tile_level_stats.to_csv(save_dir / f'tile_level_stats_{ref_col}.csv', index=True)
    return stats


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='on_als',
    default_run='evaluate_chm_with_als_and_lvis',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()
