# pip install rasterio plotly numpy affine
import numpy as np
import rasterio as rio
from affine import Affine
from pathlib import Path
import plotly.graph_objects as go
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
import json
import yaml
from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore
import hydra
from omegaconf import OmegaConf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import geopandas as gpd
from visualization._utils import get_patch_by_coords


class VisRHS:
    def __init__(self, year: int, rh_vis_param_path: str = None,
                 prediction_dir: str = None,
                 root_save_dir: str = None,
                 tiles_info_path: str = None,
                 project_folder: str = None,
                 **kwargs):
        self.project_folder = Path(project_folder).expanduser()
        self.prediction_dir = Path(prediction_dir).expanduser()
        self.root_save_dir = Path(root_save_dir).expanduser()
        self.root_save_dir.mkdir(exist_ok=True)
        self.year = year
        self.tiles_info = OmegaConf.load(tiles_info_path)
        self.tiles_info = OmegaConf.to_container(self.tiles_info, resolve=True)
        rh_vis_param_path = Path(rh_vis_param_path).expanduser()
        self.rh_vis_param = OmegaConf.load(rh_vis_param_path)

    def make_gif(self):
        cfg = self.tiles_info['gif']  # common parameters for all tiles
        rh_idxs = cfg.get('rh_idxs', range(101))
        gif_png_dir = self.root_save_dir / f'rhs_3d_pngs'
        gif_png_dir.mkdir(exist_ok=True)
        for tile_info in cfg['tiles']:  # tile-specific parameters
            n_pngs = len(list(gif_png_dir.glob(f"{tile_info['tile_id']}_*.png")))
            if n_pngs >= len(rh_idxs):
                continue
            window = rio.windows.Window(
                col_off=tile_info['col_off'],
                row_off=tile_info['row_off'],
                width=cfg['width'],
                height=cfg['height'])

            for rh_idx in rh_idxs:
                self.vis_rhs_3d(tile_info['tile_id'], rh_idx, window, gif_png_dir,
                                cfg['z_exaggeration'],
                                cfg['camera'],
                                cfg['aspectratio'],
                                cfg['cmin'], cfg['cmax'], cfg['orthographic'], tile_info['name'])

            png_files = list(gif_png_dir.glob(f"{tile_info['tile_id']}_RH*.png"))
            frames = [Image.open(p) for p in png_files]

            frames[0].save(
                self.root_save_dir / f'{tile_info['tile_id']}_RH0-100_animation.gif',
                save_all=True,
                append_images=frames[1:],
                duration=60,  # ms per frame
                loop=0        # 0 = infinite loop
            )

    def vis_rgb(self):
        zarr_path = Path(f'~/data/gvs/deploy/inference_{self.year}.zarr').expanduser()
        rgb_dir = self.root_save_dir / f'rgb_pngs'
        rgb_dir.mkdir(exist_ok=True)
        cfg = self.tiles_info['rgb']
        for time in cfg['time_idxs']:
            ds = xr.open_zarr(zarr_path, group=cfg['tile_id'], consolidated=False, chunks='auto')
            max_val = cfg.get('max_val', 2000)
            print(f'max_val: {max_val}')
            rgb = ds.s2.sel(
                band=['B04', 'B03', 'B02']).isel(
                time=time, x=slice(cfg['col_off'],
                                   cfg['col_off'] + cfg['width']),
                y=slice(cfg['row_off'],
                        cfg['row_off'] + cfg['height']))
            rgb = rgb.clip(0, max_val) / max_val
            plt.figure(figsize=(8, 4))
            img = rgb.plot.imshow(x='x', y='y', rgb='band')
            img.axes.set_aspect('equal')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.title('')
            plt.xlabel('')
            plt.ylabel('')
            plt.savefig(rgb_dir / f"{cfg['name']}_{cfg['tile_id']}_rgb_time{time}.pdf", bbox_inches='tight')
            plt.close()

    def vis_rhs(self):
        '''
        Visualize the RHs as 2D images for given RH indices of given tiles
        Args:
            rh_idxs: list of RH indices to visualize
        Returns:
            None
        '''
        save_dir = self.root_save_dir / f'rhs_2d_pngs'
        save_dir.mkdir(exist_ok=True)
        cfg = self.tiles_info['rhs']
        for rh_idx in cfg['rh_idxs']:
            window = rio.windows.Window(
                col_off=cfg['col_off'],
                row_off=cfg['row_off'],
                width=cfg['width'],
                height=cfg['height'])
            file_path = self.prediction_dir / f"{cfg['tile_id']}/RH{rh_idx}_Q1.tif"
            with rio.open(file_path) as src:
                rh = src.read(1, window=window)
                nodata = src.nodata
            vmin = self.rh_vis_param[f'rh{rh_idx}']['cmin']
            vmax = self.rh_vis_param[f'rh{rh_idx}']['cmax']
            print(f'rh{rh_idx}', vmin, vmax)
            plt.figure(figsize=(8, 4))
            rh[rh == nodata] = 0
            plt.imshow(rh, cmap='inferno', vmin=vmin, vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            cbar = plt.colorbar(shrink=1, aspect=20)
            cbar.ax.set_ylabel(f'RH{rh_idx} [m]')
            cbar.set_ticks(np.linspace(vmin, vmax, 6))
            cbar.set_ticklabels([f'{int(tick/10)}' for tick in np.linspace(vmin, vmax, 6)])
            # plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
            # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(save_dir / f"{cfg['name']}_{cfg['tile_id']}_RH{rh_idx}.pdf", bbox_inches='tight')
            plt.close()

    def vis_rhs_3d(
            self, tile, rh_idx, window, gif_png_dir, z_exaggeration=2.0, camera=None, aspectratio=(1, 1, 0.25),
            cmin=None, cmax=None, orthographic=False, title=None):
        file_path = self.prediction_dir / f'{tile}_cog/RH{rh_idx}_Q1.cog.tif'
        name = file_path.stem.split('.')[0]
        with rio.open(file_path) as src:
            rh = src.read(1, window=window, masked=True)
            T: Affine = src.window_transform(window)

        # optional downsample
        factor = max(1, int(np.ceil(max(rh.shape)/1500)))
        if factor > 1:
            rh = rh[::factor, ::factor]
            T = T * Affine.scale(factor)

        nrows, ncols = rh.shape
        rows, cols = np.mgrid[0:nrows, 0:ncols]
        X = T.c + cols*T.a + rows*T.b
        Y = T.f + cols*T.d + rows*T.e
        Z = rh.filled(0) * z_exaggeration

        xmin, xmax = float(X.min()), float(X.max())
        ymin, ymax = float(Y.min()), float(Y.max())

        # Create custom tick labels divided by 10
        tick_range = np.linspace(cmin, cmax, 6)

        fig = go.Figure(data=[go.Surface(z=Z, x=X[0, :], y=Y[:, 0], cmin=cmin, cmax=cmax,
                                         colorbar=dict(
            thickness=15,   # make bar thinner
            len=0.5,       # shorten (0–1 relative to plot height)
            y=0.4,          # center it vertically
            yanchor="middle",
            tickmode='array',
            tickvals=tick_range,
            ticktext=[f'{int(v/10)}' for v in tick_range]
        ))])

        cam = camera or dict(
            eye=dict(x=-1.2, y=-2.0, z=1.0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1),
            projection=dict(type='orthographic' if orthographic else 'perspective')
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(title=dict(text="Longitude", font=dict(size=18)), range=[xmin, xmax]),
                yaxis=dict(title=dict(text="Latitude", font=dict(size=18)),  range=[ymin, ymax]),
                zaxis=dict(
                    title=dict(text=f"RH{rh_idx} (m)", font=dict(size=18)),
                    range=[cmin, cmax],
                    tickmode='array',
                    tickvals=tick_range,
                    ticktext=[f'{int(v/10)}' for v in tick_range]
                ),
                aspectmode="manual",
                aspectratio=dict(x=aspectratio[0], y=aspectratio[1], z=aspectratio[2]),
                camera=cam,
                bgcolor="white",
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=20, t=0),  # Zero margins
            paper_bgcolor="white",
            plot_bgcolor="white",
            title=dict(text=title, y=0.75, xanchor='left', yanchor='top', font=dict(size=16))
        )
        # fig.write_html(f"output/{tile}_{name}.html")
        fig.write_image(gif_png_dir / f"{tile}_{name}.png")
        return fig

    def make_s2_rh_subplots(self):
        '''
        Make subplots of S2 and RHs for given RH indices of given tiles, mainly for comparison of different RH products
        Args:
            rh_idxs: list of RH indices to visualize
        Returns:
            None
        '''
        save_dir = self.project_folder / 'vsm_examples' / f's2_rh_subplots'
        save_dir.mkdir(exist_ok=True, parents=True)
        zarr_path = self.project_folder / 'deploy' / f'inference_{self.year}.zarr'
        cfg = self.tiles_info['s2_rh_subplots']
        # the parent directory of the prediction files (e.g. ~/data/gvs/deploy/predictions_GTiff_2020)
        pred_path = Path(cfg['pred_path']).expanduser()
        for tile_info in cfg['tiles']:
            cols = len(cfg['rh_idxs']) + 1
            rows = 1
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4))

            # Pattern: [image, image, cbar, image, cbar]
            ncols = 2 * cols - 1
            widths = []
            for j in range(cols):
                widths.append(1.0)               # image column
                if j > 0:
                    widths.append(0.05)          # cbar column right after each image (except first)

            fig = plt.figure(figsize=(10, 4))
            gs = gridspec.GridSpec(
                nrows=1,
                ncols=ncols,
                width_ratios=widths,
                wspace=0.2,  # spacing between image and colorbar columns
            )

            axes_main = []
            caxes = []

            # Create axes: main image axes at even columns; colorbar axes at odd columns
            ax = fig.add_subplot(gs[0, 0])
            axes_main.append(ax)
            for j in range(1, cols):
                ax = fig.add_subplot(gs[0, 2*j-1])
                axes_main.append(ax)
                cax = fig.add_subplot(gs[0, 2*j])
                caxes.append(cax)
            width = tile_info.get('width', cfg['width'])
            height = tile_info.get('height', cfg['height'])
            window = rio.windows.Window(
                col_off=tile_info['col_off'],
                row_off=tile_info['row_off'],
                width=width, height=height)
            img_ds = xr.open_zarr(zarr_path, group=tile_info['tile_id'], consolidated=False, chunks='auto')
            max_val = cfg.get('max_val', 2000)
            x_slice = slice(tile_info['col_off'], tile_info['col_off']+width)
            y_slice = slice(tile_info['row_off'], tile_info['row_off']+height)
            rgb = img_ds.s2.sel(band=['B04', 'B03', 'B02']).isel(time=tile_info['time_idx'], x=x_slice, y=y_slice)
            rgb = rgb.clip(0, max_val) / max_val
            rgb.plot.imshow(x='x', y='y', rgb='band',  ax=axes_main[0], add_colorbar=False)
            axes_main[0].set(title=f'Sentinel-2 RGB', xticks=[], yticks=[], aspect='equal', xlabel='', ylabel='')

            for i, rh_idx in enumerate(cfg['rh_idxs']):
                rh_file_path = pred_path / f'{tile_info['tile_id']}/RH{rh_idx}_Q1.tif'
                with rio.open(rh_file_path) as src:
                    rh = src.read(1, window=window)
                    nodata = src.nodata
                vmin = self.rh_vis_param[f'rh{rh_idx}']['cmin']
                vmax = self.rh_vis_param[f'rh{rh_idx}']['cmax']
                rh[rh == nodata] = 0
                im = axes_main[i+1].imshow(rh, cmap='inferno', vmin=vmin, vmax=vmax)
                axes_main[i+1].set(title=f'RH{rh_idx} [m]', xticks=[], yticks=[], aspect='equal', xlabel='', ylabel='')
                # Colorbar goes to the dedicated cax for this panel (the slot right of it)
                cax = caxes[i]  # note: i maps to panel i+1
                if cax is not None:
                    pos = cax.get_position()
                    # shrink to 80% height and center vertically
                    cax.set_position([pos.x0, pos.y0 + pos.height*0.15, pos.width, pos.height*0.7])
                    cb = fig.colorbar(im, cax=cax)
                    cb.set_ticks(np.linspace(vmin, vmax, 4))
                    cb.set_ticklabels([f'{int(t/10)}' for t in np.linspace(vmin, vmax, 4)])
            plt.tight_layout()
            rh_idxs_str = '_'.join([str(rh_idx) for rh_idx in cfg['rh_idxs']])
            plt.savefig(
                save_dir / f"exp_{tile_info['tile_id']}_time{tile_info['time_idx']}_RH{rh_idxs_str}_{width}x{height}.pdf", bbox_inches='tight')
            plt.close()

    def vis_density(self):
        '''
        Visualize the vertical density of energy derived from RHs (1/(RH_{i} - RH_{i-1})), the height interval that returns 1% of the total energy.
        Args:
            None
        Returns:
            None
        '''
        save_dir = self.root_save_dir / f'density_figures'
        save_dir.mkdir(exist_ok=True, parents=True)
        zarr_path = self.project_folder / 'deploy' / f'inference_{self.year}.zarr'
        
        s2_grid = gpd.read_parquet(self.project_folder / 'deploy' / 'deploy_status.parquet').to_crs("EPSG:4326")
        
        # example specific parameters
        cfg = self.tiles_info['density']
        best_image_time = cfg['best_image_time']
        buffer_m = cfg['buffer_m']
        points = cfg['points']
        
        
        vsm_patch, rgb, points_utm = get_patch_by_coords(
            points, s2_grid, self.year, buffer_m=buffer_m, prediction_dir=self.prediction_dir, input_image_dir=zarr_path,
            stac_collection=self.project_folder / f'deploy/gvsm_stac_catalog/vsm_{self.year}', best_image_time=best_image_time)
        vsm_patch = vsm_patch.compute()
        rgb = rgb.compute()
        rgb = rgb.clip(0, 2000) / 2000
        rgb = rgb.data.transpose(1, 2, 0)
        extent = [
            float(vsm_patch.x.min()),
            float(vsm_patch.x.max()),
            float(vsm_patch.y.min()),
            float(vsm_patch.y.max())
        ]
        fig, axes = plt.subplots(2, len(points_utm)+1, figsize=(18, 9), gridspec_kw={'width_ratios': [2] + [1] * len(points_utm)})
        axes[0, 0].imshow(rgb, extent=extent, origin='upper')
        axes[0, 0].set(title='RGB', xticks=[], yticks=[], aspect='equal', xlabel='', ylabel='')
        
        axes[1, 0].imshow(vsm_patch.sel(band='RH98'), cmap='inferno', extent=extent, origin='upper')
        axes[1, 0].set(title='RH98', xticks=[], yticks=[], aspect='equal', xlabel='', ylabel='')
        for i, coord in enumerate(points_utm.geometry):
            for ax in axes[:, 0]:
                ax.plot(coord.x, coord.y, 'ro', markersize=6)
                ax.text(coord.x, coord.y, str(i), color='black', fontsize=10, va='bottom')

            rhs = vsm_patch.sel(x=coord.x, y=coord.y, method='nearest')
            axes[0, i+1].plot(rhs)
            axes[0, i+1].set_title(f'Point {i}')
            density = 1/(rhs - rhs.shift(band=1))
            density = density.fillna(0)
            density[density == np.inf]=1
            axes[1, i+1].plot(density, np.arange(len(density)))
        
            
        plt.savefig(save_dir / f'density_{buffer_m}m_{best_image_time}.pdf', bbox_inches='tight')
        plt.close()
            


@dataclass
class Config:
    year: int = 2020
    prediction_dir: str = '~/data/gvs/deploy/predictions_2020'
    project_folder: str = '~/data/gvs'
    root_save_dir: str = '~/data/gvs/deploy/eu_results'
    tiles_info_path: str = 'config/vis_params/gif_examples.yaml'
    rh_vis_param_path: str = 'config/vis_params/rh_vis_param.yaml'
    task: str = 'vis_rhs'


cs = ConfigStore.instance()
cs.store(name='vis_rhs', node=Config)


@hydra.main(config_name='vis_rhs', version_base='1.2')
def main(cfg):
    vis = VisRHS(**cfg)
    if cfg.task == 'vis_rhs':
        vis.vis_rhs()
    elif cfg.task == 'vis_rgb':
        vis.vis_rgb()
    elif cfg.task == 'make_gif':
        vis.make_gif()
    elif cfg.task == 'make_s2_rh_subplots':
        vis.make_s2_rh_subplots()
    elif cfg.task == 'vis_density':
        vis.vis_density()


if __name__ == '__main__':
    main()
