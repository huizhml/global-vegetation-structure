# pip install rasterio plotly numpy affine
import numpy as np
import rasterio as rio
from affine import Affine
from pathlib import Path
import plotly.graph_objects as go
import xarray as xr
import matplotlib.pyplot as plt


def vis_rgb(tile,year, col_off, row_off, width, height, title=None):
    zarr_path = Path(f'~/data/gvs/deploy/inference_{year}.zarr').expanduser()
    ds = xr.open_zarr(zarr_path, group=tile, consolidated=False, chunks='auto')
    rgb = ds.s2.sel(band=['B04', 'B03', 'B02']).isel(time=1, x=slice(col_off, col_off+width), y=slice(row_off, row_off+height))
    rgb = rgb.clip(0, 2000) / 2000
    plt.figure(figsize=(10, 10))
    rgb.plot.imshow(x='x', y='y', rgb='band')
    plt.tight_layout()
    plt.savefig(f"/home/ksb781/data/gvs/deploy/EU_results/{tile}_{year}_rgb.png")
    plt.title(title)
    plt.close()
    
def vis_rhs(tile, year, rh_idx, window, z_exaggeration=2.0,
            camera=None, aspectratio=(1,1,0.25), cmin=None, cmax=None,
            orthographic=False, title=None):
    file_path = Path(f'~/data/gvs/deploy/predictions_{year}/{tile}_cog/RH{rh_idx}_Q1.cog.tif').expanduser()
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

    fig = go.Figure(data=[go.Surface(z=Z, x=X[0, :], y=Y[:, 0], cmin=cmin, cmax=cmax, 
                                     colorbar=dict(
                                    thickness=15,   # make bar thinner
                                    len=0.5,       # shorten (0–1 relative to plot height)
                                    y=0.5,          # center it vertically
                                    yanchor="middle"
    ))])

    cam = camera or dict(
        eye=dict(x=-1.2, y=-2.0, z=1.0),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
        projection=dict(type='orthographic' if orthographic else 'perspective')
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Longitude", range=[xmin, xmax]),
            yaxis=dict(title="Latitude",  range=[ymin, ymax]),
            zaxis=dict(title=f"RH{rh_idx} (dm)", range=[cmin, cmax]),
            aspectmode="manual",
            aspectratio=dict(x=aspectratio[0], y=aspectratio[1], z=aspectratio[2]),
            camera=cam,
        ),
        width=1200,
        height=1000,
        margin=dict(l=10, r=10, b=10, t=40),  # Adjusted margins for a tighter layout
        title=title
    )
    # fig.write_html(f"output/{tile}_{name}.html")
    fig.write_image(f"/home/ksb781/data/gvs/deploy/EU_results/{tile}_{name}.png")
    return fig

def make_gif(tile):
    from PIL import Image
    from pathlib import Path

    png_dir = Path("/home/ksb781/data/gvs/deploy/EU_results")
    # png_files = sorted(png_dir.glob(f"{tile}_*.png"))
    png_files = []
    for i in range(101):
        png_files.append(png_dir.glob(f"{tile}_RH{i}_*.png"))
    png_files = [file for sublist in png_files for file in sublist]

    frames = [Image.open(p) for p in png_files]

    frames[0].save(
        f"/home/ksb781/data/gvs/deploy/EU_results/{tile}_RH0-100_animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=60,  # ms per frame
        loop=0        # 0 = infinite loop
    )

# Define a window to read a small portion of the data
tiles = [
    # {'name': 'Dry deciduous forest', 'tile_id':  '42QXJ', 'col_off':8150, 'row_off':6620}, # dry deciduous
    # {'name': 'Evergreen forest', 'tile_id':  '19MDT', 'col_off':6244, 'row_off':8330}, # evergreen
    # {'name': 'Seasonal inundated forest', 'tile_id':  '20MKC', 'col_off':542, 'row_off':4793}, # flooded forest
    # {'name': 'Crop-/grassland', 'tile_id':  '17RML', 'col_off':3384, 'row_off':8622}, # Sorghum
    # {'name': 'Tree plantation', 'tile_id':  '49MFT', 'col_off':900, 'row_off':5800}, # plantation
    {'name': 'Plantation', 'tile_id':  '31TEJ', 'col_off':1122, 'row_off':1090, 'time': 15}, # plantation
    # {'name': 'Primary forest', 'tile_id':  '33WWN', 'col_off':5783, 'row_off':1564}, # primary forest # time=15
    # {'name': 'Naturally regenerating forest', 'tile_id':  '32TNQ', 'col_off':6885, 'row_off':7923}, # plantation # time=3
]
year = 2020
width = 100
height = 100

for tile_info in tiles:
    tile = tile_info['tile_id']
    col_off = tile_info['col_off']
    row_off = tile_info['row_off']
    name = tile_info['name']
    window = rio.windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)
    
    vis_rgb(tile, year, col_off, row_off, width, height, title=name)
    common_camera = dict(
         eye=dict(x=-0.8, y=-1.2, z=0.6), # eye=dict(x=-0.5, y=-1.5, z=0.8),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
      projection=dict(type="perspective"),
    
    )
    
    # figs = []
    # # for rh_idx in [98]:
    # for rh_idx in range(0,101):
    #     figs.append(
    #         vis_rhs(tile, year, rh_idx, window,
    #                 z_exaggeration=1,
    #                 camera=common_camera,
    #                 aspectratio=(1,1,0.4),
    #                 cmin=-50, cmax=300,
    #                 orthographic=True,
    #                 title=name)
    #     )
    # make_gif(tile)
    # print(f'animation generated for {tile}')