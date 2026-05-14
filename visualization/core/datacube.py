import os

import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import pystac
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS, Transformer
import stackstac
import datetime
from dask.utils import natural_sort_key
import rasterio
import pyvista as pv
import numpy as np

from download.core.utils import get_epsg_from_tile

pv.OFF_SCREEN = True


def plot_datacube(
    data: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    save_path: Path,
    cmap: str = "inferno",
    bg_color: str = "black",
    show_top_boundary: bool = False,
    min_contour_points: int = 50,
    aspect_ratio: float = None,
):
    """Render a 3-D datacube as a volume with Equal Earth projection
    and data-coverage boundary outlines on the base plane.

    Parameters
    ----------
    aspect_ratio : float or None
        Target width/height ratio for the final image.  e.g. 2.5 means
        the image will be 2.5× wider than tall.  None keeps the original
        proportions after auto-crop.
    """

    # Boundary / text colour: contrast with background
    line_color = "white" if bg_color == "black" else "black"

    # ── 1. latitude ascending ───────────────────────────────────────────
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[::-1, :, :]

    # ── 2. longitude: convert 0…360 → -180…180, then sort ──────────────
    if np.nanmax(lons) > 180:
        lons = ((lons + 180) % 360) - 180
        order = np.argsort(lons)
        lons = lons[order]
        data = data[:, order, :]

    data = data / 10
    ny, nx, nz = data.shape

    # ── 3. Footprint boundary polylines (lon/lat, before projection) ────
    #    Top boundary  = footprint of the last z-layer  (top face)
    #    Bottom boundary = footprint of the first z-layer (base face)
    def _contours_to_lonlat(footprint_2d, lons, lats, nx, ny,
                            min_pts=50):
        """Extract boundary polylines in lon/lat from a 2-D boolean mask.
        Contours shorter than *min_pts* vertices are discarded."""
        from skimage import measure

        fp = np.pad(footprint_2d.astype(np.float32), 1,
                     mode="constant", constant_values=0.0)
        raw = measure.find_contours(fp, 0.5)
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())
        dlon = (lon_max - lon_min) / max(nx - 1, 1)
        dlat = (lat_max - lat_min) / max(ny - 1, 1)
        polys = []
        for c in raw:
            if len(c) < min_pts:
                continue
            rows = c[:, 0] - 1
            cols = c[:, 1] - 1
            lon_pts = lon_min + cols * dlon
            lat_pts = lat_min + rows * dlat
            polys.append(np.column_stack([lon_pts, lat_pts]))
        return polys

    try:
        bottom_footprint = (~np.isnan(data[:, :, 0]))
        boundary_bot_lonlat = _contours_to_lonlat(
            bottom_footprint, lons, lats, nx, ny, min_pts=min_contour_points)
        if show_top_boundary:
            top_footprint = (~np.isnan(data[:, :, -1]))
            boundary_top_lonlat = _contours_to_lonlat(
                top_footprint, lons, lats, nx, ny, min_pts=min_contour_points)
        else:
            boundary_top_lonlat = []
    except ImportError:
        boundary_top_lonlat = []
        boundary_bot_lonlat = []

    vmin = 0
    vmax = 50
    print("vmin, vmax:", vmin, vmax)

    # ── 4. NaN → sentinel, clip ─────────────────────────────────────────
    sentinel = -1
    opacity = [0.0, 0.8, 0.8, 0.85, 0.9, 1.0]
    vol = np.nan_to_num(data, nan=sentinel)
    vol = np.clip(vol, sentinel, vmax)

    # ── 5. Project lon/lat → Equal Earth ────────────────────────────────
    transformer = Transformer.from_crs(
        "EPSG:4326", "+proj=eqearth", always_xy=True
    )

    lon2d, lat2d = np.meshgrid(lons, lats)  # (ny, nx)
    x2d, y2d = transformer.transform(lon2d, lat2d)  # metres

    # Normalise to a compact range so VTK spacing is reasonable
    x_min, x_max = float(x2d.min()), float(x2d.max())
    y_min, y_max = float(y2d.min()), float(y2d.max())
    x_range = x_max - x_min if (x_max - x_min) > 0 else 1.0
    y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0
    max_range = max(x_range, y_range)
    scale = 100.0 / max_range

    x2d_n = (x2d - x_min) * scale  # (ny, nx), range ~[0, 100]
    y2d_n = (y2d - y_min) * scale

    z_spacing = 6.0

    # ── 6. Project boundary polylines into the same normalised space ────
    z_top = (nz - 1) * z_spacing  # top layer of the volume

    def _project_polylines(lonlat_list, z_val):
        out = []
        for bl in lonlat_list:
            bx, by = transformer.transform(bl[:, 0], bl[:, 1])
            bx_n = (bx - x_min) * scale
            by_n = (by - y_min) * scale
            bz = np.full_like(bx_n, z_val)
            out.append(np.column_stack([bx_n, by_n, bz]))
        return out

    boundary_top_3d = _project_polylines(boundary_top_lonlat, z_top)
    boundary_bot_3d = _project_polylines(boundary_bot_lonlat, 0.0)

    # ── 7. Build StructuredGrid with projected + normalised coords ──────
    x3d = np.repeat(x2d_n[:, :, np.newaxis], nz, axis=2)
    y3d = np.repeat(y2d_n[:, :, np.newaxis], nz, axis=2)
    z_layers = np.arange(nz, dtype=np.float64) * z_spacing
    z3d = np.broadcast_to(
        z_layers[np.newaxis, np.newaxis, :], (ny, nx, nz)
    ).copy()

    # VTK order: (x=lon, y=lat, z=time)
    src_grid = pv.StructuredGrid(
        np.ascontiguousarray(np.transpose(x3d, (1, 0, 2))),
        np.ascontiguousarray(np.transpose(y3d, (1, 0, 2))),
        np.ascontiguousarray(np.transpose(z3d, (1, 0, 2))),
    )
    src_grid.dimensions = (nx, ny, nz)

    vol_vtk = np.transpose(vol, (1, 0, 2))  # (nx, ny, nz)
    src_grid.point_data["values"] = vol_vtk.ravel(order="F")

    # ── 8. Resample onto uniform ImageData (needed by add_volume) ───────
    x_min_n = float(x2d_n.min())
    x_max_n = float(x2d_n.max())
    y_min_n = float(y2d_n.min())
    y_max_n = float(y2d_n.max())

    uniform = pv.ImageData()
    uniform.dimensions = (nx, ny, nz)
    uniform.origin = (x_min_n, y_min_n, 0.0)
    uniform.spacing = (
        (x_max_n - x_min_n) / max(nx - 1, 1),
        (y_max_n - y_min_n) / max(ny - 1, 1),
        z_spacing,
    )

    resampled = uniform.sample(src_grid)

    # Fallback: if resampling gave empty data, assign directly
    vals = resampled.point_data.get("values")
    if vals is None or np.all(vals == 0):
        print(
            "WARNING: resampling produced no data — falling back to "
            "direct assignment."
        )
        uniform.point_data["values"] = vol_vtk.ravel(order="F")
        resampled = uniform

    print(
        f"Grid bounds: {resampled.bounds}, "
        f"value range: {resampled.point_data['values'].min():.2f} "
        f".. {resampled.point_data['values'].max():.2f}"
    )

    # ── 9. Plot ─────────────────────────────────────────────────────────
    p = pv.Plotter(off_screen=True, window_size=(2200, 1200))
    p.enable_parallel_projection()

    p.add_volume(
        resampled,
        scalars="values",
        cmap=cmap,
        clim=(sentinel, vmax),
        opacity=opacity,
        shade=False,
        show_scalar_bar=False,
    )

    # Dummy mesh for a clean scalar bar
    dummy = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    dummy["values"] = np.array([0.0, vmax])
    # ── Scalar-bar layout (single source of truth) ────────────────────
    sbar_x = 0.04           # left edge (normalised viewport)
    sbar_y = 0.6          # bottom edge
    sbar_h = 0.22          # height
    sbar_w = 0.03          # width
    sbar_title_gap = 0.02  # gap between bar top and title

    p.add_mesh(
        dummy,
        scalars="values",
        cmap=cmap,
        clim=(0, vmax),
        show_scalar_bar=True,
        opacity=0.0,
        scalar_bar_args={
            "title": "",
            "color": line_color,
            "vertical": True,
            "position_x": sbar_x,
            "position_y": sbar_y,
            "height": sbar_h,
            "width": sbar_w,
            "label_font_size": 24,
            "n_labels": 2,
            "fmt": "%.0f",
        },
    )

    # # Manually place the colorbar title, left-aligned with the bar
    # import vtk
    # title_actor = vtk.vtkTextActor()
    # title_actor.SetInput("Height [m]")
    # tp = title_actor.GetTextProperty()
    # tp.SetFontSize(30)
    # tp.SetJustificationToLeft()
    # tp.SetVerticalJustificationToBottom()
    # if line_color == "white":
    #     tp.SetColor(1, 1, 1)
    # else:
    #     tp.SetColor(0, 0, 0)
    # title_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    # title_actor.GetPositionCoordinate().SetValue(
    #     sbar_x, sbar_y + sbar_h + sbar_title_gap
    # )
    # p.renderer.AddActor2D(title_actor)

    # ── 10. Draw boundary outlines ─────────────────────────────────────
    for pts in boundary_top_3d:
        if len(pts) < 2:
            continue
        line = pv.lines_from_points(pts, close=False)
        p.add_mesh(line, color=line_color, line_width=2)

    for pts in boundary_bot_3d:
        if len(pts) < 2:
            continue
        line = pv.lines_from_points(pts, close=False)
        p.add_mesh(line, color=line_color, line_width=2)

    # ── 11. Camera ──────────────────────────────────────────────────────
    bounds = resampled.bounds
    bx0, bx1, by0, by1, bz0, bz1 = bounds

    cx = 0.5 * (bx0 + bx1)
    cy = 0.5 * (by0 + by1)
    cz = 0.5 * (bz0 + bz1)

    diag = np.sqrt((bx1 - bx0) ** 2 + (by1 - by0) ** 2 + (bz1 - bz0) ** 2)
    print(f"Grid diagonal: {diag:.2f}")

    shift_x = 18.0
    shift_y = 5.0

    p.camera_position = [
        (cx + shift_x, by0 + diag * 0.02 + shift_y, bz1 + diag * 0.1),  # eye
        (cx - 0.5 + shift_x, cy - 0.1 + shift_y, cz),  # focal point
        (0, 0, 1),  # view-up
    ]
    p.enable_parallel_projection()
    p.camera.zoom(1.2)

    p.set_background(bg_color)
    p.show(auto_close=False)
    img = p.screenshot(transparent_background=False, return_img=True)
    p.close()

    # ── 12. Auto-crop + padding ─────────────────────────────────────────
    bg_tol = 6
    pad_value = 255 if bg_color == "white" else 0
    if bg_color == "white":
        mask = img.min(axis=2) < (255 - bg_tol)
    else:
        mask = img.max(axis=2) > bg_tol
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        img = img[r0:r1, c0:c1]

    pad = 30
    img = np.pad(
        img,
        pad_width=((pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )

    # ── 13. Resize to target aspect ratio (width / height) ──────────────
    if aspect_ratio is not None:
        from PIL import Image as PILImage
        h, w = img.shape[:2]
        current_ratio = w / h
        if current_ratio < aspect_ratio:
            # too tall → shrink height, keep width
            new_h = int(round(w / aspect_ratio))
            new_w = w
        else:
            # too wide → shrink width, keep height
            new_w = int(round(h * aspect_ratio))
            new_h = h
        pil_img = PILImage.fromarray(img)
        img = np.array(pil_img.resize((new_w, new_h), PILImage.LANCZOS))

    plt.imsave(save_path, img)


# ═══════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════


def read_overview(file: str, overview_level: int = 4):
    with rasterio.open(file, overview_level=overview_level) as src:
        data = src.read(1)
    return data


def read_coords(file: str, overview_level: int = 4):
    with rasterio.open(file, overview_level=overview_level) as src:
        transform = src.transform
        cols = np.arange(src.width)
        rows = np.arange(src.height)
        lons = transform[2] + cols * transform[0]
        lats = transform[5] + rows * transform[4]
        nodata = src.nodata
    return lons, lats, nodata


def read_datacube(
    data_dir: str,
    filename_pattern: str = "*.tif",
    overview_level: int = 4,
    rh_step: int = 2,
):
    data_dir = Path(data_dir).expanduser()
    files = list(data_dir.glob(filename_pattern))
    files = [str(file) for file in files]
    files = sorted(files, key=natural_sort_key)
    data = []
    for file in files[::rh_step]:
        data_i = read_overview(file, overview_level=overview_level)
        data.append(data_i[:, :, None])
    data = np.concatenate(data, axis=2)
    lons, lats, nodata = read_coords(files[0], overview_level=overview_level)
    data = data.astype(np.float32)
    data[data == nodata] = np.nan
    return data, lons, lats


def get_patch_by_latlon(
    lat,
    lon,
    s2_grid: gpd.GeoDataFrame = None,
    year: int = 2020,
    buffer_km=3,
    q_idx=1,
):
    points = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    matched = gpd.sjoin(points, s2_grid, predicate="within", how="left")
    matched = matched.sort_values(f"type_{year}", ascending=False).iloc[0]
    tile = s2_grid[s2_grid["Name"] == matched["Name"]].iloc[0]
    tile_id = matched["Name"]
    epsg = get_epsg_from_tile(tile_id)

    tile_crs = CRS.from_epsg(epsg)
    points_utm = points.to_crs(tile_crs)

    buffer_m = (buffer_km / 2) * 1000
    points_buffer = points_utm.buffer(buffer_m)

    pred_dir = Path(
        f"~/data/gvs/deploy/predictions_{year}/{tile_id}_cog"
    ).expanduser()
    tif_files = list(pred_dir.glob(f"*_Q{q_idx}.cog.tif"))

    item = pystac.Item(
        id=tile_id,
        geometry=tile.geometry,
        bbox=tile.geometry.bounds,
        datetime=datetime.datetime(year, 1, 1, 0, 0, 0),
        properties={},
    )
    for f in tif_files:
        item.add_asset(
            f.stem, pystac.Asset(href=str(f), media_type=pystac.MediaType.COG)
        )

    image = stackstac.stack(
        item,
        epsg=epsg,
        resolution=10,
        bounds=points_buffer.total_bounds.tolist(),
    )
    return image.squeeze()


def visualize_datacube(
    save_path: str = None,
    data_dir: str = None,
    filename_pattern: str = "*.tif",
    cmap: str = "viridis",
    rh_step: int = 2,
    bg_color: str = "black",
    show_top_boundary: bool = False,
    min_contour_points: int = 50,
    aspect_ratio: float = None,
    **kwargs,
):
    data, lons, lats = read_datacube(
        data_dir, filename_pattern=filename_pattern, rh_step=rh_step
    )
    save_path = Path(save_path).expanduser()
    print(f"Read datacube with shape {data.shape}, lons {lons.shape}, lats {lats.shape}")
    plot_datacube(
        data, lons, lats, save_path, cmap=cmap, bg_color=bg_color,
        show_top_boundary=show_top_boundary,
        min_contour_points=min_contour_points,
        aspect_ratio=aspect_ratio,
    )


if __name__ == "__main__":
    data_dir = "~/data/gvs/products/prediction_intervals/2020/masked/mosaic/"
    save_path = "/projects/dereeco/data/gvs/results/vsm_datacube/every2rhs_black_bg_v2.png"
    visualize_datacube(data_dir, save_path, cmap="viridis")