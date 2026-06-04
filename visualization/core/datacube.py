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


# Region presets. Each entry bundles:
#   file       — vector boundary (.gpkg/.shp/.geojson); None = no spatial mask
#   projection — pyproj-acceptable CRS string for plot_datacube
#   camera     — {shift_x, shift_y, zoom} for the oblique 3-D view; see
#                plot_datacube for what each knob does
REGIONS = {
    "global": {
        "file": None,
        # Equal Earth — global equal-area, wide-and-short framing.
        "projection": "+proj=eqearth",
        # pitch=1 keeps the legacy oblique view; z_spacing=6 keeps the cube
        # tall relative to its footprint.
        "camera": {
            "shift_x": 18.0, "shift_y": 5.0,
            "zoom": 1.2, "pitch": 1.0,
        },
        "z_spacing": 6.0,
        "aspect_ratio": 2.0,
        "scalar_bar": {
            "position_x": 0.04, "position_y": 0.6,
            "height": 0.22, "width": 0.03,
            "label_font_size": 72, "n_labels": 2,
        },
    },
    "europe": {
        "file": "~/data/gvs/results/eu_results/countries.gpkg",
        # ETRS89-extended / LAEA Europe — the official equal-area projection
        # for pan-European statistical and cartographic products.
        "projection": "EPSG:3035",
        # Lower pitch tilts the camera more top-down; smaller z_spacing
        # compresses the time-axis so the cube isn't a tall skinny column.
        # Iterate from here.
        "camera": {
            "shift_x": 60.0, "shift_y": 10.0,
            #  standard zoom; >1 closer, <1 wider.
            "zoom": 0.6, "pitch": 1, #  view tilt, 0..1. 1 = the legacy oblique south-east view, 0 = pure top-down
        },
        "z_spacing": 4.0, # how thick each RH layer is in the normalized cube
        "exclude": ["Cyprus"],
        "clip_bbox": [
            (-12.0, 30.0, 50.0, 75.0),   # mainland Europe + UK + Scandinavia
            (-25.0, 62.0, -12.0, 67.0),  # Iceland
        ],
        "aspect_ratio": 1.7777777778,
        # Colorbar disabled — Europe panels are typically shown without one.
        # Set this to a dict (same keys as the global preset) to bring it back.
        "scalar_bar": False,
    },
}


def _load_region_geometry(
    region: str,
    region_file: str = None,
    exclude: list = None,
    clip_bbox: tuple = None,
):
    """Return a shapely (Multi)Polygon for `region` in EPSG:4326.

    `region_file` overrides the preset's `file`; `exclude` overrides the
    preset's exclude list. Excluded names are matched case-insensitively
    against the first name-like column present in the vector file.
    `clip_bbox=(lon_min, lat_min, lon_max, lat_max)` further trims the
    geometry (applied after exclude).
    """
    spec = REGIONS[region]
    path = Path(region_file or spec["file"]).expanduser()
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"region file {path} has no CRS")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    exclude = exclude if exclude is not None else spec.get("exclude", [])
    if exclude:
        candidate_cols = [
            "NAME", "name", "NAME_EN", "NAME_LONG",
            "ADMIN", "admin", "CNTR_NAME", "country", "Country",
        ]
        name_col = next((c for c in candidate_cols if c in gdf.columns), None)
        if name_col is None:
            print(
                f"WARNING: 'exclude'={exclude} set but no name column found "
                f"in {path}; available columns: {list(gdf.columns)}"
            )
        else:
            lower_exclude = {e.casefold() for e in exclude}
            keep = ~gdf[name_col].astype(str).str.casefold().isin(lower_exclude)
            dropped = int((~keep).sum())
            gdf = gdf[keep]
            print(
                f"Dropped {dropped} feature(s) matching {exclude} on "
                f"column '{name_col}'"
            )

    geom = gdf.union_all()

    clip_bbox = clip_bbox if clip_bbox is not None else spec.get("clip_bbox")
    if clip_bbox is not None:
        from shapely.geometry import box
        from shapely.ops import unary_union
        # Accept either a single bbox or a list of bboxes (union them).
        bboxes = (
            [tuple(b) for b in clip_bbox]
            if hasattr(clip_bbox[0], "__iter__")
            else [tuple(clip_bbox)]
        )
        clip_shape = unary_union([box(*bb) for bb in bboxes])
        before_bounds = geom.bounds
        geom = geom.intersection(clip_shape)
        print(
            f"Clipped region geometry to {len(bboxes)} bbox(es) {bboxes} "
            f"(bounds {before_bounds} -> {geom.bounds})"
        )
    return geom


def _mask_data_to_region(
    data, lons, lats, region: str,
    region_file: str = None,
    exclude: list = None,
    clip_bbox: tuple = None,
):
    """Crop and mask `data` to the exact shape of `region`.

    Returns (data, lons, lats) where pixels outside the region polygon are NaN
    and the arrays are cropped to the region's bounding box.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    geom = _load_region_geometry(
        region, region_file=region_file, exclude=exclude, clip_bbox=clip_bbox,
    )

    if np.nanmax(lons) > 180:
        lons = ((lons + 180) % 360) - 180
        order = np.argsort(lons)
        lons = lons[order]
        data = data[:, order, :]

    lon_min, lat_min, lon_max, lat_max = geom.bounds
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    if not lon_mask.any() or not lat_mask.any():
        raise ValueError(
            f"region '{region}' bounds {geom.bounds} do not overlap data extent"
        )
    lons_c = lons[lon_mask]
    lats_c = lats[lat_mask]
    data_c = data[lat_mask][:, lon_mask, :]

    # rasterio wants a north-up transform. Flip to descending lats temporarily
    # if needed, then flip back so the existing plot logic still sees the
    # original orientation.
    flip_lat = lats_c[0] < lats_c[-1]
    if flip_lat:
        lats_for_tx = lats_c[::-1]
        data_c = data_c[::-1, :, :]
    else:
        lats_for_tx = lats_c

    nx = len(lons_c)
    ny = len(lats_for_tx)
    dlon = (lons_c[-1] - lons_c[0]) / max(nx - 1, 1)
    dlat = (lats_for_tx[0] - lats_for_tx[-1]) / max(ny - 1, 1)
    west = lons_c[0] - dlon / 2.0
    north = lats_for_tx[0] + dlat / 2.0
    transform = from_origin(west, north, dlon, dlat)

    mask = rasterize(
        [(geom, 1)],
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    data_c[~mask] = np.nan

    if flip_lat:
        data_c = data_c[::-1, :, :]

    return data_c, lons_c, lats_c


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
    projection: str = "+proj=eqearth",
    camera_shift_x: float = 18.0,
    camera_shift_y: float = 5.0,
    camera_zoom: float = 1.2,
    camera_pitch: float = 1.0,
    z_spacing: float = 6.0,
    scalar_bar: dict = None,
):
    """Render a 3-D datacube as a volume and data-coverage boundary outlines
    on the base plane.

    Parameters
    ----------
    aspect_ratio : float or None
        Target width/height ratio for the final image.  e.g. 2.5 means
        the image will be 2.5× wider than tall.  None keeps the original
        proportions after auto-crop.
    projection : str
        Any CRS string accepted by pyproj (EPSG code, proj string, WKT).
        Defaults to Equal Earth — global equal-area. For Europe use
        "EPSG:3035" (ETRS89 LAEA).
    camera_shift_x, camera_shift_y : float
        Horizontal / "into-the-scene" offsets applied to both the eye and
        focal point, in the [0, 100] normalised projected-coord units.
        +shift_x moves eye east, +shift_y lifts the eye toward the focal.
    camera_zoom : float
        Parallel-projection zoom factor. >1 zooms in, <1 zooms out.
    camera_pitch : float
        View tilt in [0, 1]. 1 = the oblique south-east view (legacy global
        framing); 0 = pure top-down. Intermediate values blend the two.
    z_spacing : float
        Spacing between RH layers in the normalised projected grid. The xy
        extent is normalised to [0, 100], so this is also the unit per layer
        on the time axis. Smaller = shorter cube relative to its footprint.
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

    # ── 5. Project lon/lat → target CRS ─────────────────────────────────
    transformer = Transformer.from_crs(
        "EPSG:4326", projection, always_xy=True
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

    # GPU volume rendering uploads the grid as a 3-D texture, which OpenGL caps
    # at MAX_3D_TEXTURE_SIZE per axis (2048 is the spec-guaranteed minimum and
    # the limit on many GPUs). Decouple the render grid from the data grid: read
    # the data at full overview detail (so the boundary contours stay sharp),
    # but resample the volume onto a uniform grid capped at MAX_TEXTURE_DIM per
    # axis so the texture stays valid. Raise this if your GPU reports a larger
    # MAX_3D_TEXTURE_SIZE.
    MAX_TEXTURE_DIM = 2048
    gx = min(nx, MAX_TEXTURE_DIM)
    gy = min(ny, MAX_TEXTURE_DIM)
    gz = min(nz, MAX_TEXTURE_DIM)
    if (gx, gy, gz) != (nx, ny, nz):
        print(
            f"Capping render grid {(nx, ny, nz)} -> {(gx, gy, gz)} "
            f"(MAX_3D_TEXTURE_SIZE={MAX_TEXTURE_DIM})"
        )

    uniform = pv.ImageData()
    uniform.dimensions = (gx, gy, gz)
    uniform.origin = (x_min_n, y_min_n, 0.0)
    uniform.spacing = (
        (x_max_n - x_min_n) / max(gx - 1, 1),
        (y_max_n - y_min_n) / max(gy - 1, 1),
        z_spacing,
    )

    resampled = uniform.sample(src_grid)

    # Fallback: if resampling gave empty data, assign directly. Only valid when
    # the render grid matches the source grid (no capping), since the raw values
    # array is sized nx*ny*nz.
    vals = resampled.point_data.get("values")
    if vals is None or np.all(vals == 0):
        if (gx, gy, gz) == (nx, ny, nz):
            print(
                "WARNING: resampling produced no data — falling back to "
                "direct assignment."
            )
            uniform.point_data["values"] = vol_vtk.ravel(order="F")
            resampled = uniform
        else:
            print(
                "WARNING: resampling produced no data on the capped render "
                "grid; rendering may be empty."
            )

    print(
        f"Grid bounds: {resampled.bounds}, "
        f"value range: {resampled.point_data['values'].min():.2f} "
        f".. {resampled.point_data['values'].max():.2f}"
    )

    # ── 9. Plot ─────────────────────────────────────────────────────────
    p = pv.Plotter(off_screen=True, window_size=(4400, 2400))
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

    # ── Scalar-bar layout (single source of truth) ────────────────────
    # Defaults match the legacy global framing; override per-region via
    # REGIONS[region]["scalar_bar"] or pass `scalar_bar=...` directly.
    # Pass `scalar_bar=False` to suppress the colorbar entirely.
    if scalar_bar is not False:
        sbar = {
            "position_x": 0.04,
            "position_y": 0.6,
            "height": 0.22,
            "width": 0.03,
            "label_font_size": 72,
            "n_labels": 2,
        }
        if scalar_bar:
            sbar.update(scalar_bar)

        # Dummy mesh carries the scalar bar; the volume itself has it off.
        dummy = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        dummy["values"] = np.array([0.0, vmax])
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
                "position_x": sbar["position_x"],
                "position_y": sbar["position_y"],
                "height": sbar["height"],
                "width": sbar["width"],
                "label_font_size": sbar["label_font_size"],
                "n_labels": sbar["n_labels"],
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

    shift_x = camera_shift_x
    shift_y = camera_shift_y
    pitch = float(np.clip(camera_pitch, 0.0, 1.0))

    # Two camera anchors blended by `pitch`:
    #   pitch=1 -> oblique south-east view (legacy global framing)
    #   pitch=0 -> pure top-down (eye directly above the focal)
    eye_y_obl = by0 + diag * 0.02 + shift_y
    eye_z_obl = bz1 + diag * 0.1
    eye_y_top = cy - 0.1 + shift_y
    eye_z_top = bz1 + diag * 1.5

    eye = (
        cx + shift_x,
        pitch * eye_y_obl + (1 - pitch) * eye_y_top,
        pitch * eye_z_obl + (1 - pitch) * eye_z_top,
    )
    focal = (cx - 0.5 + shift_x, cy - 0.1 + shift_y, cz)
    # View-up rotates from world-+z (oblique) to world-+y (top-down) so the
    # north of the map stays "up" in the image as we tilt down.
    view_up = (0.0, 1.0 - pitch, pitch)

    p.camera_position = [eye, focal, view_up]
    p.enable_parallel_projection()
    p.camera.zoom(camera_zoom)

    p.set_background(bg_color)
    p.show(auto_close=False)
    img = p.screenshot(scale=1,transparent_background=False, return_img=True)
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

    plt.imsave(save_path, img, dpi=300)


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
    region: str = None,
    region_file: str = None,
    exclude: list = None,
    clip_bbox: tuple = None,
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
    if region is not None and region != "global":
        data, lons, lats = _mask_data_to_region(
            data, lons, lats, region,
            region_file=region_file,
            exclude=exclude,
            clip_bbox=clip_bbox,
        )
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
    cmap: str = "inferno",
    rh_step: int = 2,
    overview_level: int = 3,
    bg_color: str = "black",
    show_top_boundary: bool = True,
    min_contour_points: int = 50,
    aspect_ratio: float = None,
    region: str = None,
    region_file: str = None,
    exclude: list = None,
    clip_bbox: tuple = None,
    projection: str = None,
    camera_shift_x: float = None,
    camera_shift_y: float = None,
    camera_zoom: float = None,
    camera_pitch: float = None,
    z_spacing: float = None,
    scalar_bar: dict = None,
    **kwargs,
):
    data, lons, lats = read_datacube(
        data_dir,
        filename_pattern=filename_pattern,
        rh_step=rh_step,
        overview_level=overview_level,
        region=region,
        region_file=region_file,
        exclude=exclude,
        clip_bbox=clip_bbox,
    )
    # Resolve from REGIONS preset: explicit arg > region preset > global preset.
    preset = REGIONS.get(region, REGIONS["global"])
    if projection is None:
        projection = preset.get("projection", REGIONS["global"]["projection"])
    cam = preset.get("camera", REGIONS["global"]["camera"])
    if camera_shift_x is None:
        camera_shift_x = cam["shift_x"]
    if camera_shift_y is None:
        camera_shift_y = cam["shift_y"]
    if camera_zoom is None:
        camera_zoom = cam["zoom"]
    if camera_pitch is None:
        camera_pitch = cam.get("pitch", REGIONS["global"]["camera"]["pitch"])
    if z_spacing is None:
        z_spacing = preset.get("z_spacing", REGIONS["global"]["z_spacing"])
    if aspect_ratio is None:
        aspect_ratio = preset.get("aspect_ratio")
    if scalar_bar is None:
        scalar_bar = preset.get("scalar_bar", REGIONS["global"].get("scalar_bar"))

    save_path = Path(save_path).expanduser()
    print(f"Read datacube with shape {data.shape}, lons {lons.shape}, lats {lats.shape}")
    print(f"Projecting with CRS: {projection}")
    print(
        f"Camera: shift_x={camera_shift_x}, shift_y={camera_shift_y}, "
        f"zoom={camera_zoom}, pitch={camera_pitch}, z_spacing={z_spacing}, "
        f"aspect_ratio={aspect_ratio}"
    )
    plot_datacube(
        data, lons, lats, save_path, cmap=cmap, bg_color=bg_color,
        show_top_boundary=show_top_boundary,
        min_contour_points=min_contour_points,
        aspect_ratio=aspect_ratio,
        projection=projection,
        camera_shift_x=camera_shift_x,
        camera_shift_y=camera_shift_y,
        camera_zoom=camera_zoom,
        camera_pitch=camera_pitch,
        z_spacing=z_spacing,
        scalar_bar=scalar_bar,
    )


if __name__ == "__main__":
    data_dir = "~/data/gvs/products/prediction_intervals/2020/masked/mosaic/"
    save_path = "/projects/dereeco/data/gvs/results/vsm_datacube/every2rhs_black_bg_v2.png"
    visualize_datacube(data_dir, save_path, cmap="viridis")