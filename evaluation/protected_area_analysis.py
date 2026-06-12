"""
Protected Area Structural Integrity Analysis
=============================================
Compare FHD, ENL, RH98 between:
  - Primary forests within protected areas (reference baseline)
  - All forests within protected areas
  - All forests outside protected areas

Stratified by WWF biome.
"""

import shutil
import subprocess
import warnings
from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from scipy import stats

from const import BIOMES_BY_VALUE

warnings.filterwarnings("ignore")


# GDAL dtype names for the gdal_rasterize CLI. Cover the dtypes we'd realistically
# burn (binary masks, biome IDs, occasional float layers); anything else can be
# added on demand.
_GDAL_DTYPE = {
    "uint8": "Byte", "uint16": "UInt16", "int16": "Int16",
    "int32": "Int32", "uint32": "UInt32",
    "float32": "Float32", "float64": "Float64",
}


def _gdal_rasterize_to_cache(vector_path, profile, cache_file, attribute=None,
                             fill=0, dtype="uint8", layer=None):
    """Stream `vector_path` to a GeoTIFF at `cache_file` via the gdal_rasterize
    CLI. Used instead of geopandas + rasterio.features.rasterize because the
    latter materialises every feature as a shapely object in RAM — a 4.6 GB /
    268k-polygon gpkg blows past 32 GB before the rasterize call even starts.
    gdal_rasterize reads features in batches from OGR and writes blocks of the
    output as it goes, so peak RSS stays in the low GB regardless of input size.

    gdal_rasterize does NOT reproject vectors, so the source CRS must already
    match the target raster CRS. We check up front via fiona (metadata-only,
    no geometry read) and fail loudly with an `ogr2ogr` command the caller can
    run once if the CRSs disagree.
    """
    if not shutil.which("gdal_rasterize"):
        raise RuntimeError(
            "gdal_rasterize not on PATH. It ships with GDAL; install GDAL or "
            "pass `use_gdal_cli=False` to fall back to the in-memory path "
            "(which OOMs on multi-GB vectors).")

    with fiona.open(vector_path, layer=layer) as src:
        src_crs_dict = src.crs
    from pyproj import CRS as _PyCRS
    src_crs = _PyCRS.from_user_input(src_crs_dict) if src_crs_dict else None
    tgt_crs = profile["crs"]
    src_epsg = src_crs.to_epsg() if src_crs else None
    tgt_epsg = tgt_crs.to_epsg() if hasattr(tgt_crs, "to_epsg") else None

    # gdal_rasterize doesn't reproject. If the vector is in a different CRS we
    # pay a one-time ogr2ogr reprojection (also streamed; bounded memory) and
    # cache the reprojected gpkg next to the raster cache so subsequent cache
    # rebuilds reuse it instead of reprojecting again.
    cache_file = Path(cache_file).expanduser()
    if src_epsg and tgt_epsg and src_epsg != tgt_epsg:
        if not shutil.which("ogr2ogr"):
            raise RuntimeError("ogr2ogr not on PATH; install GDAL.")
        reproj_path = (cache_file.parent /
                       f"{Path(vector_path).stem}.reprojected_to_{tgt_epsg}.gpkg")
        if not reproj_path.exists():
            reproj_path.parent.mkdir(parents=True, exist_ok=True)
            ogr_cmd = [
                "ogr2ogr",
                "-t_srs", f"EPSG:{tgt_epsg}",
                "-f", "GPKG",
                str(reproj_path), str(vector_path),
            ]
            if layer:
                ogr_cmd += [layer]
            print(f"  reprojecting EPSG:{src_epsg} -> EPSG:{tgt_epsg}")
            print(f"  $ {' '.join(ogr_cmd)}")
            subprocess.run(ogr_cmd, check=True)
        else:
            print(f"  reusing reprojected vector: {reproj_path}")
        # ogr2ogr preserves the source layer name in the output gpkg, so the
        # caller's `layer` arg keeps working. For a single-layer shp input
        # `layer` was None and stays None — gdal_rasterize picks the only
        # layer in the reprojected gpkg automatically.
        vector_path = reproj_path

    t = profile["transform"]
    xres, yres = t.a, -t.e  # transform.e is negative for north-up rasters
    xmin, ymax = t.c, t.f
    xmax = xmin + t.a * profile["width"]
    ymin = ymax + t.e * profile["height"]

    gdal_dtype = _GDAL_DTYPE.get(str(dtype))
    if gdal_dtype is None:
        raise ValueError(f"unsupported dtype for gdal_rasterize: {dtype!r}")

    cache_file = Path(cache_file).expanduser()
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gdal_rasterize",
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-tr", str(xres), str(yres),
        "-init", str(fill),
        "-a_nodata", str(fill),
        "-ot", gdal_dtype,
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
    ]
    cmd += (["-a", attribute] if attribute else ["-burn", "1"])
    if layer:
        cmd += ["-l", layer]
    cmd += [str(vector_path), str(cache_file)]

    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_raster(path, band=1):
    """Load one band of a GeoTIFF, return array, profile, and transform.
    Nodata pixels are cast to float NaN so downstream masking is uniform."""
    with rasterio.open(path) as src:
        data = src.read(band)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data.astype(float))
    return data, profile, transform


def rasterize_vector(vector_path, profile, attribute=None, fill=0,
                     dtype="uint8", layer=None, cache_file=None,
                     use_gdal_cli=True):
    """Rasterize a vector file onto the raster grid defined by `profile`.

    If `attribute` is given, burn that attribute value; otherwise burn 1.
    `layer` picks a specific layer from multi-layer formats (e.g. GeoPackage).

    When `cache_file` is given:
      - if it exists, the rasterization is skipped and the array is read from
        the cached single-band GeoTIFF;
      - if it doesn't, the rasterization runs once and writes the cache so
        every subsequent run hits the fast path. The cache is grid-bound — its
        shape must match `profile`; delete it to rebuild on a new grid.

    For the build step, `use_gdal_cli=True` (default) shells out to the
    `gdal_rasterize` CLI, which streams features from disk and keeps RSS in
    the low GB. The in-memory `gpd.read_file` + `rasterio.features.rasterize`
    path is kept as a fallback for small vectors when no `cache_file` is given,
    but it OOMs on multi-GB / hundred-thousand-polygon inputs (peak ~3-5× the
    on-disk file size from shapely object overhead).
    """
    out_shape = (profile["height"], profile["width"])
    vector_path = Path(vector_path).expanduser()

    if cache_file is not None:
        cache_file = Path(cache_file).expanduser()
        if cache_file.exists():
            with rasterio.open(cache_file) as src:
                data = src.read(1)
            if data.shape != out_shape:
                raise ValueError(
                    f"cache {cache_file} has shape {data.shape}, target grid is "
                    f"{out_shape}; delete the cache to rebuild on the new grid."
                )
            print(f"  loaded rasterized cache: {cache_file}")
            return data

        if use_gdal_cli:
            _gdal_rasterize_to_cache(
                vector_path, profile, cache_file,
                attribute=attribute, fill=fill, dtype=dtype, layer=layer,
            )
            with rasterio.open(cache_file) as src:
                return src.read(1)

    # In-memory fallback: only safe for small vectors.
    print(f"  rasterizing {vector_path} in-memory...")
    gdf = gpd.read_file(vector_path, layer=layer) if layer else gpd.read_file(vector_path)
    if gdf.crs != profile["crs"]:
        gdf = gdf.to_crs(profile["crs"])
    if attribute:
        shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf[attribute])]
    else:
        shapes = [(geom, 1) for geom in gdf.geometry]
    out = rasterize(
        shapes, out_shape=out_shape, transform=profile["transform"],
        fill=fill, dtype=dtype,
    )
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_profile = {
            **profile, "count": 1, "dtype": dtype, "nodata": fill,
            "compress": "lzw", "tiled": True,
        }
        with rasterio.open(cache_file, "w", **cache_profile) as dst:
            dst.write(out, 1)
        print(f"  wrote rasterized cache: {cache_file}")
    return out


def sample_raster_at_points(raster_shape, transform, points_gdf):
    """Return (rows, cols, valid) of pixel coordinates for each point. `valid`
    masks points falling outside the raster footprint."""
    rows, cols = rowcol(transform, points_gdf.geometry.x, points_gdf.geometry.y)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    h, w = raster_shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return rows, cols, valid


# Shared per-metric colour palette: diversity metrics use the cool ramp,
# RH98 sits in the warm contrast so the "canopy height misses degradation"
# story reads at a glance.
_METRIC_COLORS = {
    "fhd": "#2166ac", "fhd_rel": "#1b7837",
    "enl": "#4393c3", "cr": "#92c5de", "rh98": "#d6604d",
}


def plot_biome_comparison(results_df, metrics, output_path):
    """Per-biome bar plot of Cohen's d (vs. primary forest) for each metric,
    split into protected vs. outside-protected panels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = [m.upper() for m in metrics]
    colors = [_METRIC_COLORS.get(m, "#888888") for m in metrics]

    for ax, group in zip(axes, ["in_pa", "outside_pa"]):
        group_data = results_df[results_df["group"] == group]
        pivot = group_data.pivot(
            index="biome", columns="metric", values="cohens_d"
        ).reindex(columns=metrics)
        if len(pivot) == 0:
            continue
        pivot.plot(kind="bar", ax=ax, width=0.8, color=colors)
        ax.set_ylabel("Cohen's d (vs. primary forest)")
        ax.set_xlabel("")
        title = "Protected areas" if group == "in_pa" else "Outside protected areas"
        ax.set_title(f"{title}: structural deviation from primary forest")
        ax.legend(labels)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_headline_bars(pcts: dict, output_path):
    """Bar chart of the % of PA forest pixels below biome-primary P10, one
    bar per metric. Takes an ordered dict so the column order matches
    `metrics` from the caller."""
    labels = [m.upper() for m in pcts]
    values = list(pcts.values())
    colors = [_METRIC_COLORS.get(m, "#888888") for m in pcts]

    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 1, 4))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylabel("% of PA forests below\nbiome primary forest P10")
    ax.set_title("Structural degradation in protected areas")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Headline plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main runnable
# ---------------------------------------------------------------------------
def analyze_protected_areas(
    diversity_file: str,
    rh98_file: str,
    wdpa_file: str,
    biomes_file: str,
    naturalness_file: str,
    save_dir: str,
    fhd_band: int = 1,
    enl_band: int = 3,
    cr_band: int = 4,
    bin_size: float = 5.0,
    biome_attribute: str = "BIOME_NUM",
    wdpa_layer: str = None,
    biomes_layer: str = None,
    wdpa_cache_file: str = None,
    biomes_cache_file: str = None,
    forest_biomes: list = None,
    forest_threshold: float = 5.0,
    naturalness_col: str = "naturalness",
    primary_forest_value: int = 1,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    naturalness_crs: str = "EPSG:4326",
    min_primary_samples: int = 100,
    **kwargs,
):
    """Compare FHD / ENL / RH98 between primary forest, protected forest, and
    unprotected forest, stratified by WWF biome.

    Parameters
    ----------
    diversity_file : Multi-band 1 km GeoTIFF holding the diversity metrics in
        band order [fhd, enl1d, enl2d, cr]. `fhd_band` and `enl_band` pick the
        two bands actually used by the analysis.
    rh98_file : 1 km single-band GeoTIFF of RH98, stored as int16 decimetres;
        converted to float metres in-memory. Must share the same grid as
        `diversity_file`.
    fhd_band, enl_band, cr_band : 1-indexed band numbers within
        `diversity_file`. Defaults: 1 (fhd), 3 (enl2d), 4 (cr). Use 2 for
        enl1d.
    bin_size : Height-bin size used when FHD was computed, in metres. Drives
        the `fhd_rel = FHD / ln(RH98 / bin_size)` normalisation, which puts
        FHD in [0, 1] by dividing out the height-dependent max entropy. Must
        match the `bin_width` used by `evaluation/diversity_maps.py`.
    wdpa_file : WDPA protected area boundaries (shapefile / geopackage).
    biomes_file : WWF biome polygons; `biome_attribute` is burned to the grid.
    naturalness_file : CSV of Lesiv et al. (2022) forest naturalness point
        annotations with `lon_col`, `lat_col`, and `naturalness_col` columns.
    save_dir : Output directory for `biome_results.csv` and the plots.
    biome_attribute : Vector attribute on `biomes_file` to burn as biome id.
    wdpa_layer, biomes_layer : Layer name to read from multi-layer GeoPackages
        (e.g. WDPA's `WDPA_poly_<release>`). Leave `None` for single-layer
        shapefiles.
    wdpa_cache_file, biomes_cache_file : Path to a single-band GeoTIFF cache of
        the rasterized layer. On first run the vector is reprojected and
        rasterized, then written to the cache; on subsequent runs the cache is
        loaded directly. Strongly recommended for big static layers (the WDPA
        gpkg is multi-GB / hundreds of thousands of polygons).
    forest_biomes : Biome ids to analyse. Defaults to the forested biomes in
        the WWF scheme (1, 2, 3, 4, 5, 6, 12, 14).
    forest_threshold : RH98 height (metres) above which a pixel counts as
        forest. Replaces a separate forest mask raster.
    naturalness_col : Column on `naturalness_file` carrying the naturalness
        class label.
    primary_forest_value : Value of `naturalness_col` flagging primary forest.
    lon_col, lat_col : Longitude / latitude columns of `naturalness_file`.
    naturalness_crs : CRS the lon/lat columns are in.
    min_primary_samples : Skip biomes with fewer primary-forest pixels.
    """
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    forest_biomes = forest_biomes or [1, 2, 3, 4, 5, 6, 12, 14]
    metrics = ["fhd", "fhd_rel", "enl", "cr", "rh98"]

    # --- 1. Load rasters -------------------------------------------------
    # FHD, ENL, CR come from the same 4-band diversity GeoTIFF
    # (band order: 1=fhd, 2=enl1d, 3=enl2d, 4=cr); RH98 is a separate file.
    print("Loading raster data...")
    diversity_file = Path(diversity_file).expanduser()
    rh98_file = Path(rh98_file).expanduser()
    fhd, profile, transform = load_raster(diversity_file, band=fhd_band)
    enl, _, _ = load_raster(diversity_file, band=enl_band)
    cr, _, _ = load_raster(diversity_file, band=cr_band)
    rh98, _, _ = load_raster(rh98_file)
    # rh98_file is stored as int16 in decimetres; convert to float metres so
    # the threshold and stats below are in real height units.
    rh98 = rh98.astype(float) / 10.0
    assert fhd.shape == enl.shape == cr.shape == rh98.shape, (
        f"Shape mismatch: FHD={fhd.shape}, ENL={enl.shape}, "
        f"CR={cr.shape}, RH98={rh98.shape}"
    )
    print(f"  Raster shape: {fhd.shape[0]} x {fhd.shape[1]}")

    # --- 2. Rasterize vector layers --------------------------------------
    print("Rasterizing WDPA protected areas...")
    protected_mask = rasterize_vector(
        wdpa_file, profile, layer=wdpa_layer, cache_file=wdpa_cache_file,
    )

    print("Rasterizing WWF biomes...")
    biome_raster = rasterize_vector(
        biomes_file, profile, attribute=biome_attribute, layer=biomes_layer,
        cache_file=biomes_cache_file,
    )

    print("Loading forest naturalness (point annotations)...")
    nat_df = pd.read_csv(naturalness_file)
    nat_gdf = gpd.GeoDataFrame(
        nat_df,
        geometry=gpd.points_from_xy(nat_df[lon_col], nat_df[lat_col]),
        crs=naturalness_crs,
    )
    if nat_gdf.crs != profile["crs"]:
        nat_gdf = nat_gdf.to_crs(profile["crs"])

    primary_pts = nat_gdf[nat_gdf[naturalness_col] == primary_forest_value].copy()
    print(f"  Total naturalness points: {len(nat_gdf):,}")
    print(f"  Primary forest points:    {len(primary_pts):,}")

    p_rows, p_cols, p_valid = sample_raster_at_points(fhd.shape, transform, primary_pts)
    primary_pixel_set = set(zip(p_rows[p_valid], p_cols[p_valid]))
    print(f"  Primary forest pixels (unique 1km cells): {len(primary_pixel_set):,}")

    primary_raster = np.zeros_like(fhd, dtype=bool)
    for r, c in primary_pixel_set:
        primary_raster[r, c] = True

    # --- 3. Build pixel-level dataframe ----------------------------------
    # Forest mask is derived from RH98 height instead of a separate raster:
    # pixels taller than `forest_threshold` metres count as forest.
    print("Building pixel dataframe...")
    is_forest = rh98 > forest_threshold
    valid = (is_forest & np.isfinite(fhd) & np.isfinite(enl)
             & np.isfinite(cr) & np.isfinite(rh98))
    rows, cols = np.where(valid)
    # Relative entropy: FHD / max possible FHD given canopy height. n_bins =
    # rh98 / bin_size; max entropy is ln(n_bins). Pixels with n_bins <= 1
    # have no diversity to measure (single bin) -> mark as NaN so the per-
    # biome nanmean/nanpercentile path skips them cleanly.
    n_bins = rh98[valid] / bin_size
    with np.errstate(divide="ignore", invalid="ignore"):
        fhd_rel_vals = np.where(n_bins > 1, fhd[valid] / np.log(n_bins), np.nan)
    df = pd.DataFrame({
        "row": rows,
        "col": cols,
        "fhd": fhd[valid],
        "fhd_rel": fhd_rel_vals,
        "enl": enl[valid],
        "cr": cr[valid],
        "rh98": rh98[valid],
        "biome": biome_raster[valid],
        "protected": protected_mask[valid].astype(bool),
        "is_primary": primary_raster[valid],
    })
    df["group"] = np.where(df["protected"], "in_pa", "outside_pa")

    print(f"  Total forest pixels:    {len(df):,}")
    print(f"  Protected:              {df['protected'].sum():,}")
    print(f"  Primary forest:         {df['is_primary'].sum():,}")
    print(f"  Primary + protected:    {(df['is_primary'] & df['protected']).sum():,}")

    # --- 4. Per-biome analysis ------------------------------------------
    print("\nPer-biome analysis:")
    print("=" * 80)
    results = []
    for biome_id in forest_biomes:
        biome_name = BIOMES_BY_VALUE.get(biome_id, {}).get("name", f"Biome {biome_id}")
        bdf = df[df["biome"] == biome_id]
        primary = bdf[bdf["is_primary"]]
        in_pa = bdf[bdf["group"] == "in_pa"]
        outside = bdf[bdf["group"] == "outside_pa"]

        if len(primary) < min_primary_samples:
            print(f"\n{biome_name}: skipped (only {len(primary)} primary samples)")
            continue

        print(f"\n{biome_name} (n_primary={len(primary):,}, "
              f"n_pa={len(in_pa):,}, n_outside={len(outside):,})")

        for metric in metrics:
            ref_values = primary[metric].values
            ref_mean = np.nanmean(ref_values)
            ref_std = np.nanstd(ref_values)
            ref_p10 = np.nanpercentile(ref_values, 10)

            for group_name, group_df in [("in_pa", in_pa), ("outside_pa", outside)]:
                grp_values = group_df[metric].values
                grp_mean = np.nanmean(grp_values)

                pooled_std = np.sqrt((ref_std**2 + np.nanstd(grp_values)**2) / 2)
                cohens_d = (ref_mean - grp_mean) / pooled_std if pooled_std > 0 else 0
                pct_below = np.nanmean(grp_values < ref_p10) * 100

                if len(grp_values) > 1 and len(ref_values) > 1:
                    t_stat, p_val = stats.ttest_ind(
                        ref_values, grp_values, equal_var=False, nan_policy="omit"
                    )
                else:
                    t_stat, p_val = np.nan, np.nan

                results.append({
                    "biome": biome_name,
                    "biome_id": biome_id,
                    "metric": metric,
                    "group": group_name,
                    "ref_mean": ref_mean,
                    "group_mean": grp_mean,
                    "diff": ref_mean - grp_mean,
                    "cohens_d": cohens_d,
                    "pct_below_p10": pct_below,
                    "p_value": p_val,
                    "n_ref": len(ref_values),
                    "n_group": len(grp_values),
                })

            print(f"  {metric:>5s}: primary={ref_mean:.2f}, "
                  f"PA={np.nanmean(in_pa[metric]):.2f}, "
                  f"outside={np.nanmean(outside[metric]):.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / "biome_results.csv", index=False)
    if results_df.empty:
        n_primary_total = int(df["is_primary"].sum())
        biomes_present = sorted(df["biome"].unique().tolist())
        raise RuntimeError(
            "No biome passed `min_primary_samples` — nothing to analyse.\n"
            f"  primary forest pixels (all biomes): {n_primary_total:,}\n"
            f"  biome ids present in pixel grid:    {biomes_present}\n"
            f"  forest_biomes filter:               {forest_biomes}\n"
            f"  min_primary_samples:                {min_primary_samples}\n"
            "Common causes:\n"
            "  - `primary_forest_value` does not match `naturalness_col` "
            "encoding (Lesiv et al. uses 11 for primary natural forest, not 1)\n"
            "  - `biome_attribute` burns a column whose values don't overlap "
            "`forest_biomes` (WWF uses BIOME 1..14; some derivatives renumber)\n"
            "  - `forest_threshold` too high so no pixels qualify as forest"
        )

    # --- 5. Global headline numbers -------------------------------------
    print("\n" + "=" * 80)
    print("GLOBAL HEADLINE NUMBERS")
    print("=" * 80)
    for m in metrics:
        df[f"below_p10_{m}"] = False
    for biome_id in forest_biomes:
        bdf = df[df["biome"] == biome_id]
        primary = bdf[bdf["is_primary"]]
        if len(primary) < min_primary_samples:
            continue
        for metric in metrics:
            ref_p10 = np.nanpercentile(primary[metric].values, 10)
            mask = (df["biome"] == biome_id) & df["protected"]
            df.loc[mask, f"below_p10_{metric}"] = df.loc[mask, metric] < ref_p10

    pa_pixels = df[df["protected"]]
    n_pa = len(pa_pixels)
    pcts = {}
    if n_pa > 0:
        for m in metrics:
            pcts[m] = pa_pixels[f"below_p10_{m}"].mean() * 100
        print(f"\nAmong protected area forests (n={n_pa:,}):")
        for m in metrics:
            print(f"  {pcts[m]:5.1f}% below biome primary forest P10 in {m.upper()}")
        if "rh98" in pcts:
            divs = [m for m in metrics if m != "rh98"]
            div_max = max(pcts[m] for m in divs) if divs else 0
            print(f"\n  -> If diversity metrics (max={div_max:.0f}%) >> "
                  f"RH98 ({pcts['rh98']:.0f}%),")
            print(f"     canopy height underestimates structural degradation.")

    # --- 6. Visualization -----------------------------------------------
    plot_biome_comparison(results_df, metrics, save_dir / "biome_comparison.png")
    if n_pa > 0:
        plot_headline_bars(pcts, save_dir / "headline.png")

    # --- 7. Summary table -----------------------------------------------
    print("\n\nSummary table (Cohen's d by biome and metric, in_pa):")
    print("-" * 60)
    summary = results_df[results_df["group"] == "in_pa"].pivot(
        index="biome", columns="metric", values="cohens_d"
    ).round(3)
    print(summary.to_string())
    print("\nDone. Results saved to:", save_dir)
    return results_df


# ============================================================================
# Hydra entrypoint
# ============================================================================
import hydra
from config.loader import register
from config.runner import run_cli

register(
    Path(__file__).resolve().parents[1] / 'config' / 'eval' / 'config.yaml',
    section='protected_area_analysis',
    default_run='analyze_protected_areas',
)


@hydra.main(config_name='no_log', version_base='1.2', config_path='../config/base')
def main(cfg):
    run_cli(cfg)


if __name__ == '__main__':
    main()
