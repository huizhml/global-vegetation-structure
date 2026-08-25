"""
Project-wide constants, loaded from ``const/hyperparams.yaml``.

To add or change a constant, edit the YAML — no code change needed here unless
you want to expose a derived value or an alias.
"""
from pathlib import Path
import yaml

_HP_PATH = Path(__file__).with_name("hyperparams.yaml")
with _HP_PATH.open() as _f:
    _HP = yaml.safe_load(_f)


# ---------------------------------------------------------------------------
# Nodata sentinels
# ---------------------------------------------------------------------------
VSM_NODATA     = int(_HP["vsm_nodata"])         # int16 nodata for VSM rasters / inputs
INDICES_NODATA = float(_HP["indices_nodata"])   # nodata for diversity-index outputs

# ---------------------------------------------------------------------------
# Vertical-profile / height constants
# ---------------------------------------------------------------------------
MAX_HEIGHT_METERS = float(_HP["max_height_meters"])
RH100_INDEX       = int(_HP["rh100_index"])
RH98_INDEX        = int(_HP["rh98_index"])

# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------
LAT_MEAN     = float(_HP["lat_mean"])
LAT_STD      = float(_HP["lat_std"])
LON_SIN_MEAN = float(_HP["lon_sin_mean"])
LON_SIN_STD  = float(_HP["lon_sin_std"])
LON_COS_MEAN = float(_HP["lon_cos_mean"])
LON_COS_STD  = float(_HP["lon_cos_std"])
SLOPE_MEAN   = float(_HP["slope_mean"])
SLOPE_STD    = float(_HP["slope_std"])

# ---------------------------------------------------------------------------
# Sentinel-2 SCL
# ---------------------------------------------------------------------------
SCL_EXCLUDE_LABELS = list(_HP["scl_exclude_labels"])
SCL_WATER          = int(_HP["scl_water"])

# ---------------------------------------------------------------------------
# ESA World Cover — derived from the unified table in hyperparams.yaml.
# Every name below is a view into the same source rows.
# ---------------------------------------------------------------------------
_ESA_TABLE      = [dict(e) for e in _HP["esa_wc"]]
_ESA_BY_SHORT   = {e["short"]: e for e in _ESA_TABLE}

ESA_WC_TABLE    = _ESA_TABLE                                # list[dict]: idx, raw, full, short
ESA_WC_BY_IDX   = {e["idx"]: e for e in _ESA_TABLE}         # lookup by predicted-label idx
ESA_WC          = {e["full"]: e["raw"] for e in _ESA_TABLE} # full-name → raw (raw-ascending)
ESA_WC_SHORT    = {e["idx"]: e["short"]                     # idx → short name (idx-ordered)
                   for e in sorted(_ESA_TABLE, key=lambda r: r["idx"])}

# Convenience scalars used as mask values across datasets/* and postprocessing/*
ESA_SNOW         = _ESA_BY_SHORT["Snow"]["idx"]
ESA_BUILT_UP     = _ESA_BY_SHORT["Built"]["idx"]
ESA_WATER        = _ESA_BY_SHORT["Water"]["idx"]

ESA_SNOW_RAW     = _ESA_BY_SHORT["Snow"]["raw"]
ESA_BUILT_UP_RAW = _ESA_BY_SHORT["Built"]["raw"]
ESA_WATER_RAW    = _ESA_BY_SHORT["Water"]["raw"]
ESA_UNKNOWN_RAW  = _ESA_BY_SHORT["unknown"]["raw"]

ESA_DATETIME = str(_HP["esa_datetime"])

del _ESA_TABLE, _ESA_BY_SHORT

# ---------------------------------------------------------------------------
# Biomes
# ---------------------------------------------------------------------------
BIOMES          = [dict(b) for b in _HP["biomes"]]
BIOMES_BY_VALUE = {int(b["value"]): b for b in BIOMES}  # lookup biome dict by its 'value' int

# ---------------------------------------------------------------------------
# GEDI processing
# ---------------------------------------------------------------------------
KEY_RHS        = tuple(_HP["key_rhs"])           # full set (postprocessing)
KEY_RHS_EVAL   = list(_HP["key_rhs_eval"])       # subset for diversity-index eval
CHM_COLS       = list(_HP["chm_cols"])
GEDI_META_COLS = list(_HP["gedi_meta_cols"])
COVERAGE_BEAMS = list(_HP["coverage_beams"])
POWER_BEAMS    = list(_HP["power_beams"])

# ---------------------------------------------------------------------------
# External project IDs
# ---------------------------------------------------------------------------
LUMI_PROJECT = int(_HP["lumi_project"])

# ---------------------------------------------------------------------------
# Plotting / visualisation
# ---------------------------------------------------------------------------
PALETTE        = list(_HP["palette"])
# Per-band cmin/cmax/cmap for VSM thumbnails (diversity indices + rh*_q1).
# rh*_q1 cmin/cmax are in METERS — see hyperparams.yaml note on dm conversion.
VSM_VIS_PARAMS = {k: dict(v) for k, v in _HP["vsm_vis_params"].items()}

# Central font sizes (points). Every plot reads these so a single edit in
# hyperparams.yaml restyles all figures. Keys: ticks, label, title, annot,
# legend, colorbar.
FONT_SIZES = {k: int(v) for k, v in _HP["font_sizes"].items()}

# Central figure sizes (inches, (width, height)). Same idea as FONT_SIZES:
# every plot picks a preset so a single edit restyles all figures. Keys:
# mini, small, square, medium, wide, large, panel, strip.
FIGURE_SIZES = {k: tuple(v) for k, v in _HP["figure_sizes"].items()}


def set_plot_fonts(**overrides):
    """Push :data:`FONT_SIZES` into matplotlib rcParams.

    Call once before plotting so figure elements that don't pass an explicit
    size inherit the central defaults. Pass keyword overrides to bump a single
    category for the current process, e.g. ``set_plot_fonts(title=20)``. The
    overrides are written back into :data:`FONT_SIZES` in place, so code that
    reads ``FONT_SIZES['annot']`` directly (e.g. an explicit ``fontsize=``
    argument) picks them up too — not just elements that fall back to rcParams.
    Returns the effective size dict. Raises ``KeyError`` on an unknown category.
    """
    import logging
    import matplotlib as mpl

    # pdf.fonttype 42 (below) makes savefig run fontTools' subsetter, which
    # logs every pruned table at INFO. Under Hydra's INFO root logger that
    # buries real output in "glyf pruned" noise, so keep it at WARNING.
    logging.getLogger('fontTools').setLevel(logging.WARNING)

    unknown = set(overrides) - set(FONT_SIZES)
    if unknown:
        raise KeyError(
            f"unknown font category {sorted(unknown)}; "
            f"valid keys: {sorted(FONT_SIZES)}"
        )

    FONT_SIZES.update(overrides)  # mutate in place so direct reads see overrides
    mpl.rcParams.update({
        "font.size":        FONT_SIZES["annot"],
        "axes.titlesize":   FONT_SIZES["title"],
        "axes.labelsize":   FONT_SIZES["label"],
        "xtick.labelsize":  FONT_SIZES["ticks"],
        "ytick.labelsize":  FONT_SIZES["ticks"],
        "legend.fontsize":  FONT_SIZES["legend"],
        "figure.titlesize": FONT_SIZES["title"],
        # Embed TrueType subsets (with a ToUnicode cmap) instead of the default
        # Type 3 fonts, so exported PDF/PS text stays selectable and copies as
        # correct Unicode even after the figure is re-embedded in another doc.
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
    })
    return dict(FONT_SIZES)


def fewer_ticks(ax, axis: str = 'both', nbins: int = 3, prune=None) -> None:
    """Thin matplotlib Axes major ticks to ~``nbins`` per axis.

    Defaults thin both axes for plain numeric plots. For categorical /
    grouped axes (boxplot/violin/bar x-axis, heatmap rows/cols), pass
    ``axis='x'`` or ``axis='y'`` to thin only the *other* axis and leave
    the grouped one's explicit ticks intact.
    """
    from matplotlib.ticker import MaxNLocator
    if axis not in ('x', 'y', 'both'):
        raise ValueError(f"axis must be 'x', 'y', or 'both' (got {axis!r})")
    if axis in ('x', 'both'):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune=prune))
    if axis in ('y', 'both'):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune=prune))


del _HP, _HP_PATH, _f, Path, yaml
