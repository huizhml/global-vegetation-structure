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


del _HP, _HP_PATH, _f, Path, yaml
