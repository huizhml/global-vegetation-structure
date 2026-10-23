import os
import re
import time
import ee
import json
import logging
from typing import List
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import dask
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgp
import matplotlib.pyplot as plt
from shapely.geometry import shape
import requests
from retry import retry
from io import StringIO
from dotenv import load_dotenv
import pyarrow.parquet as pq
import xarray as xr
import pystac
import pystac_client
import planetary_computer
from xrspatial import slope
from shapely.geometry import box
from dask.distributed import Client, LocalCluster

from download.core import DaskDownloader
from download.core.utils import authenticate, ensure_authenticated, shapely_to_geojson, ee_fc_to_gpd, exact_shot_number, get_aux_df, row_to_stac_item, get_patch, get_epsg_from_tile, buffer_and_snap_bounds, get_total_bounds, check_unfinished_files, read_parquets_with_file_name
from download.core.constants import dtypes
from tools.task_state import TaskState
from tools.mgrs_tile import mgrs_100km
load_dotenv()


def is_parquet_ok(path):
    try:
        pq.read_metadata(path)   # only reads footer + schema
        return True
    except Exception:
        return False


# authenticate()  # TODO: only authenticate when needed

stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace)

logger = logging.getLogger(__name__)
GEDI_START = pd.Timestamp('2018-01-01')


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def download_gedi_table_index( year: int, out_file: Path, rewrite: bool = False):
    
    '''
    Download the GEDI table index
    Reason to keep this data local:
    1. At the time of writing, we need to sample a small subset of global GEDI data for Sentinel-2 tile level bias correction.
        There are 20406 and 10228 orbits in 2020 and 2024, the table index data has only three columns, table_id, start_time, end_time.
        It's a very small data
    2. We want to do sample the GEDI footprints from the tile-level growing season, it's easier to do the table_id (orbit_id) filtering locally.
    Output:
        * gedi_table_index_file: the path to the GEDI table index file, should have columns: system:index, table_id, start_time, end_time, geometry

    '''
    out_file.parent.mkdir(exist_ok=True, parents=True)
    if out_file.exists() and not rewrite:
        gdf = gpd.read_parquet(out_file)
        return gdf
    gedi_table_index = ee.FeatureCollection("LARSE/GEDI/GEDI02_A_002_INDEX") \
        .filter(ee.Filter.stringContains('time_start', str(year)))
    download_id = ee.data.getTableDownloadId({'table': gedi_table_index, 'fileFormat': 'csv'})
    res = requests.get(ee.data.makeTableDownloadUrl(download_id))
    if res.status_code == 200:
        data = StringIO(res.content.decode('utf-8'))
        df = pd.read_csv(data)
        geom = df['.geo'].apply(lambda x: shape(json.loads(x)))
        gdf = gpd.GeoDataFrame(df, geometry=geom)
        gdf = gdf.drop(columns=['.geo'])
        gdf.to_parquet(out_file)
        return gdf
    else:
        raise requests.HTTPError(f'Failed to download {out_file}')

# State info prefix of a tile done by a download with the antimeridian-safe
# orbit join (2026-10-15 on): ALL_MONTHS = every orbit over it, GROWING_MONTHS =
# the orbits in its growing months (all months when none is). Done lines
# without a prefix come from earlier growing-months runs.
ALL_MONTHS = 'all months'
GROWING_MONTHS = 'growing months'


def water_filter_expr(max_water_persistence: float = None, drop_no_dem: bool = True) -> str:
    """GEE filter expression that drops shots on water ('' = keep all).

    No DEM (digital_elevation_model == -999999) is open sea in practice: flat
    ~3 m "canopies" with a land pft_class next to islands and coasts.
    landsat_water_persistence (% of Landsat observations with water) catches
    lakes and rivers, which do have a DEM.
    """
    parts = (['digital_elevation_model>-1000'] if drop_no_dem else []) + \
            ([f'landsat_water_persistence<{max_water_persistence}'] if max_water_persistence is not None else [])
    return ' && '.join(parts)


def shot_filter_expr(sensitivity_threshold: float = None, max_water_persistence: float = None,
                     drop_no_dem: bool = True) -> str:
    """GEE filter for the shots download_all_valid keeps (quality, leaf-on,
    sensitivity, not on water)."""
    expr = 'quality_flag==1 && degrade_flag==0 && region_class>0 && leaf_off_flag!=1'
    if sensitivity_threshold is not None:
        expr = f'{expr} && sensitivity>={sensitivity_threshold}'
    water = water_filter_expr(max_water_persistence, drop_no_dem)
    return f'{expr} && {water}' if water else expr


# GEDI pft_class = MODIS MCD12Q1 LC_Type5 of the 1 km cell: 1-4 trees, 5 shrub,
# 6 grass, 7-8 cropland, 9 urban, 11 barren kept; 0 water, 10 snow/ice dropped.
# Until 2026-10 downloads kept 1-8 only; 9 and 11 were added to those years
# with add_pft_classes.
VALID_PFT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]


# Every property of the GEDI L2A tables in GEE. Tables differ: e.g. the
# *_003_04_V002 granules have no orbit_number, minor_frame_number,
# shot_number_within_beam, local_beam_azimuth, local_beam_elevation. A CSV
# download takes its columns from the first feature, so a merged request
# (orbits_fc) must name them all, or a batch led by such a table drops these
# five columns for every orbit in it (2026-10-22).
GEDI_COLUMNS = ['system:index', '.geo', 'beam', 'degrade_flag', 'delta_time', 'digital_elevation_model',
                'digital_elevation_model_srtm', 'elev_highestreturn', 'elev_lowestmode', 'elevation_bias_flag',
                'energy_total', 'landsat_treecover', 'landsat_water_persistence', 'lat_highestreturn',
                'leaf_off_doy', 'leaf_off_flag', 'leaf_on_cycle', 'leaf_on_doy', 'local_beam_azimuth',
                'local_beam_elevation', 'lon_highestreturn', 'minor_frame_number', 'modis_nonvegetated',
                'modis_nonvegetated_sd', 'modis_treecover', 'modis_treecover_sd', 'num_detectedmodes',
                'orbit_number', 'pft_class', 'quality_flag', 'region_class', 'selected_algorithm',
                'selected_mode', 'selected_mode_flag', 'sensitivity', 'shot_number', 'shot_number_within_beam',
                'solar_azimuth', 'solar_elevation', 'surface_flag', 'urban_focal_window_size',
                'urban_proportion'] + [f'rh{i}' for i in range(101)]


def pft_label(classes) -> str:
    '''[1, 2, ..., 9, 11] -> "1-9,11", for state lines.'''
    classes, runs = sorted(int(c) for c in classes), []
    for c in classes:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ','.join(f'{a}-{b}' if b > a else f'{a}' for a, b in runs)
# GEE fails to build a table download above some size (~38k shots with all
# 143 columns, ~76k with 58: it is bytes; 2026-10 benchmark, see
# tools/gedi_merge_benchmark.py); such a request is split in two and retried.
SPLITTABLE_ERRORS = ('No space left on device',)


def orbits_fc(orbit_ids: list, geom: dict, shot_filter: str, columns: list = None,
              pft_classes: list = VALID_PFT) -> ee.FeatureCollection:
    '''The shots of all orbit_ids over geom as one FeatureCollection, so they
    come in one download. filterBounds goes on each orbit table before the
    merge, so each uses its own spatial index.'''
    fcs = [ee.FeatureCollection(oid).filterBounds(geom) for oid in orbit_ids]
    fc = fcs[0] if len(fcs) == 1 else ee.FeatureCollection(fcs).flatten()
    fc = fc.filter(shot_filter).filter(ee.Filter.inList('pft_class', list(pft_classes)))
    return fc if columns is None else fc.select(columns)


def fetch_orbits(orbit_ids: list, geom: dict, shot_filter: str, shots_per_orbit: float = None,
                 max_shots: int = 20000, max_orbits: int = 50, first_batch: int = 5,
                 prior_weight: float = 3, columns: list = None, pft_classes: list = VALID_PFT):
    '''Download the shots of orbit_ids over geom in few requests, each with as
    many orbits as are expected to fit in max_shots.

    The expected shots per orbit start at shots_per_orbit (from another year,
    see shots_per_orbit_prior; None = first_batch orbits in the first
    request) and follow what the requests so far returned, the prior counting
    as prior_weight orbits. A request GEE cannot build (SPLITTABLE_ERRORS) is
    split in two, down to single orbits.

    An orbit table the GEE index lists but GEE does not have (e.g.
    GEDI02_A_2022362115234_O22900_01_..., 2026-10) is left out of its request
    and logged as `missing`.

    Returns (shots or None, log): log has one dict per request with orbits,
    shots (for a failed request: what its two halves returned), seconds, ok,
    and missing (the table it failed on) if any.
    '''
    log, frames = [], []

    def fetch_batch(batch):
        start = time.time()
        entry = dict(orbits=len(batch), shots=None, seconds=None, ok=True)
        log.append(entry)
        try:
            df = ee_fc_to_gpd(orbits_fc(batch, geom, shot_filter, columns, pft_classes),
                              selectors=GEDI_COLUMNS if columns is None else None)
        except ee.EEException as e:
            m = re.search(r"Collection asset '([^']+)' not found", str(e))
            if not m or m.group(1) not in batch:
                raise
            entry.update(ok=False, seconds=round(time.time() - start, 1), missing=m.group(1))
            rest = [o for o in batch if o != m.group(1)]
            entry['shots'] = fetch_batch(rest) if rest else 0
            return entry['shots']
        except requests.HTTPError as e:
            entry.update(ok=False, seconds=round(time.time() - start, 1))
            if len(batch) == 1 or not any(m in str(e) for m in SPLITTABLE_ERRORS):
                raise
            half = len(batch) // 2
            entry['shots'] = fetch_batch(batch[:half]) + fetch_batch(batch[half:])
            return entry['shots']
        entry['seconds'] = round(time.time() - start, 1)
        entry['shots'] = 0 if df is None else len(df)
        if df is not None:
            frames.append(df)
        return entry['shots']

    pending = list(orbit_ids)
    seen_orbits = seen_shots = 0
    while pending:
        if seen_orbits == 0:
            expected = shots_per_orbit
        elif shots_per_orbit is None:
            expected = seen_shots / seen_orbits
        else:
            expected = (seen_shots + prior_weight * shots_per_orbit) / (seen_orbits + prior_weight)
        n = first_batch if expected is None else int(max_shots // max(expected, 1))
        n = min(max(n, 1), max_orbits)
        batch, pending = pending[:n], pending[n:]
        seen_shots += fetch_batch(batch)
        seen_orbits += len(batch)
    shots = pd.concat(frames, ignore_index=True) if frames else None
    return shots, log


def target_orbits(pairs: pd.DataFrame, orbit_months: str):
    '''Which (tile, orbit) pairs of tile_orbit_pairs a download takes: all for
    'all'; for 'growing' the orbits in the tile's growing months, all months
    for a tile with none there. Returns (mask over pairs, tiles without
    growing-month orbits).'''
    no_growing = set(pairs['Name']) - set(pairs.loc[pairs['in_growing'], 'Name'])
    if orbit_months == 'all':
        return pd.Series(True, index=pairs.index), no_growing
    return pairs['in_growing'] | pairs['Name'].isin(no_growing), no_growing


def state_shots(state_dir: str) -> dict:
    '''{tile: shots} of the tiles a download_all_valid state has a count for
    ("... 5321 shots ..." or "no valid shots").'''
    shots = {}
    for name, (status, _, info) in TaskState(state_dir, create=False).load().items():
        if status != 'done':
            continue
        m = re.search(r'(\d+) shots', info)
        if m:
            shots[name] = int(m.group(1))
        elif 'no valid shots' in info:
            shots[name] = 0
    return shots


def shots_per_orbit_prior(s2_table: gpd.GeoDataFrame, state_dir: str, index_file: str, year: int,
                          orbit_months: str = 'growing') -> dict:
    '''Expected shots per orbit for each S2 tile, from the download of `year`:
    its shots in the tile (download state) / its orbits over the tile. The
    number of orbits changes a lot between years (2019 starts in April, GEDI
    was off from 2023-03 to 2024-04), the shots per orbit much less.
    Tiles without a count in that state are left out.'''
    shots = state_shots(state_dir)
    index = download_gedi_table_index(year=year, out_file=Path(index_file).expanduser())
    index = index.set_crs(epsg=4326, allow_override=True)
    pairs = tile_orbit_pairs(s2_table[s2_table['Name'].isin(shots.keys())], index)
    mask, _ = target_orbits(pairs, orbit_months)
    n_orbits = pairs[mask].groupby('Name').size()
    return {name: shots[name] / k for name, k in n_orbits.items() if k > 0}


def organize_erda_copy(save_dir: str, staging_dir: str, plan_file: str, float_tiles_file: str,
                       state_dir: str, **kwargs):
    '''Move a year copied from ERDA (staging_dir) into save_dir, except the
    tiles of plan_file: those were repaired in save_dir, their ERDA versions
    stay in staging_dir. Lists the tiles whose shot_number is float64 (every id
    rounded, see exact_shot_number) in float_tiles_file, from the ERDA files,
    since a repaired file has int64 ids even where they are rounded.
    Only tiles done in state_dir are moved: e.g. the halves of antimeridian
    tiles on ERDA belong to tiles that were reset and fetched again.'''
    save_dir, staging_dir = Path(save_dir).expanduser(), Path(staging_dir).expanduser()
    files = sorted(staging_dir.glob('*.parquet'))
    plan = set(pd.read_csv(Path(plan_file).expanduser(), sep='\t')['tile'])
    float_tiles = [p.stem for p in files if str(pq.read_schema(p).field('shot_number').type) == 'double']
    Path(float_tiles_file).expanduser().write_text(''.join(f'{t}\n' for t in float_tiles))
    done = {k for k, (st, _, _) in TaskState(state_dir, create=False).load().items() if st == 'done'}
    moved, clash, not_done = 0, [], []
    for p in files:
        if p.stem in plan:
            continue
        if p.stem not in done:
            not_done.append(p.stem)
            continue
        dst = save_dir / p.name
        if dst.exists():
            clash.append(p.stem)
            continue
        os.replace(p, dst)
        moved += 1
    print(f'{staging_dir}: {len(files)} files, {moved} moved to {save_dir}, '
          f'{sum(p.stem in plan for p in files)} left (repaired here), {len(clash)} already there {clash[:5]}, '
          f'{len(not_done)} left (not done in the state) {not_done[:10]}')
    print(f'{len(float_tiles)} tiles with float64 shot_number -> {float_tiles_file}')


def finalize_gedi_states(s2_grid_file: str, state_dirs: list, **kwargs):
    '''Before a year is copied elsewhere: every S2 tile in each state's
    expected.tsv again (runs on a subset of tiles rewrite it to that subset),
    and all log lines merged into snapshot.tsv.'''
    names = gpd.read_parquet(Path(s2_grid_file).expanduser())['Name']
    for d in state_dirs:
        st = TaskState(d, create=False)
        st.set_expected(names, source=str(s2_grid_file))
        st.compact(idle_s=0)
        state = st.load()
        print(f'{d}: {len(state)} tiles, ' + ', '.join(
            f'{s}: {n}' for s, n in pd.Series([v[0] for v in state.values()]).value_counts().items()))


def reset_gedi_tiles(save_dir: str, state_dir: str, tiles_file: str, backup_dir: str, **kwargs):
    '''Forget tiles so that the next download_all_valid fetches them whole
    (e.g. to replace rounded ids): parquet moved to backup_dir, state reset.'''
    save_dir, backup_dir = Path(save_dir).expanduser(), Path(backup_dir).expanduser()
    tiles = Path(tiles_file).expanduser().read_text().split()
    backup_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for t in tiles:
        f = save_dir / f'{t}.parquet'
        if f.exists():
            os.replace(f, backup_dir / f.name)
            moved += 1
    TaskState(state_dir, create=False).mark_many(tiles, status='reset')
    print(f'{len(tiles)} tiles reset in {state_dir}, {moved} parquets moved to {backup_dir}')


def water_mask(df: pd.DataFrame, max_water_persistence: float = None, drop_no_dem: bool = True) -> pd.Series:
    """True for shots water_filter_expr keeps, for data downloaded without it."""
    keep = pd.Series(True, index=df.index)
    if drop_no_dem:
        keep &= df['digital_elevation_model'] > -1000
    if max_water_persistence is not None:
        keep &= df['landsat_water_persistence'] < max_water_persistence
    return keep


def tile_orbit_pairs(s2_table: gpd.GeoDataFrame, gedi_index: gpd.GeoDataFrame) -> pd.DataFrame:
    """Every (S2 grid row, GEDI orbit table) pair whose footprints intersect.

    Returns columns row (index into s2_table), Name, table_id, month (of the
    table's time_start), in_growing (month in the tile's growing_months) and
    wrapped (only found with the tile shifted by +-360 deg, see below).

    Granules crossing the antimeridian come with continuous longitudes beyond
    180 in the GEE index (about 8% of them, e.g. 179 -> 200), so on their own
    they miss every tile west of -180 they actually cover (zone 01/02 tiles,
    the western half of tiles split at 180, ...). Each tile is therefore also
    tested shifted by -+360 deg. Downloads before 2026-10-15 did not do this:
    their orbits are the pairs with wrapped == False.
    """
    if 'month' not in gedi_index:
        gedi_index = gedi_index.assign(month=gedi_index['time_start'].str[5:7].astype(int))
    tiles = s2_table[['Name', 'geometry', 'growing_months']].rename_axis('row')
    shifted = [tiles.assign(geometry=tiles.translate(xoff=dx), wrapped=dx != 0) for dx in (0, 360, -360)]
    tiles = gpd.GeoDataFrame(pd.concat(shifted), geometry='geometry', crs=s2_table.crs)
    pairs = gpd.sjoin(tiles, gedi_index[['table_id', 'month', 'geometry']],
                      predicate='intersects', how='inner')
    # a pair found unshifted keeps wrapped == False
    pairs = pairs.reset_index().sort_values('wrapped').drop_duplicates(['row', 'table_id'])
    pairs['in_growing'] = [m in {int(g) for g in gm}
                           for m, gm in zip(pairs['month'], pairs['growing_months'])]
    return pd.DataFrame(pairs[['row', 'Name', 'table_id', 'month', 'in_growing', 'wrapped']]).sort_index()


class GEDI(DaskDownloader):
    """
    A class for filtering, sampling(stratified, per orbit & per cell) and downloading GEDI data from Google Earth Engine.

    Attributes:
        nSampledPerKm2 (float): The number of GEDI points to sample per km^2.
        raster (ee.ImageCollection): The GEDI raster data collection in Google Earth Engine.
        grid (ee.FeatureCollection): The MGRS grid feature collection in GEE, each grid cell has assigned 'count'(filtered GEDI Points), 'landmass'.
        year (int): The year of the GEDI data to download.
        dataFolder (pathlib.Path): The local folder to save the downloaded GEDI data.
        keepLeafOff255 (bool): Whether to keep GEDI points with leaf off flag of 255.

    Methods:
        download_all_valid(s2_grid_file: str, gedi_table_index_file: str, save_dir: str = None, **kwargs):
            Download all valid GEDI points for all S2 tiles.
            Valid: (quality flag == 1, degrade flag == 0, region class > 0, leaf off flag != 1, sensitivity >= 0.95) for the growing season
            NOTE: keep a local copy for future use, better indexing and faster retrieval.
            Args:
                * s2_grid_file: the path to the S2 table file, should have columns: Name, geometry, growing_months
                * gedi_table_index_file: the path to the GEDI table index file, should have columns: system:index, table_id, start_time, end_time, geometry
                * save_dir: the path to save the downloaded GEDI data
            Returns:
                * None

        add_slope(s2_grid_file: str, gedi_table_index_file: str, save_dir: str = None, **kwargs):
            Add slope to the GEDI data, where the slope is calculated from the DEM data (Copernicus DEM + NASADEM).
            NOTE: slope information indicates how representative the GEDI vertical profile is, higher slope means less representative.
            Args:
                * location_dir: the path to the location files
                * dem_meta_file: the path to the DEM metadata file
                * save_dir: the path to save the GEDI data with slope
                NOTE: the save directory should be the same as the location files directory
            Returns:
                * None

        sample_subset(s2_grid_file: str, save_dir: str = None, correction_number_per_tile: int = 2000, exclude_used_gedi_points: bool = False, used_gedi_points_dir: str = None, **kwargs):
            Sample GEDI points for GVS correction. 
            NOTE: slope filter can be added here
            Args:
                * s2_grid_file: the path to the S2 table file, should have columns: Name, geometry, growing_months
                * save_dir: the path to save the sampled GEDI data
                * correction_number_per_tile: the number of GEDI points to sample per tile
                * exclude_used_gedi_points: whether to exclude used GEDI points
                * used_gedi_points_dir: the path to the used GEDI points file
            Returns:
                * None

        check_slope_distribution(save_dir: str):
            Check the distribution of the slope data
            Args:
                * save_dir: the path to save the GEDI data with slope
            Returns:
                * None


        getTableAssetIds(export=False):
            Returns a list of unique table asset IDs for GEDI orbits from GEE rasterized version of GEDI within the specified year.

        getValidZoneCodes(export=False):
            Returns a list of valid zone codes of MRGS grid cells where GEDI points are present.

        download(processes=40, maxTries=3):
            Downloads GEDI data for all valid MGRS grid cells in the specified year.

        filterGEDI(asset_id):
            Returns a filtered GEDI feature collection for specified table asset ID.
            Filter by quality flag, degrade flag, region class, and growing season.

        sampleCell(zone_code):
            Downloads GEDI data for the specified MGRS grid cell.

        getSampleRatio(cell):
            Returns the number of GEDI points to sample for the specified MGRS grid cell.
    """
    

    # this is obtained from the total number of points we want to sample per year(75M) and the total landmass in the world
    nSampledPerKm2 = 0.7033933006651656
    GEDI_START = pd.Timestamp('2018-01-01')

    def __init__(self, year=2019,
                 # all needed
                 save_dir: str = None,
                 # original download
                 mgrs_file: str = 'GEDI/mgrs_with_count_and_orbits.parquet',
                 # download all valid
                 s2_grid_file: str = None,
                 gedi_table_index_file: str = None,
                 state_dir: str = None,
                 # add slope
                 location_dir: str = None,
                 dem_meta_file: str = None,
                 # check slope distribution
                 deploy_status_dir: str = None,
                 input_dir: str = None,
                 used_parq_dir: str = None,
                 # optional
                 flag_dir: str = None,
                 key_file: str = None, npartitions=100,
                 sensitivity_threshold: float = 0.95,
                 orbit_months: str = 'growing',
                 tiles: list = None,
                 max_water_persistence: float = None,
                 drop_no_dem: bool = False,
                 # orbits merged per GEE request, see fetch_orbits
                 max_shots_per_request: int = 20000,
                 max_orbits_per_request: int = 50,
                 prior_year: int = None,
                 prior_state_dir: str = None,
                 prior_index_file: str = None,
                 # download_all_valid: only print what it would fetch
                 dry_run: bool = False,
                 # pft_class kept by the GEE query (VALID_PFT)
                 pft_classes: list = None,
                 # add_pft_classes: fetch only these classes, per-tile outcome
                 # in add_pft_state_dir; tiles in exclude_tiles_file skipped
                 add_pft: list = None,
                 add_pft_state_dir: str = None,
                 exclude_tiles_file: str = None,
                 # one tile name per line, instead of tiles
                 tiles_file: str = None,
                 n_parallel=40, row_group_size=100, random_state: int = 42, rewrite: bool = False, **kwargs):
        """
        Initializes a GEDI object.

        Args:
            year (int, optional): The year of the GEDI data to download. Defaults to 2019.
            dataFolder (str, optional): The local folder to save the downloaded GEDI data. Defaults to 'gedi'.
            keepLeafOff255 (bool, optional): Whether to keep GEDI points with leaf off flag of 255. Defaults to False.
        """
        super().__init__(n_parallel=n_parallel, max_retries=3, **kwargs)
        self.mgrs_file = mgrs_file
        self.s2_grid_file = s2_grid_file
        self.gedi_table_index_file = gedi_table_index_file
        self.location_dir = location_dir
        self.dem_meta_file = dem_meta_file
        self.deploy_status_dir = deploy_status_dir
        self.input_dir = input_dir
        self.used_parq_dir = used_parq_dir
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(exist_ok=True, parents=True)
        # Per-tile outcome of download_all_valid; see tools/task_state.py.
        self.state = TaskState(state_dir) if state_dir is not None else None

        # flag_dir is only used by the legacy download_zone path; other methods
        # (e.g. download_all_valid) don't need it.
        if flag_dir is not None:
            self.flag_dir = Path(flag_dir).expanduser()
            self.flag_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.flag_dir = None
        self.npartitions = npartitions
        self.n_parallel = n_parallel
        self.row_group_size = row_group_size
        self.year = year
        self.sensitivity_threshold = sensitivity_threshold
        assert orbit_months in ('all', 'growing'), orbit_months
        self.orbit_months = orbit_months
        # only these S2 tiles (e.g. a trial run into a copy of a few tiles); None = all
        if tiles_file is not None:
            tiles = Path(tiles_file).expanduser().read_text().split()
        self.tiles = None if tiles is None else [str(t) for t in tiles]
        # shots on water (see water_filter_expr) are also dropped from the rows
        # of a parquet that a download appends to
        self.max_water_persistence = max_water_persistence
        self.drop_no_dem = drop_no_dem
        self.filter = shot_filter_expr(sensitivity_threshold, max_water_persistence, drop_no_dem)
        self.max_shots_per_request = max_shots_per_request
        self.max_orbits_per_request = max_orbits_per_request
        # shots per orbit from another year's download size the first request
        # of a tile; None = no prior
        self.prior_year = prior_year
        self.prior_state_dir = prior_state_dir
        self.prior_index_file = prior_index_file
        self.dry_run = dry_run
        self.pft_classes = list(VALID_PFT if pft_classes is None else pft_classes)
        self.add_pft = None if add_pft is None else list(add_pft)
        self.add_pft_state = TaskState(add_pft_state_dir) if add_pft_state_dir is not None else None
        self.exclude_tiles_file = exclude_tiles_file
        self.random_state = random_state
        self.rewrite = rewrite
        self.key_file = key_file

        self._ensure_authenticated()

    def _ensure_authenticated(self):
        """Initialize Earth Engine in the current process with this run's key_file.

        Must be called from inside worker tasks too — dask workers are separate
        processes that don't run __init__, and importing the package can
        authenticate them with the default key as a side effect.
        """
        ensure_authenticated(self.key_file)  # per-run key file, falls back to env/default

    @dask.delayed
    def download_zone(self, zone):
        self._ensure_authenticated()  # runs on a worker process; ensure this run's key
        print('downloading ', zone['MGRS_UTM'])
        if zone[f'count_{self.year}'] == 0:
            logger.info(f"no GEDI points in {zone['MGRS_UTM']}")
            return None
        flag = self.flag_dir / f'{zone["MGRS_UTM"]}_{self.year}_done'
        if flag.exists() and not self.rewrite:
            logger.info(f"{flag} exists")
            return None
        zone_dir = self.save_dir / zone["MGRS_UTM"]
        zone_dir.mkdir(exist_ok=True, parents=True)
        geom = ee.Geometry.BBox(*zone['geometry'].bounds).toGeoJSON()
        last_coords = geom['coordinates'][0][0].copy()
        geom['coordinates'][0].append(last_coords)
        sample_ratio = zone['landmass'] * self.nSampledPerKm2/zone[f'count_{self.year}'] + 0.001
        total = sampled = 0
        new_tracks = defaultdict(list)
        for track_id in zone[f'tracks_{self.year}']:
            filename = track_id.split('/')[-1]
            if (zone_dir / f'{filename}.parquet').exists() and not self.rewrite:
                continue
            fc = ee.FeatureCollection(track_id).filterBounds(geom).filter(self.filter)
            orbit_size = fc.size().getInfo()
            if orbit_size > 0:
                total += orbit_size
                sampled += self.download_orbit(fc, sample_ratio, zone_dir, filename)
                new_tracks[self.year].append(track_id)

        logger.info(f'{zone["MGRS_UTM"]}, Total: {total}, sampled: {sampled}, #want: {total * sample_ratio}')
        logger.info(zone["MGRS_UTM"], new_tracks)
        if len(os.listdir(zone_dir)) > 0:
            flag.touch()
        else:
            os.rmdir(zone_dir)  # remove empty directory

    @retry(requests.HTTPError, tries=10, delay=1)
    def download_orbit(self, fc, sample_ratio, zone_dir, filename):
        file = zone_dir / f'{filename}.parquet'
        if file.exists() and not self.rewrite:
            df = gpd.read_parquet(file)
            return len(df)
        if sample_ratio < 1:
            fc = fc.randomColumn('random', seed=self.random_state).filter(ee.Filter.lte('random', sample_ratio))
        sampled = fc.size().getInfo()
        if sampled > 0:
            download_id = ee.data.getTableDownloadId({'table': fc, 'fileFormat': 'csv'})
            res = requests.get(ee.data.makeTableDownloadUrl(download_id))
            if res.status_code == 200:
                data = StringIO(res.content.decode('utf-8'))
                df = pd.read_csv(data)
                df['.geo'] = df['.geo'].apply(lambda x: shape(json.loads(x)))
                df = df.rename(columns={'.geo': 'geometry'})
                df = gpd.GeoDataFrame(df, geometry='geometry')
                df = df[['geometry', *dtypes.keys()]]
                df = df.astype(dtypes)
                df['date'] = pd.to_timedelta(df['delta_time'], unit='s') + GEDI_START
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                df.to_parquet(file, row_group_size=self.row_group_size, engine='pyarrow')
                print(f'{filename} saved')
            else:
                raise requests.HTTPError(f'Failed to download {filename}')
        return sampled

    def download(self):
        """
        Downloads GEDI data for all valid MGRS grid cells in the specified year.
        """
        cluster = LocalCluster()
        client = Client(cluster)  # timeout
        print(client)
        mgrs_file = Path(self.mgrs_file).expanduser()
        if not mgrs_file.exists():
            raise FileNotFoundError(f'{mgrs_file} not found')
        mgrs_df = gpd.read_parquet(self.mgrs_file)
        # row = mgrs_df[mgrs_df['MGRS_UTM']=='04L'].iloc[0]
        # self.download_zone(row).compute()
        self.schedule_tasks(mgrs_df, self.download_zone)

    def download_all_valid(self):
        '''
        Per-tile pipeline: one task per S2 tile, each task makes a small GEE
        call per overlapping orbit (filterBounds = single tile polygon).

        Why per-tile and not per-orbit
        ------------------------------
        A per-orbit refactor was tried (one GEE call per orbit, filterBounds =
        bbox of all overlapping pending tiles, then local sjoin to split per
        tile). It was ~5–10x SLOWER in practice. Measured timings:

            orbit X (96 tiles, 71k pts): gee=251s sjoin=0.2s write=1.7s
            orbit Y (54 tiles, 77k pts): gee=252s sjoin=0.3s write=1.3s
            orbit Z (30 tiles, 62k pts): gee=256s sjoin=0.2s write=0.9s

        Two compounding causes:
          1. bbox(union of tiles) for orbits crossing continents ≈ whole region
             → GEE returns the full orbit's points, not the subset we need;
          2. GEE has a large per-request fixed cost that's worse to amortize
             into a few big calls than into many small parallel calls — small
             single-tile filterBounds calls (5-vertex polygon, ~few k pts)
             are fast and parallelize well across workers.

        Per-tile keeps GEE happy: small payload, simple filter polygon, lots
        of small tasks. Downside: S2 overlap regions produce duplicate shots
        across tile parquets — use `dedup_shots` as a post-processing pass
        if a globally-deduped dataset is needed.

        Within a tile, the orbits are merged server-side into few requests
        (fetch_orbits): one request per orbit is mostly fixed per-request
        cost (2026-10 benchmark, tools/gedi_merge_benchmark.py: 10 orbits per
        request was 2-5x faster with ~9x fewer requests). A request is kept
        under max_shots_per_request with the shots per orbit of prior_year
        (shots_per_orbit_prior) and of the tile's requests so far, since GEE
        cannot build a download much above ~38k shots.

        Which orbits (orbit_months)
        ---------------------------
        'growing' (default): orbits whose month is in the tile's
        growing_months (s2_grid_file); a tile with no such orbit takes the
        orbits of all months instead. 'all': every orbit over the tile.
        Shots in leaf-off conditions are dropped per shot by leaf_off_flag != 1
        either way, and shots on water by max_water_persistence / drop_no_dem.

        Orbits come from tile_orbit_pairs, which also finds the antimeridian
        granules (wrapped) that downloads before 2026-10-15 missed. A tile
        those downloads did (parquet, or done in the state without a prefix)
        is repaired in place: only orbits it should have and was not given
        (wrapped ones, other months for 'all', all months for a tile without
        growing-month orbits) are fetched and appended to its parquet (dedup
        on shot_number); its old rows get the same water filter. This assumes
        s2_grid_file and the orbit index are the ones those downloads used.

        With state_dir set, every S2 tile ends up in the state: done or failed
        with the error. Done info starts with "growing months:" or
        "all months:" (e.g. "growing months: 5321 shots (+830 from 12
        orbits)"), "no valid shots" or "no GEDI orbit over tile"; lines
        without that prefix come from earlier downloads and are redone. A
        year is complete when every tile is done with the prefix of the
        orbit_months it was run with ("all months" also counts for growing).

        Args (via __init__ / yaml):
            * s2_grid_file: S2 table with columns Name, geometry, growing_months
            * gedi_table_index_file: GEDI orbit index (downloaded if missing)
            * state_dir: TaskState directory (optional)
            * orbit_months: 'growing' or 'all'
            * tiles: only these tiles (trial runs), None = all
            * max_water_persistence, drop_no_dem: water filter, see water_filter_expr
            * dry_run: print the tiles and orbits it would fetch, with the
                expected shots and requests; writes only that list
                (fetch_plan.tsv in the state dir)
        '''
        if not self.dry_run:
            cluster = LocalCluster()
            client = Client(cluster)  # timeout
            print(client)

        gedi_index, s2_table = self._load_index_and_grid()
        if self.state is not None and not self.dry_run:
            # every S2 tile ends up done or failed; state_summary checks against this
            self.state.set_expected(s2_table['Name'], source=str(self.s2_grid_file))

        for tmp in self.save_dir.glob('.*.parquet.tmp'):   # left by a killed run
            if not self.dry_run:
                tmp.unlink()
        files = {p.stem for p in self.save_dir.glob('*.parquet') if is_parquet_ok(p)}
        state = self.state.load() if self.state is not None else {}
        done = {k: info for k, (st, _, info) in state.items() if st == 'done'}
        tag = ALL_MONTHS if self.orbit_months == 'all' else GROWING_MONTHS
        tagged = {k: info.split(':')[0] for k, info in done.items()
                  if info.startswith((ALL_MONTHS, GROWING_MONTHS))}
        # all months covers growing months; growing months is complete for 'growing' only
        complete = {k for k, t in tagged.items() if t == ALL_MONTHS or t == tag}
        # downloaded before the antimeridian-safe join: its growing-month,
        # unshifted orbits are in (a parquet without a state line counts too)
        legacy = (files | done.keys()) - tagged.keys()
        pending = s2_table[~s2_table['Name'].isin(complete)].reset_index(drop=True)
        print(f'orbit_months={self.orbit_months}: {len(complete)} tiles done, {len(pending)} pending')
        if len(pending) == 0:
            print('All tiles done.')
            return

        # One-shot sjoin (pending × orbits), replaces the per-tile linear
        # intersects() scan over the full orbit index (one R-tree sjoin).
        pairs = tile_orbit_pairs(pending, gedi_index)
        any_orbit = set(pairs['Name'])
        # a tile with no orbit in its growing months takes the orbits of all months
        target, no_growing = target_orbits(pairs, self.orbit_months)
        is_legacy = pairs['Name'].isin(legacy)
        is_growing_tagged = pairs['Name'].map(tagged).eq(GROWING_MONTHS)
        fetched = ((is_legacy & pairs['in_growing'] & ~pairs['wrapped'])
                   | (is_growing_tagged & (pairs['in_growing'] | pairs['Name'].isin(no_growing))))
        pairs = pairs[target & ~fetched]
        orbits_per_row = pairs.groupby('row')['table_id'].apply(list).to_dict()

        # Tiles crossing the antimeridian are two rows with the same Name (25
        # in the grid). One task per Name, querying each part on its own: the
        # bbox of their union would span all longitudes.
        parts_per_tile = defaultdict(list)
        for i, row in pending.iterrows():
            orbit_ids = orbits_per_row.get(i, [])
            if orbit_ids:
                parts_per_tile[row['Name']].append((row['geometry'], orbit_ids))
        nothing_to_fetch = sorted(set(pending['Name']) - parts_per_tile.keys())
        note = lambda n: ', no orbit in growing months: all months' if (
            self.orbit_months == 'growing' and n in no_growing) else ''
        prior = {}
        if self.prior_year is not None and parts_per_tile:
            prior = shots_per_orbit_prior(pending[pending['Name'].isin(parts_per_tile.keys())],
                                          self.prior_state_dir, self.prior_index_file,
                                          self.prior_year, self.orbit_months)
            print(f'shots per orbit from {self.prior_year}: {len(prior)} of {len(parts_per_tile)} tiles')
        if self.dry_run:
            self._print_plan(parts_per_tile, files, prior, nothing_to_fetch, any_orbit)
            return
        tasks = [self.get_orbit_for_one_s2_tile(name, parts, append=name in files, note=note(name),
                                                shots_per_orbit=prior.get(name))
                 for name, parts in parts_per_tile.items()]
        if self.state is not None and nothing_to_fetch:
            # tells GEDI coverage (e.g. beyond ~51.6 deg) apart from the rest
            uncovered = [n for n in nothing_to_fetch if n not in any_orbit]
            if uncovered:
                self.state.mark_many(uncovered, info=f'{tag}: no GEDI orbit over tile')
            for n in nothing_to_fetch:
                if n in any_orbit:      # every orbit it needs was fetched before
                    rows = pq.ParquetFile(self.save_dir / f'{n}.parquet').metadata.num_rows if n in files else 0
                    self.state.mark_done(n, f'{tag}: {rows} shots (nothing missing{note(n)})'
                                         if rows else f'{tag}: no valid shots{note(n)}')
        print(f'Scheduling {len(tasks)} tile fetches')
        if tasks:
            nfailed, _ = self.schedule_tasks(delayed_tasks=tasks)
            print(f'{nfailed} tiles failed after retries')
        if self.state is not None:
            self.state.compact()
            print(f'state: {self.state.path}')

    def _load_index_and_grid(self):
        '''The GEDI orbit index of self.year and the S2 grid (only self.tiles if set).'''
        gedi_index = download_gedi_table_index(year=self.year, out_file=Path(self.gedi_table_index_file).expanduser())
        gedi_index = gedi_index.drop(columns=['system:index', 'time_end'])
        gedi_index['month'] = gedi_index['time_start'].str[5:7].astype(int)
        gedi_index = gedi_index.set_crs(epsg=4326, allow_override=True)
        s2_table = gpd.read_parquet(Path(self.s2_grid_file).expanduser())
        if self.tiles is not None:
            s2_table = s2_table[s2_table['Name'].isin(self.tiles)].reset_index(drop=True)
            print(f'tiles: {s2_table["Name"].nunique()} of {len(self.tiles)} requested are in the grid')
        return gedi_index, s2_table

    def add_pft_classes(self):
        '''
        Append the shots of the pft classes self.add_pft (e.g. [9, 11]) to the
        tiles of a year downloaded without them: over every orbit
        download_all_valid takes for the tile (target_orbits), only those
        classes, so no shot comes twice. Old rows get the water filter, as in
        a repair; a tile without a parquet gets one if it has such shots.

        Per-tile outcome in add_pft_state_dir ("pft 9,11: N shots (+n from k
        orbits)"), done tiles are skipped on a rerun. Tiles in
        exclude_tiles_file are skipped: e.g. tiles just downloaded whole with
        these classes. Do not run it while a download writes to the same year.
        '''
        assert self.add_pft and self.add_pft_state is not None, 'add_pft and add_pft_state_dir are required'
        assert not set(self.add_pft) & set(range(1, 9)), f'{self.add_pft}: classes 1-8 are in every download'
        label = pft_label(self.add_pft)
        client = Client(LocalCluster())
        print(client)
        gedi_index, s2_table = self._load_index_and_grid()
        self.add_pft_state.set_expected(s2_table['Name'], source=str(self.s2_grid_file))
        exclude = set()
        if self.exclude_tiles_file is not None:
            exclude = set(Path(self.exclude_tiles_file).expanduser().read_text().split())
        done = {k for k, (st, _, _) in self.add_pft_state.load().items() if st == 'done'}
        todo = s2_table[~s2_table['Name'].isin(done | exclude)].reset_index(drop=True)
        print(f'pft {label}: {len(done)} tiles done, {len(exclude)} excluded, {todo["Name"].nunique()} to do')
        for tmp in self.save_dir.glob('.*.parquet.tmp'):   # left by a killed run
            tmp.unlink()
        pairs = tile_orbit_pairs(todo, gedi_index)
        target, _ = target_orbits(pairs, self.orbit_months)
        orbits_per_row = pairs[target].groupby('row')['table_id'].apply(list).to_dict()
        parts_per_tile = defaultdict(list)
        for i, row in todo.iterrows():
            if orbits_per_row.get(i):
                parts_per_tile[row['Name']].append((row['geometry'], orbits_per_row[i]))
        no_orbit = sorted(set(todo['Name']) - parts_per_tile.keys())
        if no_orbit:
            self.add_pft_state.mark_many(no_orbit, info=f'pft {label}: no GEDI orbit over tile')
        tasks = [self.add_pft_one_tile(name, parts) for name, parts in parts_per_tile.items()]
        print(f'Scheduling {len(tasks)} tile fetches')
        if tasks:
            nfailed, _ = self.schedule_tasks(delayed_tasks=tasks)
            print(f'{nfailed} tiles failed after retries')
        self.add_pft_state.compact()
        print(f'state: {self.add_pft_state.path}')

    @dask.delayed
    def add_pft_one_tile(self, name: str, parts: list):
        '''add_pft_classes for one tile (parts as in get_orbit_for_one_s2_tile).'''
        try:
            res = self._fetch_one_s2_tile(name, parts, append=True, pft_classes=self.add_pft)
        except Exception as e:
            self.add_pft_state.mark_failed(name, repr(e))
            raise
        self.add_pft_state.mark_done(name, f'pft {pft_label(self.add_pft)}: {self._result_info(res, parts)}')

    def _result_info(self, res: tuple, parts: list, note: str = '') -> str:
        '''State info of a _fetch_one_s2_tile result.'''
        n, n_new, n_dup, n_water, missing = res
        n_orbits = sum(len(ids) for _, ids in parts)
        extra = (f', {n_dup} already there' if n_dup else '') + \
                (f', {n_water} old shots on water dropped' if n_water else '') + \
                (f', {len(missing)} orbits not in GEE: {" ".join(m.split("/")[-1] for m in missing)}'
                 if missing else '') + note
        return f'{n} shots (+{n_new} from {n_orbits} orbits{extra})' if n else f'no valid shots{extra}'

    def _print_plan(self, parts_per_tile: dict, files: set, prior: dict, nothing_to_fetch: list, any_orbit: set):
        '''dry_run of download_all_valid: what it would fetch, with the shots
        and requests expected from the prior (fetch_orbits).'''
        plan = pd.DataFrame([dict(tile=name, orbits=sum(len(ids) for _, ids in parts),
                                  repair=name in files, shots_per_orbit=prior.get(name))
                             for name, parts in parts_per_tile.items()],
                            columns=['tile', 'orbits', 'repair', 'shots_per_orbit'])
        plan['shots'] = plan['orbits'] * plan['shots_per_orbit']
        plan['requests'] = np.maximum(np.ceil(plan['shots'] / self.max_shots_per_request),
                                      np.ceil(plan['orbits'] / self.max_orbits_per_request))
        known = plan['shots_per_orbit'].notna()
        print(f'\n===== dry run {self.year}, orbit_months={self.orbit_months}')
        print(f'tiles to fetch: {len(plan)} ({int(plan["repair"].sum())} appended to an existing parquet, '
              f'{int((~plan["repair"]).sum())} new), {int(plan["orbits"].sum())} orbits')
        print(f'nothing to fetch: {len(nothing_to_fetch)} tiles '
              f'({sum(n not in any_orbit for n in nothing_to_fetch)} without any GEDI orbit)')
        print(f'with a {self.prior_year} prior: {int(known.sum())} tiles, {int(plan.loc[known, "orbits"].sum())} orbits, '
              f'~{plan.loc[known, "shots"].sum() / 1e6:.1f}M shots in ~{int(plan.loc[known, "requests"].sum())} requests')
        print(f'without a prior: {int((~known).sum())} tiles, {int(plan.loc[~known, "orbits"].sum())} orbits')
        by = plan.groupby('repair')[['orbits', 'shots', 'requests']].sum()
        print(by.rename(index={True: 'append', False: 'new'}).to_string())
        # the one file a dry run writes: the tiles to fetch (e.g. to copy just
        # their parquets for a repair run elsewhere with tiles=...)
        plan_file = (self.state.path if self.state is not None else self.save_dir.parent) / 'fetch_plan.tsv'
        plan.to_csv(plan_file, sep='\t', index=False)
        print(f'saved {plan_file}')

    @dask.delayed
    def get_orbit_for_one_s2_tile(self, name: str, parts: list, append: bool = False, note: str = '',
                                  shots_per_orbit: float = None):
        '''
        Download all valid GEDI shots for one S2 tile across its pre-computed
        list of overlapping orbits.

        Args:
            * name: S2 tile name
            * parts: [(geometry, orbit_ids)], one per grid row with this name
                (two for tiles split at the antimeridian). orbit_ids are GEE
                FeatureCollection IDs of the orbits that intersect the part and
                still have to be fetched (growing months only for
                orbit_months='growing'; computed once in download_all_valid).
            * append: add the shots to the tile's existing parquet (orbits an
                earlier download missed or left out)
            * note: appended to the state info
            * shots_per_orbit: expected, sizes the first request (fetch_orbits)
        '''
        try:
            res = self._fetch_one_s2_tile(name, parts, append, shots_per_orbit)
        except Exception as e:
            # re-raised so schedule_tasks still retries; a later success
            # overrides this line in the state
            if self.state is not None:
                self.state.mark_failed(name, repr(e))
            raise
        if self.state is None or res is None:
            return
        tag = ALL_MONTHS if self.orbit_months == 'all' else GROWING_MONTHS
        # the pft classes of this fetch (downloads before 2026-10-22: 1-8)
        info = self._result_info(res, parts, f', pft {pft_label(self.pft_classes)}{note}')
        self.state.mark_done(name, f'{tag}: {info}')

    def _fetch_one_s2_tile(self, name: str, parts: list, append: bool = False, shots_per_orbit: float = None,
                           pft_classes: list = None):
        '''Returns (shots in the parquet, shots added, fetched shots that were
        already there, old shots dropped as water, orbit tables GEE does not
        have), or None when the parquet already existed (not append).
        pft_classes: fetched classes, default self.pft_classes.'''
        self._ensure_authenticated()  # runs on a worker process; ensure this run's key
        file = self.save_dir / f'{name}.parquet'
        if not append and file.exists() and not self.rewrite and is_parquet_ok(file):
            print(f'{file} exists and is ok')
            return None
        old = gpd.read_parquet(file) if append and file.exists() else None
        n_water = 0
        if old is not None:     # same water filter as the GEE query
            keep = water_mask(old, self.max_water_persistence, self.drop_no_dem)
            n_water = int((~keep).sum())
            old = old[keep]
        n_old = 0 if old is None else len(old)
        if old is not None:     # float64 in files where one orbit had float ids
            old['shot_number'] = exact_shot_number(old['shot_number'])
        if old is not None and 'mgrs_tile' not in old:     # written before add_mgrs
            old['mgrs_tile'] = mgrs_100km(old.geometry.x.values, old.geometry.y.values)
        data, log = [], []
        for geometry, orbit_ids in parts:
            shots, part_log = fetch_orbits(orbit_ids, shapely_to_geojson(geometry), self.filter,
                                           shots_per_orbit, self.max_shots_per_request,
                                           self.max_orbits_per_request,
                                           pft_classes=pft_classes or self.pft_classes)
            log += part_log
            if shots is not None:
                data.append(shots)
        missing = [r['missing'] for r in log if 'missing' in r]
        n_split = sum(not r['ok'] for r in log) - len(missing)
        print(f'{name}: {sum(len(ids) for _, ids in parts)} orbits in {len(log)} requests'
              + (f' ({n_split} too large, split)' if n_split else '')
              + (f' ({len(missing)} not in GEE: {missing})' if missing else ''))
        if len(data) == 0:
            print(f'No new GEDI points found for {name}')
            if n_water:
                tmp = file.with_name(f'.{file.name}.tmp')
                old.to_parquet(tmp)
                os.replace(tmp, file)
            return n_old, 0, 0, n_water, missing
        data = pd.concat(data)
        if len(parts) > 1:      # a shot on the seam between two parts comes twice
            # with delta_time: float ids (exact_shot_number) repeat within an orbit
            data = data.drop_duplicates(subset=['shot_number', 'delta_time'])
        data = data.reset_index(drop=True)
        data['system:index'] = data['system:index'].astype(str)
        # S2 tiles overlap, so one shot lands in several tile files; this is
        # the one MGRS square it belongs to (see tools/mgrs_tile.py)
        data['mgrs_tile'] = mgrs_100km(data.geometry.x.values, data.geometry.y.values)
        n_dup = 0
        if old is not None:
            # a column that is text on one side (e.g. system:index from an older
            # download) must be text on both, or the parquet write fails
            for c in old.columns.intersection(data.columns).drop('geometry', errors='ignore'):
                if (old[c].dtype == object) != (data[c].dtype == object):
                    old[c], data[c] = old[c].astype(str), data[c].astype(str)
            # fetched shots already in the file mean an orbit (or pft class)
            # was fetched twice; with delta_time, as float ids repeat
            key = ['shot_number', 'delta_time']
            dup = pd.MultiIndex.from_frame(data[key]).isin(pd.MultiIndex.from_frame(old[key]))
            n_dup = int(dup.sum())
            data = pd.concat([old, data[~dup]], ignore_index=True)
        # temp file + rename: a killed worker or a concurrent reader (add_mgrs,
        # upload_erda) never sees a half-written <tile>.parquet
        tmp = file.with_name(f'.{file.name}.tmp')
        data.to_parquet(tmp)
        os.replace(tmp, file)
        print(f'Saved {len(data)} GEDI points for {name} after filtering')
        return len(data), len(data) - n_old, n_dup, n_water, missing

    # --- Orphaned helpers from the per-orbit refactor (see download_all_valid
    # docstring for why that approach was abandoned). Kept in case a future
    # attempt revisits orbit-level fetches with a smarter strategy (precise
    # union polygon push to GEE, batched tile clusters, etc.). Not currently
    # called from anywhere.
    @retry(requests.HTTPError, tries=6, delay=2, backoff=2, max_delay=60)
    def _download_region(self, orbit_id: str, bounds):
        '''Download orbit shots whose footprint intersects the given bbox
        (minx, miny, maxx, maxy). Retries with exponential backoff to ride
        out transient EE export-capacity errors.'''
        region = ee.Geometry.Rectangle(list(bounds))
        fc = (ee.FeatureCollection(orbit_id)
              .filterBounds(region)
              .filter(self.filter)
              .filter(ee.Filter.inList('pft_class', list(range(1, 9)))))
        return ee_fc_to_gpd(fc)

    def _download_orbit(self, orbit_id: str, bounds, depth: int = 0, max_depth: int = 6):
        '''Download an orbit's shots over `bounds`, recursively splitting the
        bbox into quadrants when EE cannot materialize the table (e.g. HTTP 500
        "No space left on device" from an export too large for EE scratch).'''
        try:
            return self._download_region(orbit_id, bounds)
        except requests.HTTPError:
            if depth >= max_depth:
                raise
            minx, miny, maxx, maxy = bounds
            midx, midy = (minx + maxx) / 2, (miny + maxy) / 2
            quads = [
                (minx, miny, midx, midy),
                (midx, miny, maxx, midy),
                (minx, midy, midx, maxy),
                (midx, midy, maxx, maxy),
            ]
            frames = []
            for q in quads:
                df = self._download_orbit(orbit_id, q, depth + 1, max_depth)
                if df is not None and len(df):
                    frames.append(df)
            if not frames:
                return None
            return pd.concat(frames, ignore_index=True)

    def add_slope(self):
        '''
        Add slope to the GEDI data, where the slope is calculated from the DEM data (Copernicus DEM + NASADEM).
        NOTE: slope information indicates how representative the GEDI vertical profile is, higher slope means less representative.
        Args:
            * location_dir: the path to the location files
                NOTE: the location files should be in the format of <tile_id>.parquet
            * dem_meta_file: the path to the DEM metadata file
                NOTE: the DEM metadata file should be in the format of <collection_id>_items.parquet
            * save_dir: the path to save the GEDI data with slope
                NOTE: the save directory should be the same as the location files directory
        '''
        cluster = LocalCluster()
        client = Client(cluster)  # timeout
        print(client)
        location_files = Path(self.location_dir).expanduser().glob('*.parquet')
        dem_meta_file = Path(self.dem_meta_file).expanduser()
        self.dem_df = get_aux_df(dem_meta_file, 'cop-dem-glo-30', time_col='datetime')
        self.dem_df['datetime'] = self.dem_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

        location_files = check_unfinished_files(location_files, self.save_dir, check_exists=True)
        test_tile = '10TEL'
        location_files = [file for file in location_files if test_tile in file.stem]
        tasks = []
        for file in location_files:
            tasks.append(self.get_dem(file))
        self.schedule_tasks(delayed_tasks=tasks)

    @dask.delayed
    @retry(requests.HTTPError, tries=10, delay=1)
    def get_dem(self, file: Path):
        '''
        Get the DEM data for the GEDI points
        #TODO: not sure how it works for large tiles (more than 40k points)
        '''
        output_file = self.save_dir / f'{file.stem}.parquet'
        if output_file.exists() and not self.rewrite:
            print(f'{output_file} exists, skipping')
            return
        tile_id = file.stem
        epsg = get_epsg_from_tile(tile_id)
        loc_df = gpd.read_parquet(file)
        loc_ = loc_df.to_crs(epsg)
        total_bounds = buffer_and_snap_bounds(loc_.geometry, 60, 30)  # buffer 2 pixels
        total_bounds = get_total_bounds(total_bounds)
        bbox = box(*loc_df.geometry.total_bounds)
        dem_df = self.dem_df[self.dem_df.geometry.intersects(bbox)]

        if not dem_df.geometry.union_all().covers(bbox):
            dem_items = api.search(collections=['nasadem'], intersects=bbox).item_collection()
            dem_items.asset_name = 'elevation'
        else:
            dem_items = row_to_stac_item(dem_df, ['datetime'])
            dem_items = pystac.item_collection.ItemCollection(dem_items)
            dem_items.asset_name = 'data'

        if len(dem_items.items) == 0:
            loc_df['slope'] = np.nan
            loc_df.to_parquet(output_file)
            print(f'{output_file} does not have DEM data, skipping')
            return

        dem_image = get_patch(
            dem_items.items, [dem_items.asset_name],
            resolution=30, epsg=epsg, bounds=total_bounds, dtype='float32', fill_value=np.float32(np.nan))
        dem_image = dem_image.max(dim='time', skipna=True).squeeze()
        if dem_image.shape[0] == 0:
            loc_df['slope'] = np.nan
            loc_df.to_parquet(output_file)
            print(f'{output_file} does not have DEM data, skipping')
            return
        dem_image.attrs['res'] = 30
        slope_da = slope(dem_image)
        target_x = xr.DataArray(loc_.geometry.x.values, dims="points")
        target_y = xr.DataArray(loc_.geometry.y.values, dims="points")
        sampled_values = slope_da.sel(x=target_x, y=target_y, method='nearest')
        sampled_values = sampled_values.compute()
        loc_df['slope'] = sampled_values.values
        loc_df.to_parquet(output_file)

    def check_slope_distribution(self):
        '''
        Check the distribution of the slope data
        '''
        input_dir = Path(self.input_dir).expanduser()
        columns = ['geometry', 'slope']
        df = read_parquets_with_file_name(input_dir, columns)
        before_slope_filter = df.groupby('Name')['geometry'].count().reset_index(name='count')
        after_slope_filter = df[df['slope'] < 20].groupby('Name')['geometry'].count().reset_index(name='count')
        valid_tiles_before = before_slope_filter[before_slope_filter['count'] > 200]['Name'].unique()
        valid_tiles_after = after_slope_filter[after_slope_filter['count'] > 200]['Name'].unique()
        affected_tiles = set(valid_tiles_before) - set(valid_tiles_after)
        with open(input_dir.parent / 'figures/affected_tiles_by_slope_filter.txt', 'w') as f:
            for tile in affected_tiles:
                f.write(f'{tile}\n')
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        before_slope_filter['count'].hist(ax=axes[0], bins=100, label='before slope filter')
        after_slope_filter['count'].hist(ax=axes[1], bins=100, label='after slope filter')
        for ax in axes:
            ax.set_yscale('log')
            ax.set_ylim(0, 1e5)
            ax.set_xlabel('Number of GEDI points')
            ax.set_ylabel('#tiles')
            ax.legend()
        plt.savefig(input_dir.parent / 'figures/slope_distribution.png')
        
        deploy_status_dir = Path(self.deploy_status_dir).expanduser()
        s2_files = list(deploy_status_dir.glob('deploy_status_*.parquet'))
        latest_s2_file = sorted(s2_files, key=lambda x: pd.to_datetime(x.stem.split('_')[2]))[-1]
        s2_df = gpd.read_parquet(latest_s2_file)
        after_slope_filter = after_slope_filter.set_index('Name')
        after_slope_filter = after_slope_filter.rename(columns={'count': f'gedi_count_after_slope_filter_{self.year}'})
        s2_df = s2_df.join(after_slope_filter)
        s2_df[f'gedi_count_after_slope_filter_{self.year}'] = s2_df[f'gedi_count_after_slope_filter_{self.year}'].fillna(0).astype(int)
        s2_df.to_parquet(latest_s2_file.with_stem(f'deploy_status_{datetime.now().strftime("%Y-%m-%dT%H")}'))
        return 



    def get_used_gedi_points(self):
        '''
        Get the GEDI points used for GVS training, evaluation and calibration
        '''
        cluster = LocalCluster()
        client = Client(cluster)  # timeout
        print(client)
        s2_tile_file = Path(self.s2_grid_file).expanduser()
        used_parq_dir = Path(self.used_parq_dir).expanduser()
        s2_tile = gpd.read_parquet(s2_tile_file, columns=['Name', 'geometry'])
        temp_dir = self.save_dir / 'temp'
        temp_dir.mkdir(exist_ok=True, parents=True)
        gedi_parq_files = list(used_parq_dir.glob('*_v1.parquet'))
        for gedi_parq_file in gedi_parq_files:
            gedi_df = pd.read_parquet(gedi_parq_file, columns=['lat', 'lon'])
            split_name = gedi_parq_file.stem.split('_')[0]
            self.get_used_gedi_points_for_one_parquet(gedi_df, s2_tile, temp_dir, split_name)
        # merge points from all splits into one file

        @dask.delayed
        def process_tile(tile_id):
            parquet_files = list(temp_dir.glob(f'{tile_id}*.parquet'))
            file = self.save_dir / f'{tile_id}.parquet'
            if len(parquet_files) == 0:  # or (file.exists() and is_parquet_ok(file)):
                return
            gedi_df = dgp.read_parquet(parquet_files, gather_spatial_partitions=False).compute()
            gedi_df.to_parquet(file)
            print(f'Saved {len(gedi_df)} GEDI points to {file}')

        tasks = []
        for tile_id in s2_tile['Name']:
            tasks.append(process_tile(tile_id))
        self.schedule_tasks(delayed_tasks=tasks)

    def get_used_gedi_points_for_one_parquet(
            self, df: pd.DataFrame, s2_tile: pd.DataFrame, temp_dir: Path, split_name: str):
        '''
        Get the GEDI points used for GVS training, evaluation and calibration
        '''
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')
        gdf_ = gpd.sjoin(gdf, s2_tile, how='left', predicate='intersects')
        gdf_.groupby('Name').apply(lambda x: x.to_parquet(temp_dir / f'{x.Name.iloc[0]}_{split_name}.parquet'))
        return df

    def get_tiles_covered_by_gedi(self):
        '''
        Get the tiles covered by GEDI

        Args:
            * s2_grid_file: the path to the S2 tiles file, should have columns: Name, geometry
            * gedi_table_index_file: the path to the GEDI table index file, should have columns: system:index, table_id, start_time, end_time, geometry
            * save_dir: the path to save the tiles covered by GEDI

        Output:
            * tiles_covered_by_gedi.txt: a text file with the tile names in the GEDI range
            Each line is a tile name
        '''
        s2_grid_file = Path(self.s2_grid_file).expanduser()
        s2_tiles = gpd.read_parquet(s2_grid_file)
        gedi_table_index_file = Path(self.gedi_table_index_file).expanduser()
        gedi_table_index = gpd.read_parquet(gedi_table_index_file)

        tiles_covered_by_gedi = s2_tiles[s2_tiles.intersects(gedi_table_index['geometry'])]['Name'].unique()
        tiles_covered_by_gedi.to_csv(self.save_dir / 'tiles_covered_by_gedi.txt', header=None, index=None, sep=' ', mode='w')


def get_unique_shotnumbers(tiles: List[Path]):
    
    records = []
    for t in tiles:
        shots = pq.read_table(t, columns=["shot_number"]).to_pandas()
        shots["_tile"] = t.name
        records.append(shots)

    all_shots = pd.concat(records)
    print(f"Total rows: {len(all_shots)}, unique shot_number: {all_shots['shot_number'].nunique()}")

    # 每个 shot_number 只保留第一个出现的 tile
    keep = all_shots.drop_duplicates(subset=["shot_number"], keep="first")
    print(f'Before: {len(all_shots)}, after: {len(keep)}')
    return keep


def dedup_shots(parq_dir: str, **kwargs):
    '''
    Remove duplicated shoots. Not applied, for furture use
    '''
    parq_dir = Path(parq_dir).expanduser()
    tiles = sorted(parq_dir.glob("*.parquet"))
    keep = get_unique_shotnumbers(tiles)
    for t in tiles:
        tile_name = t.name
        keep_shots = keep[keep["_tile"] == tile_name]["shot_number"]
        
        df = pd.read_parquet(t)
        before = len(df)
        df = df[df["shot_number"].isin(keep_shots.values)]
        after = len(df)
        
        if before != after:
            print(f"{tile_name}: {before} → {after} (removed {before - after})")
            df.to_parquet(str(t) + ".tmp")
            os.replace(str(t) + ".tmp", str(t))



def _drop_water_one(fp: Path, max_water_persistence, drop_no_dem):
    cols = ['digital_elevation_model', 'landsat_water_persistence']
    head = pq.read_table(fp, columns=cols, partitioning=None).to_pandas()
    keep = water_mask(head, max_water_persistence, drop_no_dem)
    if keep.all():
        return fp.stem, len(keep), len(keep)
    df = gpd.read_parquet(fp)
    df = df[keep.to_numpy()]
    tmp = fp.with_name(f'.{fp.name}.tmp')
    df.to_parquet(tmp)
    os.replace(tmp, fp)
    return fp.stem, len(keep), len(df)


def drop_water_shots(parq_dir: str, max_water_persistence: float = 50, drop_no_dem: bool = True,
                     workers: int = 8, **kwargs) -> pd.DataFrame:
    '''
    Apply the download water filter (water_filter_expr) in place to tile
    parquets downloaded without it. Idempotent: a file with nothing to drop is
    not rewritten. Writes <parq_dir>_state/water_filter_<UTC time>.tsv with
    the shots per tile before and after. Do not run it while a download
    writes to the same directory.
    '''
    from concurrent.futures import ProcessPoolExecutor
    parq_dir = Path(parq_dir).expanduser()
    files = sorted(p for p in parq_dir.glob('*.parquet') if is_parquet_ok(p))
    print(f'{len(files)} tiles in {parq_dir}, filter: {water_filter_expr(max_water_persistence, drop_no_dem)}')
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_drop_water_one, files, [max_water_persistence] * len(files),
                             [drop_no_dem] * len(files), chunksize=8))
    report = pd.DataFrame(rows, columns=['tile', 'shots_before', 'shots_after'])
    state_dir = Path(f'{parq_dir}_state')
    state_dir.mkdir(exist_ok=True)
    out = state_dir / f'water_filter_{datetime.utcnow():%Y%m%dT%H%M%SZ}.tsv'
    report.to_csv(out, sep='\t', index=False)
    before, after = report['shots_before'].sum(), report['shots_after'].sum()
    print(f'{(report.shots_before > report.shots_after).sum()} tiles rewritten, '
          f'{before - after} of {before} shots dropped ({(before - after) / max(before, 1):.2%}) -> {out}')
    return report
