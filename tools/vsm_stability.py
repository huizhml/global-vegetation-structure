"""Stability test of the VSM predictions on a small set of S2 tiles per WWF biome.

First use (2024): re-predict a few tiles per biome with the current deploy code and
compare, at the 2024 GEDI shots of the conformal cal/test sets, three versions:

    A  new prediction, no per-tile shift
    B  the bias_corrected release (original + per-tile mean shift + blending)
    C  A + per-tile, per-RH median(GEDI - Q1), from the tile's other GEDI shots

    select_stability_tiles   tiles per biome from the cal/test tiles of the CP points
    eval_stability           coverage / bias / MAE per variant, RH and biome

    python -m tools.run run=select_stability_tiles
    python -m tools.run run=eval_stability

Second use (2017-2026): inter-annual stability per biome. The same random pixels of
each tile are read from the yearly prediction of every year (new prediction with the
current code, 2020 from the release), plus the prediction at the GEDI shots of the
years with GEDI. The prediction keeps only these points and writes no GeoTIFF
(S2DatasetStream sample_points_file, run_lumi_inference.sh predict_sample_worklist).

    select_stability_tiles        (keep_file + wwf_biomes: extend the 2024 tiles)
    write_stability_worklist      tile x year lines for the LUMI array
    make_stability_pixels         fixed random pixels per tile (kept)
    write_stability_points        pixels + GEDI shots of one tile-year, kept by the prediction
    sample_stability_tile         prediction at the pixels / GEDI shots of one tile-year
    summarize_stability_years     year offsets, per-pixel SD, GEDI accuracy per biome
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import shapely
from pyproj import Transformer

from tools.cp_points import _read_band

SPLITS = ('cal', 'test')
RHS = tuple(range(101))
VARIANTS = ('A', 'B', 'C')


def _tile_biomes(points_dir: Path) -> pd.DataFrame:
    """One row per tile: split, number of shots, majority WWF biome (1-14) of its shots."""
    rows = []
    for split in SPLITS:
        for f in sorted((points_dir / split).glob('*.parquet')):
            b = pq.read_table(f, columns=['BIOME']).column('BIOME').to_pandas()
            b = b[b.between(1, 14)]
            rows.append(dict(tile=f.stem, split=split, n_shots=len(b),
                             biome=int(b.mode().iloc[0]) if len(b) else 0,
                             biome_share=float((b == b.mode().iloc[0]).mean()) if len(b) else 0.))
    return pd.DataFrame(rows)


def _wwf_tiles(grid_file: Path, biome_file: Path, biome: int, n_points: int, min_share: float,
               seed: int) -> pd.DataFrame:
    """Tiles whose land (points in a WWF biome 1-14) is >= min_share of WWF biome `biome`,
    from n_points random points per tile; candidates = tiles with a point of that biome on a
    coarse 5x5 lattice first. -> tile, biome_share, land_share."""
    grid = gpd.read_parquet(grid_file)[['Name', 'geometry']].to_crs(epsg=4326)
    eco = gpd.read_file(biome_file)[['BIOME', 'geometry']].to_crs(epsg=4326)
    lattice = _points_in(grid, 25, seed, regular=True)
    cand = gpd.sjoin(lattice, eco[eco.BIOME == biome], predicate='within').tile.unique()
    pts = _points_in(grid[grid.Name.isin(cand)], n_points, seed)
    j = gpd.sjoin(pts, eco, predicate='within', how='left').drop_duplicates('point_key')
    j['land'] = j.BIOME.between(1, 14)
    rows = []
    for tile, g in j.groupby('tile'):
        land = g[g.land]
        rows.append(dict(tile=tile, land_share=len(land) / len(g),
                         biome_share=float((land.BIOME == biome).mean()) if len(land) else 0.))
    out = pd.DataFrame(rows)
    return out[(out.biome_share >= min_share) & (out.land_share >= 0.5)]


def _points_in(grid: gpd.GeoDataFrame, n: int, seed: int, regular: bool = False) -> gpd.GeoDataFrame:
    """n points per tile (Name) in lon/lat, uniform (or a sqrt(n) x sqrt(n) lattice) over the
    tile footprint; antimeridian tiles (several rows per Name) split by part area."""
    rows = []
    for name, g in grid.groupby('Name'):
        rng = np.random.default_rng([seed, int.from_bytes(name.encode(), 'little')])
        geoms = list(g.geometry)
        areas = np.array([geom.area for geom in geoms])
        counts = np.round(n * areas / areas.sum()).astype(int)
        counts[np.argmax(areas)] += n - counts.sum()
        for geom, k in zip(geoms, counts):
            if k <= 0:
                continue
            x0, y0, x1, y1 = geom.bounds
            if regular:
                m = int(np.ceil(np.sqrt(k)))
                xs, ys = np.meshgrid(np.linspace(x0, x1, m + 2)[1:-1], np.linspace(y0, y1, m + 2)[1:-1])
                cand = gpd.points_from_xy(xs.ravel(), ys.ravel())
                xy = [(p.x, p.y) for p in cand if geom.contains(p)]
            else:
                xy = []
                while len(xy) < k:
                    xs = rng.uniform(x0, x1, 4 * k)
                    ys = rng.uniform(y0, y1, 4 * k)
                    inside = shapely.contains_xy(geom, xs, ys)
                    xy += list(zip(xs[inside], ys[inside]))
                xy = xy[:k]
            rows += [(name, x, y) for x, y in xy]
    df = pd.DataFrame(rows, columns=['tile', 'lon', 'lat'])
    df['point_key'] = np.arange(len(df))
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)


def select_stability_tiles(points_dir: str, out_file: str, per_biome: int = 3,
                           min_shots: int = 500, seed: int = 42, keep_file: str = None,
                           wwf_biomes=(), grid_file: str = None, biome_file: str = None,
                           wwf_points: int = 200, wwf_min_share: float = 0.8, **kwargs):
    """per_biome random tiles of each WWF biome (majority biome of the tile's CP shots,
    cal and test tiles together) with at least min_shots shots; all of them when a biome
    has fewer. keep_file: an earlier selection (tsv) whose tiles are kept, only the rest
    up to per_biome is drawn. wwf_biomes: biomes without CP tiles (Tundra) are drawn from
    the S2 grid by WWF polygon share instead (split 'none', n_shots 0).
    Writes <out_file> (tsv: tile, split, biome, n_shots, biome_share, source) and the
    plain tile list next to it (.txt)."""
    points_dir = Path(points_dir).expanduser()
    tiles = _tile_biomes(points_dir)
    ok = tiles[(tiles.biome > 0) & (tiles.n_shots >= min_shots)]
    keep = pd.read_csv(Path(keep_file).expanduser(), sep='\t') if keep_file else ok.iloc[:0]
    keep = keep.assign(source='cp')
    rng = np.random.default_rng(seed)
    picked = [keep]
    for biome, g in ok.groupby('biome'):
        g = g[~g.tile.isin(keep.tile)].sort_values('tile')
        need = per_biome - int((keep.biome == biome).sum())
        if need > 0:
            picked.append(g.iloc[np.sort(rng.choice(len(g), size=min(need, len(g)), replace=False))].assign(source='cp'))
    for biome in wwf_biomes:
        have = sum(int((p.biome == biome).sum()) for p in picked)
        if have >= per_biome:
            continue
        w = _wwf_tiles(Path(grid_file).expanduser(), Path(biome_file).expanduser(), int(biome),
                       wwf_points, wwf_min_share, seed).sort_values('tile')
        w = w[~w.tile.isin(pd.concat(picked).tile)]
        print(f'biome {biome}: {len(w)} WWF candidate tiles (share >= {wwf_min_share})')
        w = w.iloc[np.sort(rng.choice(len(w), size=min(per_biome - have, len(w)), replace=False))]
        picked.append(pd.DataFrame(dict(tile=w.tile, split='none', n_shots=0, biome=int(biome),
                                        biome_share=w.biome_share, source='wwf')))
    sel = pd.concat(picked)[['tile', 'split', 'biome', 'n_shots', 'biome_share', 'source']]
    sel = sel.sort_values(['biome', 'tile'])

    out_file = Path(out_file).expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out_file, sep='\t', index=False)
    out_file.with_suffix('.txt').write_text('\n'.join(sel.tile) + '\n')
    summary = tiles.groupby('biome').agg(tiles=('tile', 'size'), eligible=('n_shots', lambda n: int((n >= min_shots).sum())))
    print(summary.join(sel.groupby('biome').size().rename('picked'), how='outer').fillna(0).astype(int).to_string())
    print(f'{len(sel)} tiles -> {out_file}')
    print(sel.to_string(index=False))


# ---------------------------------------------------------------------------
# evaluation


def _read_at(cog_dir: Path, lon: np.ndarray, lat: np.ndarray, quantiles, threads: int) -> dict:
    """{f'RH{rh}_Q{q}': metres at the points} for all RHs and the given quantiles."""
    with rasterio.open(cog_dir / 'RH0_Q1.tif') as ref:
        xs, ys = Transformer.from_crs(4326, ref.crs, always_xy=True).transform(lon, lat)
        rows, cols = rasterio.transform.rowcol(ref.transform, xs, ys)
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        with ThreadPoolExecutor(threads) as pool:
            futs = {pool.submit(_read_band, str(cog_dir / f'RH{rh}_Q{q}.tif'), ref, rows, cols): f'RH{rh}_Q{q}'
                    for rh in RHS for q in quantiles}
            return {futs[f]: f.result() for f in as_completed(futs)}


def _eval_tile(tile: str, points_file: Path, gedi_file: Path, cog_dir: Path, out_dir: Path,
               n_delta_shots: int, sensitivity_min: float, seed: int, threads: int):
    """Evaluation points with A (new prediction) next to B (release) -> <out_dir>/points/<tile>.parquet;
    per-RH median(GEDI - A Q1) of the tile's other GEDI shots -> <out_dir>/deltas/<tile>.parquet."""
    pts = gpd.read_parquet(points_file)
    new = _read_at(cog_dir, pts.lon.values, pts.lat.values, (0, 1, 2), threads)
    pts = pts.assign(**{f'A_{k}': v for k, v in new.items()})

    # delta shots: same filters as the CP points, minus the evaluation shots
    shots = gpd.read_parquet(gedi_file, columns=['shot_number', 'sensitivity', 'mgrs_tile', 'geometry']
                             + [f'rh{r}' for r in RHS],
                             filters=[('mgrs_tile', '==', tile), ('sensitivity', '>=', sensitivity_min)])
    shots = shots[~shots.shot_number.isin(pts.shot_number)].to_crs(epsg=4326)
    if len(shots) > n_delta_shots:
        rng = np.random.default_rng([seed, int.from_bytes(tile.encode(), 'little')])
        shots = shots.iloc[np.sort(rng.choice(len(shots), size=n_delta_shots, replace=False))]
    q1 = _read_at(cog_dir, shots.geometry.x.values, shots.geometry.y.values, (1,), threads)
    deltas = []
    for r in RHS:
        err = shots[f'rh{r}'].values - q1[f'RH{r}_Q1']
        err = err[~np.isnan(err)]
        deltas.append(dict(tile=tile, rh=r, n=len(err), delta=float(np.median(err)) if len(err) else 0.))
    for sub in ('points', 'deltas'):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    pts.to_parquet(out_dir / 'points' / f'{tile}.parquet')
    pd.DataFrame(deltas).to_parquet(out_dir / 'deltas' / f'{tile}.parquet')
    return f'{len(pts)} points, {len(shots)} delta shots'


def _metrics(y, q0, q1, q2) -> dict:
    lo, hi = np.minimum(q0, q2), np.maximum(q0, q2)
    return dict(n=len(y), coverage=np.mean((lo <= y) & (y <= hi)), above=np.mean(y > hi),
                below=np.mean(y < lo), bias=np.median(q1 - y), mae=np.mean(np.abs(q1 - y)),
                width=np.mean(hi - lo))


def summarize_stability(out_dir: Path, tiles: pd.DataFrame) -> pd.DataFrame:
    """Per variant x RH x group (tile, biome, all): metrics on points valid in every variant."""
    rows = []
    for t in tiles.itertuples():
        pf = out_dir / 'points' / f'{t.tile}.parquet'
        if not pf.exists():
            continue
        p = pd.read_parquet(pf)
        d = pd.read_parquet(out_dir / 'deltas' / f'{t.tile}.parquet').set_index('rh').delta
        for r in RHS:
            y = p[f'rh{r}'].values
            v = {'B': [p[f'RH{r}_Q{q}'].values for q in (0, 1, 2)],
                 'A': [p[f'A_RH{r}_Q{q}'].values for q in (0, 1, 2)]}
            v['C'] = [a + d[r] for a in v['A']]
            ok = ~np.isnan(y) & np.all([~np.isnan(a) for vv in v.values() for a in vv], axis=0)
            for name, (q0, q1, q2) in v.items():
                rows.append(dict(variant=name, rh=r, tile=t.tile, biome=t.biome,
                                 **_metrics(y[ok], q0[ok], q1[ok], q2[ok])))
    per_tile = pd.DataFrame(rows)

    def pooled(g):      # point-weighted mean of the per-tile metrics (bias: weighted mean of medians)
        w = g.n / g.n.sum()
        return pd.Series({'n': g.n.sum(), 'tiles': len(g),
                          **{m: (g[m] * w).sum() for m in ('coverage', 'above', 'below', 'bias', 'mae', 'width')}})
    by_biome = per_tile.groupby(['variant', 'rh', 'biome']).apply(pooled, include_groups=False).reset_index()
    overall = per_tile.groupby(['variant', 'rh']).apply(pooled, include_groups=False).reset_index().assign(biome=0)
    return per_tile, pd.concat([overall, by_biome], ignore_index=True)


def eval_stability(tiles_file: str, points_dir: str, gedi_dir: str, cog_root: str, out_dir: str,
                   report_dir: str, n_delta_shots: int = 5000, sensitivity_min: float = 0.95, seed: int = 42,
                   workers: int = 8, threads: int = 8, **kwargs):
    """Variants A / B / C at the CP points of the tiles in tiles_file (tsv from
    select_stability_tiles). points_dir holds <split>/<tile>.parquet (CP points with the
    release Q0/Q1/Q2 = B), cog_root <tile>/RH*_Q*.tif of the new prediction (A).
    Writes <out_dir>/points and <out_dir>/deltas per tile, and the summaries
    <report_dir>/stability_per_tile.parquet, stability_summary.tsv (biome 0 = all tiles).
    Tiles already evaluated are skipped."""
    points_dir, gedi_dir = Path(points_dir).expanduser(), Path(gedi_dir).expanduser()
    cog_root, out_dir = Path(cog_root).expanduser(), Path(out_dir).expanduser()
    tiles = pd.read_csv(Path(tiles_file).expanduser(), sep='\t')
    todo = [t for t in tiles.itertuples()
            if not (out_dir / 'points' / f'{t.tile}.parquet').exists()
            and len(list((cog_root / t.tile).glob('RH*_Q[012].tif'))) == 303]
    missing = [t.tile for t in tiles.itertuples() if len(list((cog_root / t.tile).glob('RH*_Q[012].tif'))) < 303]
    print(f'{len(tiles)} tiles, {len(todo)} to evaluate, {len(missing)} without a complete prediction {missing}')
    with ProcessPoolExecutor(workers) as pool:
        futs = {pool.submit(_eval_tile, t.tile, points_dir / t.split / f'{t.tile}.parquet',
                            gedi_dir / f'{t.tile}.parquet', cog_root / t.tile, out_dir,
                            n_delta_shots, sensitivity_min, seed, threads): t.tile for t in todo}
        for f in as_completed(futs):
            try:
                print(futs[f], f.result())
            except Exception as e:
                print(futs[f], f'FAILED {e!r}')

    per_tile, summary = summarize_stability(out_dir, tiles)
    report_dir = Path(report_dir).expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    per_tile.to_parquet(report_dir / 'stability_per_tile.parquet')
    summary.to_csv(report_dir / 'stability_summary.tsv', sep='\t', index=False)
    s = summary[(summary.biome == 0) & summary.rh.isin([0, 10, 25, 50, 75, 90, 95, 98, 99, 100])]
    pd.set_option('display.width', 200, 'display.precision', 3)
    for m in ('coverage', 'bias', 'mae', 'width'):
        print(f'\n{m} (all tiles)')
        print(s.pivot(index='rh', columns='variant', values=m).to_string())


# ---------------------------------------------------------------------------
# inter-annual stability (2017-2026)

YEAR_RHS = (10, 25, 50, 75, 90, 98)
QUANTILES = (0, 1, 2)


def write_stability_worklist(tiles_file: str, years, skip_years, out_file: str, **kwargs):
    """<out_file> (tsv, header tile, year): one line per tile and year in `years` minus
    `skip_years` (taken from an existing product instead). Line N+1 = array task N."""
    tiles = pd.read_csv(Path(tiles_file).expanduser(), sep='\t')
    years = [y for y in years if y not in set(skip_years)]
    wl = pd.DataFrame([(t, y) for y in years for t in tiles.tile], columns=['tile', 'year'])
    out_file = Path(out_file).expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    wl.to_csv(out_file, sep='\t', index=False)
    print(f'{len(tiles)} tiles x {len(years)} years {years} = {len(wl)} lines -> {out_file}')


def make_stability_pixels(tiles_file: str, grid_file: str, biome_file: str, out_dir: str,
                          n_pixels: int = 20000, seed: int = 42, **kwargs):
    """n_pixels random locations per tile, uniform over the tile footprint of the S2 grid
    -> <out_dir>/<tile>.parquet (point_id, lon, lat, BIOME = WWF biome of the location,
    0 outside every polygon). Every year is read at these locations. Existing files are
    kept, so the locations never change once written."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = pd.read_csv(Path(tiles_file).expanduser(), sep='\t')
    todo = [t for t in tiles.tile if not (out_dir / f'{t}.parquet').exists()]
    print(f'{len(tiles)} tiles, {len(todo)} without pixels')
    if not todo:
        return
    grid = gpd.read_parquet(Path(grid_file).expanduser())[['Name', 'geometry']].to_crs(epsg=4326)
    eco = gpd.read_file(Path(biome_file).expanduser())[['BIOME', 'geometry']].to_crs(epsg=4326)
    for tile in todo:
        pts = _points_in(grid[grid.Name == tile], n_pixels, seed)
        j = gpd.sjoin(pts, eco, predicate='within', how='left').drop_duplicates('point_key')
        df = pd.DataFrame(dict(point_id=np.arange(len(j), dtype=np.int32), lon=j.lon.values, lat=j.lat.values,
                               BIOME=j.BIOME.fillna(0).astype(np.int16).values))
        df.to_parquet(out_dir / f'{tile}.parquet', index=False)
        print(tile, len(df), 'pixels, biomes', df.BIOME.value_counts().to_dict())


def _band_paths(tile: str, year: int, pred_dir, manifest, products_root, rhs, quantiles) -> dict:
    """{'RH<r>_Q<q>': path}: <pred_dir>/RH<r>_Q<q>.tif, or from a release manifest
    (tools/release_manifest.py: q<q>_root relative to products_root, file_suffix)."""
    if manifest is None:
        d = Path(str(pred_dir).format(tile=tile, year=year)).expanduser()
        return {f'RH{r}_Q{q}': d / f'RH{r}_Q{q}.tif' for r in rhs for q in quantiles}
    m = pd.read_csv(Path(str(manifest).format(year=year)).expanduser(), sep='\t').set_index('tile_id')
    if tile not in m.index:
        raise FileNotFoundError(f'{tile} not in {manifest}')
    row, root = m.loc[tile], Path(products_root).expanduser()
    return {f'RH{r}_Q{q}': root / row[f'q{q}_root'] / tile / f'RH{r}_Q{q}{row.file_suffix}'
            for r in rhs for q in quantiles}


def _read_points(paths: dict, lon, lat, threads: int) -> dict:
    """{band: metres} at lon/lat, NaN for nodata / outside; the grid of the first Q1 band is
    the reference (all bands must share it)."""
    ref_path = next(p for k, p in paths.items() if k.endswith('_Q1'))
    with rasterio.open(ref_path) as ref:
        xs, ys = Transformer.from_crs(4326, ref.crs, always_xy=True).transform(lon, lat)
        rows, cols = rasterio.transform.rowcol(ref.transform, xs, ys)
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        with ThreadPoolExecutor(threads) as pool:
            futs = {pool.submit(_read_band, str(p), ref, rows, cols): k for k, p in paths.items()}
            return {futs[f]: f.result() for f in as_completed(futs)}


def _gedi_shots(tile: str, year: int, gedi_dir, rhs, n_gedi: int, sensitivity_min: float, seed: int):
    """Up to n_gedi random shots (sensitivity >= sensitivity_min) of the tile-year, or None.
    Same seed rule for every year and source, so a rerun picks the same shots."""
    gf = Path(str(gedi_dir).format(year=year)).expanduser() / f'{tile}.parquet' if gedi_dir else None
    if gf is None or not gf.exists():
        return None
    shots = gpd.read_parquet(gf, columns=['shot_number', 'sensitivity', 'geometry'] + [f'rh{r}' for r in rhs],
                             filters=[('sensitivity', '>=', sensitivity_min)]).to_crs(epsg=4326)
    if len(shots) > n_gedi:
        rng = np.random.default_rng([seed, year, int.from_bytes(tile.encode(), 'little')])
        shots = shots.iloc[np.sort(rng.choice(len(shots), size=n_gedi, replace=False))]
    if not len(shots):
        return None
    return pd.DataFrame({'shot_number': shots.shot_number.values, 'lon': shots.geometry.x.values,
                         'lat': shots.geometry.y.values, **{f'rh{r}': shots[f'rh{r}'].values for r in rhs}})


def write_stability_points(tile: str, year: int, pixels_dir: str, out_file: str, gedi_dir: str = None,
                           rhs=YEAR_RHS, n_gedi: int = 5000, sensitivity_min: float = 0.95, seed: int = 42,
                           **kwargs):
    """Points the prediction of one tile-year is kept at (predict with
    --data.init_args.sample_points_file): the fixed pixels (kind 'pixel', point_id) and
    the year's GEDI shots (kind 'gedi', shot_number, rh<r>) -> <out_file> (parquet)."""
    year = int(year)
    px = pd.read_parquet(Path(pixels_dir).expanduser() / f'{tile}.parquet')
    parts = [pd.DataFrame({'kind': 'pixel', 'point_id': px.point_id.values, 'lon': px.lon.values, 'lat': px.lat.values})]
    g = _gedi_shots(tile, year, gedi_dir, rhs, n_gedi, sensitivity_min, seed)
    if g is not None:
        parts.append(g.assign(kind='gedi', point_id=-1))
    pts = pd.concat(parts, ignore_index=True)
    out_file = Path(out_file).expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pts.to_parquet(out_file, index=False)
    print(f'{tile} {year}: {len(px)} pixels, {0 if g is None else len(g)} GEDI shots -> {out_file}')


def _from_samples(samples: pd.DataFrame, keys) -> dict:
    """{band: metres} from the int16 dm columns written by the prediction (32767 = nodata)."""
    out = {}
    for k in keys:
        v = samples[k].values.astype(np.float32)
        v[v == 32767] = np.nan
        out[k] = v / 10.
    return out


def sample_stability_tile(tile: str, year: int, pixels_dir: str, out_dir: str, pred_dir: str = None,
                          samples_file: str = None, manifest: str = None, products_root: str = None,
                          gedi_dir: str = None, rhs=YEAR_RHS, quantiles=QUANTILES, n_gedi: int = 5000,
                          sensitivity_min: float = 0.95, seed: int = 42, threads: int = 8, **kwargs):
    """Prediction of one tile-year at the fixed pixels -> <out_dir>/pixel_values/<year>/<tile>.parquet
    (point_id + RH<r>_Q<q>, metres) and at the GEDI shots (when the tile-year has them)
    -> <out_dir>/gedi_values/<year>/<tile>.parquet (shot_number, lon, lat, rh<r>, RH<r>_Q<q>).
    Source: samples_file (the prediction kept at the points of write_stability_points, no
    GeoTIFF), or rasters in pred_dir ({tile}, {year} allowed) or a release manifest (2020)."""
    year = int(year)
    out_dir = Path(out_dir).expanduser()
    keys = sorted((f'RH{r}_Q{q}' for r in rhs for q in quantiles), key=_band_order)
    px = pd.read_parquet(Path(pixels_dir).expanduser() / f'{tile}.parquet')
    if samples_file:
        smp = pd.read_parquet(Path(samples_file).expanduser())
        sp = smp[smp.kind == 'pixel'].set_index('point_id').loc[px.point_id.values]
        vals = _from_samples(sp, keys)
        sg = smp[smp.kind == 'gedi']
        gedi = sg[['shot_number', 'lon', 'lat'] + [f'rh{r}' for r in rhs]].reset_index(drop=True) if len(sg) else None
        gvals = _from_samples(sg, keys) if len(sg) else None
    else:
        paths = _band_paths(tile, year, pred_dir, manifest, products_root, rhs, quantiles)
        missing = [str(p) for p in paths.values() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f'{tile} {year}: {len(missing)} bands missing, e.g. {missing[0]}')
        vals = _read_points(paths, px.lon.values, px.lat.values, threads)
        gedi = _gedi_shots(tile, year, gedi_dir, rhs, n_gedi, sensitivity_min, seed)
        gvals = _read_points(paths, gedi.lon.values, gedi.lat.values, threads) if gedi is not None else None

    out = pd.DataFrame({'point_id': px.point_id.values, **{k: vals[k] for k in keys}})
    f = out_dir / 'pixel_values' / str(year) / f'{tile}.parquet'
    f.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f.with_suffix('.tmp'), index=False)
    f.with_suffix('.tmp').rename(f)
    msg = f'{tile} {year}: {len(out)} pixels, {int(out[f"RH{rhs[-1]}_Q1"].notna().sum())} valid'
    if gedi is not None:
        g = pd.concat([gedi.reset_index(drop=True), pd.DataFrame({k: gvals[k] for k in keys})], axis=1)
        gfo = out_dir / 'gedi_values' / str(year) / f'{tile}.parquet'
        gfo.parent.mkdir(parents=True, exist_ok=True)
        g.to_parquet(gfo, index=False)
        msg += f', {len(g)} GEDI shots'
    print(msg)


def _band_order(col: str):
    rh, q = col[2:].split('_Q')
    return int(rh), int(q)


def _robust_sd(x, axis):
    """1.4826 x median absolute deviation from the median (SD for normal data, robust to a
    single changed year)."""
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def summarize_stability_years(tiles_file: str, values_dir: str, report_dir: str, rhs=YEAR_RHS,
                              min_valid_share: float = 0.5, vegetated_rh98: float = 5.,
                              pixels_dir: str = None, **kwargs):
    """Per tile, RH: the pixels valid in every year of the tile (years with < min_valid_share
    valid pixels are dropped as 'no prediction'). pixels_dir given: only pixels whose own WWF
    biome is the tile's biome (coastal tiles hold sea and neighbour biomes). Writes to report_dir:
      stability_years_tiles.tsv   tile x year x rh: year offset = median over pixels of
                                  (Q1 - the pixel's median over years), median Q0-Q2 width
      stability_pixels.tsv        tile x rh: median / p90 over pixels of the per-pixel SD and
                                  robust SD over years, all pixels and vegetated ones
                                  (multi-year median RH98 >= vegetated_rh98 m)
      stability_gedi.tsv          biome x year x rh: bias median(Q1 - GEDI), MAE, [Q2, Q0] coverage
      stability_years_biome.tsv / stability_pixels_biome.tsv   the same per biome (median over tiles)"""
    values_dir, report_dir = Path(values_dir).expanduser(), Path(report_dir).expanduser()
    tiles = pd.read_csv(Path(tiles_file).expanduser(), sep='\t')
    year_rows, pix_rows, gedi_rows = [], [], []
    for t in tiles.itertuples():
        files = sorted((values_dir / 'pixel_values').glob(f'*/{t.tile}.parquet'))
        if not files:
            continue
        frames = {int(f.parent.name): pd.read_parquet(f).set_index('point_id') for f in files}
        if pixels_dir:
            px_b = pd.read_parquet(Path(pixels_dir).expanduser() / f'{t.tile}.parquet').set_index('point_id').BIOME
            ids = px_b.index[px_b == t.biome]
            frames = {y: v.loc[v.index.intersection(ids)] for y, v in frames.items()}
        years = sorted(frames)
        for r in rhs:
            q1 = np.stack([frames[y][f'RH{r}_Q1'].values for y in years])          # years x pixels
            width = np.stack([np.abs(frames[y][f'RH{r}_Q0'].values - frames[y][f'RH{r}_Q2'].values)
                              for y in years])
            keep = np.isfinite(q1).mean(axis=1) >= min_valid_share
            for y, k in zip(years, keep):
                if not k:
                    year_rows.append(dict(tile=t.tile, biome=t.biome, year=y, rh=r, status='no prediction'))
            yk = [y for y, k in zip(years, keep) if k]
            q1, width = q1[keep], width[keep]
            ok = np.all(np.isfinite(q1), axis=0)
            if len(yk) < 2 or ok.sum() == 0:
                continue
            q1, width = q1[:, ok], width[:, ok]
            med = np.median(q1, axis=0)
            dev = q1 - med
            veg_q1 = np.stack([frames[y][f'RH{rhs[-1]}_Q1'].values for y in yk])[:, ok]
            veg = np.median(veg_q1, axis=0) >= vegetated_rh98
            for i, y in enumerate(yk):
                year_rows.append(dict(tile=t.tile, biome=t.biome, year=y, rh=r, status='ok', n_pixels=int(ok.sum()),
                                      offset=float(np.median(dev[i])), mean_q1=float(q1[i].mean()),
                                      width=float(np.median(width[i])), n_vegetated=int(veg.sum()),
                                      offset_vegetated=float(np.median(dev[i, veg])) if veg.sum() >= 100 else np.nan))
            sd, rsd = q1.std(axis=0, ddof=1), _robust_sd(q1, axis=0)
            for name, m in (('all', np.ones_like(veg)), ('vegetated', veg)):
                if m.sum() == 0:
                    continue
                pix_rows.append(dict(tile=t.tile, biome=t.biome, rh=r, pixels=name, n_years=len(yk),
                                     years=','.join(map(str, yk)), n_pixels=int(m.sum()),
                                     median_q1=float(np.median(med[m])),
                                     sd_median=float(np.median(sd[m])), sd_p90=float(np.percentile(sd[m], 90)),
                                     rsd_median=float(np.median(rsd[m])), rsd_p90=float(np.percentile(rsd[m], 90)),
                                     tile_mean_sd=float(np.std(q1[:, m].mean(axis=1), ddof=1))))
        for f in sorted((values_dir / 'gedi_values').glob(f'*/{t.tile}.parquet')):
            g = pd.read_parquet(f)
            for r in rhs:
                y, q0, q1, q2 = (g[c].values for c in (f'rh{r}', f'RH{r}_Q0', f'RH{r}_Q1', f'RH{r}_Q2'))
                ok = np.isfinite(y) & np.isfinite(q0) & np.isfinite(q1) & np.isfinite(q2)
                if ok.sum():
                    gedi_rows.append(dict(tile=t.tile, biome=t.biome, year=int(f.parent.name), rh=r,
                                          **_metrics(y[ok], q0[ok], q1[ok], q2[ok])))

    report_dir.mkdir(parents=True, exist_ok=True)
    yr, px, gd = pd.DataFrame(year_rows), pd.DataFrame(pix_rows), pd.DataFrame(gedi_rows)
    yr.to_csv(report_dir / 'stability_years_tiles.tsv', sep='\t', index=False)
    px.to_csv(report_dir / 'stability_pixels.tsv', sep='\t', index=False)
    ok = yr[yr.status == 'ok']
    by_year = ok.groupby(['biome', 'year', 'rh']).agg(
        tiles=('tile', 'size'), offset=('offset', 'median'),
        offset_q25=('offset', lambda v: v.quantile(.25)), offset_q75=('offset', lambda v: v.quantile(.75)),
        tiles_vegetated=('offset_vegetated', 'count'), offset_vegetated=('offset_vegetated', 'median'),
        width=('width', 'median')).reset_index()
    by_year.to_csv(report_dir / 'stability_years_biome.tsv', sep='\t', index=False)
    px.groupby(['biome', 'rh', 'pixels']).agg(
        tiles=('tile', 'size'), n_years=('n_years', 'median'), median_q1=('median_q1', 'median'),
        sd_median=('sd_median', 'median'), rsd_median=('rsd_median', 'median'), sd_p90=('sd_p90', 'median'),
        tile_mean_sd=('tile_mean_sd', 'median')).reset_index().to_csv(
        report_dir / 'stability_pixels_biome.tsv', sep='\t', index=False)
    if len(gd):
        def pooled(g):
            w = g.n / g.n.sum()
            return pd.Series({'n': g.n.sum(), 'tiles': len(g),
                              **{m: (g[m] * w).sum() for m in ('coverage', 'above', 'below', 'bias', 'mae', 'width')}})
        gb = pd.concat([gd.groupby(['year', 'rh']).apply(pooled, include_groups=False).reset_index().assign(biome=0),
                        gd.groupby(['biome', 'year', 'rh']).apply(pooled, include_groups=False).reset_index()])
        gb.to_csv(report_dir / 'stability_gedi.tsv', sep='\t', index=False)
    print(f'{ok.tile.nunique()} tiles, {len(ok)} tile-year-rh rows, '
          f'{int((yr.status == "no prediction").sum()) if len(yr) else 0} without prediction -> {report_dir}')
    pd.set_option('display.width', 220, 'display.precision', 2)
    if len(by_year):
        print(f'\nyear offset RH{rhs[-1]} (median over tiles, m)')
        print(by_year[by_year.rh == rhs[-1]].pivot(index='biome', columns='year', values='offset').to_string())
        print(f'\nyear offset RH{rhs[-1]}, vegetated pixels (median over tiles, m)')
        print(by_year[by_year.rh == rhs[-1]].pivot(index='biome', columns='year', values='offset_vegetated').to_string())


def _stability_images(tile, year, grid, max_cloud_cover, n_images):
    """The images the yearly deploy picks for one tile-year (same query and selection as
    S2DatasetStream.download_tile_by_api), summarised."""
    from types import SimpleNamespace
    import planetary_computer
    import pystac_client
    from datasets.zarr_dataset_deploy import S2DatasetStream
    row = grid.loc[tile]
    b = row.geometry.bounds
    api = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                                    modifier=planetary_computer.sign_inplace)
    items = api.search(collections='sentinel-2-l2a', bbox=[b[0], b[1], min(b[2], 180), b[3]],
                       datetime=f'{year}-01-01/{year}-12-31',
                       query={'eo:cloud_cover': {'lt': max_cloud_cover}, 's2:nodata_pixel_percentage': {'lt': 90},
                              's2:mgrs_tile': {'eq': tile}}).item_collection()
    out = dict(tile=tile, year=year, n_candidates=0, n_selected=0)
    if not len(items):
        return out, None
    df = gpd.GeoDataFrame.from_features(items.to_dict(), crs='epsg:4326')
    df['id'] = df['s2:product_uri'].str.replace(r'_[A-Z]\d{4}', '', regex=True).str.replace('.SAFE', '')
    df = df[pd.to_datetime(df['datetime']).dt.month.isin(list(row.growing_months))]
    out['n_candidates'] = int(df.id.nunique())
    if not len(df):
        return out, None
    sel = S2DatasetStream._pick_best_images(SimpleNamespace(n_iamges_per_tile=n_images), df.copy())
    plat = sel.platform.str[-1].str.upper()
    out.update(n_selected=len(sel), cloud_selected_mean=float(sel['eo:cloud_cover'].mean()),
               cloud_selected_max=float(sel['eo:cloud_cover'].max()),
               nodata_selected_mean=float(sel['s2:nodata_pixel_percentage'].mean()),
               n_s2a=int((plat == 'A').sum()), n_s2b=int((plat == 'B').sum()), n_s2c=int((plat == 'C').sum()),
               baselines=','.join(sorted(sel['s2:processing_baseline'].astype(str).unique())))
    sel = pd.DataFrame(dict(tile=tile, year=year, item_id=sel['id'].values, datetime=sel['datetime'].values,
                            cloud=sel['eo:cloud_cover'].values, nodata=sel['s2:nodata_pixel_percentage'].values,
                            platform=sel.platform.values, baseline=sel['s2:processing_baseline'].astype(str).values))
    return out, sel


def count_stability_images(tiles_file: str, s2_grid_file: str, out_file: str, selected_file: str = None,
                           years=tuple(range(2017, 2027)), max_cloud_cover: float = 90,
                           n_images: int = 20, n_workers: int = 16, **kwargs):
    """Per tile-year of the stability test: how many growing-season images the deploy could
    choose from and the cloud cover of the (up to 20) it picks -> out_file (tsv); the picked
    images themselves -> selected_file (parquet: tile, year, item_id, ...)."""
    tiles = pd.read_csv(Path(tiles_file).expanduser(), sep='\t').tile.tolist()
    grid = gpd.read_parquet(Path(s2_grid_file).expanduser(), columns=['Name', 'growing_months', 'geometry']).set_index('Name')
    rows, picked = [], []
    with ThreadPoolExecutor(n_workers) as ex:
        futs = {ex.submit(_stability_images, t, y, grid, max_cloud_cover, n_images): (t, y) for t in tiles for y in years}
        for f in as_completed(futs):
            try:
                row, sel = f.result()
                rows.append(row)
                if sel is not None:
                    picked.append(sel)
            except Exception as e:  # keep going; the row records the failure
                rows.append(dict(tile=futs[f][0], year=futs[f][1], error=str(e)[:200]))
    if selected_file:
        sel = pd.concat(picked).sort_values(['tile', 'year', 'datetime'])
        Path(selected_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
        sel.to_parquet(Path(selected_file).expanduser(), index=False)
        print(f'{len(sel)} selected images -> {selected_file}')
    df = pd.DataFrame(rows).sort_values(['tile', 'year'])
    out = Path(out_file).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)
    print(df.groupby('year')[['n_candidates', 'n_selected', 'cloud_selected_mean']].median().round(1))
    print(f'{len(df)} tile-years -> {out}')


# ---------- clouds at the sample pixels: SCL (what the deploy masked with) vs Cloud Score+ ----------

def _datatake(item_id: str) -> str:
    """S2B_MSIL2A_20240602T034541_R104_T49UDT_20240602T091002 -> 20240602T034541"""
    return item_id.split('_')[2]


def _scl_tile_year(tile, year, ids, pts, out_file):
    import planetary_computer
    import pystac_client
    api = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                                    modifier=planetary_computer.sign_inplace)
    items = {i.id: i for i in api.search(collections='sentinel-2-l2a', ids=list(ids)).items()}
    cols, crs, rc = {}, None, None
    for iid in ids:
        it = items.get(iid)
        if it is None:
            continue
        with rasterio.open(it.assets['SCL'].href) as src:
            if src.crs != crs:  # all images of a tile share one CRS; computed once
                crs = src.crs
                x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(pts.lon.values, pts.lat.values)
                r, c = rasterio.transform.rowcol(src.transform, x, y)
                r, c = np.asarray(r), np.asarray(c)
                inside = (r >= 0) & (r < src.height) & (c >= 0) & (c < src.width)
                rc = (np.clip(r, 0, src.height - 1), np.clip(c, 0, src.width - 1), inside)
            a = src.read(1)
        v = a[rc[0], rc[1]].astype(np.uint8)
        v[~rc[2]] = 0
        cols[iid] = v
    df = pd.DataFrame({'point_id': pts.point_id.values, **cols})
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file.with_suffix('.tmp'), index=False)
    out_file.with_suffix('.tmp').rename(out_file)
    return len(cols), len(ids)


def sample_stability_scl(selected_file: str, pixels_dir: str, out_dir: str, n_workers: int = 16, **kwargs):
    """SCL class of every selected image at the fixed sample pixels, read from Planetary Computer
    (the same SCL the deploy masked with) -> out_dir/<year>/<tile>.parquet (point_id + one uint8
    column per item_id; 0 = nodata or outside). Existing files are skipped."""
    sel = pd.read_parquet(Path(selected_file).expanduser())
    out_dir = Path(out_dir).expanduser()
    todo = [(t, y, g.item_id.tolist()) for (t, y), g in sel.groupby(['tile', 'year'])
            if not (out_dir / str(y) / f'{t}.parquet').exists()]
    print(f'{len(todo)} tile-years to read')
    pts = {t: pd.read_parquet(Path(pixels_dir).expanduser() / f'{t}.parquet') for t in sel.tile.unique()}
    with ThreadPoolExecutor(n_workers) as ex:
        futs = {ex.submit(_scl_tile_year, t, y, ids, pts[t], out_dir / str(y) / f'{t}.parquet'): (t, y)
                for t, y, ids in todo}
        for i, f in enumerate(as_completed(futs)):
            t, y = futs[f]
            try:
                n, m = f.result()
                print(f'[{i + 1}/{len(todo)}] {t} {y}: {n}/{m} images', flush=True)
            except Exception as e:
                print(f'[{i + 1}/{len(todo)}] {t} {y}: FAILED {str(e)[:200]}', flush=True)


def _cloudscore_tile_year(ee, tile, year, ids, pts, out_file, max_request=50331648, n_threads=16):
    """Cloud Score+ cs_cdf of the selected images at the points: the images stacked into one
    image (one band each) and downloaded with computePixels on the tile's own 10 m grid, in blocks
    under the request-size limit; only the point values of each block are kept."""
    col = (ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
           .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
           .filter(ee.Filter.stringEndsWith('system:index', f'T{tile}')))
    index = col.aggregate_array('system:index').getInfo()
    asset = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED/'
    bands, names = [], []
    for iid in ids:
        m = [i for i in index if i.startswith(_datatake(iid))]  # one datatake may be split into granules
        if m:  # by asset id: filtering the global collection per band is slow
            im = [ee.Image(asset + i).select('cs_cdf') for i in m]
            bands.append((im[0] if len(im) == 1 else ee.ImageCollection(im).mosaic()).rename(iid))
            names.append(iid)
    vals = np.full((len(pts), len(names)), np.nan, dtype=np.float32)
    if names:
        proj = ee.Image(asset + [i for i in index if i.startswith(_datatake(names[0]))][0]).select('cs_cdf').projection().getInfo()
        t = proj['transform']  # [10, 0, x0, 0, -10, y0]: the S2 tile grid
        # cs_cdf x 250 as uint8, 255 = no data
        img = ee.Image.cat(bands).multiply(250).round().unmask(255).toUint8()
        x, y = Transformer.from_crs('EPSG:4326', proj['crs'], always_xy=True).transform(pts.lon.values, pts.lat.values)
        col_, row_ = np.floor((np.asarray(x) - t[2]) / t[0]).astype(int), np.floor((np.asarray(y) - t[5]) / t[4]).astype(int)
        # the request limit counts 2 bytes per pixel and band
        k = int(np.ceil(10980 / np.floor(np.sqrt(max_request / (2 * len(names))))))
        n = int(np.ceil(10980 / k))

        def block(ij):
            bi, bj = ij
            m = (row_ // n == bi) & (col_ // n == bj)
            if not m.any():
                return
            a = ee.data.computePixels({'expression': img, 'fileFormat': 'NUMPY_NDARRAY', 'grid': {
                'dimensions': {'width': n, 'height': n}, 'crsCode': proj['crs'],
                'affineTransform': {'scaleX': t[0], 'shearX': 0, 'translateX': t[2] + bj * n * t[0],
                                    'shearY': 0, 'scaleY': t[4], 'translateY': t[5] + bi * n * t[4]}}})
            v = np.stack([a[b] for b in names], axis=-1)[row_[m] - bi * n, col_[m] - bj * n].astype(np.float32)
            v[v == 255] = np.nan
            vals[m] = v / 250
        with ThreadPoolExecutor(n_threads) as ex:
            list(ex.map(block, [(i, j) for i in range(k) for j in range(k)]))
    df = pd.DataFrame({'point_id': pts.point_id.values})
    df = pd.concat([df, pd.DataFrame(vals, columns=names)], axis=1)
    for iid in ids:  # images without Cloud Score+ stay NaN
        if iid not in df:
            df[iid] = np.float32(np.nan)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file.with_suffix('.tmp'), index=False)
    out_file.with_suffix('.tmp').rename(out_file)
    return len(names), len(ids)


def sample_stability_cloudscore(selected_file: str, pixels_dir: str, out_dir: str, key_file: str,
                                n_workers: int = 2, n_threads: int = 16, part: int = 0, n_parts: int = 1, **kwargs):
    """Cloud Score+ cs_cdf (GEE, 0 = cloudy .. 1 = clear) of every selected image at the fixed
    sample pixels, matched by datatake and tile (n_workers tile-years at a time, n_threads block
    requests each) -> out_dir/<year>/<tile>.parquet (point_id + one
    float column per item_id; NaN = no Cloud Score+). Existing files are skipped. part / n_parts:
    this run's share of the tile-years, to spread them over GEE projects (one key_file each)."""
    import ee
    from download.core.utils import authenticate
    authenticate(key_file)
    sel = pd.read_parquet(Path(selected_file).expanduser())
    out_dir = Path(out_dir).expanduser()
    todo = [(t, y, g.item_id.tolist()) for (t, y), g in sel.groupby(['tile', 'year'])
            if not (out_dir / str(y) / f'{t}.parquet').exists()]
    todo = todo[part::n_parts]
    print(f'{len(todo)} tile-years to sample (part {part} of {n_parts})')
    pts = {t: pd.read_parquet(Path(pixels_dir).expanduser() / f'{t}.parquet') for t in sel.tile.unique()}
    with ThreadPoolExecutor(n_workers) as ex:
        futs = {ex.submit(_cloudscore_tile_year, ee, t, y, ids, pts[t], out_dir / str(y) / f'{t}.parquet',
                          n_threads=n_threads): (t, y)
                for t, y, ids in todo}
        for i, f in enumerate(as_completed(futs)):
            t, y = futs[f]
            try:
                n, m = f.result()
                print(f'[{i + 1}/{len(todo)}] {t} {y}: {n}/{m} images with Cloud Score+', flush=True)
            except Exception as e:
                print(f'[{i + 1}/{len(todo)}] {t} {y}: FAILED {str(e)[:200]}', flush=True)


def summarize_stability_clouds(tiles_file: str, values_dir: str, scl_dir: str, cloudscore_dir: str,
                               report_dir: str, pixels_dir: str, cs_clear: float = 0.6,
                               vegetated_rh98: float = 5., **kwargs):
    """How clean were the observations the deploy used? Per pixel and tile-year, among the
    selected images: n_used = SCL not excluded (what entered the median), n_leak = used but
    Cloud Score+ cs_cdf < cs_clear, leak_share = n_leak / n_used (images with Cloud Score+ only).
    Joined with the RH98 Q1 deviation from the pixel's multi-year median (as in
    summarize_stability_years: own-biome pixels valid in every year).
      stability_clouds_tiles.tsv  tile x year: median n_used, mean leak_share (vegetated pixels),
                                  share of SCL-masked pixels Cloud Score+ calls clear, offset
      stability_clouds_bins.tsv   year x leak_share bin: pixels, median and median |deviation|"""
    from const import SCL_EXCLUDE_LABELS
    tiles = pd.read_csv(Path(tiles_file).expanduser(), sep='\t')
    values_dir, scl_dir, cs_dir = (Path(d).expanduser() for d in (values_dir, scl_dir, cloudscore_dir))
    excl = np.array(SCL_EXCLUDE_LABELS)
    tile_rows, bin_rows = [], []
    bins = [-0.01, 0, 0.25, 0.5, 1.0]
    labels = ['0', '(0, 0.25]', '(0.25, 0.5]', '(0.5, 1]']
    for t in tiles.itertuples():
        files = sorted((values_dir / 'pixel_values').glob(f'*/{t.tile}.parquet'))
        if not files:
            continue
        px = pd.read_parquet(Path(pixels_dir).expanduser() / f'{t.tile}.parquet').set_index('point_id')
        ids = px.index[px.BIOME == t.biome]
        q1 = pd.DataFrame({int(f.parent.name): pd.read_parquet(f).set_index('point_id').RH98_Q1 for f in files}).reindex(ids)
        q1 = q1.loc[:, q1.notna().mean() >= 0.5].dropna()
        if q1.shape[1] < 2 or not len(q1):
            continue
        med = q1.median(axis=1)
        veg = med >= vegetated_rh98
        dev = q1.sub(med, axis=0)
        for y in q1.columns:
            fs, fc = scl_dir / str(y) / f'{t.tile}.parquet', cs_dir / str(y) / f'{t.tile}.parquet'
            if not (fs.exists() and fc.exists()):
                continue
            scl = pd.read_parquet(fs).set_index('point_id').reindex(q1.index)
            cs = pd.read_parquet(fc).set_index('point_id').reindex(q1.index)
            common = [c for c in scl.columns if c in cs.columns]
            sv, cv = scl[common].to_numpy(), cs[common].to_numpy()
            used = ~np.isin(sv, excl)
            has_cs = np.isfinite(cv)
            n_used = used.sum(1)
            n_used_cs = (used & has_cs).sum(1)
            n_leak = (used & has_cs & (cv < cs_clear)).sum(1)
            masked_cs = (~used) & (sv != 0) & has_cs
            leak_share = np.where(n_used_cs > 0, n_leak / np.maximum(n_used_cs, 1), np.nan)
            v = veg.to_numpy()
            d = dev[y].to_numpy()
            tile_rows.append(dict(
                tile=t.tile, biome=t.biome, year=y, n_images=len(common),
                images_with_cs=int(has_cs.any(0).sum()), n_vegetated=int(v.sum()),
                n_used_median=float(np.median(n_used[v])) if v.any() else np.nan,
                leak_share_mean=float(np.nanmean(leak_share[v])) if v.any() else np.nan,
                masked_but_clear=float(((cv >= cs_clear) & masked_cs).sum() / max(masked_cs.sum(), 1)),
                offset_vegetated=float(np.median(d[v])) if v.sum() >= 100 else np.nan))
            ls = pd.cut(pd.Series(leak_share[v]), bins, labels=labels)
            for lab in labels:
                m = (ls == lab).to_numpy()
                if m.sum():
                    bin_rows.append(dict(tile=t.tile, biome=t.biome, year=y, leak_bin=lab, n_pixels=int(m.sum()),
                                         dev_median=float(np.median(d[v][m])),
                                         absdev_median=float(np.median(np.abs(d[v][m])))))
    report_dir = Path(report_dir).expanduser()
    tiles_df, bins_df = pd.DataFrame(tile_rows), pd.DataFrame(bin_rows)
    tiles_df.to_csv(report_dir / 'stability_clouds_tiles.tsv', sep='\t', index=False)
    bins_df.to_csv(report_dir / 'stability_clouds_bins.tsv', sep='\t', index=False)
    print(tiles_df.groupby('year')[['n_used_median', 'leak_share_mean', 'masked_but_clear', 'offset_vegetated']]
          .median().round(3))
    w = bins_df.assign(wd=bins_df.dev_median * bins_df.n_pixels, wa=bins_df.absdev_median * bins_df.n_pixels)
    g = w.groupby('leak_bin', observed=True)[['n_pixels', 'wd', 'wa']].sum()
    print((g.assign(dev=g.wd / g.n_pixels, absdev=g.wa / g.n_pixels)[['n_pixels', 'dev', 'absdev']]).round(2))
    x = tiles_df.dropna(subset=['offset_vegetated', 'leak_share_mean'])
    print('Spearman offset vs leak_share (tile-years):', round(x.offset_vegetated.corr(x.leak_share_mean, method='spearman'), 2))
    print(f'-> {report_dir}')


S2_BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')


def _read_overview(href, res, tries=4):
    """Band of a Planetary Computer asset at `res` m (overview, else native), signed afresh on every
    try: SAS tokens expire and the blob store answers 403 under load."""
    import time
    import planetary_computer
    for k in range(tries):
        try:
            with rasterio.open(planetary_computer.sign(href.split('?')[0])) as src:
                f = int(round(res / src.res[0]))
                f = f if f in src.overviews(1) else 1
                h, w = src.height // f, src.width // f
                return src.read(1, out_shape=(h, w)), src.crs, src.transform, src.height, src.width
        except rasterio.errors.CRSError:
            raise
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(15 * (k + 1))


def _reflectance_tile_year(tile, year, ids, pts, out_file, res=80):
    """Reflectance of the selected images at the points, read from the COG overviews at `res` m
    (bands without such an overview at their native resolution). Harmonized as in the deploy
    (acquisitions from 2022-01-25: clip at 1000, minus 1000)."""
    import planetary_computer
    import pystac_client
    api = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                                    modifier=planetary_computer.sign_inplace)
    items = {i.id: i for i in api.search(collections='sentinel-2-l2a', ids=list(ids)).items()}
    cols, meta, xy = {}, [], None
    for iid in ids:
        it = items.get(iid)
        if it is None:
            continue
        new = iid.split('_')[2][:8] >= '20220125'
        try:
            vals = {}
            for b in S2_BANDS:
                a, crs, t, H, W = _read_overview(it.assets[b].href, res)
                if xy is None or xy[0] != crs:
                    x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(
                        pts.lon.values, pts.lat.values)
                    xy = (crs, np.asarray(x), np.asarray(y))
                h, w = a.shape
                r = np.floor((t.f - xy[2]) / (-t.e * H / h)).astype(int)
                c = np.floor((xy[1] - t.c) / (t.a * W / w)).astype(int)
                inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
                v = a[np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)].astype(np.int32)
                v[~inside] = 0
                if new:
                    v = np.where(v == 0, 0, np.clip(v, 1000, None) - 1000)
                vals[b] = v.astype(np.uint16)
        except Exception as e:  # e.g. the 50JLT 2026 image without projection metadata
            print(f'{tile} {year} {iid}: {str(e)[:120]}', flush=True)
            continue
        for b, v in vals.items():
            cols[f'{b}:{iid}'] = v
        p = it.properties
        meta.append(dict(item_id=iid, platform=p.get('platform'), sun_zenith=p.get('s2:mean_solar_zenith'),
                         relative_orbit=p.get('sat:relative_orbit')))
    df = pd.DataFrame({'point_id': pts.point_id.values, **cols})
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file.with_suffix('.tmp'), index=False)
    pd.DataFrame(meta).to_csv(out_file.with_suffix('.images.tsv'), sep='\t', index=False)
    out_file.with_suffix('.tmp').rename(out_file)
    return len(meta), len(ids)


def sample_stability_reflectance(selected_file: str, pixels_dir: str, out_dir: str, res: int = 80,
                                 n_workers: int = 16, **kwargs):
    """Surface reflectance (12 bands, harmonized as in the deploy) of every selected image at the
    fixed sample pixels, from the Planetary Computer COG overviews at `res` m
    -> out_dir/<year>/<tile>.parquet (point_id + uint16 '<band>:<item_id>' columns, 0 = nodata)
    and <tile>.images.tsv (platform, sun zenith, orbit per image). Existing files are skipped."""
    sel = pd.read_parquet(Path(selected_file).expanduser())
    out_dir = Path(out_dir).expanduser()
    todo = [(t, y, g.item_id.tolist()) for (t, y), g in sel.groupby(['tile', 'year'])
            if not (out_dir / str(y) / f'{t}.parquet').exists()]
    print(f'{len(todo)} tile-years to read')
    pts = {t: pd.read_parquet(Path(pixels_dir).expanduser() / f'{t}.parquet') for t in sel.tile.unique()}
    with ThreadPoolExecutor(n_workers) as ex:
        futs = {ex.submit(_reflectance_tile_year, t, y, ids, pts[t], out_dir / str(y) / f'{t}.parquet', res): (t, y)
                for t, y, ids in todo}
        for i, f in enumerate(as_completed(futs)):
            t, y = futs[f]
            try:
                n, m = f.result()
                print(f'[{i + 1}/{len(todo)}] {t} {y}: {n}/{m} images', flush=True)
            except Exception as e:
                print(f'[{i + 1}/{len(todo)}] {t} {y}: FAILED {str(e)[:200]}', flush=True)


def summarize_stability_reflectance(tiles_file: str, values_dir: str, scl_dir: str, cloudscore_dir: str,
                                    reflectance_dir: str, report_dir: str, pixels_dir: str,
                                    cs_clear: float = 0.6, vegetated_rh98: float = 5., **kwargs):
    """Year offsets of the inputs, defined like the RH98 year offset: per pixel the median
    reflectance over the clear observations of the year, minus the pixel's median over years;
    median over the vegetated own-biome pixels (valid RH98 in every year).
    Clear: SCL not excluded (mask 'scl', as in the deploy) or also cs_cdf >= cs_clear ('scl_cs');
    'scl_cs_S2A' / 'scl_cs_S2B': the same with one platform's images only.
      stability_reflectance_tiles.tsv  tile x year x mask: offset of each band and NDVI
                                       (reflectance x 1e4; NDVI x 1e4), RH98 offset, sun zenith"""
    from const import SCL_EXCLUDE_LABELS
    tiles = pd.read_csv(Path(tiles_file).expanduser(), sep='\t')
    values_dir, scl_dir, cs_dir, rf_dir = (Path(d).expanduser() for d in
                                           (values_dir, scl_dir, cloudscore_dir, reflectance_dir))
    excl = np.array(SCL_EXCLUDE_LABELS)
    feats = list(S2_BANDS) + ['NDVI']
    rows = []
    for t in tiles.itertuples():
        files = sorted((values_dir / 'pixel_values').glob(f'*/{t.tile}.parquet'))
        if not files:
            continue
        px = pd.read_parquet(Path(pixels_dir).expanduser() / f'{t.tile}.parquet').set_index('point_id')
        ids = px.index[px.BIOME == t.biome]
        q1 = pd.DataFrame({int(f.parent.name): pd.read_parquet(f).set_index('point_id').RH98_Q1 for f in files}).reindex(ids)
        q1 = q1.loc[:, q1.notna().mean() >= 0.5].dropna()
        if q1.shape[1] < 2 or not len(q1):
            continue
        med = q1.median(axis=1)
        veg = (med >= vegetated_rh98).to_numpy()
        if veg.sum() < 100:
            continue
        rh_off = q1.sub(med, axis=0)[veg].median()
        per_year = {}   # mask -> year -> (n_pixels, n_feats) per-pixel medians
        sun = {}
        for y in q1.columns:
            fr, fs, fc = (d / str(y) / f'{t.tile}.parquet' for d in (rf_dir, scl_dir, cs_dir))
            if not (fr.exists() and fs.exists()):
                continue
            rf = pd.read_parquet(fr).set_index('point_id').reindex(q1.index[veg])
            img = pd.read_csv(fr.with_suffix('.images.tsv'), sep='\t')
            if not len(img):
                continue
            sun[y] = img.sun_zenith.mean()
            scl = pd.read_parquet(fs).set_index('point_id').reindex(q1.index[veg])
            cs = pd.read_parquet(fc).set_index('point_id').reindex(q1.index[veg]) if fc.exists() else None
            iids = [i for i in img.item_id if i in scl.columns]
            a = np.stack([rf[[f'{b}:{i}' for i in iids]].to_numpy(np.float32) for b in S2_BANDS], -1)  # px, img, band
            a[a == 0] = np.nan
            ndvi = (a[..., 7] - a[..., 3]) / (a[..., 7] + a[..., 3]) * 1e4
            a = np.concatenate([a, ndvi[..., None]], -1)
            clear = ~np.isin(scl[iids].to_numpy(), excl)
            csv = cs.reindex(columns=iids).to_numpy() if cs is not None else np.full(clear.shape, np.nan)
            clear_cs = clear & (csv >= cs_clear)
            plat = img.set_index('item_id').platform.reindex(iids).to_numpy()
            masks = {'scl': clear, 'scl_cs': clear_cs}
            for p in ('Sentinel-2A', 'Sentinel-2B'):
                masks[f'scl_cs_S2{p[-1]}'] = clear_cs & (plat == p)[None, :]
            for k, m in masks.items():
                b = np.where(m[..., None], a, np.nan)
                with np.errstate(all='ignore'):
                    per_year.setdefault(k, {})[y] = np.nanmedian(b, axis=1)
        if 'scl' not in per_year:
            continue
        base = {k: np.nanmedian(np.stack(list(v.values())), axis=0) for k, v in per_year.items()
                if k in ('scl', 'scl_cs')}
        for k, v in per_year.items():
            bk = base['scl_cs' if k.startswith('scl_cs') else 'scl']
            for y, m in v.items():
                with np.errstate(all='ignore'):
                    off = np.nanmedian(m - bk, axis=0)
                rows.append(dict(tile=t.tile, biome=t.biome, year=y, mask=k,
                                 n_pixels=int(np.isfinite(m[:, 3]).sum()), rh98_offset=float(rh_off[y]),
                                 sun_zenith=sun.get(y), **{f: float(o) for f, o in zip(feats, off)}))
    out = pd.DataFrame(rows)
    report_dir = Path(report_dir).expanduser()
    out.to_csv(report_dir / 'stability_reflectance_tiles.tsv', sep='\t', index=False)
    print(out[out['mask'] == 'scl'].groupby('year')[['rh98_offset', 'B02', 'B04', 'B08', 'B11', 'NDVI', 'sun_zenith']]
          .median().round(1))
    print(f'-> {report_dir}')


def summarize_stability_per_image(tiles_file: str, values_dir: str, per_image_dir: str, selected_file: str,
                                  report_dir: str, pixels_dir: str, var: str = 'RH98_Q1',
                                  vegetated_rh98: float = 5., nodata: int = 32767, **kwargs):
    """Year offset of every single image on the per-image test tiles: median over the vegetated
    own-biome pixels of (the image's value - the pixel's median of the yearly values over years),
    the yearly values from values_dir/pixel_values (as in summarize_stability_years).
    Also checks that the yearly values of the per-image run equal values_dir.
      stability_per_image.tsv  tile, year, item_id, date, platform, baseline, scene cloud,
                               n_pixels (valid in the image), offset, year_offset"""
    tiles = pd.read_csv(Path(tiles_file).expanduser(), sep='\t').set_index('tile')
    values_dir, per_dir = Path(values_dir).expanduser(), Path(per_image_dir).expanduser()
    sel = pd.read_parquet(Path(selected_file).expanduser()).set_index('item_id')
    rows = []
    for f in sorted((per_dir / 'point_samples').glob('*/*_per_image.parquet')):
        tile, year = f.stem.replace('_per_image', ''), int(f.parent.name)
        biome = tiles.loc[tile, 'biome']
        files = sorted((values_dir / 'pixel_values').glob(f'*/{tile}.parquet'))
        px = pd.read_parquet(Path(pixels_dir).expanduser() / f'{tile}.parquet').set_index('point_id')
        ids = px.index[px.BIOME == biome]
        q1 = pd.DataFrame({int(g.parent.name): pd.read_parquet(g).set_index('point_id')[var] for g in files}).reindex(ids)
        q1 = q1.loc[:, q1.notna().mean() >= 0.5].dropna()
        med = q1.median(axis=1)
        veg = med[med >= vegetated_rh98].index
        new = per_dir / 'pixel_values' / str(year) / f'{tile}.parquet'
        if new.exists():
            a = pd.read_parquet(new).set_index('point_id')[var].reindex(q1.index)
            same = np.isclose(a, q1[year], equal_nan=True).mean()
            print(f'{tile} {year}: yearly {var} equal to the stability run at {same:.4f} of pixels')
        per = pd.read_parquet(f)
        per = per[per.kind == 'pixel'].set_index('point_id').reindex(veg)
        year_off = float((q1.loc[veg, year] - med[veg]).median())
        for c in [c for c in per.columns if c.startswith(f'{var}:')]:
            v = per[c].astype(float)
            v[v == nodata] = np.nan
            d = v / 10 - med[veg]
            date, iid = c.split(':', 1)[1].split('_', 1)
            s = sel.loc[iid] if iid in sel.index else None
            rows.append(dict(tile=tile, biome=biome, year=year, item_id=iid, date=date,
                             platform=iid[:3], baseline=None if s is None else s.baseline,
                             scene_cloud=None if s is None else s.cloud,
                             n_pixels=int(d.notna().sum()), offset=float(d.median()), year_offset=year_off))
    out = pd.DataFrame(rows)
    report_dir = Path(report_dir).expanduser()
    out.to_csv(report_dir / 'stability_per_image.tsv', sep='\t', index=False)
    print(out.groupby(['tile', 'year']).agg(n=('offset', 'size'), image_median=('offset', 'median'),
                                            year_offset=('year_offset', 'first')).round(2))
    print(f'-> {report_dir}')
