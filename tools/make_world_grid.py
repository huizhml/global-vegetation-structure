from pathlib import Path

import geodatasets
import geopandas as gpd
import numpy as np
from shapely.geometry import box


def make_world_land_grid(
    out_file: str,
    resolution: float = 1.0,
    land_source: str = 'naturalearth.land',
    clip_to_land: bool = False,
    biome_file: str = None,
    biome_cols: tuple = ('BIOME',),
    **kwargs,
):
    '''
    Build a regular lon/lat grid over the globe, keep only cells intersecting
    land, and save as a GeoParquet for later spatial aggregation.

    Args:
        out_file: output path (.parquet). Parent dirs are created.
        resolution: cell size in degrees (default 1.0 -> 1°x1°).
        land_source: geodatasets key for the land polygons.
        clip_to_land: if True, clip each cell to the land boundary; otherwise
            keep the full square cell (faster, preferred for aggregation).
        biome_file: optional path to the WWF ecoregions shapefile
            (e.g. ~/data/GEDI/ecoregions/wwf_terr_ecos.shp). If given,
            tag each cell with the biome containing its centroid.
        biome_cols: columns to carry over from the biome file.
    '''
    out_file = Path(out_file).expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    land = gpd.read_file(geodatasets.get_path(land_source)).to_crs('EPSG:4326')
    land_union = land.union_all()

    lons = np.arange(-180.0, 180.0, resolution)
    lats = np.arange(-90.0, 90.0, resolution)
    cells = [box(x, y, x + resolution, y + resolution) for x in lons for y in lats]
    centroids_lon = [x + resolution / 2 for x in lons for _ in lats]
    centroids_lat = [y + resolution / 2 for _ in lons for y in lats]

    grid = gpd.GeoDataFrame(
        {
            'cell_id': np.arange(len(cells), dtype=np.int32),
            'lon': centroids_lon,
            'lat': centroids_lat,
            'geometry': cells,
        },
        crs='EPSG:4326',
    )

    keep = grid.sindex.query(land_union, predicate='intersects')
    grid = grid.iloc[np.sort(np.unique(keep))].reset_index(drop=True)

    if clip_to_land:
        grid = gpd.clip(grid, land).reset_index(drop=True)

    if biome_file is not None:
        biome_file = Path(biome_file).expanduser()
        ecoregions = gpd.read_file(biome_file)
        if biome_cols is not None:
            ecoregions = ecoregions[biome_cols]
        ecoregions = ecoregions.to_crs('EPSG:4326')
        # Centroid-based join: each cell gets exactly one biome (the one
        # containing its centroid), avoiding double-counting from polygon overlap.
        centroids = gpd.GeoDataFrame(
            {'cell_id': grid['cell_id']},
            geometry=gpd.points_from_xy(grid['lon'], grid['lat']),
            crs='EPSG:4326',
        )
        tagged = gpd.sjoin(centroids, ecoregions, how='left', predicate='within')
        tagged = tagged.drop_duplicates(subset='cell_id').drop(columns=['index_right', 'geometry'])
        grid = grid.merge(tagged, on='cell_id', how='left')

    grid.to_parquet(out_file)
    print(f'Saved {len(grid):,} land cells at {resolution}° to {out_file}')
    return grid


if __name__ == '__main__':
    make_world_land_grid(
        out_file='~/data/gvs/grids/world_1deg_land_grid.parquet',
        resolution=1.0,
    )
