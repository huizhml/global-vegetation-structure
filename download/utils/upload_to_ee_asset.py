from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import ee
import geemap
from download._utils import authenticate

authenticate()

parquet_fps = Path('/home/ksb781/data/GVS/train_subsets').expanduser().glob('train*.parquet')
asset_folder = 'projects/ee-omegazhanghui/assets/GVS'

for fp in parquet_fps:
    df = pd.read_parquet(fp)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    fc = geemap.geopandas_to_ee(gdf)
    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description='upload_training_reference',
        assetId=f'{asset_folder}/{fp.stem}'
    )
    print(task)
    task.start()
    print(f'Exporting {fp.stem} to Earth Engine')


