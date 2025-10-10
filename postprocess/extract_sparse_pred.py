import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
import dask_geopandas as dgpd
import h5py
from pathlib import Path
from pyproj import Transformer

tile_id = '35NMA'
year = 2020
h5_file = f'~/data/gvs/GEDI_S2_h5/{tile_id[:3]}.h5'
h5_file = Path(h5_file).expanduser()
h5_file = h5py.File(h5_file, 'r')
# gdf_ = dgpd.read_parquet(f'~/data/gvs/split_test0.1_cal0.1_val0.1_seed42_v1/index_table_val/*.parquet', gather_spatial_partitions=False)
# gdf_ = gdf_.compute()
# gdf = gdf_[gdf_['s2_tile'] == tile_id]
gdf = gpd.read_parquet(f'~/data/gvs/split_test0.1_cal0.1_val0.1_seed42_v1/index_table_val/35N.parquet')
gdf = gdf[gdf['s2_tile'] == tile_id].iloc[:40000]
# gdf = gdf.drop_duplicates(subset=['geometry'])#.iloc[:40000]
rhs = []
# gdf =  gdf.sort_values(['path', 'in_partition_idx'])
# gdf['in_partition_idx'] = gdf.groupby('path').cumcount()
for i, row in gdf.iterrows():
    rhs_ = h5_file[f'{row["path"][4:]}/rhs'][row['in_partition_idx'], -1]
    shot_number = h5_file[f'{row["path"][4:]}/shot_number'][row['in_partition_idx']]
    if int(shot_number) != int(row['shot_number']):
        import ipdb; ipdb.set_trace()
    rhs.append(rhs_)
rhs = np.array(rhs).reshape(200, 200)

pred_tile = f'/home/ksb781/data/gvs/deploy/predictions_{year}/{tile_id}/35NMA_qr5e6gaq_None_7.RH100_Q1.geo.tif'
# Open raster
with rasterio.open(pred_tile) as src:
    transform = src.transform
    raster_crs = src.crs
    data = src.read(1)  # assuming single-band
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    # Convert each point to row, col and set mask
    
    selected_pixels = []
    for geo in gdf.geometry:
        x, y = geo.x, geo.y
        x, y = transformer.transform(x, y)
        # Ensure x,y are in same CRS as the raster
        row, col = rowcol(transform, x, y)
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            selected_pixels.append(data[row, col])

    # Apply mask to the data
    selected_pixels = np.array(selected_pixels)
    selected_pixels = selected_pixels.reshape(200, 200)
    
    profile = src.profile
    profile['width'] = 200
    profile['height'] = 200
    profile['crs'] = raster_crs
    profile['transform'] = transform
    profile['dtype'] = 'float32'
    profile['compress'] = None
    
    with rasterio.open(f'/home/ksb781/data/gvs/deploy/predictions_{year}/{tile_id}/sparse_pred_reshaped.tif', 'w', **profile) as dst:
        dst.write(selected_pixels, indexes=1)
    with rasterio.open(f'/home/ksb781/data/gvs/deploy/predictions_{year}/{tile_id}/sparse_ref_reshaped.tif', 'w', **profile) as dst:
        dst.write(rhs, indexes=1)

    np.save(f'/home/ksb781/data/gvs/deploy/predictions_{year}/{tile_id}/sparse_pred.npy', selected_pixels)
errs=(selected_pixels-rhs)**2
print(f'mean error: {errs.mean()}')
print(f'rmse: {np.sqrt(errs.mean())}')
import ipdb; ipdb.set_trace()

# `selected_pixels` now contains values under the point locations