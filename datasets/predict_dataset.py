import pystac
import planetary_computer
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from utils._stackstac import stack
import xbatcher
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader, OrderOption
from ffcv.fields import NDArrayField, IntField, FloatField
import xbatcher
import pystac_client
from pyproj import Transformer
import pandas as pd
import geopandas as gpd
from xrspatial import slope

from pystac_client.stac_api_io import StacApiIO
from download.s2_download import harmonize_to_old


bands = [
    'B01', 'B04', 'B03'#, 'B02',
    #  'B05', 'B06', 'B07', 'B08', 'B8A',
    # 'B09', 'B11', 'B12'
]
band_size = len(bands)


def utm_to_wgs84(bounds, utm_epsg=32632, wgs84_epsg=4326):
    """
    Convert bounds from UTM (EPSG:32632) to WGS84 (EPSG:4326).
    
    Parameters:
    - bounds: Tuple of (min_x, min_y, max_x, max_y) in UTM coordinates.
    - utm_epsg: EPSG code for the input UTM projection (default: 32632).
    - wgs84_epsg: EPSG code for WGS84 (default: 4326).
    
    Returns:
    - Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84.
    """
    transformer = Transformer.from_crs(utm_epsg, wgs84_epsg, always_xy=True)
    
    min_x, min_y, max_x, max_y = bounds
    
    # Transform all corners
    bottom_left = transformer.transform(min_x, min_y)
    bottom_right = transformer.transform(max_x, min_y)
    top_left = transformer.transform(min_x, max_y)
    top_right = transformer.transform(max_x, max_y)
    
    # Calculate new bounds
    min_lon = min(bottom_left[0], bottom_right[0], top_left[0], top_right[0])
    min_lat = min(bottom_left[1], bottom_right[1], top_left[1], top_right[1])
    max_lon = max(bottom_left[0], bottom_right[0], top_left[0], top_right[0])
    max_lat = max(bottom_left[1], bottom_right[1], top_left[1], top_right[1])
    
    return min_lon, min_lat, max_lon, max_lat

class XarrayDataset(Dataset):
    def __init__(self):
        super().__init__()
        item_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20240806T102559_R108_T32TMT_20240806T134756"

        # Load the individual item metadata and sign the assets
        item = pystac.Item.from_file(item_url)
        signed_item = planetary_computer.sign(item)

        img = stack(signed_item, resolution=10, properties=False, dtype='int16', fill_value=np.int16(32767))
        # Open one of the data assets (other asset keys to use: 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP', 'visual')
        img = harmonize_to_old(img)
        img = img.drop_vars(['gsd', 'title','common_name', 'center_wavelength', 'full_width_half_max', 'proj:bbox'])
        img = img.sel(band=bands)

        stac_api_io = StacApiIO()
        stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'
        api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)
        bbox_of_interest = utm_to_wgs84(img.spec.bounds, utm_epsg=img.spec.epsg)
        search = api.search(collections=["esa-worldcover"],
            bbox=bbox_of_interest,
            datetime="2021-01-01/2021-12-01")
        items = search.item_collection()
        wc_img = stack(items, ['map'], resolution=10, epsg=img.spec.epsg, bounds=img.spec.bounds, dtype='int16', fill_value=np.int16(0), 
                    properties=False)
        wc_img = wc_img.max(dim='time', skipna=True).squeeze()
        wc_img = wc_img.assign_coords({'band': 'esa_wc'})
        wc_img = wc_img.drop_vars(['created', 'raster:bands', 'proj:shape', 'title', 'description' ])
        img_da = xr.concat([img, wc_img], dim='band', coords='minimal', compat='override', combine_attrs='drop')        

        search = api.search(collections=["cop-dem-glo-30"], bbox=bbox_of_interest)
        items = search.item_collection()
        dem_img = stack(items, ['data'], resolution=30, epsg=img.spec.epsg, bounds=img.spec.bounds, dtype='float32', fill_value=np.float32(0), 
                    properties=False)
        dem_img = dem_img.max(dim='time', skipna=True).squeeze()
        dem_img = dem_img.assign_coords({'band': 'slope'})
        slope_img = slope(dem_img)
        w, h = slope_img.shape[-2:]
        slope_da = slope_img.interp(x=img_da.x, y=img_da.y)
        ds = xr.Dataset({'data': img_da, 'slope': slope_da})
        ds = ds.pad(x=16, y=16, constant_values=0)
        x_coord_before = []
        x_coord_after = []
        y_coord_before = []
        y_coord_after = []
        for i in range(16):
            x_coord_before.append(img_da.x.values[0] - (16 - i) * 10)
            x_coord_after.append(img_da.x.values[-1] + (i + 1) * 10)
            y_coord_before.append(img_da.y.values[0] + (16 - i) * 10)
            y_coord_after.append(img_da.y.values[-1] - (i + 1) * 10)
        x_coord = np.concatenate([x_coord_before, img_da.x.values, x_coord_after])
        y_coord = np.concatenate([y_coord_before, img_da.y.values, y_coord_after])
        df = pd.DataFrame({'x': x_coord, 'y': y_coord})
        df['geometry'] = gpd.points_from_xy(df['x'], df['y'])
        gdf = gpd.GeoDataFrame(df, crs='EPSG:32632')
        gdf = gdf.to_crs('EPSG:4326')

        s = ds.slope.shape[0]
        ds = ds.assign_coords({'x': np.arange(s), 'y': np.arange(s)})
        # latlon = np.stack(latlon, axis=0)
        # latlon_da = xr.DataArray(latlon, dims=['latlon', 'y', 'x'])
        ds = ds.assign(lat=('x', gdf.geometry.x), lon=('y', gdf.geometry.y))
        # img = img.chunk({'band': band_size, 'x': 512, 'y': 512})
        self.data = xbatcher.BatchGenerator(ds, 
                             input_dims={'band':band_size+1, 'x':512,'y':512},
                             batch_dims={'band':band_size+1, 'x':512,'y':512}, input_overlap={'x': 16, 'y': 16}, 
                             preload_batch=True)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx.item()]
        return img.data.data[0], img.slope.data, img.lat.data, img.lon.data
    


def write_beton(out_file, dataset, shuffle_indices=False):
	# Pass a type for each data field
	print("Writing dataset to", out_file)
	writer = DatasetWriter(out_file, {
		'image': NDArrayField(dtype=np.dtype("int16"), shape=(band_size+1, 512, 512)),
        'slope': NDArrayField(dtype=np.dtype("float64"), shape=(512, 512)),
        'lat': NDArrayField(dtype=np.dtype("float64"), shape=(512,)),
        'lon': NDArrayField(dtype=np.dtype("float64"), shape=(512,)),
		})
	writer.from_indexed_dataset(dataset, shuffle_indices=shuffle_indices)



if __name__ == '__main__':
    dataset = XarrayDataset()
    # dataset[np.int32(0)]
    write_beton('output/test1.beton', dataset, shuffle_indices=False)
