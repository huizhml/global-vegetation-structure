import os
from osgeo import gdal
import geopandas as gpd
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import rasterio
import time
import dask
import pystac
import stackstac
import xarray as xr
import rioxarray
import numpy as np
import dask.dataframe as dd
from shapely.geometry import shape, box
import dask.array as da
import pystac_client
import planetary_computer
from pystac_client.stac_api_io import StacApiIO
from download._dask_downloader import DaskDownloader
from postprocess.translate import translate_tile

gdal.UseExceptions()

stac_api_io = StacApiIO()
stac_endpoint = 'https://planetarycomputer.microsoft.com/api/stac/v1'


def init_gtiff(prediction_fp: Path, tile_info: dict, options: list):
    prediction_fp = prediction_fp.with_suffix('.tif')
    if prediction_fp.exists():
        os.remove(prediction_fp)
    driver = gdal.GetDriverByName('GTiff')
    tiff_output = driver.Create(
        str(prediction_fp),
        xsize=tile_info['width'],
        ysize=tile_info['height'],
        bands=1,
        eType=gdal.GDT_Int16,
        options=options
    )
    tiff_output.SetGeoTransform(tile_info['transform'])
    tiff_output.SetProjection(tile_info['crs'])
    tiff_output.SetMetadataItem('Year', str(tile_info['year']))
    tiff_output.SetMetadataItem('Sentinel-2 tile', tile_info['tile_id'])
    return tiff_output

class VSMCorrection(DaskDownloader):
    def __init__(self, year: int, n_parallel: int = 100, corrected_pred_dir: str=None,
                 correction_stats_dir:str=None, stac_collection_dir:str=None, 
                 s2_grid_file:str=None, distance_map_dir:str=None, output_dir:str=None, 
                 chunksize:int=1024,
                 costal_tiles_file:str=None,
                 **kwargs):
        
        super().__init__(n_parallel=n_parallel, **kwargs)
        self.year = year
        if costal_tiles_file is not None:
            self.costal_tiles_file = Path(costal_tiles_file).expanduser()
        else:
            self.costal_tiles_file = None
        self.corrected_pred_dir = Path(corrected_pred_dir).expanduser()
        self.correction_stats_dir = Path(correction_stats_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.stac_collection_dir = Path(stac_collection_dir).expanduser()
        self.distance_map_dir = Path(distance_map_dir).expanduser()
        self.s2_grid_file = Path(s2_grid_file).expanduser()
        self.s2_grid = gpd.read_parquet(self.s2_grid_file, columns=['Name', 'geometry'])
        self._sindex = self.s2_grid.sindex
        self.chunksize = chunksize
    
    
    def find_intersecting_s2_tiles(self, current_tile: pystac.Item):
        '''
        Find intersecting S2 tiles
        '''
        geom = shape(current_tile.geometry)
        idx = list(self._sindex.query(geom, predicate="intersects"))
        return self.s2_grid.iloc[idx]['Name'].to_numpy()
    
    
    def create_distance_map_item(self, map_item: pystac.Item, dist_map_file: str):
        '''
        Create distance map item
        '''
        dist_map_file = Path(dist_map_file).expanduser()
        dist_item = pystac.Item(
            id=f'{map_item.id}',
            geometry=map_item.geometry,
            bbox=map_item.bbox,
            datetime=map_item.datetime,
            properties={
                'proj:epsg': map_item.properties['proj:epsg'],
                'raster:bands': map_item.properties['raster:bands']
            }
        )
        dist_item.add_asset('distance_to_border', pystac.Asset(
            href=f'file://{str(dist_map_file)}',
            media_type="image/tiff; application=geotiff; profile=cloud-optimized",
            title=f'Distance map - 10m',
            roles=["data"],
        ))
        return dist_item
    
    def get_water_snow_mask(self, tile: pystac.Item):
        '''
        Mask snow and water predictions if it is costal tile
        '''
        with open(self.costal_tiles_file, 'r') as f:
            costal_tiles = f.read().splitlines()
        tile_id = tile.id.split("_")[0]
        if tile_id not in costal_tiles:
            return None
        
        api = pystac_client.Client.open(stac_endpoint, modifier=planetary_computer.sign_inplace, stac_io=stac_api_io)
        search = api.search(collections='esa-worldcover', bbox=tile.bbox, datetime=f'2021-01-01/2021-12-31')
        items = search.item_collection()
        if len(items) == 0:
            raise ValueError(f'No ESA World Cover items found for tile {tile_id}')
        wc_image = stackstac.stack(items, ['map'], bounds=tile.assets['RH98_Q1'].extra_fields['proj:bbox'], epsg=tile.properties['proj:epsg'], resolution=10, dtype='uint16', fill_value=np.uint16(0),rescale=False)
        wc_image = wc_image.max(dim='time', skipna=True).squeeze()
        assert wc_image.shape == tile.assets['RH98_Q1'].extra_fields['proj:shape']
        water_snow_mask = wc_image.isin([0, 70, 80]) # nodata, snow and ice, water
        return water_snow_mask
    
    def update_href(self, item: pystac.Item, use_flash: bool = True):
        '''
        Update href for all assets in the item
        '''
        if len(item.assets) == 1:
            file_path = self.distance_map_dir / f'{item.id.split("_")[0]}.tif'
            item.assets['distance_to_border'].href = f'file://{file_path}'
            return item
        old_dir = Path(item.assets['RH98_Q1'].href.replace('file://', '')).parent
        if use_flash:
            new_dir = Path.home() / f'flash/data/gvs/deploy/predictions_gtiff_{self.year}/{old_dir.parts[-1]}' 
        else:
            new_dir = Path.home() / f'data/gvs/deploy/predictions_gtiff_{self.year}/{old_dir.parts[-1]}' 
        for asset in item.assets:
            new_path = new_dir / Path(item.assets[asset].href).name.replace('.tif', '_uncompressed.tif')
            item.assets[asset].href = f'file://{str(new_path)}'
        return item
    
        
    
    def run_correction_and_blending(self, tile_id: str):
        '''
        Run correction and blending for a single tile and band. And extract points for conformal prediction, save median prediction. It'll be the final version
        Pre-requisite:
        - Generated distance maps for all tiles
        - Calculated residuals for all tiles, and saved the residuals for each RH for later correction step.
        '''
        current_tile = pystac.Item.from_file(str(self.stac_collection_dir / f'{tile_id}_{self.year}/{tile_id}_{self.year}.json'))
        bbox_local = current_tile.assets['RH98_Q1'].extra_fields['proj:bbox']
        dist_map_file = self.distance_map_dir / f'{tile_id}.tif'
        
        intersecting_s2_tiles = self.find_intersecting_s2_tiles(current_tile)
        intersect_items = []
        intersect_dist_items = []
        biases = []
        for tile in intersecting_s2_tiles:
            intersect_tile = pystac.Item.from_file(str(self.stac_collection_dir / f'{tile}_{self.year}/{tile}_{self.year}.json'))
            intersect_items.append(intersect_tile)
            dist_map_file = self.distance_map_dir / f'{tile}.tif'
            intersect_dist_item = self.create_distance_map_item(intersect_tile, dist_map_file)
            intersect_dist_items.append(intersect_dist_item)
            if 'home' not in str(Path.home()): # on lumi
                intersect_tile = self.update_href(intersect_tile, use_flash=False)
                intersect_dist_item = self.update_href(intersect_dist_item, use_flash=False)
            
            tile_image = stackstac.stack(
                [intersect_tile],
                assets=['RH98_Q1'],
                epsg=current_tile.properties['proj:epsg'],
                resolution=10, bounds=bbox_local, rescale=False, dtype='float32', fill_value=np.float32(np.nan))
            
            if tile_image.shape[0] == 0:
                print(f'{tile} has no overlap with current tile {tile_id}')
                continue
            
            # Apply bias correction if GEDI reference data is available
            correct_stats_file = self.correction_stats_dir / f'{tile}.npz'
            if correct_stats_file.exists():
                bias = np.load(correct_stats_file)['bias']
            else:
                bias = np.zeros(101)
            biases.append(bias)
        # get normalized weights for blending, same for each band
        dist_images = stackstac.stack(
            intersect_dist_items, assets=['distance_to_border'], chunksize=self.chunksize,
            epsg=current_tile.properties['proj:epsg'], bounds=bbox_local,
            resolution=10, rescale=False, dtype='float32', fill_value=np.float32(np.nan))
        weights = dist_images.where(dist_images.notnull(), 0)
        sum_w = weights.sum(dim='time')
        weights_normalized = weights / sum_w
        # weights_normalized = weights_normalized.persist()
        biases = np.vstack(biases).repeat(3, axis=1)
        intersect_images = stackstac.stack(
                intersect_items, chunksize=self.chunksize,
                # assets=[f'RH{i}_Q1' for i in range(101)],
                epsg=current_tile.properties['proj:epsg'], bounds=bbox_local,
                resolution=10, rescale=False, dtype='float32', fill_value=np.float32(np.nan))

        intersect_images = intersect_images + biases[:, :, None, None]
        blended = (intersect_images * weights_normalized.data).sum(dim='time')
        blended = blended.round().astype(np.int16).fillna(np.int16(32767))

        water_snow_mask = self.get_water_snow_mask(current_tile)
        if water_snow_mask is not None:
            blended = blended.where(~water_snow_mask, 32767)
            
        corrected_pred_dir = self.corrected_pred_dir / f'{tile_id}'
        corrected_pred_dir.mkdir(parents=True, exist_ok=True)
        tile_shape = current_tile.assets['RH98_Q1'].extra_fields['proj:shape']
        profile = {
            'driver': 'GTiff',
            'width': tile_shape[1],
            'height': tile_shape[0],
            'count': 1,
            'dtype': 'int16',
            'crs': f"EPSG:{current_tile.properties['proj:epsg']}",
            'transform': current_tile.assets['RH98_Q1'].extra_fields['proj:transform'],
            'tiled': True,
            'blockxsize': self.chunksize,
            'blockysize': self.chunksize,
            'compress': None,
            'num_threads': 1,
            'predictor': 2,
            'interleave': 'BAND',
            'nodata': 32767,
        }
        
        
        def write_block(block):
            band_name = block.band.values[0]
            output_file = corrected_pred_dir / f'{band_name}.tif'
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(block.data.squeeze(), indexes=1)
            return xr.DataArray(
                    np.zeros((1, 1,1), dtype="bool"),
                    dims=('band',"x", "y"),
                    name="block_status",
                )
            
        band_names =  blended.band.values # ['RH48_Q2'] # + list(blended.band.values)[:16]
        blended = blended.sel(band=band_names)
        chunk_size = 10980
        blended = blended.chunk({'band': 1, 'x': chunk_size, 'y': chunk_size})
        num_bands = blended.shape[0]
        output_template = xr.DataArray(
            da.zeros((num_bands, 1, 1), dtype="bool", chunks=(1,1,1)),   # each block returned a reduced array, to save RAM
            dims=('band', "x", 'y'),
            name="block_status",
        ) 
        
        blended = blended.map_blocks(write_block, template=output_template)
        dask.compute(blended)
        
        translate_dir = self.corrected_pred_dir.with_name(self.corrected_pred_dir.name + '_cog') / f'{tile_id}'
        translate_dir.mkdir(parents=True, exist_ok=True)
        translate_tile(corrected_pred_dir, translate_dir, profile="ZSTD")


    # ------------------------------------------------------------
    #     Following functions are for testing purposes
    # ------------------------------------------------------------
    def test_reading(self):
        '''
        Test reading from 303 cog files vs. one compressed geotiff file with 303 bands
        '''
        def func(block):
            return xr.DataArray(
                    np.zeros((1, 1,1), dtype="bool"),
                    dims=('band',"x", "y"),
                    name="block_status",
                )
        template = xr.DataArray(
            da.zeros((303, 11,11), dtype="bool", chunks=(1,1,1)),
            dims=('band',"x", "y"),
            name="block_status",
        )
        tile_id = '19NBF'
        gtiff_dir = Path(f'~/data/gvs/deploy/predictions_gtiff_2020/{tile_id}').expanduser()
        gtiff_dir = gtiff_dir.glob('*.tif')
        ds = [xr.open_dataset(gtiff_file, chunks={'band': 1, 'x': 1024, 'y': 1024}) for gtiff_file in gtiff_dir]
        ds = xr.concat(ds, dim='band')
        t0 = time.time()
        ds = ds.map_blocks(func, template=template)
        ds.compute()
        t1 = time.time()
        print(f'Time taken to read from 303 uncompressed geotiff files: {t1 - t0} seconds')
        
        cog_dir = Path(f'~/data/gvs/deploy/predictions_2020/{tile_id}').expanduser()
        cog_files = cog_dir.glob('*.tif')
        ds = [xr.open_dataset(cog_file, chunks={}) for cog_file in cog_files]
        ds = xr.concat(ds, dim='band')
        t0 = time.time()
        ds = ds.map_blocks(func, template=template)
        ds.compute()
        t1 = time.time()
        print(f'Time taken to read from 303 cog files: {t1 - t0} seconds')
        tif_file = Path(f'~/data/gvs/deploy/predictions_zip_2020/{tile_id}.tif').expanduser()
        ds = xr.open_dataset(tif_file, chunks={})
        t1 = time.time()
        ds = ds.map_blocks(func, template=template)
        ds.compute()
        t2 = time.time()
        print(f'Time taken to read from one compressed geotiff file: {t2 - t1} seconds')
    
    
@dataclass
class AggGediToS2:
    year: int = 2024
    s2_grid_file: str = '~/data/gvs/s2_tiles_with_growing_months.parquet'
    corrected_pred_dir: str = f'~/data/gvs/deploy/predictions_corrected_blended_{year}'
    correction_stats_dir: str = f'~/data/gvs/deploy/correction/tile_stats_{year}'
    stac_collection_dir: str = '~/data/gvs/deploy/gvsm_stac_catalog/vsm_local'
    distance_map_dir: str = '~/data/gvs/deploy/blending/distance_maps'
    costal_tiles_file: str = '~/data/gvs/deploy/tiles_coastal_regions.txt'
    tile_id: str = '20MRS'
    n_parallel: int = 2
    chunksize: int = 1024
    task: str = 'run_correction_and_blending'
    output_dir: str = f'~/data/gvs/deploy/predictions_corrected_blended_{year}'

cs = ConfigStore.instance()
cs.store(name='agg_gedi_to_s2', node=AggGediToS2)


@hydra.main(config_name='agg_gedi_to_s2', version_base="1.2")
def main(cfg):
    time_start = time.time()
    vsm_correction = VSMCorrection(**cfg)
    if cfg.task == 'run_correction_and_blending':
        vsm_correction.run_correction_and_blending(cfg.tile_id)
    elif cfg.task == 'test_reading':
        vsm_correction.test_reading()
    time_end = time.time()
    print(f'Time taken: {time_end - time_start} seconds')


if __name__ == '__main__':
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=4, threads_per_worker=4, dashboard_address=":9020")
    client = Client(cluster)
    print(client)
    main()
