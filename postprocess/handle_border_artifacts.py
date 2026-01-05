import os
import re
import datetime
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
import subprocess
import shutil
from pystac_client.stac_api_io import StacApiIO
from download._dask_downloader import DaskDownloader
from postprocess.translate import translate_tile
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="rio_cogeo\\.profiles",
    message="Non-standard compression schema: zstd.*",
)
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
                 correction_stats_dir:str=None, stac_collection_dir:str=None, flag_dir:str=None,
                 s2_grid_file:str=None, distance_map_dir:str=None, 
                 chunksize:int=1024,
                 costal_tiles_file:str=None,
                 use_flash: bool = False,
                 rhs_idx: str = 'all_rhs',
                 **kwargs):
        
        super().__init__(n_parallel=n_parallel, **kwargs)
        self.year = year
        if costal_tiles_file is not None:
            self.costal_tiles_file = Path(costal_tiles_file).expanduser()
        else:
            self.costal_tiles_file = None
        self.corrected_pred_dir = Path(corrected_pred_dir).expanduser()
        self.correction_stats_dir = Path(correction_stats_dir).expanduser()
        self.stac_collection_dir = Path(stac_collection_dir).expanduser()
        self.distance_map_dir = Path(distance_map_dir).expanduser()
        self.s2_grid_file = Path(s2_grid_file).expanduser()
        self.s2_grid = gpd.read_parquet(self.s2_grid_file, columns=['Name', 'geometry'])
        self._sindex = self.s2_grid.sindex
        self.chunksize = chunksize
        self.use_flash = use_flash
        self.flag_dir = Path(flag_dir).expanduser()
        self.flag_dir.mkdir(parents=True, exist_ok=True)
        self.on_lumi = 'home' not in str(Path.home())
        self.flash_dir = Path.home() / f'flash/data/gvs/deploy/predictions_gtiff_{self.year}'
        self.scratch_dir = Path.home() / f'data/gvs/deploy/predictions_gtiff_{self.year}'
        key_rhs = [0, 10, 25, 50, 75, 95, 98, 100] 
        if rhs_idx == 'all_rhs':
            self.key_rhs = None
        elif rhs_idx == 'key_rhs':
            self.key_rhs = [f'RH{rh}_Q{q}' for rh in key_rhs for q in range(0, 3)]
        elif rhs_idx == 'rest_rhs':
            all_rhs = list(range(101))
            rest_rhs = [rh for rh in all_rhs if rh not in key_rhs]
            self.key_rhs = [f'RH{rh}_Q{q}' for rh in rest_rhs for q in range(0, 3)]
        else:
            raise ValueError(f"Invalid rhs_idx: {rhs_idx}")
        self.tiles = []
        for config_file in Path('~/data/gvs/deploy/tiles_by_zone_for_postprocess/').expanduser().glob('*.txt'):
            with open(config_file, 'r') as f:
                tiles = f.read().splitlines()
                self.tiles.extend(tiles)
    
    
    def find_intersecting_s2_tiles(self, current_tile: pystac.Item):
        '''
        Find intersecting S2 tiles
        '''
        geom = shape(current_tile.geometry)
        idx = list(self._sindex.query(geom, predicate="intersects"))
        tiles = self.s2_grid.iloc[idx]['Name'].to_numpy()
        tiles = [tile for tile in tiles if tile in self.tiles]
        return tiles
    
    
    def copy_data_from_lumio(self, tile_ids: list[str]):
        '''
        Copy data from lumio to local
        '''
        t0 = time.time()
        correct_tiles = 0
        if self.use_flash:
            local = self.flash_dir
        else:
            local = self.scratch_dir
        for tile_id in tile_ids:
            # check if the local file exists
            if (self.flash_dir / f'{tile_id}').exists():
                file_count = len(list((self.flash_dir / f'{tile_id}').glob('*.tif')))
                if file_count == 303:
                    correct_tiles += 1
                    continue
            elif (self.scratch_dir / f'{tile_id}').exists():
                file_count = len(list((self.scratch_dir / f'{tile_id}').glob('*.tif')))
                if file_count == 303:
                    correct_tiles += 1
                    continue
            print(f"Copying tile {tile_id}")
            zone_name = tile_id[:3].lower()
            bucket_name = f"{zone_name}-{self.year}"
            remote = f"lumi-465001846-private:{bucket_name}/predictions_GTiff_{self.year}/{tile_id}"
            os.system(f"rclone copy {remote} {local/f'{tile_id}'} --transfers=16 --checkers=16 --multi-thread-streams=4")
            local_files = list(local.glob(f'{tile_id}/*.tif'))
            if len(local_files) == 303:
                correct_tiles += 1
        if correct_tiles < len(tile_ids):
            self.copy_data_from_lumio(tile_ids)
        t1 = time.time()
        print(f"LUMI-O ➡️ Local - {len(tile_ids)} tiles: {t1 - t0} seconds")
    
    
    def copy_data_to_lumio(self, tile_id: str, translate_dir: Path = None):
        '''
        Copy data to lumio
        '''
        t0 = time.time()
        zone_name = tile_id[:3].lower()
        bucket_name = f"{zone_name}-{self.year}"
        remote = f"lumi-465001846-private:{bucket_name}/{tile_id}"
        os.system(f"rclone sync {translate_dir} {remote} --transfers=16 --checkers=16 --multi-thread-streams=4")
        t1 = time.time()
        # check if remote file are exactly the same as local file
        rclone_command = ['rclone', 'check', '--checkers=16', '--checksum', '--stats-one-line', remote, translate_dir]
        task = subprocess.run(rclone_command, capture_output=True, text=True)
        
        if task.returncode != 0:
            raise ValueError(f"Failed to check if remote file {remote} is exactly the same as local file {translate_dir}")
        log_output = task.stdout + task.stderr
        DIFF_PATTERN = r': (\d+) differences found'
        match = re.search(DIFF_PATTERN, log_output)
        if match:
            difference_count = int(match.group(1))
            print(f"✅ Extracted difference count: {difference_count}")
            if difference_count > 0:
                print(f"🚨 FAILURE: Found {difference_count} differences/corruptions.")
                self.copy_data_to_lumio(tile_id)
            else:
                print("✨ SUCCESS: All files verified as identical and uncorrupted. Deleting local corrected predictions.")
                shutil.rmtree(translate_dir) # cog files
                shutil.rmtree(self.corrected_pred_dir / f'{tile_id}') # geotiff files
                (self.flag_dir / f"{tile_id}_done").touch()
        else:
            raise ValueError(f"Could not find the summary line in rclone output.")
        print(f"LUMI-O ⬅️ Local - {tile_id}: {t1 - t0} seconds")
        
    def delete_local_file(self, tile_ids: list[str]):
        '''
        Delete local input files
        '''
        for tile_id in tile_ids:
            current_tile = pystac.Item.from_file(str(self.stac_collection_dir / f'{tile_id}_{self.year}/{tile_id}_{self.year}.json'))
            intersecting_tiles = self.find_intersecting_s2_tiles(current_tile)
            # only if all intersecting tiles are done, delete the local file
            if all(os.path.exists(self.flag_dir / f'{tile}_done') for tile in intersecting_tiles):
                print(f"ALL tiles are done, deleting local file for tile {tile_id}")
                #  TODO: delete the original predictions on lumi-o
                if (self.flash_dir / f'{tile_id}').exists():
                    shutil.rmtree(self.flash_dir / f'{tile_id}')
                elif (self.scratch_dir / f'{tile_id}').exists():
                    shutil.rmtree(self.scratch_dir / f'{tile_id}')
                else:
                    raise ValueError(f"Tile {tile_id} does not exist in flash or scratch directory")
    
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
            # raise ValueError(f'No ESA World Cover items found for tile {tile_id}')
            print(f'No ESA World Cover items found for tile {tile_id}')
            return None
        wc_image = stackstac.stack(items, ['map'], bounds=tile.assets['RH98_Q1'].extra_fields['proj:bbox'], epsg=tile.properties['proj:epsg'], resolution=10, dtype='uint16', fill_value=np.uint16(0),rescale=False)
        print(wc_image.shape)
        if wc_image.shape[0] == 0:
            print(f'{tile_id} has no overlap with ESA World Cover')
            return None
        
        wc_image = wc_image.max(dim='time', skipna=True).squeeze()
        assert wc_image.shape == tuple(tile.assets['RH98_Q1'].extra_fields['proj:shape'])
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
        tile_id = old_dir.parts[-1]
        flash_dir = self.flash_dir / f'{tile_id}'
        scratch_dir = self.scratch_dir / f'{tile_id}'
        if flash_dir.exists() and len(list(flash_dir.glob('*.tif'))) == 303:
            new_dir = f'file://{str(flash_dir)}/' 
        elif scratch_dir.exists() and len(list(scratch_dir.glob('*.tif'))) == 303:
            new_dir = f'file://{str(scratch_dir)}/' 
        else:
            # read directory from s3
            new_dir = self.get_bucket_url(tile_id, project_id=465001846)
            # raise ValueError(f"Tile {tile_id} does not exist in flash or scratch directory")
        for asset in item.assets:
            new_path = new_dir + Path(item.assets[asset].href).name.replace('.tif', '_uncompressed.tif')
            item.assets[asset].href = new_path
        return item
    
    def get_bucket_url(self, tile_id: str, project_id: int=465001846):
        '''
        Get bucket url for a tile
        '''
        zone_name = tile_id[:3].lower()
        bucket_name = f"{zone_name}-{self.year}"
        return f"https://{project_id}.lumidata.eu/{bucket_name}/predictions_GTiff_{self.year}/{tile_id}/"
        
    
    def run_correction_and_blending(self, tile_id: str):
        '''
        Run correction and blending for a single tile and band. And extract points for conformal prediction, save median prediction. It'll be the final version
        Pre-requisite:
        - Generated distance maps for all tiles
        - Calculated residuals for all tiles, and saved the residuals for each RH for later correction step.
        '''
        if (self.flag_dir / f'{tile_id}_done').exists():
            print(f"Tile {tile_id} already processed, skipping")
            return
  
        stac_file = self.stac_collection_dir / f'{tile_id}_{self.year}/{tile_id}_{self.year}.json'
        if not stac_file.exists():
            print(f"Stac file {stac_file} does not exist, skipping")
            return
        current_tile = pystac.Item.from_file(str(stac_file))
        bbox_local = current_tile.assets['RH98_Q1'].extra_fields['proj:bbox']
        dist_map_file = self.distance_map_dir / f'{tile_id}.tif'
        
        intersecting_s2_tiles = self.find_intersecting_s2_tiles(current_tile)
        # if self.on_lumi:
        #     self.copy_data_from_lumio(intersecting_s2_tiles)
        cluster = LocalCluster(n_workers=4, threads_per_worker=4, dashboard_address=":9020")
        client = Client(cluster)
        print(client)
        intersect_items = []
        intersect_dist_items = []
        biases = []
        for tile in intersecting_s2_tiles:
            intersect_stac_file = self.stac_collection_dir / f'{tile}_{self.year}/{tile}_{self.year}.json'
            if not intersect_stac_file.exists():
                print(f"Stac file {intersect_stac_file} does not exist, skipping")
                continue
            intersect_tile = pystac.Item.from_file(str(intersect_stac_file))
            intersect_items.append(intersect_tile)
            dist_map_file = self.distance_map_dir / f'{tile}.tif'
            intersect_dist_item = self.create_distance_map_item(intersect_tile, dist_map_file)
            intersect_dist_items.append(intersect_dist_item)
            if self.on_lumi: # on lumi
                intersect_tile = self.update_href(intersect_tile)
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
        print(f"Loaded {len(biases)} bias files")
        # get normalized weights for blending, same for each band
        dist_images = stackstac.stack(
            intersect_dist_items, assets=['distance_to_border'], chunksize=self.chunksize,
            epsg=current_tile.properties['proj:epsg'], bounds=bbox_local,
            resolution=10, rescale=False, dtype='float32', fill_value=np.float32(np.nan))
        weights = dist_images.where(dist_images.notnull(), 0)
        sum_w = weights.sum(dim='time')
        weights_normalized = xr.where(sum_w > 0, weights / sum_w, np.nan).transpose(*weights.dims) # masked area like water, the sum can be 0
        # weights_normalized = weights_normalized.persist()
        biases = np.vstack(biases).repeat(3, axis=1)
        intersect_images = stackstac.stack(
                intersect_items, chunksize=self.chunksize,
                # assets=[f'RH{i}_Q1' for i in range(101)],
                epsg=current_tile.properties['proj:epsg'], bounds=bbox_local,
                resolution=10, rescale=False, dtype='float32', fill_value=np.float32(np.nan))

        intersect_images = intersect_images + biases[:, :, None, None]
        blended = (intersect_images * weights_normalized.data).sum(dim='time', min_count=1) #!!!!! skipna=True is the default, and it'll return 0 if all are nan, we need min_count=1 to be able to mask water, built-up, snow
        blended = blended.round().fillna(32767).astype(np.int16)

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
            
        band_names =  self.key_rhs or blended.band.values # blended.band.values # ['RH48_Q2'] # + list(blended.band.values)[:16]
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
        translate_tile(corrected_pred_dir, translate_dir, profile="ZSTD") # TODO: add list of bands, key RHs are prioritized
        if self.on_lumi:
            self.copy_data_to_lumio(tile_id, translate_dir)
            self.delete_local_file(intersecting_s2_tiles)
        else:
            (self.flag_dir / f"{tile_id}_done").touch()

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
    flag_dir: str = f'~/data/gvs/deploy/flags_postprocess_{year}'
    costal_tiles_file: str = '~/data/gvs/deploy/tiles_coastal_regions.txt'
    tile_id: str = '20MRS'
    n_parallel: int = 2
    chunksize: int = 2048
    use_flash: bool = False
    rhs_idx: str = 'all_rhs'
    task: str = 'run_correction_and_blending'
    
disable_outputs = {
    "hydra": {
        "run": {"dir": "."},
        "output_subdir": None,
        "job_logging": {"enabled": False},
        "hydra_logging": {"enabled": False},
    }
}

cs = ConfigStore.instance()
cs.store(name='agg_gedi_to_s2', node=AggGediToS2)
cs.store(group="hydra", name="disable_logging", node=disable_outputs)

@hydra.main(config_path=None,config_name='agg_gedi_to_s2', version_base="1.2")
def main(cfg):
    print("Start at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    time_start = time.time()
    vsm_correction = VSMCorrection(**cfg)
    if cfg.task == 'run_correction_and_blending':
        vsm_correction.run_correction_and_blending(cfg.tile_id)
    elif cfg.task == 'test_reading':
        vsm_correction.test_reading()
    time_end = time.time()
    print(f'Time taken: {time_end - time_start} seconds')
    print("-" * 30)


if __name__ == '__main__':
    from dask.distributed import Client, LocalCluster
    main()
