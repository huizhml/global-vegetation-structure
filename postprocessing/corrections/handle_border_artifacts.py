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
from postprocessing.core.translate import translate_tile
from postprocessing.corrections.blending import Blending
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

KEY_RHS = (0, 10, 25, 50, 75, 95, 98, 100)
ESA_WORLD_COVER_WATER_SNOW_MASK = (0, 70, 80) # nodata, snow and ice, water


class VSMCorrection(Blending):
    def __init__(self, year: int, 
                 tile_id: str=None,
                 flag_dir:str=None,
                 output_dir: str=None,
                 stac_collection_dir:str=None, 
                 s2_grid_file:str=None, 
                 distance_map_dir:str=None, 
                 chunksize:int=1024,
                 total_tiles_file:str=None,
                 costal_tiles_file:str=None,
                 use_flash: bool = False,
                 rhs_idx: str = 'all_rhs',
                 small_area: bool = True,
                 **kwargs):
        
        super().__init__(year=year, tile_id=tile_id, s2_grid_file=s2_grid_file, stac_collection_dir=stac_collection_dir, distance_map_dir=distance_map_dir, output_dir=output_dir, small_area=small_area, chunksize=chunksize, **kwargs)

        self.use_flash = use_flash
        self.flag_dir = Path(f'{flag_dir}/{rhs_idx}').expanduser()
        self.flag_dir.mkdir(parents=True, exist_ok=True)
        
        self.geotiff_dir, self.cog_dir = self._init_output_dir(output_dir)
        self.total_tiles = self._load_total_tiles(total_tiles_file)
        self.is_costal_tile = self._check_if_costal_tile(costal_tiles_file)
        self.key_rhs = self._init_rhs_idx(rhs_idx)
        
    def _init_output_dir(self, output_dir: str):
        '''
        Initialize output directory
        '''
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        geotiff_dir = output_dir / f'geotiff/{self.tile_id}'
        geotiff_dir.mkdir(parents=True, exist_ok=True)
        cog_dir = output_dir / f'cog/{self.tile_id}'
        cog_dir.mkdir(parents=True, exist_ok=True)
        return geotiff_dir, cog_dir
        
    
    def _init_rhs_idx(self, rhs_idx: str):
        '''
        Initialize RHS index
        Args:
            rhs_idx: str, 'all_rhs', 'key_rhs', 'rest_rhs', 'rh98'
        '''
        if rhs_idx == 'all_rhs':
            return None
        elif rhs_idx == 'key_rhs':
            return [f'RH{rh}_Q{q}' for rh in KEY_RHS for q in range(0, 3)]
        elif rhs_idx == 'rest_rhs':
            all_rhs = list(range(101))
            rest_rhs = [rh for rh in all_rhs if rh not in KEY_RHS]
            return [f'RH{rh}_Q{q}' for rh in rest_rhs for q in range(0, 3)]
        elif rhs_idx == 'rh98':
            return [f'RH98_Q1']
        else:
            raise ValueError(f"Invalid rhs_idx: {rhs_idx}")
    
    def _load_total_tiles(self, total_tiles_file: str):
        '''
        Load total tiles from file. 
        Not all tiles from the S2 grid file have images for the given year, therefore we keep a list of all tiles available for the given year.
        '''
        total_tiles_file = Path(total_tiles_file).expanduser()
        with open(total_tiles_file, 'r') as f:
            total_tiles = f.read().splitlines()
        return total_tiles
    
    def _check_if_costal_tile(self, costal_tiles_file: str):
        '''
        Load costal tiles from file
        '''
        costal_tiles_file = Path(costal_tiles_file).expanduser()
        with open(costal_tiles_file, 'r') as f:
            costal_tiles = f.read().splitlines()
        return self.tile_id in costal_tiles
    
    
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
    
    
    def copy_data_to_lumio(self):
        '''
        Copy data to lumio
        '''
        t0 = time.time()
        zone_name = self.tile_id[:3].lower()
        bucket_name = f"{zone_name}-{self.year}"
        remote = f"lumi-465001846-private:{bucket_name}/{self.tile_id}"
        os.system(f"rclone sync {self.translate_dir} {remote} --transfers=16 --checkers=16 --multi-thread-streams=4")
        t1 = time.time()
        # check if remote file are exactly the same as local file
        rclone_command = ['rclone', 'check', '--checkers=16', '--checksum', '--stats-one-line', remote, self.translate_dir]
        task = subprocess.run(rclone_command, capture_output=True, text=True)
        
        if task.returncode != 0:
            raise ValueError(f"Failed to check if remote file {remote} is exactly the same as local file {self.translate_dir}")
        log_output = task.stdout + task.stderr
        DIFF_PATTERN = r': (\d+) differences found'
        match = re.search(DIFF_PATTERN, log_output)
        if match:
            difference_count = int(match.group(1))
            print(f"✅ Extracted difference count: {difference_count}")
            if difference_count > 0:
                print(f"🚨 FAILURE: Found {difference_count} differences/corruptions.")
                self.copy_data_to_lumio(self.translate_dir)
            else:
                print("✨ SUCCESS: All files verified as identical and uncorrupted. Deleting local corrected predictions.")
                shutil.rmtree(self.translate_dir) # cog files
                shutil.rmtree(self.output_dir / f'{self.tile_id}') # geotiff files
                (self.flag_dir / f"{self.tile_id}_done").touch()
        else:
            raise ValueError(f"Could not find the summary line in rclone output.")
        print(f"LUMI-O ⬅️ Local - {self.tile_id}: {t1 - t0} seconds")
        
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
    
    def get_water_snow_mask(self, tile: pystac.Item):
        '''
        Mask snow and water predictions if it is costal tile
        '''
        tile_id = tile.id.split("_")[0]
        
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
        water_snow_mask = wc_image.isin(ESA_WORLD_COVER_WATER_SNOW_MASK) # nodata, snow and ice, water
        return water_snow_mask
    

    
    def run_blending_and_mask(self):
        '''
        Run correction and blending for a single tile and band. And extract points for conformal prediction, save median prediction. It'll be the final version
        Pre-requisite:
        - Generated distance maps for all tiles
        - Calculated residuals for all tiles, and saved the residuals for each RH for later correction step.
        '''
        if (self.flag_dir / f'{self.tile_id}_done').exists():
            print(f"Tile {self.tile_id} already processed, skipping")
            return
  
        blended, current_tile, intersecting_s2_tiles = self.run_blending()

            
        if self.is_costal_tile:
            snow_water_mask = self.get_water_snow_mask(current_tile)
            blended = blended.where(~snow_water_mask, 32767)
            
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
            output_file = self.geotiff_dir / f'{band_name}.tif'
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
        translate_tile(self.geotiff_dir, self.cog_dir, profile="ZSTD") # TODO: add list of bands, key RHs are prioritized
        if self.on_lumi:
            self.copy_data_to_lumio(self.translate_dir)
            self.delete_local_file(intersecting_s2_tiles)
        else:
            (self.flag_dir / f"{self.tile_id}_done").touch()

            
        

@dataclass
class AggGediToS2:
    year: int = 2020
    s2_grid_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    output_dir: str = f'~/data/gvs/predictions/{year}/blended'
    correction_stats_dir: str = f'~/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/{year}/none'
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'
    distance_map_dir: str = '~/data/gvs/assets/blending/distance_maps'
    flag_dir: str = f'~/data/gvs/state/{year}/blended'
    costal_tiles_file: str = '~/data/gvs/assets/worklists/tiles_coastal_regions.txt'
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
    main()
