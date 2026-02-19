from dataclasses import dataclass
from typing import List
from functools import lru_cache
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
import rasterio
import pystac
import stackstac
import xarray as xr
from shapely.geometry import shape
import matplotlib.pyplot as plt
import geopandas as gpd
import dask.array as da
from rasterio.crs import CRS
import dask
import time

NO_DATA = 32767

@lru_cache(maxsize=16)
def create_distance_arr(shape: tuple):
    rows = np.arange(shape[0], dtype=np.uint16)
    cols = np.arange(shape[1], dtype=np.uint16)

    # distance to top/bottom for each row; left/right for each col
    dy = np.minimum(rows, (shape[0] - 1) - rows)      # shape (H,)
    dx = np.minimum(cols, (shape[1] - 1) - cols)      # shape (W,)
    arr = np.minimum(dy[:, None], dx[None, :])  # shape (H, W), uint16
    return arr[None, :, :]

def create_distance_map(item_file: Path, output_file: Path):
    item = pystac.Item.from_file(str(item_file))
    distance_arr = create_distance_arr(tuple(item.assets['RH98_Q1'].extra_fields['proj:shape']))
    shape = item.assets['RH98_Q1'].extra_fields['proj:shape']
    profile = {
        "driver": "GTiff",
        "dtype": np.uint16,
        "count": 1,
        "width": shape[1],
        "height": shape[0],
        "crs": CRS.from_epsg(item.properties['proj:epsg']),
        "transform": rasterio.Affine(*item.assets['RH98_Q1'].extra_fields['proj:transform']),
        "compress": "zstd",
        "predictor": 2,
        "blockxsize": 256,
        "blockysize": 256,
        "tiled": True,
        "interleave": "band",
    }
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(distance_arr)
    print(f'Saved distance map to {output_file}')

def create_distance_map_for_all_tiles(stac_collection_dir: str, save_dir: str, **kwargs):
    '''
    Create distance maps for all tiles in the S2 grid
    '''
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    stac_collection_dir = Path(stac_collection_dir).expanduser()
    
    tile_items = stac_collection_dir.glob(f'*_2024') #NOTE: 2024 has more tiles, and contains all 2020 tiles
    unfinished_tiles = []
    for tile_item in tile_items:
        tile_id = tile_item.stem.split('_')[0]
        if (save_dir / f'{tile_id}.tif').exists():
            continue
        item_file = tile_item / f'{tile_id}_2024.json'
        unfinished_tiles.append(item_file)
    if len(unfinished_tiles) == 0:
        print('All distance maps already created')
        return
    print(f'Creating distance maps for {len(unfinished_tiles)} tiles')
    tasks  = []
    for item_file in unfinished_tiles:
        tile_id = item_file.stem.split('_')[0]
        output_file = save_dir / f'{tile_id}.tif'
        tasks.append(dask.delayed(create_distance_map)(item_file, output_file))
    dask.compute(*tasks)
    

def create_distance_map_item(map_item: pystac.Item, dist_map_file: str):
    '''
    Create distance map STAC item from an existing STAC item and a distance map file
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



class Blending:
    def __init__(self, year: int, 
                 tile_id: str, 
                 s2_grid_file: str, 
                 stac_collection_dir: str, 
                 distance_map_dir: str,
                 output_dir: str, 
                 small_area: bool = True, 
                 chunksize: int = 1024, **kwargs):
        self.year = year
        self.tile_id = tile_id
        self.chunksize = chunksize
        
        self.distance_map_dir = Path(distance_map_dir).expanduser()
        self.s2_grid_file = Path(s2_grid_file).expanduser()
        self.stac_collection_dir = Path(stac_collection_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.on_lumi = 'home' not in str(Path.home())
        
        self.s2_grid, self._sindex = self._init_s2_grid(s2_grid_file)
        

    def _init_s2_grid(self, s2_grid_file: str):
        '''
        Read S2 grid dataframe and spatial index
        '''
        s2_grid_file = Path(s2_grid_file).expanduser()
        s2_grid = gpd.read_parquet(s2_grid_file, columns=['Name', 'geometry'])
        return s2_grid, s2_grid.sindex

    def update_href(self, item: pystac.Item):
        '''
        Update href for all assets in the item
        '''
        if len(item.assets) == 1:
            file_path = self.distance_map_dir / f'{item.id.split("_")[0]}.tif'
            item.assets['distance_to_border'].href = f'file://{file_path}'
            return item
        old_dir = Path(item.assets['RH98_Q1'].href.replace('file://', '')).parent
        tile_id = old_dir.parts[-1]
        new_dir = self.get_bucket_url(tile_id, project_id=465001846)
            # raise ValueError(f"Tile {tile_id} does not exist in flash or scratch directory")
        for asset in item.assets:
            new_path = new_dir + Path(item.assets[asset].href).name.replace('.tif', '_uncompressed.tif')
            item.assets[asset].href = new_path
        return item
    
    def get_bucket_url(self, project_id: int=465001846):
        '''
        Get bucket url for a tile
        '''
        zone_name = self.tile_id[:3].lower()
        bucket_name = f"{zone_name}-{self.year}"
        return f"https://{project_id}.lumidata.eu/{bucket_name}/predictions_GTiff_{self.year}/{self.tile_id}/"
        

    def find_intersecting_s2_tiles(self, current_tile: str):
        '''
        Find intersecting S2 tiles
        '''
        tile_geom = self.s2_grid[self.s2_grid['Name'] == current_tile]['geometry'].iloc[0]
        geom = shape(tile_geom)
        idx = list(self._sindex.query(geom, predicate="intersects"))
        tiles = self.s2_grid.iloc[idx]['Name'].unique()
        return tiles
    
    def load_intersecting_images(self, current_tile: pystac.Item, intersecting_s2_tiles: list[str]):
        '''
        Load intersecting S2 images
        '''
        bbox_local = current_tile.assets['RH98_Q1'].extra_fields['proj:bbox']
        intersect_items = []
        intersect_dist_items = []
        for tile in intersecting_s2_tiles:
            intersect_stac_file = self.stac_collection_dir / f'{tile}_{self.year}/{tile}_{self.year}.json'
            if not intersect_stac_file.exists():
                print(f"Stac file {intersect_stac_file} does not exist, skipping")
                continue
            intersect_tile = pystac.Item.from_file(str(intersect_stac_file))
            intersect_items.append(intersect_tile)
            dist_map_file = self.distance_map_dir / f'{tile}.tif'
            if not dist_map_file.exists():
                create_distance_map(intersect_stac_file, dist_map_file)
            intersect_dist_item = create_distance_map_item(intersect_tile, dist_map_file)
            intersect_dist_items.append(intersect_dist_item)
            if self.on_lumi: # on lumi
                intersect_tile = self.update_href(intersect_tile)
                intersect_dist_item = self.update_href(intersect_dist_item)
            
        return intersect_items, intersect_dist_items, bbox_local
    
    def plot_images(self, images: xr.DataArray, save_dir: Path, name: str, colorbar: bool = False):
        '''
        Plot images
        '''
        if images.shape[0] == 1:
            plt.figure()
            plt.imshow(images.squeeze())
            if colorbar:
                plt.colorbar()
            plt.savefig(save_dir / f'{name}.png')
            plt.close()
            return
        _, axes = plt.subplots(3,3, figsize=(10, 10))
        for i, tile_id in enumerate(images.id.data):
            row = i // 3
            col = i % 3
            axes[row, col].imshow(images.isel(time=i, band=0).data)
            axes[row, col].set_title(tile_id)
        plt.tight_layout()
        plt.savefig(save_dir / f'{name}.png')
        plt.close()
    
    def blend_one_tile(self, intersect_items: List[pystac.Item], intersect_dist_items: List[pystac.Item], bbox_local: tuple, epsg: int):
        '''
        Blend one tile
        '''
        dist_images = stackstac.stack(
            intersect_dist_items, assets=['distance_to_border'], chunksize=self.chunksize,
            epsg=epsg, bounds=bbox_local,
            resolution=10, rescale=False, dtype='float32', fill_value=np.float32(np.nan))
        weights = dist_images.where(dist_images.notnull(), 0)
        sum_w = weights.sum(dim='time')
        weights_normalized = xr.where(sum_w > 0, weights / sum_w, np.nan).transpose(*weights.dims) # masked area like water, the sum can be 0
        # weights_normalized = weights_normalized.persist()
        intersect_images = stackstac.stack(
                intersect_items, chunksize=self.chunksize,
                # assets=[f'RH{i}_Q1' for i in range(101)],
                epsg=epsg, bounds=bbox_local,
                resolution=10, rescale=False, dtype='float32', fill_value=np.float32(np.nan))

        blended = (intersect_images * weights_normalized.data).sum(dim='time', min_count=1) #!!!!! skipna=True is the default, and it'll return 0 if all are nan, we need min_count=1 to be able to mask water, built-up, snow
        blended = blended.round().fillna(NO_DATA).astype(np.int16)
        return dist_images, intersect_images, blended
        
    
    
    def verify_blending(self, small_area: bool = True):
        '''
        Verify blending on a small area & large tiles
        '''
        
        save_dir = Path(f'~/data/gvs/deploy/blending').expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        stac_file = self.stac_collection_dir / f'{self.tile_id}_{self.year}/{self.tile_id}_{self.year}.json'
        if not stac_file.exists():
            print(f"Stac file {stac_file} does not exist, skipping")
            return
        current_tile = pystac.Item.from_file(str(stac_file))
        intersecting_s2_tiles = self.find_intersecting_s2_tiles(self.tile_id)
        intersect_items, intersect_dist_items, bbox_local = self.load_intersecting_images(current_tile, intersecting_s2_tiles)
        dist_images, intersect_images, blended = self.blend_one_tile(intersect_items, intersect_dist_items, bbox_local, current_tile.properties['proj:epsg'])
        intersect_images = intersect_images.sel(band=['RH98_Q1'])
        blended = blended.sel(band=['RH98_Q1'])
        if small_area:
            blended = blended.isel(x=slice(0, 2048), y=slice(8932, 10980))
            dist_images = dist_images.isel(x=slice(0, 2048), y=slice(8932, 10980))
            intersect_images = intersect_images.isel(x=slice(0, 2048), y=slice(8932, 10980))
        
        blended = blended.compute()
        blended = blended.astype(np.float32)
        blended = np.where(blended==NO_DATA, np.nan, blended)
        dist_images = dist_images.compute()
        intersect_images = intersect_images.compute()
        current_tile_idx = np.argmax(intersect_images.id.data == current_tile.id)
        diff = blended - intersect_images.isel(time=current_tile_idx)
        save_dir = self.output_dir / 'figures'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_images(intersect_images, save_dir, 'predictions')
        self.plot_images(dist_images, save_dir, 'distance_to_border')
        self.plot_images(blended, save_dir, 'blended')
        self.plot_images(blended, save_dir, 'blended')
        self.plot_images(diff.data, save_dir, 'difference', colorbar=True)

        if not small_area:
            with rasterio.open(current_tile.assets['RH98_Q1'].href) as src:
                profile = src.profile
            print(f'saved to {save_dir / f'{current_tile.id}_blended.tif'}')
            with rasterio.open((save_dir / f'{current_tile.id}_blended.tif'), 'w', **profile) as dst:
                dst.write(blended.squeeze(), 1)

    def run_blending(self):
        '''
        Run blending
        '''
        stac_file = self.stac_collection_dir / f'{self.tile_id}_{self.year}/{self.tile_id}_{self.year}.json'
        if not stac_file.exists():
            print(f"Stac file {stac_file} does not exist, skipping")
            return
        current_tile = pystac.Item.from_file(str(stac_file))
        intersecting_s2_tiles = self.find_intersecting_s2_tiles(self.tile_id)
        intersect_items, intersect_dist_items, bbox_local = self.load_intersecting_images(current_tile, intersecting_s2_tiles)
        _, _, blended = self.blend_one_tile(intersect_items, intersect_dist_items, bbox_local, current_tile.properties['proj:epsg'])
        return blended, current_tile, intersecting_s2_tiles

@dataclass
class Config:
    stac_collection_dir: str = '~/data/gvs/products/gvsm_stac_catalog/vsm_local'
    distance_map_dir: str = '~/data/gvs/assets/blending/distance_maps'
    s2_grid_file: str = '~/data/gvs/state/s2_tiles_with_growing_months.parquet'
    
    output_dir: str = '~/data/gvs/sanity_checks/blending'
    tile_id: str = '21MXQ'
    year: int = 2020
    
    linear_correct: bool = False
    small_area: bool = True
    task: str = 'create_distance_maps'


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_name="config", version_base='1.2')
def main(cfg):
    print(cfg)
    blending = Blending(**cfg)
    t0 = time.time()
    if cfg.task == 'create_distance_maps':
        blending.verify_blending(small_area=cfg.small_area)
    print(f'Time taken: {time.time() - t0} seconds')
if __name__ == '__main__':
    main()
