import os
import datetime
import re
from dataclasses import dataclass
import numpy as np
from hydra.core.config_store import ConfigStore
from pathlib import Path
import rasterio
from rasterio.warp import transform_bounds, transform_geom
from dotenv import load_dotenv
import pystac
from tqdm import tqdm
import dask
from const import LUMI_PROJECT
load_dotenv('.planetarycomputer/settings.env')


class StacCatalog:
    '''
    Create a STAC catalog for both years.
    Predictions for different year will be organized in different items, i.e., {tile_id}_{year}
    data_dir: 
    '''
    
    def __init__(self, collection_id:str=None, catalog_dir: str=None, data_source: str=None, data_dir: str=None,  new_predictions_dir: str=None, **kwargs): 
        self.collection_id = f'{collection_id}_{data_source}'
        
        self.catalog_dir = Path(f'{catalog_dir}').expanduser()
        self.catalog_dir.mkdir(exist_ok=True)
        self.data_dir = Path(f'{data_dir}').expanduser()
        self.new_predictions_dir = new_predictions_dir
        
    def create_catalog(self):
        catalog  = pystac.Catalog(id='gvsm', description='Global Vegetation Structure Model')
        self.collection = self.create_collection()
        catalog.add_child(self.collection)
        self.add_items_to_collection()
                # 4. Normalize HREFs (choose a directory structure)
        catalog.normalize_hrefs(str(self.catalog_dir))

        # 5. Save everything (catalog.json, collection.json, item.jsons)
        catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED, dest_href=str(self.catalog_dir))


    def create_collection(self):
        collection = pystac.Collection(id=self.collection_id, 
                                       description=f'Global Vegetation Structure Model',
                                       extent=pystac.Extent(
                                        spatial=pystac.SpatialExtent([[-180, -90, 180, 90]]),
                                        temporal=pystac.TemporalExtent([[datetime.datetime(2020,1,1), datetime.datetime(2024,12,31)]]), # TODO: would this be misleading, the temporal coverage is actually 2020 and 2024
                                        ))
        return collection
    
    def get_tile_pred_dir(self, year: int=2024, tile: str=None):
        if year == 2020:
            pred_dir = self.data_dir / f'2020/original/tiles/cog/{tile}'
            if pred_dir.exists():
                return f'file://{str(pred_dir)}'
            else:
                raise ValueError(f'No predictions found for tile {tile} in year {year}')
        elif year == 2024:
            zone = tile[:3].lower()
            return f'https://{LUMI_PROJECT}.lumidata.eu/{zone}-{year}/{year}/original/tiles/{tile}' # NOTE: for future use. TODO: how to check if the files exist?
        else:
            raise ValueError(f'Invalid year: {year}')
    
    def add_items_to_collection(self):
        tiles = os.listdir(self.data_dir / f'2024/original/tiles/cog') # 2024 has the most tiles, including all tiles from 2020
        for tile in tiles:
            path = self.data_dir / tile
            tile_info = self.get_raster_info(str(path))
            self.create_stac_item(2020, tile, tile_info)
            self.create_stac_item(2024, tile, tile_info)

    
    def get_raster_info(self, path: str):
        with rasterio.open(f'{path}/RH98_Q1.tif') as dataset:
            src_crs = dataset.crs
            bounds = dataset.bounds  # left, bottom, right, top
            shape = dataset.shape
            transform = dataset.transform[:6]
            resolution = dataset.res[0]
            nodata = dataset.nodata
            data_type = dataset.dtypes[0]

        # Build a polygon from the bounds in native CRS
        src_poly = {
            "type": "Polygon",
            "coordinates": [[
                [bounds.left,  bounds.bottom],
                [bounds.left,  bounds.top],
                [bounds.right, bounds.top],
                [bounds.right, bounds.bottom],
                [bounds.left,  bounds.bottom],
            ]]
        }

        # Transform to WGS84 for STAC
        geom_4326 = transform_geom(src_crs, "EPSG:4326", src_poly, precision=6)
        minx, miny, maxx, maxy = transform_bounds(
            src_crs, "EPSG:4326", *bounds, densify_pts=21
        )
        bbox_4326 = [minx, miny, maxx, maxy]
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
            
        return {
            'src_crs': src_crs,
            'geom_4326': geom_4326,
            'bbox_4326': bbox_4326,
            'bbox': bbox,
            'shape': shape,
            'transform': transform,
            'resolution': resolution,
            'nodata': nodata,
            'data_type': data_type
        }
    
    def create_stac_item(self, year, tile, tile_info: dict): # TODO: add q_idx
        #NOTE: for future use, every updated prediction should also be translated to cog, to avoid copies of the large amount of data
        pred_dir = self.get_tile_pred_dir(year, tile)
        # if year == 2020:
        #     masked_dir = Path(f'~/data/gvs/deploy/predictions_gtiff_masked_2020/{tile}').expanduser()
        #     gtiff_dir = Path(f'~/data/gvs/deploy/predictions_gtiff_2020/{tile}').expanduser()
        #     cog_dir = Path(f'~/data/gvs/deploy/predictions_2020/{tile}').expanduser()
        #     if masked_dir.exists(): # NOTE: all folders have 303 files, I didn't check the files inside the folder
        #         path = masked_dir
        #     elif gtiff_dir.exists():
        #         path = gtiff_dir
        #     elif cog_dir.exists():
        #         path = cog_dir
        #     else:
        #         print(f'No predictions found for tile {tile} in year {year}')
        #         return
        
        # elif year == 2024:
        #     path = Path(f'~/data/gvs/deploy/predictions_gtiff_2024/{tile}').expanduser()
        #     if path.exists():
        #         path = path
        #     else:
        #         raise ValueError(f'No predictions found for tile {tile} in year {year}')
        item = pystac.Item(
            id=f'{tile}_{year}',
            geometry=tile_info['geom_4326'],
            bbox=tile_info['bbox_4326'],
            datetime=datetime.datetime.strptime(f'{year}-01-01T00:00:00Z', '%Y-%m-%dT%H:%M:%S%z'),
            properties={
                'proj:epsg': tile_info['src_crs'].to_epsg(),
                'raster:bands': [
                    {
                        'nodata': tile_info['nodata'],
                        'data_type': tile_info['data_type'],
                        'spatial_resolution': tile_info['resolution']
                    }
                ]
            }
        )

        for rh_idx in range(101):
            for q_idx in range(3):
                item.add_asset(
                    f"RH{rh_idx}_Q{q_idx}",
                    pystac.Asset(
                        href=f"{pred_dir}/RH{rh_idx}_Q{q_idx}.tif",
                        media_type="image/tiff; application=geotiff; profile=cloud-optimized",
                        title=f'Median RH {rh_idx} - 10m',
                        roles=["data"],
                        extra_fields={
                            'proj:bbox': tile_info['bbox'],
                            'proj:shape': tile_info['shape'],
                            'proj:transform': tile_info['transform'],
                            'gsd': tile_info['resolution']
                        },
                    )
            )
        # image = stackstac.stack(item, resolution=10)
        # test = image.isel(time=0, band=0).compute() #@2025-10-02, tested, works well, can load image from tif files
        self.collection.add_item(item)

    def update_collection(self):
        '''
        Usually the updated predictions should overwrite the original predictions, to make further operation logic simpler. Old predictions can be achieved in a separate folder.
        This function is mainly for case that the original prediction folder is no longer editable, e.g., in LUMI old project.
        '''
        year = re.search(r'\d{4}', self.new_predictions_dir).group(0)
        year = int(year)
        new_predictions_dir = Path(self.new_predictions_dir).expanduser()
        collection_dir = f'{self.catalog_dir}/{self.collection_id}'
        collection_dir = Path(collection_dir).expanduser()
        # item_dirs = collection_dir.glob(f'*_{year}')
        tiles = os.listdir(new_predictions_dir)
        tasks = []
        for tile_id in tiles:
            tasks.append(dask.delayed(self.update_item)(tile_id, year, new_predictions_dir))
        dask.compute(*tasks)

    def update_item(self, tile_id: str, year: int=2024, new_predictions_dir: str=None):
        item_path = f'{self.catalog_dir}/{self.collection_id}/{tile_id}_{year}/{tile_id}_{year}.json'
        item = pystac.Item.from_file(item_path)
        if len(list((new_predictions_dir/tile_id).glob('RH*Q*.tif'))) != 303:
            print(f'Not enough predictions for tile {tile_id} in year {year}, skipping')
            return
        for rh_idx in range(101):
            for q_idx in range(3):
                item.assets[f'RH{rh_idx}_Q{q_idx}'].href = f"file://{new_predictions_dir / tile_id / f'RH{rh_idx}_Q{q_idx}.tif'}"
        item.save_object(dest_href=item_path)
        print(f'Updated STAC item saved to {item_path}')

    
    # def update_item(self, tile_id: str, year: int=2024):
    #     item_path = f'{self.catalog_dir}/{self.collection_id}/{tile_id}_{year}/{tile_id}_{year}.json'
    #     item = pystac.Item.from_file(item_path)
    #     gtif_dir = Path(f'~/data/gvs/predictions/{year}/original/tiles/geotiff/{tile_id}').expanduser()
    #     cog_dir = Path(f'~/data/gvs/predictions/{year}/original/tiles/cog/{tile_id}').expanduser()
    #     file_path = item.assets['RH98_Q1'].href.replace('file://', '')
    #     file_path = Path(file_path).expanduser()
    #     if file_path.exists() and len(list(file_path.parent.glob('*.tif'))) == 303:
    #         print('Local file exists and is complete, skip updating')
    #         return
    #     if gtif_dir.exists() and len(list(gtif_dir.glob('*.tif'))) == 303:
    #         pred_dir = gtif_dir
    #         print(f'GTiff directory exists and is complete, updating')
    #     elif cog_dir.exists() and len(list(cog_dir.glob('*.tif'))) == 303:
    #         pred_dir = cog_dir
    #         print(f'COG directory exists and is complete, updating')
    #     elif self.year == 2024:
    #         pred_dir = gtif_dir
    #         print(f'2024 predictions directory exists and is complete, updating')
    #     else:
    #         raise ValueError(f'No predictions found for tile {tile_id} in year {self.year}')
    #     # Update asset hrefs
    #     for key, asset in item.assets.items():
    #         asset.href = f'file://{pred_dir / Path(asset.href).name}'

    #     # 🔑 Save updated item back to disk
    #     item.save_object(dest_href=item_path)

    #     print(f'Updated STAC item saved to {item_path}')