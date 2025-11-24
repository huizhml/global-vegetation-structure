import os
import datetime
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import rasterio
from rasterio.warp import transform_bounds, transform_geom
from dotenv import load_dotenv
import pystac
from tqdm import tqdm

load_dotenv('.planetarycomputer/settings.env')


class StacCatalog:
    '''
    Create a STAC catalog for both years.
    Predictions for different year will be organized in different items, i.e., {tile_id}_{year}
    '''
    
    def __init__(self, collection_id:str=None, year: int=2020, catalog_dir: str=None, data_source: str=None, **kwargs): 
        self.collection_id = f'{collection_id}_{data_source}'
        self.year = year
        
        self.catalog_dir = Path(f'{catalog_dir}').expanduser()
        self.catalog_dir.mkdir(exist_ok=True)
        
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
                                       description=f'Global Vegetation Structure Model - {self.year}',
                                       extent=pystac.Extent(
                                        spatial=pystac.SpatialExtent([[-180, -90, 180, 90]]),
                                        temporal=pystac.TemporalExtent([[datetime.datetime(self.year,1,1), datetime.datetime(self.year,12,31)]]),
                                        ))
        return collection
    
    def add_items_to_collection(self):
        # 2024 has the most tiles, and all tiles from 2020
        tiles = os.listdir(Path('~/data/gvs/deploy/predictions_gtiff_2024').expanduser())
        for tile in tiles:
            path = Path(f'~/data/gvs/deploy/predictions_gtiff_2024/{tile}').expanduser()
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
        if year == 2020:
            masked_dir = Path(f'~/data/gvs/deploy/predictions_gtiff_masked_2020/{tile}').expanduser()
            gtiff_dir = Path(f'~/data/gvs/deploy/predictions_gtiff_2020/{tile}').expanduser()
            cog_dir = Path(f'~/data/gvs/deploy/predictions_2020/{tile}').expanduser()
            if masked_dir.exists(): # NOTE: all folders have 303 files, I didn't check the files inside the folder
                path = masked_dir
            elif gtiff_dir.exists():
                path = gtiff_dir
            elif cog_dir.exists():
                path = cog_dir
            else:
                print(f'No predictions found for tile {tile} in year {year}')
                return
        
        elif year == 2024:
            path = Path(f'~/data/gvs/deploy/predictions_gtiff_2024/{tile}').expanduser()
            if path.exists():
                path = path
            else:
                raise ValueError(f'No predictions found for tile {tile} in year {year}')
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

        # Add an asset later
        # if 'https' in path:
        #     href_prefix = ''
        # else:
        href_prefix = 'file://'
        for rh_idx in range(101):
            for q_idx in range(3):
                item.add_asset(
                    f"RH{rh_idx}_Q{q_idx}",
                    pystac.Asset(
                        href=f"{href_prefix}{path}/RH{rh_idx}_Q{q_idx}.tif",
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


@dataclass
class Config:
    collection_id: str ='vsm'
    catalog_dir: str = '~/data/gvs/deploy/gvsm_stac_catalog'
    data_source: str = 'local'
    task: str = 'create_catalog'
    
    
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
    
@hydra.main(config_name="config", version_base='1.2')
def main(cfg):
    print(cfg)
    stac_collection = StacCatalog(**cfg)
    if cfg.task == 'create_catalog':
        stac_collection.create_catalog()
    else:
        raise ValueError(f'Invalid task: {cfg.task}')

    
if __name__ == "__main__":
    main()  