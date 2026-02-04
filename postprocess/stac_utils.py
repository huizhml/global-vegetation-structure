# stac_utils.py
from pathlib import Path
import pystac

def create_distance_map_item(map_item: pystac.Item, dist_map_file: str | Path) -> pystac.Item:
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


def make_bucket_url(tile_id: str, year: int, project_id: int = 465001846) -> str:
    zone_name = tile_id[:3].lower()
    bucket_name = f"{zone_name}-{year}"
    return f"https://{project_id}.lumidata.eu/{bucket_name}/predictions_GTiff_{year}/{tile_id}/"


def update_item_hrefs(
    item: pystac.Item,
    *,
    distance_map_dir: Path,
    year: int,
    project_id: int = 465001846,
) -> pystac.Item:
    """
    Rewrites hrefs to point to local flash/scratch if present & complete, else bucket URL.
    Handles both distance map items (single asset) and prediction items.
    """
    if len(item.assets) == 1 and "distance_to_border" in item.assets:
        file_path = distance_map_dir / f"{item.id.split('_')[0]}.tif"
        item.assets["distance_to_border"].href = f"file://{file_path}"
        return item

    old_dir = Path(item.assets["RH98_Q1"].href.replace("file://", "")).parent
    tile_id = old_dir.parts[-1]
    base = make_bucket_url(tile_id, year=year, project_id=project_id)
    for a in item.assets:
        new_path = base + Path(item.assets[a].href).name.replace(".tif", "_uncompressed.tif")
        item.assets[a].href = new_path
    return item