import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import hydra
import geemap
import ee
from pathlib import Path
from download._utils import authenticate
from google.cloud import storage
import time

authenticate()

class GEE:
    def __init__(self, gee_asset: str):
        self.gee_asset = gee_asset

    def get_gee_asset(self):
        return self.gee_asset
    
    def upload_geoparquet(self, geoparquet_file: str, description: str='upload_geoparquet'):
        geoparquet_file = Path(geoparquet_file).expanduser()
        df = gpd.read_parquet(geoparquet_file)
        df["geometry"] = df["geometry"].apply(lambda geom: geom if geom.is_empty else shapely.force_2d(geom))
        temp_file = geoparquet_file.with_suffix('.geojson')
        df = df.to_file(driver='GeoJSON', filename=temp_file)
        # upload to GCS
        bucket_name = 'gs://gvs-deploy-status'
        gcs_path = f'{bucket_name}/{geoparquet_file.stem}.geojson'
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(temp_file)
        # ingest to Earth Engine
        asset_id = f'{self.gee_asset}/{geoparquet_file.stem}'
        ingestion_task = ee.data.startIngestion(ee.data.newTaskId()[0], {
            "id": asset_id,
            "sources": [{
                "primaryPath": gcs_path,
                "files": [gcs_path]
            }]
        })
        print(f'Ingesting {geoparquet_file} to {asset_id}')
        # Optional: monitor progress
        while True:
            status = ee.data.getTaskStatus(ingestion_task["id"])[0]
            print("State:", status["state"])
            if status["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            time.sleep(10)
        import ipdb; ipdb.set_trace()
        return asset_id

@dataclass
class MyConfig:
    gee_asset: str = 'projects/gisproject-1/assets/GVS_deploy_status'
    geoparquet_file: str = '~/data/gvs/deploy/deploy_status.parquet'

cs = ConfigStore.instance()
cs.store(name='gee', node=MyConfig)

@hydra.main(config_name='gee', version_base='1.2')
def main(cfg):
    gee = GEE(cfg.gee_asset)
    gee.upload_geoparquet(cfg.geoparquet_file)
    
    
if __name__ == '__main__':
    main()
    
