import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import hydra
from pathlib import Path
import tempfile
import subprocess
import time

class ERDA:
    def __init__(self):
        pass

    
    def upload_geoparquet_as_geojson(self, geoparquet_file: str):
        geoparquet_file = Path(geoparquet_file).expanduser()
        df = gpd.read_parquet(geoparquet_file)
        df["geometry"] = df["geometry"].apply(lambda geom: geom if geom.is_empty else shapely.force_2d(geom))
        df.to_file(driver='GeoJSON', filename=geoparquet_file.with_suffix('.geojson'))
        with tempfile.NamedTemporaryFile("w", delete=False) as batch:
            batch.write("mkdir GVS/deploy_status\n")
            batch.write("cd GVS/deploy_status\n")
            batch.write(f"put {geoparquet_file.with_suffix('.geojson')}\n")
            batch_path = batch.name

        subprocess.run(["sftp", "-b", batch_path, "ucph-erda"], check=True)


@dataclass
class MyConfig:
    geoparquet_file: str = '~/data/gvs/deploy/deploy_status.parquet'

cs = ConfigStore.instance()
cs.store(name='gee', node=MyConfig)

@hydra.main(config_name='gee', version_base='1.2')
def main(cfg):
    erda = ERDA()
    erda.upload_geoparquet_as_geojson(cfg.geoparquet_file)
    
    
if __name__ == '__main__':
    main()
    
