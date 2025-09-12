import os
import time
import logging
import hydra
import h5py
import dask
import dask.bag as db
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from dask.utils import natural_sort_key
from datatree.io import _iter_nc_groups
from h5netcdf.legacyapi import Dataset as h5Dataset

logger = logging.getLogger(__name__)

class IndexTableGenerater:
    """
    Generate an index table for each zone, will be saved as out_idx_dir/{zone}.parquet
    Saved index table will have columns: path, in_partition_idx, s2_tile, lat, lon, sensitivity, shot_number
    Using dask to process zones in parallel. Thus using h5_dir/{zone}.h5 as input.
    """

    def __init__(self, h5_dir:str='~/data/GEDI', out_idx_dir:str='~/data/index_table') -> None:
        """
        * h5_dir: folder where {zone}.h5 is
        * out_idx_dir: folder where {zone}.parquet is saved to
        """
        self.h5_dir = Path(h5_dir).expanduser()
        self.out_idx_dir = Path(out_idx_dir).expanduser()
        self.out_idx_dir.mkdir(parents=True, exist_ok=True)
        h5_files = self.h5_dir.glob('*.h5')
        zones = [f.stem for f in h5_files]
        self.target_files = [str(self.out_idx_dir/f'{z}.parquet') for z in zones]
        self.target_files = sorted(self.target_files, key=natural_sort_key)
        if debug:
            self.target_files = self.target_files[:1] # for testing
            print('target_files:', self.target_files)
    
    def __call__(self):
        print('generating index tables for ', self.unfinished_files)
        if len(self.unfinished_files) == 0:
            logger.info('All zones have index tables generated.')
            return
        files_db = db.from_sequence(self.unfinished_files)
        files_db.map(self.gen_index_table_for_zone).compute()


    @property
    def unfinished_files(self):
        files = []
        for fp in self.target_files:
            if not os.path.exists(fp):
                files.append(fp)
        return files

    def gen_index_table_for_zone(self, index_table_file, save=True):
        """
        Go through all groups in the HDF5 file and generate index table.

        Args:
            index_table_file (str): The overall index table for zones used.
            save (bool, optional): Whether to save the index tables as CSV files. Defaults to True.

        Returns:
            None
        """
        index_table_file = Path(index_table_file)
        zone = index_table_file.stem
        index_df = []
        with h5Dataset(self.h5_dir/f'{zone}.h5', mode='r') as ncds:
            with h5py.File(self.h5_dir/f'{zone}.h5',) as data:
                for group in _iter_nc_groups(ncds):
                    if len(group.split('/')) == 3:
                        s2_ids = data[f'{group}/id'][:].astype('U')
                        latlon = data[f'{group}/latlon'][:]
                        sensitivity = data[f'{group}/gedi_attrs'][:, 24]
                        shot_number = data[f'{group}/shot_number'][:].astype('U')
                        index_df.append([f'/{zone}{group}', s2_ids,latlon[:,0],latlon[:,1], sensitivity, shot_number])
        index_df = pd.DataFrame(index_df, columns=['path','s2_tile', 'lat', 'lon', 'sensitivity', 'shot_number'])
        index_df = index_df.explode(['s2_tile', 'lat', 'lon', 'sensitivity', 'shot_number'])
        index_df['s2_tile'] = index_df['s2_tile'].str[33:38]
        index_df['in_partition_idx'] = index_df.groupby('path').cumcount()
        before = len(index_df)
        index_df = index_df[index_df['sensitivity'] >= 0.95]
        print(f'filtered out {before - len(index_df)} shots with sensitivity < 0.95 for zone {zone}')
        index_gdf = gpd.GeoDataFrame(index_df, geometry=gpd.points_from_xy(index_df.lon, index_df.lat), crs='EPSG:4326')


        if save:
            index_gdf.to_parquet(index_table_file)
            print(f'index table saved to: ', index_table_file)




@dataclass
class MyConfig:
    h5_dir: str = '~/data/GEDI/GEDI_S2_h5s_original'
    out_idx_dir: str = '~/data/GEDI/geo_index_table_with_sensitivity'


cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_name="my_config", version_base="1.2")
def main(cfg):
    # check if the zone has Sentinel-2 candidates gathered
    from dask.distributed import Client, LocalCluster
    # might fix the communication error caused by I/O. ref: https://github.com/dask/distributed/issues/3129#issuecomment-1684858307
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 30})
    dask.config.set({"distributed.scheduler.active-memory-manager.MALLOC_TRIM_THRESHOLD_": 0})
    cluster = LocalCluster()
    client = Client(cluster)
    print(client)

    t0 = time.time()
    IndexTableGenerater(**cfg)()
    
    logger.info(f'time taken: {time.time() - t0}')
    client.close()

if __name__ == '__main__':
    main()