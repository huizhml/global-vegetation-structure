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
from dask.utils import natural_sort_key
from datatree.io import _iter_nc_groups
from h5netcdf.legacyapi import Dataset as h5Dataset

logger = logging.getLogger(__name__)

class IndexTableGenerater:
    """
    Using dask to process zones in parallel.
    Generate an index table for each zone.
    """

    def __init__(self, h5_dir:str='~/data/GEDI', index_table_dir:str='~/data/index_table') -> None:
        """
        * h5_dir: folder where {zone}.h5 is
        * index_table_dir: folder where {zone}.parquet is saved to
        """
        self.h5_dir = Path(h5_dir).expanduser()
        self.index_table_dir = Path(index_table_dir).expanduser()
        self.index_table_dir.mkdir(parents=True, exist_ok=True)
        h5_files = self.h5_dir.glob('*.h5')
        zones = [f.stem for f in h5_files]
        self.target_files = [str(self.index_table_dir/f'{z}.parquet') for z in zones]
        self.target_files = sorted(self.target_files, key=natural_sort_key)
    
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
                        index_df.append([f'/{zone}{group}', s2_ids,latlon[:,0],latlon[:,1]])
        index_df = pd.DataFrame(index_df, columns=['path','s2_tile', 'lat', 'lon'])
        index_df = index_df.explode(['s2_tile', 'lat', 'lon'])
        index_df['s2_tile'] = index_df['s2_tile'].str[33:38]
        index_df['in_partition_idx'] = index_df.groupby('path').cumcount()
        index_gdf = gpd.GeoDataFrame(index_df, geometry=gpd.points_from_xy(index_df.lon, index_df.lat), crs='EPSG:4326')

        if save:
            index_gdf.to_parquet(index_table_file)
            print(f'index table saved to: ', index_table_file)


@hydra.main(config_path="../config", config_name="s2_download", version_base="1.2")
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
    bestS2Finder = IndexTableGenerater(index_table_dir='~/data/geo_index_table')()
    
    logger.info(f'time taken: {time.time() - t0}')
    client.close()

if __name__ == '__main__':
    main()