import time
import os
import ee
import json
from typing import Any, Dict
from pathlib import Path
import json
import pandas as pd
import dask.dataframe as dd
import geopandas as gpd

def authenticate():
    key_file = os.environ['KEY_FILE']
    key = json.load(open(key_file))
    credentials = ee.ServiceAccountCredentials(key['client_email'], key_file)
    ee.Initialize(credentials, url='https://earthengine-highvolume.googleapis.com')
    print('Authenticated Earth Engine successfully') 

def get_missing_fc(missing_table:Path=None):
    '''
    Return the list of assets in LARSE/GEDI/GEDI02_A_002 but not in LARSE/GEDI/GEDI02_A_002_INDEX
    LARSE/GEDI/GEDI02_A_002: a folder containing all GEDI02_A_002 assets as feature collections
    LARSE/GEDI/GEDI02_A_002_INDEX: a feature collection containing most GEDI02_A_002 orbits(boundaries) as features
    
    Args:
        missing_table (Path): Path to the missing table CSV file (optional)
    
    Returns:
        DataFrame: A DataFrame containing the list of missing assets
    
    '''
    if missing_table.exists():
        missing = pd.read_csv(missing_table)
    else:
        assets = ee.data.listAssets('LARSE/GEDI/GEDI02_A_002')['assets']
        assets = pd.DataFrame(assets)
        assets_indexed = ee.data.listFeatures({'assetId': 'LARSE/GEDI/GEDI02_A_002_INDEX', 'fileFormat':'PANDAS_DATAFRAME'})
        missing = [asset['id'] for asset in assets if asset['id'] not in assets_indexed['table_id']]
        missing = assets[~assets['id'].isin(assets_indexed['table_id'])]
    return missing

def add_missing_fc(zone: Dict[str, Any], missing_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Add the id of missing feature collections to mgrs dataframe .

    Args:
        zone (Dict[str, Any]): The zone dictionary containing information about the zone.
        missing_df (pd.DataFrame): The DataFrame containing the missing feature collections.

    Returns:
        Dict[str, Any]: The updated zone dictionary with the missing feature collections added.
    """
    zone.loc['tracks'] = json.loads(zone['tracks'])
    for id in missing_df['id']:
        geom = ee.Geometry.BBox(*zone['geometry'].bounds)
        flag = ee.FeatureCollection(id).filterBounds(geom).size().getInfo()
        if flag > 0:
            zone['tracks'].append(id)
    return zone

class MGRS:
    """
    Class for handling MGRS (Military Grid Reference System) data.

    Attributes:
        gee_asset (str): The Earth Engine asset ID for the MGRS grid with landmass and GEDI asset IDs(not complete).
        mgrs_file (Path): The path to the file containing the MGRS dataframe.
        use_dask (bool): Flag indicating whether to use Dask for reading the file.
        npartitions (int): The number of partitions to use when using Dask.
        missing_file (Path): The path to the file containing the missing GEE assets(feature collections).

    Methods:
        __init__(mgrs_file:Path, missing_file:Path, use_dask:bool=False, npartitions=60)
            Initialize the MGRS object.
        get_mgrs()
            Retrieves the MGRS data.
        update_mgrs(mgrs_df, missing_file:Path)
            Update the MGRS data to include all GEDI assets.
    """

    def __init__(self, mgrs_file:Path, missing_file:Path, use_dask:bool=False, npartitions=60):
        """
        Initialize the MGRS object.

        Parameters:
            mgrs_file (Path): The path to the file containing the MGRS data.
            missing_file (Path): The path to the file containing the missing data.
            use_dask (bool, optional): Flag indicating whether to use Dask for reading the file. Defaults to False.
            npartitions (int, optional): The number of partitions to use when using Dask. Defaults to 60.
        """
        self.gee_asset = 'projects/gisproject-1/assets/mgrs_with_landmass_and_gedi_counts'
        self.mgrs_file = mgrs_file
        self.use_dask = use_dask
        self.npartitions = npartitions
        self.missing_file = missing_file

    def get_mgrs(self):
        """
        Retrieves the MGRS (Military Grid Reference System) data.
        If it's not already downloaded, it will be downloaded from Earth Engine.
        Currently, the GEE asset is MGRS grid with landmass and GEDI asset IDs(only from LARSE/GEDI/GEDI02_A_002_INDEX).
        If download from GEE, it needs to be updated to include all GEDI assets.

        Returns:
            mgrs_df: The MGRS data as a GeoDataFrame.
        """
        
        if not self.mgrs_file.exists():
            print('Downloading MGRS data from Earth Engine...')
            mgrs = ee.FeatureCollection(self.gee_asset)
            mgrs_df = ee.data.computeFeatures({'expression': mgrs, 'fileFormat': 'GEOPANDAS_GEODATAFRAME'})
            mgrs_df = self.update_mgrs(mgrs_df, self.missing_file)
        else:
            mgrs_df = gpd.read_parquet(self.mgrs_file)
        if self.use_dask:
            mgrs_df = dd.from_pandas(mgrs_df, npartitions=self.npartitions)
                
        return mgrs_df
    
    def update_mgrs(self, mgrs_df, mssing_file:Path):
        """
        Update the MGRS data to include all GEDI assets.

        Parameters:
            mgrs_df: The MGRS data as a GeoDataFrame.
            missing_file (Path): The path to the file containing the missing data.
        """
        from dask.distributed import Client, LocalCluster
        from dask import config
        config.set({'interface': 'lo'}) 
        cluster = LocalCluster()
        client = Client(cluster)#timeout

        print(f'Updating MGRS data to include all GEDI assets...')
        mgrs_df = dd.from_pandas(mgrs_df, npartitions=self.npartitions)
        missing_df = get_missing_fc(mssing_file)
        meta = {'geometry': 'object', 'MGRS_UTM': 'str', 'landmass': 'uint16', 'tracks': 'object'}
        mgrs_df = mgrs_df[['geometry', 'MGRS_UTM', 'landmass', 'tracks']].apply(add_missing_fc, axis=1, args=(missing_df,), meta=meta).compute()
        mgrs_df.to_parquet(self.mgrs_file)

    def remove_empty_tracks(self, mgrs_df):
        """
        Remove empty tracks from the MGRS data.

        Parameters:
            mgrs_df: The MGRS data as a GeoDataFrame.
        """
        invalid_id = 'LARSE/GEDI/GEDI02_A_002/GEDI02_A_2022362115234_O22900_01_T06690_02_003_02_V002'
        mgrs_df['tracks'] = mgrs_df['tracks'].apply(lambda x: x[x!=invalid_id])
        mgrs_df.to_parquet(self.mgrs_file)


# authenticate()

if __name__ == '__main__':
    data_dir = 'GEDI'
    mgrs_file = Path.home() /data_dir/ 'mgrs_with_tracks.parquet'
    missing_df = Path.home() /data_dir/ 'missing.csv'
    mgrs = MGRS(mgrs_file, missing_df)
    mgrs_df = mgrs.get_mgrs()
    mgrs.remove_empty_tracks(mgrs_df)
    print(mgrs_df.head())
