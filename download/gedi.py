import os
import ee
import logging
import requests
import multiprocessing
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import shape
from pathlib import Path
import tqdm
from retry import retry
import dask.dataframe as dd
import dask.array as da
import geemap
import hvplot.pandas
import matplotlib.pyplot as plt
import hydra

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

class GEDI:
    """
    A class for filtering, sampling(stratified, per orbit & per cell) and downloading GEDI data from Google Earth Engine.

    Attributes:
        nSampledPerKm2 (float): The number of GEDI points to sample per km^2.
        raster (ee.ImageCollection): The GEDI raster data collection in Google Earth Engine.
        grid (ee.FeatureCollection): The MGRS grid feature collection in GEE, each grid cell has assigned 'count'(filtered GEDI Points), 'landmass'.
        year (int): The year of the GEDI data to download.
        dataFolder (pathlib.Path): The local folder to save the downloaded GEDI data.
        keepLeafOff255 (bool): Whether to keep GEDI points with leaf off flag of 255.

    Methods:
        getTableAssetIds(export=False):
            Returns a list of unique table asset IDs for GEDI orbits from GEE rasterized version of GEDI within the specified year.

        getValidZoneCodes(export=False):
            Returns a list of valid zone codes of MRGS grid cells where GEDI points are present.

        download(processes=40, maxTries=3):
            Downloads GEDI data for all valid MGRS grid cells in the specified year.

        filterGEDI(asset_id):
            Returns a filtered GEDI feature collection for specified table asset ID.
            Filter by quality flag, degrade flag, region class, and growing season.

        sampleCell(zone_code):
            Downloads GEDI data for the specified MGRS grid cell.

        getSampleRatio(cell):
            Returns the number of GEDI points to sample for the specified MGRS grid cell.
    """

    nSampledPerKm2 = 0.66793882312 # this is obtained from the total number of points we want to sample per year(75M) and the total landmass in the world

    def __init__(self, year=2019, dataFolder='gedi', keepLeafOff255=True, debug=False):
        """
        Initializes a GEDI object.

        Args:
            year (int, optional): The year of the GEDI data to download. Defaults to 2019.
            dataFolder (str, optional): The local folder to save the downloaded GEDI data. Defaults to 'gedi'.
            keepLeafOff255 (bool, optional): Whether to keep GEDI points with leaf off flag of 255. Defaults to False.
        """
        self.raster = ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
        self.mgrs = ee.FeatureCollection('projects/gisproject-1/assets/gedi_count_mgrs_aggregated_landmass')
        assetId = 'gedi_count_mgrs_aggregated_leaf_off_0and255' if keepLeafOff255 else 'gedi_count_mgrs_aggregated_leaf_off_0'
        self.grid = ee.FeatureCollection(f"projects/gisproject-1/assets/{assetId}").filter('count_leaf_on > 0')
        
        self.year = year
        self.dataFolder = Path.home() / dataFolder
        self.dataFolder.mkdir(exist_ok=True)
        self.keepLeafOff255 = keepLeafOff255
        self.debug = debug

        if os.path.exists(self.dataFolder / f'gedi_table_asset_ids_{self.year}.txt'):
            with open(self.dataFolder / f'gedi_table_asset_ids_{self.year}.txt', 'r') as f:
                self.allAssetIds = f.read().split('\n') 
        else:
            self.allAssetIds = self.getTableAssetIds(export=True)

        if os.path.exists(self.dataFolder / f'mgrs_with_count_{self.year}.gpkg'):
            gpd.read_file(self.dataFolder / f'mgrs_with_count_{self.year}.gpkg', driver='GPKG')
            # with open(self.dataFolder / f'mgrs_zones_{self.year}.txt', 'r') as f:
            #     self.MGRScells = f.read().split('\n')
        else:
            self.MGRScells = self.getValidZoneCodes(export=True)
        
        if (self.dataFolder / f'assetIdsForEachZone_{self.year}.json').exists():    
            with open(self.dataFolder / f'assetIdsForEachZone_{self.year}.json', 'r') as f:
                text = f.read().replace('}{', ',')
                self.cellAssetIds = json.loads(text)
        else:
            self.cellAssetIds = None
        
        if os.path.exists(self.dataFolder / f'SampleTable{self.year}.csv'):
            self.sampleTable = pd.read_csv(self.dataFolder / f'SampleTable{self.year}.csv')
        else:
            self.sampleTable = self.getSampleTable(export=True)

    def getTableAssetIds(self, export=False):
        """
        Returns a list of unique table asset IDs for GEDI tracks from GEE rasterized version of GEDI within the specified year.

        Args:
            export (bool, optional): If True, exports the list of table asset IDs to a text file. Defaults to False.

        Returns:
            list: A list of unique table asset IDs for GEDI data within the specified year.
        """
        
        gedi = self.raster.filterDate(f'{self.year}-01-01', f'{self.year+1}-01-01')
        allAssetIds = (gedi.aggregate_array('table_asset_ids')
                                .flatten()
                                .distinct()
                                .getInfo())
        if export:
            with open(self.dataFolder / f'gedi_table_asset_ids_{self.year}.txt', 'w') as f:
                f.write('\n'.join(allAssetIds))
        return allAssetIds
    

    def getLeafOnCount(self, cellname):
        cell = self.mgrs.filter(f"MGRS_UTM == '{cellname}'").first()
        count = 0
        for assetId in self.allAssetIds:
            fc = ee.FeatureCollection(assetId).filterBounds(cell.geometry())
            fc = self.filterGEDI(fc)
            count += fc.size().getInfo()

        print(cellname, count)
        return {'MGRS_UTM': cellname, 'count_leaf_on': count}
        
    def getValidZoneCodes(self, export=False):
        """
        Returns a list of valid zone codes of MRGS grid cells where GEDI points are present.
        
        Args:
            export (bool, optional): If True, exports the list of zone codes to a text file. Defaults to False.
        
        Returns:
            list: A list of valid zone codes for GEDI vector data download.
        """
        cellnames = self.mgrs.aggregate_array('MGRS_UTM').getInfo()
        grid = geemap.ee_to_gdf(self.mgrs)
        
        pool = multiprocessing.Pool(processes=40)

        for cellname in cellnames:
            counts = pool.apply_async(self.getLeafOnCount, (cellname,))
        
        pool.close()
        pool.join()
        
        counts = counts.get()
        counts = pd.DataFrame.from_records(counts, columns=['MGRS_UTM', 'count_leaf_on'])
        grid = grid.merge(counts, on='MGRS_UTM', how='left')
        grid = grid[grid['count_leaf_on'] > 0]
        cellnames = grid['MGRS_UTM'].tolist()

        if export:
            with open(self.dataFolder / f'mgrs_with_count_{self.year}.gpkg', 'w') as f:
                grid.to_file(f, driver='GPKG')

        return cellnames
    
    def getSampleTable(self, export=False, update=False, plot=False):
        if update:
            # check for empty */sampled*.json files

            sampledTable = []
            for cellname in os.listdir(self.dataFolder):
                nCsv = len(list(self.dataFolder.glob(f'{cellname}/*.csv')))
                if os.path.isdir(self.dataFolder / cellname) and nCsv > 0:
                    sampled = pd.read_json(self.dataFolder / cellname / f'sampled{self.year}.json')
                    duplicated = sampled[sampled['filename'].duplicated()]
                    if len(duplicated) > 0:
                        logger.info(f'cell {cellname} has duplicated files: {duplicated}')
                    sampledTable.append({'MGRS_UTM': cellname, 'nSampled': sampled['nSampled'].sum(), 'nTotalInCell': sampled.get('nTotalInCell', []).sum()})
            sampledDf = pd.DataFrame.from_records(sampledTable, columns=['MGRS_UTM', 'nSampled', 'nTotalInCell'])

            with open(self.dataFolder / f'SampleTable{self.year}.csv', 'r') as f:
                df = pd.read_csv(f)
                df = df.merge(sampledDf, on='MGRS_UTM', how='left')

            with open(self.dataFolder / f'SampledTable{self.year}.csv', 'w') as f:
                df.to_csv(f, index=False)
        
        if plot:
            if (self.dataFolder / f'SampledTable{self.year}.csv').exists():
                df = pd.read_csv(self.dataFolder / f'SampledTable{self.year}.csv')
            else:
                df = self.getSampleTable(update=True)
            grid = geemap.ee_to_gdf(self.grid)
            grid = grid.merge(df, on='MGRS_UTM', how='left')
            print(grid.head())
            grid['sampledPerKm2'] = grid['nSampled'] / grid['landmass_x']
            plotargs = dict(tiles='CartoLight', frame_height=450, geo=True, cmap='plasma')
            plot = grid.hvplot(c='nSampled', hover_cols=['MGRS_UTM', 'nSampled'], title=f'GEDI counts after sampling, {self.year}', **plotargs) + \
                    grid.hvplot(c='sampledPerKm2', hover_cols=['MGRS_UTM', 'sampledPerKm2'], title=f'GEDI density (per KM^2) after sampling, {self.year}', **plotargs) + \
                    grid.hvplot(c='nWanted', hover_cols=['MGRS_UTM', 'nWanted'], title=f'Expected GEDI counts (landmass * n_km2), before sampling, {self.year}', **plotargs) + \
                    grid.hvplot(c='nTotal', hover_cols=['MGRS_UTM', 'nTotal'], title=f'GEDI counts before sampling, {self.year}', **plotargs)
            plot = plot.cols(1)
            hvplot.save(plot, self.dataFolder / 'sampledCountMap.html')
            logger.info(f'GEDI count map saved to {self.dataFolder / "sampledCountMap.html"}')

        else:

            property_names = self.grid.first().propertyNames().sort().getInfo()
            data = self.grid.map(lambda f: ee.Feature(None, f.toDictionary(property_names)))
            data = [x["properties"] for x in data.getInfo()["features"]]
            df = pd.DataFrame(data)
            df = df.drop(['count'], axis=1)
            df = df.rename(columns={'count_leaf_on': 'nTotal'})
            df['nWanted'] = df['landmass'] * self.nSampledPerKm2
            df['sampleRatio'] = df['nWanted'] / df['nTotal']
            df['nWanted'] = df.round({'nWanted': 0})['nWanted'].astype(int)
            
            if export:
                df.to_csv(self.dataFolder / f'SampleTable{self.year}.csv', index=False)
        return df


    def plotHistogram(self):
        ddf = dd.read_csv(self.dataFolder / f'*/*.csv', usecols=['rh98', 'pft_class', '.geo'], blocksize=64e6)
        h, bins = da.histogram(ddf['rh98'], bins=100).compute()
        plt.stairs(h, bins)
        plt.savefig('rh98.png')

        print('plot saved')


    def download(self, cellNames=None, processes=40):
        """
        Downloads GEDI data for all valid MGRS grid cells in the specified year.

        Args:
            processes (int, optional): The number of processes to use for parallel downloading. Defaults to 40.
        """
        if self.debug:
            import ipdb; ipdb.set_trace()
            self.sampleCell('25S')
        cellNames = cellNames or self.MGRScells
        pool = multiprocessing.Pool(processes=processes)
        for code in cellNames:
            pool.apply_async(self.sampleCell, (code, ))

        pool.close()
        pool.join()


    def sampleCell(self, cellname):
        """
        Downloads GEDI data for the specified MGRS grid cell, looping through all GEDI orbits in the cell.

        Args:
            cellname (str): The MGRS grid cell name.

        Returns:
            list: A list of files that failed to download.
        """
        cell = self.grid.filter(ee.Filter.eq('MGRS_UTM', cellname)).first()
        sampleRatio = self.getSampleRatio(cell)
        folder = self.dataFolder / cellname
        folder.mkdir(exist_ok=True)
        
        res = []
        assetIds = []
        # If there's a record of assetIds for the cell, use that, otherwise use allAssetIds
        cellAssetIds = self.cellAssetIds  and self.cellAssetIds.get(cellname)
        cellAssetIds = cellAssetIds or self.allAssetIds
        for asset_id in cellAssetIds:
            # if already downloaded, skip
            filename = folder /f"{asset_id.split('/')[-1]}.csv"
            if filename.exists():
                logger.info(f'{filename} already exists, skipping...')
                continue
            fc = ee.FeatureCollection(asset_id).filterBounds(cell.geometry())
            fc = self.filterGEDI(fc)
            l = fc.size().getInfo()
            if l > 0:
                res.append(self.sampleAsset(fc, filename, sampleRatio, l))
                assetIds.append(asset_id) # Failed files also added to the list

        # update the assetIds for the cell and save
        if cellAssetIds is None:
            self.cellAssetIds = self.cellAssetIds | {cellname: assetIds}
            with open(self.dataFolder / f'assetIdsForEachZone_{self.year}.json', 'w') as f:
                json.dump(self.cellAssetIds, f)
            

        # current cell is finished, save some summary file
        sampled, failed = [], []
        for f in res:
            f['filename'] = str(f['filename'])
            if f.get('errorMsg') is not None:
                failed.append(f)
            else:
                sampled.append(f)

        if len(sampled) > 0:
            with open(folder / f'sampled{self.year}.json', 'w') as f:
                json.dump(sampled, f)

        if len(failed) > 0:    
            with open(folder / f'failed{self.year}.json', 'w') as f:
                json.dump(failed, f)

        logger.info(f'>>>>>>>>> Finish grid cell: {cellname} <<<<<<<<<<')

    @retry(tries=10, delay=1)
    def sampleAsset(self, fc, filename, sampleRatio, l):
        """
        Downloads GEDI points for a given asset(orbit) inside a cell to a CSV file.
        The asset id and cell code is embedded in the filename. fc is the filtered feature collection.
        The sampleRatio is determined by the landmass of the cell and the number of points in the cell.

        Args:
            fc (ee.FeatureCollection): GEDI points of an orbit in the cell.
            filename (str): The name of the file to save the CSV data to.
            sampleRatio (float): The ratio of wanted #points to total #point in a cell.

        Returns:
            dict/None: A dictionary containing the filename and error message (if any).
        """
        if sampleRatio < 1:
            fc = fc.randomColumn().filter(ee.Filter.lte('random', sampleRatio))
        sampled = fc.size().getInfo()
        if sampled > 0:
            downloadId = ee.data.getTableDownloadId({'table': fc,'format': 'csv', 'filename': 'test.csv'})
            res = requests.get(ee.data.makeTableDownloadUrl(downloadId))
            if res.status_code == 200:
                with open(filename, 'wb') as fd:
                    fd.write(res.content)
                logger.info(f'file saved to {filename}')
            else:
                logger.error(f'error downloading {filename}, error: {res.content}')
                return {
                    'filename': filename.relative_to(self.dataFolder),
                    'errorMsg': res.content,
                }
        return {
            'filename': filename.relative_to(self.dataFolder), 
            'nSampled': sampled, # include 0 sampled entries as well, but not saving the file
            'nTotalInCell': l,
        }

    def filterGEDI(self, fc):
        """
        Returns a filtered GEDI feature collection for given table asset ID.
        Filter by quality flag, degrade flag, region class, and growing season.

        Args:
            idx (str): A table asset ID.

        Returns:
            ee.FeatureCollection: A filtered GEDI feature collection.
        """
        growingSeasonFilter = 'leaf_off_flag == 0 || leaf_off_flag == 255' if self.keepLeafOff255 else 'leaf_off_flag == 0'
        return (fc.filter('quality_flag == 1 && degrade_flag == 0 && region_class > 0') # quality flag and water mask
                .filter(growingSeasonFilter))    

    def getSampleRatio(self, cell):
        
        """
        Returns the number of GEDI points to sample for the specified MGRS grid cell.

        Args:
            cell (ee.Feature): The MGRS grid cell feature.

        Returns:
            int: The number of GEDI points to sample.
        """
        nSampledPerGrid = ee.Number(cell.get('landmass')).multiply(self.nSampledPerKm2).round().getInfo()
        totalPerCell = cell.get('count_leaf_on').getInfo()
        sampleRatio = nSampledPerGrid / totalPerCell + 0.001
        return sampleRatio

    def visualize(self):
        """
        Visualizes the GEDI data.
        """

        pass

    
    
# Maybe not needed functions, for bug fixing purposes
    def getSampleTableForCell(self, processes=100):
        """
        If the cell/sample*.json is missing or empty for a cell, this function will return a sample table for that cell.
        """
        pool = multiprocessing.Pool(processes=processes)
        for cellname in os.listdir(self.dataFolder):
            pool.apply_async(self.readCell, (cellname, ))
        pool.close()
        pool.join()

    def readCell(self, cellname):
        sampleFile = self.dataFolder / cellname / f'sampled{self.year}.json'
        if sampleFile.exists():
            with open(sampleFile, 'r') as f:
                sampled = f.read()
                
            if len(sampled) == 0:
                count = []
                for csvfile in self.dataFolder.glob(f'{cellname}/*.csv'):
                    df = pd.read_csv(csvfile)
                    nSampled = len(df)
                    count.append({
                        'filename': str(csvfile.relative_to(self.dataFolder)),
                        'nSampled': nSampled,
                        'nTotalInCell': 0
                    })
                with open(sampleFile, 'w') as f:
                    json.dump(count, f)
                print(f'file saved to {sampleFile}')
                logger.info(f'cell {cellname} has empty sample file, now updated')

    def removeEmptyCSV(self, addParentFolder=False):
        """
        For fixing the bug where 0 points are sampled but a csv file is still created.
        Downloaded csv files shouldn't be empty anymore.
        """
        if addParentFolder:
            for folder in os.listdir(self.dataFolder):
                if os.path.isdir(self.dataFolder / folder):
                    sampled = pd.read_json(self.dataFolder / folder / f'sampled{self.year}.json')
                    sampled['filename'] = sampled['filename'].apply(lambda x: f'{folder}/{x}')
                    sampled.to_json(self.dataFolder / folder / f'sampled{self.year}.json', orient='columns') # orient='records'

        sampleAssetTable = dd.read_json(self.dataFolder / f'*/sampled{self.year}.json', orient='columns') # orient='values'
        emptyFiles = sampleAssetTable[sampleAssetTable['sampled'] == 0]
        emptyFiles = emptyFiles.compute()
        emptyFiles['filename'] = emptyFiles['filename'].apply(lambda x: self.dataFolder / f'{x}.csv')
        # emptyFiles['filename'].apply(lambda x: os.remove(x))
        if len(emptyFiles) == 0:
            print(f'No empty files in {self.dataFolder}')
            return
        for i, file in enumerate(emptyFiles['filename']):
            try:
                pd.read_csv(file)
            except:
                print(f'{i}, Cannot read {file}, is empty or corrupted. Now removing...')
                os.remove(file)

    def downloadListOfFailedFiles(self, listOfFiles=None, processes=40):
        """
        Downloads GEDI data for a list of files failed before, from which we know which cell and which asset to sample from.
        This function uses multiprocessing to download GEDI data for a list of files. 
        
        Args:
        - listOfFiles (list): A list of file names to download.
        
        Returns:
        - res (list): A list of files that still failed to download.
        """
        if listOfFiles is None:
            listOfFiles = []
            for file in self.dataFolder.glob('**/failed*.json'):
                with open(file, 'r') as f:
                    listOfFiles.extend(json.load(f))
        res = []
        pool = multiprocessing.Pool(processes=processes)
        for f in listOfFiles:
            filepath = f['filename'] if isinstance(f, dict) else f
            cell = self.grid.filter(ee.Filter.eq('MGRS_UTM', filepath.parent.name)).first()
            fc = (ee.FeatureCollection('LARSE/GEDI/GEDI02_A_002/' + filepath.stem)
                .filterBounds(cell.geometry()))
            l = fc.size().getInfo()
            if l == 0:
                continue
            sampleRatio = self.getSampleRatio(cell)
            res.append(pool.apply_async(self.sampleAsset, (fc, filepath, sampleRatio, l)))
        pool.close()
        pool.join()

        # update the json file for failed cells
        for f in res:
            msg = f.get()
            if msg.get('errorMsg') is not None:
                with open(self.dataFolder / f'failed{self.year}', 'a') as f:
                    msg['filename'] = str(msg['filename'])
                    json.dump(msg, f)
            
            if msg.get('nSampled') is not None:
                with open(self.dataFolder / msg['filename'].parent.name / f'sampled{self.year}.json', 'a') as f:
                    msg['filename'] = str(msg['filename'])
                    json.dump(msg, f)

                # not consider the case when no points are sampled for a cell, another function will check that
        # TODO: remmove failed.json from each cell folder
        # no need to update assetIdsForEachZone.json for failed files, already added

    def checkUnfinishedCellsAndDownload(self, processes=40, maxTries=3):
        """
        The main downloading process somehow stopped, and some cells are not finished downloading, 
        which is indicated by the presence of the json file. We save a json file listing the number of points for each csv file once a cell is finished downloading.
        This function checks the json file and re-download the data for those cells.
        """
        # check the json file and get the list of zones failed before
        failedfiles = self.dataFolder.glob('**/failed*.json')

        with open(self.dataFolder / f'assetIdsForEachZone_{self.year}.json', 'r') as f:
            downloaded = json.load(f)

        failedCells = [cellName for cellName in self.MGRScells if cellName not in downloaded.keys()]
        self.download(cellNames=failedCells, processes=processes, maxTries=maxTries)
    
    def checkEmptyFilesAndDownload(self, maxTries=3):
        """
        Checks for empty files in the data folder and downloads data for those files.
        
        If there are no empty files, the function returns without doing anything.
        If there are empty files, a JSON file is created with the names of the empty files.
        Data is then downloaded for each empty file using multiprocessing.
        
        Args:
        - self: instance of the GediVectorDownload class
        
        Returns:
        - None
        """
        
        emptyFiles = []
        for f in self.dataFolder.glob('**/*.csv'): # when there's no error message and the file is empty
            try:
                pd.read_csv(f)
            except:
                emptyFiles.append(f)
        if len(emptyFiles) == 0:
            logger.info('no empty files')
            return
        
        with open(self.dataFolder / f'emptyFiles{self.year}.json', 'a') as f:
            json.dump([f.name for f in emptyFiles], f)

        # with open('emptyFiles.txt', 'r') as f:
        #     emptyFiles = [Path(f) for f in f.read().split('\n') if f.endswith('.csv')]

        # try maxTries times for failed files
        for i in range(maxTries):
            logger.info(f'>>>>>>>>> retrying failed files, try {i+1} <<<<<<<<<<')
            emptyFiles = self.downloadListOfFiles(emptyFiles)
            if len(emptyFiles) == 0:
                break

        if len(emptyFiles) > 0:
            with open(self.dataFolder / f'maybeTooBigFiles{self.year}.json', 'w') as f:
                json.dump([f['filename'].name for f in emptyFiles], f)
            logger.info(f'>>>>>>>>> {len(emptyFiles)} files still failed to download <<<<<<<<<<')
    
    def readData(self, zone_code):
        """
        Reads the downloaded GEDI data for a given zone code.

        Args:
            zone_code (str): A zone code.

        Returns:
            gdf: A GeoDataFrame containing the GEDI data for the given zone code.
        """
        folder = self.dataFolder / zone_code
        with open(folder / f'sampled{self.year}.json', 'r') as f:
            sampled = json.load(f)
        okFiles = [f['filename'] for f in sampled if f['sampled'] > 0]
        df = pd.concat([pd.read_csv(folder / f'{f}.csv') for f in okFiles])
        import ipdb; ipdb.set_trace()
        gdf = gpd.GeoDataFrame(df.drop(['.geo'], axis=1), crs={'init': 'epsg:4326'},geometry=[shape(json.loads(geo)) for geo in df['.geo']])
        return gdf
    
    def readAllData(self):
        """
        Reads all downloaded GEDI data.

        Returns:
            gdf: A GeoDataFrame containing all the downloaded GEDI data.
        """
        dfList = (pd.read_csv(f) for f in self.dataFolder.glob('**/*.csv') if is_non_zero_file(f))
        df = pd.concat([pd.read_csv(f) for f in dfList])
        gdf = gpd.GeoDataFrame(df.drop(['.geo'], axis=1), crs={'init': 'epsg:4326'},geometry=[shape(json.loads(geo)) for geo in df['.geo']])
        return gdf
    
    def checkDownloadedFile(self, filepath):
        """
        check if the downloaded file should be empty
        """
        filepath = Path(filepath)
        zone_code = filepath.parent.name
        cell = self.grid.filter(ee.Filter.eq('MGRS_UTM', zone_code)).first()
        fc = (ee.FeatureCollection('LARSE/GEDI/GEDI02_A_002/' + filepath.stem)
                .filterBounds(cell.geometry()))
        fc = self.filterGEDI(fc)
        l = fc.size().getInfo()
        sampleRatio = self.getSampleRatio(cell)
        import ipdb; ipdb.set_trace()
        return l * sampleRatio < 1
    

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    key = json.load(open(cfg.keyFile))
    credentials = ee.ServiceAccountCredentials(key['client_email'], cfg.keyFile)
    ee.Initialize(credentials)
    gedi = GEDI(**cfg.init)

    if cfg.task == 'download':
        gedi.download(**cfg.download)
    elif cfg.task == 'visualize':
        gedi.getSampleTable(plot=True)
    else:
        logger.error(f'Task not recognized.\nreceived: {cfg.task} \nexpected: download or visualize')


if __name__ == '__main__':
    main()


