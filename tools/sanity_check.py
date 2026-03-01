from typing import List
from pathlib import Path
import logging
import numpy as np
import pyarrow.parquet as pq
import dask
import dask.dataframe as dd
import dask_geopandas as dgp


def setup_logger(log_file: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    logger.propagate = False
    c_handler = logging.StreamHandler()      # Console handler
    f_handler = logging.FileHandler(log_file, mode='w')  # File handler (mode='w' overwrites, 'a' appends)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

def try_read_gdf(parquet_dir: List[Path], logger: logging.Logger = None):
    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        return dgp.read_parquet(parquet_dir, gather_spatial_partitions=False)
    except ValueError as e:
        logger.warning(f'Failed to read {parquet_dir[0]} as geopandas dataframe, using dask dataframe instead: {e}')
        return dd.read_parquet(parquet_dir)
    
def get_nrows(parquet_dir: Path):
    return pq.ParquetFile(parquet_dir).metadata.num_rows
    

def check_npoints_for_two_datasets(source_dir: str = None, target_dir: str = None, **kwargs):
    '''
    Check the number of files and points for two parquetdatasets
    The two datasets should have the same file naming
    Args:
        source_dir: Path to the soruce dataset
        target_dir: Path to the target dataset
        report_file: Path to the report file
    Returns:
        None
    '''

    source_dir = Path(source_dir).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()
    report_file = target_dir.parent / f'sanity_check_for_{target_dir.stem}.log'
    logger = setup_logger(report_file)
    logger.info(f'Source dataset: {source_dir}')
    logger.info(f'Target dataset: {target_dir}')
    logger.info(f'Step 1: Check the number of files for two datasets')
    source_files = list(source_dir.glob('*.parquet'))
    target_files = list(target_dir.glob('*.parquet'))
    logger.info(f'Number of files in {source_dir}: {len(source_files)}')
    logger.info(f'Number of files in {target_dir}: {len(target_files)}')
    if len(source_files) != len(target_files):
        source_file_names = [file.stem for file in source_files]
        target_file_names = [file.stem for file in target_files]
        missing_files = set(source_file_names) - set(target_file_names)
        extra_files = set(target_file_names) - set(source_file_names)
        logger.warning(f'The number of files in source and target datasets is not the same')
        logger.warning(f'Missing files: {missing_files}')
        logger.warning(f'Extra files: {extra_files}')

    logger.info(f'Step 2: Check the number of points for two datasets')
    source_files = [source_dir / f'{file.stem}.parquet' for file in target_files]
    source_nrows = [dask.delayed(get_nrows)(file) for file in source_files]
    target_nrows = [dask.delayed(get_nrows)(file) for file in target_files]
    source_nrows = dask.compute(*source_nrows)
    target_nrows = dask.compute(*target_nrows)
    source_n = sum(source_nrows)
    target_n = sum(target_nrows)

    if source_n != target_n:
        logger.warning(f'The total number of points in source and target datasets is not the same')
        logger.warning(f'Source number of points: {source_n}')
        logger.warning(f'Target number of points: {target_n}')
    else:
        if np.all(source_nrows == target_nrows):
            logger.info(f'The total number of points is identical in the common files of the source and target datasets: \n{source_n}')
        else:
            logger.warning(f'The total number of points is not identical in the common files of the source and target datasets: \n{source_nrows} != {target_nrows}\n')


def check_total_points_for_two_partitioned_datasets(source_dir: str = None, target_dir: str = None, **kwargs):
    '''
    Check the total number of points for two partitioned datasets
    The two datasets doesn't have the same file naming  
    Args:
        source_dir: Path to the soruce dataset
        target_dir: Path to the target dataset
    Returns:
        None
    '''
    source_dir = Path(source_dir).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()
    report_file = target_dir.parent / f'sanity_check_for_{target_dir.stem}.log'
    logger = setup_logger(report_file)
    logger.info(f'Source dataset: {source_dir}')
    logger.info(f'Target dataset: {target_dir}')
    logger.info(f'Check the total number of points for two datasets')
    source_nrows = [dask.delayed(get_nrows)(file) for file in source_dir.glob('*.parquet')]
    target_nrows = [dask.delayed(get_nrows)(file) for file in target_dir.glob('*.parquet')]
    source_nrows = dask.compute(*source_nrows)
    target_nrows = dask.compute(*target_nrows)
    source_n = sum(source_nrows)
    target_n = sum(target_nrows)
    if source_n != target_n:
        logger.warning(f'The total number of points in source and target datasets is not the same')
        logger.warning(f'Source number of points: {source_n}')
        logger.warning(f'Target number of points: {target_n}')
    else:
        logger.info(f'The total number of points is identical in the common files of the source and target datasets: \n{source_n} == {target_n}\n')


def check_total_points_for_two_partitioned_data_hiarchy(source_dir: str = None, target_dir: str = None, **kwargs):
    '''
    Check the total number of points for two partitioned datasets
    The two datasets doesn't have the same file naming  
    Args:
        source_dir: Path to the soruce dataset
        target_dir: Path to the target dataset
    Returns:
        None
    '''
    source_dir = Path(source_dir).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()
    report_file = target_dir / f'sanity_check_for_{target_dir.stem}.log'
    logger = setup_logger(report_file)
    logger.info(f'Source dataset: {source_dir}')
    logger.info(f'Target dataset: {target_dir}')
    logger.info(f'Check the total number of points for two datasets')
    source_df = try_read_gdf(source_dir, logger)
    source_n = source_df.size.compute()
    target_subdirs = [d for d in target_dir.rglob('*') if d.is_dir()]
    target_n = 0
    for tdir in target_subdirs:
        target_df = try_read_gdf(tdir.glob('*.parquet'), logger)
        target_n = target_df.size.compute()
    if target_n != source_n:
        logger.warning(f'The total number of points in source and target datasets is not the same')
        logger.warning(f'Source number of points: {source_n}')
        logger.warning(f'Target number of points: {target_n}')
    else:
        logger.info(f'The total number of points in source and target datasets is the same')
