from pathlib import Path
import logging
import pyarrow.parquet as pq
import dask



def setup_logger(log_file: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()      # Console handler
    f_handler = logging.FileHandler(log_file, mode='w') # File handler (mode='w' overwrites, 'a' appends)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

def check_npoints_for_two_datasets(source_dir: str=None, target_dir: str=None):
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

    source_dir = Path(source_dir).expanduser()
    target_dir = Path(target_dir).expanduser()
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

    @dask.delayed
    def check_npoints_for_one_file(file_name: str):
        source_file = source_dir / f'{file_name}.parquet'
        target_file = target_dir / f'{file_name}.parquet'
        source_npoints = pq.ParquetFile(source_file).metadata.num_rows
        target_npoints = pq.ParquetFile(target_file).metadata.num_rows
        if source_npoints != target_npoints:
            return target_file.stem
        else:
            return None
    tasks = [check_npoints_for_one_file(file.stem) for file in target_files]
    res = dask.compute(*tasks)
    res = [r for r in res if r is not None]
    if len(res) > 0:
        logger.warning(f'The number of points in source and target datasets is not the same for the following tiles: {res}')
    else:
        logger.info(f'The number of points in source and target datasets is the same for all files')