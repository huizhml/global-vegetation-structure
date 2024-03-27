import logging
from typing import Dict, Callable, Tuple
from pathlib import Path
from collections import defaultdict
from dask.distributed import get_client, as_completed
import dask.dataframe as dd
import pandas as pd

logger = logging.getLogger(__name__)
class DaskDownloader:
    """
    A class for downloading data using Dask.
    """

    def __init__(self, n_parallel: int = 60, max_retries: int = 3, **kwargs):
        self.n_parallel = n_parallel
        self.max_retries = max_retries

    def schedule_tasks(self, ddf: dd.DataFrame = None, delayed_tasks: Callable = None, args:Tuple=()):
        """
        Schedule Dask tasks manually to control the level of parallelism and enable retry for failed tasks.
        """
        assert isinstance(ddf, pd.DataFrame) and callable(delayed_tasks) or isinstance(ddf, (dd.DataFrame, dd.Series)), \
                'when delayed_tasks is not none, ddf must be pd.DataFrame'
        client = get_client()
        futures = []
        if isinstance(ddf, (dd.DataFrame, dd.Series)):
            n_parallel = min(self.n_parallel, ddf.npartitions)
            total_tasks = ddf.npartitions
        elif isinstance(ddf, pd.DataFrame):
            n_parallel = min(self.n_parallel, len(ddf))
            total_tasks = len(ddf)

        for i in range(n_parallel):
            if delayed_tasks:
                row = ddf.iloc[i]
                future = client.compute(delayed_tasks(row, *args))
            else:
                future = client.compute(ddf.get_partition(i))
            futures.append(future)

        futures_monitor = as_completed(futures, with_results=False)
        n_left = total_tasks - n_parallel
        retry_counter: Dict[str, int] = defaultdict(int)
        nfailed = 0
        while futures_monitor.count() > 0:
            f = next(futures_monitor)
            if f.status == 'error':
                if retry_counter[f.key] < self.max_retries:
                    try:
                        f.retry()
                        futures_monitor.add(f)
                        retry_counter[f.key] += 1
                    except Exception as e:
                        print(e)
                        # TODO: key eror in self.futures[key] when first retry, why? related to distributed.scheduler - ERROR - Couldn't gather keys: {('sum-aggregate-ce2045d27a178c14f0a6884069ecef48', 0): 'processing'}?
                        f.retry()
                        futures_monitor.add(f)
                        retry_counter[f.key] += 1
                    continue
                else:
                    logger.info(f'Failed to download {f.key} after {self.max_retries} retries.')
                    nfailed += 1

            f.release()
            if n_left > 0:
                if delayed_tasks:
                    row = ddf.iloc[total_tasks - n_left]
                    future = client.compute(delayed_tasks(row, *args))
                else:
                    future = client.compute(
                        ddf.get_partition(total_tasks - n_left))
                futures_monitor.add(future)
                logger.info(f'************ partition {total_tasks - n_left} submitted ****************')
                logger.info(f'{futures_monitor.count()} in processing, {n_left} waiting')
                n_left -= 1
            else:
                logger.info(f'************ all partitions submitted ****************')
                logger.info(f'{futures_monitor.count()} in processing, 0 waiting')
        return nfailed
