from typing import Dict
from pathlib import Path
from collections import defaultdict
from dask.distributed import get_client, as_completed

class DaskDownloader:
    """
    A class for downloading data using Dask.
    """
    def __init__(self, n_parallel:int = 60, max_retries:int=3, **kwargs):
        self.n_parallel = n_parallel
        self.max_retries = max_retries

    def schedule_tasks(self, ddf):
        """
        Schedule Dask tasks manually to control the level of parallelism and enable retry for failed tasks.
        """
        client = get_client()
        futures = []
        n_parallel = min(self.n_parallel, ddf.npartitions)
        for i in range(n_parallel):
            futures.append(client.compute(ddf.get_partition(i)))

        futures_monitor = as_completed(futures, with_results=False)
        n_left = ddf.npartitions - n_parallel

        retry_counter: Dict[str, int] = defaultdict(int)
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
                        f.retry() #TODO: key eror in self.futures[key] when first retry, why? related to distributed.scheduler - ERROR - Couldn't gather keys: {('sum-aggregate-ce2045d27a178c14f0a6884069ecef48', 0): 'processing'}?
                        futures_monitor.add(f)
                        retry_counter[f.key] += 1   
                    continue
                else:
                    print(f'Failed to download {f.key} after {self.max_retries} retries.')
                    # TODO: save the failed tasks to a file and retry later
            f.release()
            if n_left > 0:
                future = client.compute(ddf.get_partition(ddf.npartitions - n_left))
                print(ddf.npartitions)
                futures_monitor.add(future)
                print(f'************ partition {ddf.npartitions - n_left} submitted ****************')
                print(f'{futures_monitor.count()} in processing, {n_left} waiting')
                n_left -= 1
            
