import os
import math
from pathlib import Path

data_folder = Path.home() / 'GEDI2019'
config = []
with open(data_folder / 'download_config.txt', 'a') as f:
    for zone in os.listdir(data_folder):
        n_partitions = len(list((data_folder / zone).glob('partition_*.parquet')))
        if n_partitions ==0:
            continue
        print(n_partitions)
        n_cores = math.floor(math.log(n_partitions, 2)) 
        n_cores= min(2**n_cores, 128) # maximum #cores we can get
        n_parallel = n_cores + 20
        f.write(f'{zone} {n_partitions} {n_cores} {n_parallel} \n')
