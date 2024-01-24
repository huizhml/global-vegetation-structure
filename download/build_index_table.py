import os
import time
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import h5py
import h5netcdf
import tables


def get_group_paths(node, zone=None):
    if isinstance(node, tables.link.ExternalLink):
        zone = node._v_name
        node = node()
        
    # stop when reaching the group contains different variable groups (data variables in xarray)
    # there're some variable groups (dims actually in xarray) cannot be interpreted correctly here
    sizes =[]
    if isinstance(node, tables.Group) and len(node._v_pathname.split('/')) <= 2: 
        for child in node:
            paths = get_group_paths(child, zone)
            sizes.extend(paths)
    else:
        size = node['input'].nrows
        return [[f'{zone}{node._v_pathname}', i] for i in range(size)]
    return sizes

data_dir = Path.home() / 'data/GEDI'
h5_file = data_dir / 'merged_file.h5'
index = []
with tables.open_file(h5_file, 'r') as file:
    table = []
    for f_link in file.root:
        node = f_link()
        zone = f_link._v_name
        paths = get_group_paths(node, zone)
        table.extend(paths)
df = pd.DataFrame(np.array(table), columns=['path', 'in_partition_id'], dtype={'path': 'str', 'in_partition_id': 'uint16'})

print()

