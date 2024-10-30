
import pandas as pd

import h5py

# Replace 'yourfile.h5' with the path to your HDF5 file
file_path = '/home/ksb781/data/GEDI/GEDI_S2/11T.h5'

def inspect_hdf5_group(group, path=''):
    """
    Recursively inspects HDF5 groups and prints chunk size of datasets.
    """
    for key, item in group.items():
        item_path = f"{path}/{key}"  # Build the path string
        if isinstance(item, h5py.Dataset):
            # This item is a dataset, check if it is chunked
            if item.chunks:
                print(f"Dataset {item_path} has chunk size: {item.chunks}")
            else:
                print(f"Dataset {item_path} is not chunked")
        elif isinstance(item, h5py.Group):
            # This item is a group, recurse into it
            inspect_hdf5_group(item, item_path)

# Replace 'yourfile.h5' with the path to your HDF5 file

# Open the HDF5 file
with h5py.File(file_path, 'r') as file:
    # Start inspecting from the root group
    inspect_hdf5_group(file)