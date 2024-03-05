import os
import time
from typing import Union, List
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
import tables
import h5py
from datatree.io import _iter_nc_groups
from h5netcdf.legacyapi import Dataset as h5Dataset


class S2Dataset(Dataset):

    def __init__(self, h5_file: Union[str, Path], index_table:Path, transform=None):
            """
            Initialize the S2 dataset.

            Args:
                h5_file (Union[str, Path]): The path to the HDF5 file of a single zone or a merged HDF5 file. Should be placed under the same folder as the zone h5 files.
                use_zones (Union[str, List[str]], optional): The zones to form the HDF5 file. Defaults to TEST_ZONES.
                cache_dir (str, optional): The directory to store the cached index table. Defaults to None.
                transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.
            """
            self.transform = transform or ToTensor()
            self.h5_file = tables.open_file(h5_file)
            self.index_table = pd.read_csv(index_table)


    def __len__(self):
        return len(self.index_table)

    def __getitem__(self, idx):
        row = self.index_table.iloc[idx]
        image = self.h5_file.root[f'{row.zone}/{row.year}/{row.partition_idx}/image'][row.in_partition_idx]
        label = self.h5_file.root[f'{row.zone}/{row.year}/{row.partition_idx}/rhs'][row.in_partition_idx]

        image = image.astype('float')
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    h5_files = 'data/GEDI/merged_file_test.h5'
    dataset = S2Dataset(h5_files, cache_dir='data/GEDI')
    dataset.h5_file.close()
    # for i in range(len(dataset)):
    #     img, label = dataset[i]
    


# %%
