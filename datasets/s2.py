
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import h5py

import warnings
from tables import DataTypeWarning
warnings.filterwarnings('ignore', category=DataTypeWarning)

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
        self.transform = transform
        # self.h5_file = h5py.File(h5_file, mode='r')
        self.h5_file_path = h5_file
        self.index_table = pd.read_csv(index_table)


    def __len__(self):
        return len(self.index_table)

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_file_path, mode='r')

        row = self.index_table.loc[idx]
        image = self.h5_file[f'{row.path}/image'][row.in_partition_idx]
        image = image.astype(np.int16)
        wc = image[13]
        image = image[:12]
        label = self.h5_file[f'{row.path}/rhs'][row.in_partition_idx][:]
        slope = self.h5_file[f'{row.path}/slope'][row.in_partition_idx]
        
        image = image.astype('float')
        if self.transform:
            image = self.transform(image)
        # self.h5_file.close()
        return image, label, wc, slope
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()



if __name__ == '__main__':
    h5_files = Path.home() / 'data/GEDI/merged_file_test.h5'
    index_table = 'cache/train_index_table.csv'
    dataset = S2Dataset(h5_files, index_table)
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        print(img.shape)
        break
    dataset.h5_file.close()


# %%
