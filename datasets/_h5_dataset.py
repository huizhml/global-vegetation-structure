
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import h5py


coverage_beams = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011']
power_beams = ['BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']

class S2Dataset(Dataset):


    def __init__(self, h5_file: Union[str, Path], index_table, transform=None):
        """
        Initialize the S2 dataset.

        Args:
            h5_file (Union[str, Path]): The path to the HDF5 file of a single zone or a merged HDF5 file. Should be placed under the same folder as the zone h5 files.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.
        """
        self.transform = transform
        # self.h5_file = h5py.File(h5_file, mode='r')
        self.h5_file_path = h5_file
        self.index_table = index_table

    def __len__(self):
        return len(self.index_table)

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_file_path, mode='r')
        
        row = self.index_table.iloc[idx]
        image = self.h5_file[f'{row.path}/image'][row.in_partition_idx]
        # image = image.astype(np.int16)
        wc = image[13]
        image = image[:12]
        label = self.h5_file[f'{row.path}/rhs'][row.in_partition_idx]
        slope = self.h5_file[f'{row.path}/slope'][row.in_partition_idx]
        slope = slope.astype(np.float32)
        latlon = self.h5_file[f'{row.path}/latlon'][row.in_partition_idx]
        sensitivity = self.h5_file[f'{row.path}/gedi_attrs'][row.in_partition_idx, 24]
        shot_number = self.h5_file[f'{row.path}/shot_number'][row.in_partition_idx]
        if int(shot_number) != int(row.shot_number):
            raise ValueError(f'Error: something wrong with the index table, {shot_number} != {row.shot_number}')
        # image = image.astype('float')
        if self.transform:
            image = self.transform(image)
        # self.h5_file.close()
        return image, label, wc, slope, latlon, sensitivity, shot_number
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()



if __name__ == '__main__':
    import pandas as pd
    h5_files = Path.home() / 'data/GEDI/train.h5'
    index_table = Path.home() / 'data/GEDI/index_table_train_subsets/train0.parquet'
    index_table = pd.read_parquet(index_table)
    dataset = S2Dataset(h5_files, index_table)
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        print(img.shape)
        break
    dataset.h5_file.close()


# %%
