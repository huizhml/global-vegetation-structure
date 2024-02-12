from pathlib import Path
import h5py
import os

def merge_h5_files(file_list, output_file):
    with h5py.File(output_file, 'w') as h5_out:
        for file in file_list:
            group_name = os.path.basename(file).split('.')[0]  # Create a group name based on file name
            h5_out[group_name] = h5py.ExternalLink(file, '/')

# file_list = (Path.home() / 'data/GEDI').glob('*.h5')
file_list = ['32M', '32N', '33M', '33P', '32P', '33N']
file_list = [Path.home() / 'data/GEDI' /f'{f}.h5' for f in file_list]
output_file = Path.home() / 'data/GEDI/merged_file.h5'  # Name of the output merged file

merge_h5_files(file_list, output_file)