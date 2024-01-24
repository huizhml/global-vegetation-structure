from pathlib import Path
import h5py
import os

def merge_h5_files(file_list, output_file):
    with h5py.File(output_file, 'w') as h5_out:
        for file in file_list:
            group_name = os.path.basename(file).split('.')[0]  # Create a group name based on file name
            h5_out[group_name] = h5py.ExternalLink(file, '/')
            # with h5py.File(file, 'r') as h5_in:
            #     group = h5_out.create_group(group_name)
            #     # Copy all datasets from the current file to the group
            #     for dset in h5_in:
            #         for p in h5_in[dset]:
            #             group[dset] = h5_in[dset][...]

# file_list = (Path.home() / 'data/GEDI/01G').glob('partition_*.h5')
file_list = ['32M', '32N', '33M', '33P']
file_list = [Path.home() / 'data/GEDI' /f'{f}.h5' for f in file_list]
output_file = Path.home() / 'data/GEDI/merged_file_test.h5'  # Name of the output merged file

merge_h5_files(file_list, output_file)