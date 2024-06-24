from pathlib import Path
import seaborn as sns
import h5py

def merge_all_zones(h5_dir:str='~/data/GEDI', merged_h5_file:str='~/data/GVS.h5'):
    """
    Merge all zones in h5_dir into a single h5 file.
    """
    h5_dir = Path(h5_dir).expanduser()
    merged_h5_file = Path(merged_h5_file).expanduser()

    if merged_h5_file.exists():
        mode = 'r+'
    else:
        mode = 'w'
    zones = list(h5_dir.glob('*.h5'))
    with h5py.File(merged_h5_file, mode) as h5_out:
        for zone in zones:
            zone = zone.stem
            print(zone)
            if zone in h5_out:
                print(f"Group '{zone}' already exists in the destination file.")
            else:
                # Copy the entire source file under a new group in the destination file
                with h5py.File(h5_dir/f'{zone}.h5', 'r') as h5_in:
                    h5_in.copy('/', h5_out, name=zone)

if __name__ == '__main__':
    merge_all_zones(merged_h5_file='~/flash/data/GVS.h5')