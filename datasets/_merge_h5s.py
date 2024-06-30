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

dst_conf = {'image': {
    'shape': (14,15,15),
    'dtype': 'uint16',
    }, 
    'rhs':{
        'shape': (101,),
        'dtype': 'float32',
        },
    'slope':{
        'shape': (15,15),
        'dtype': 'float64',
        },
    'gedi_attrs':{
        'shape': (30,),
        'dtype': 'float64'
        }, 
    'latlon':{
        'shape': (2,),
        'dtype': 'float64'
    },
    'delta_day':{
        'shape': (),
        'dtype': 'uint16'
    },
    'defective_cover':{
        'shape': (),
        'dtype': 'float32'
    },
    'id':{
        'shape': (),
        'dtype': h5py.string_dtype(encoding='utf-8') # 'str'
    },
    'shot_number':{
        'shape': (),
        'dtype': 'uint64'
    }
    }

def append_to_dataset(dataset, data):
    dataset.resize(len(dataset) + len(data), axis=0)
    dataset[-len(data):] = data

def merge_partitions(zone='02K', h5_dir='~/data/GEDI'):
    """
    Merge all zones in h5_dir into a single h5 file.
    """
    h5_dir = Path(h5_dir).expanduser()
    out_h5 = Path(f'~/flash/data/{zone}.h5').expanduser()

    with h5py.File(out_h5, 'w') as h5_out:
        datasets = []
        for name, config in dst_conf.items():
            # import ipdb; ipdb.set_trace()
            dst = h5_out.create_dataset(name, 
                                    shape=(0,) + config['shape'], 
                                    maxshape=(None,) + config['shape'], 
                                    chunks=(512,) +config['shape'], 
                                    dtype=config['dtype'],
                                    compression="gzip", #lzf
                                    compression_opts=7)
            datasets.append(dst)
        with h5py.File(h5_dir/f'{zone}.h5', 'r') as h5_in:
            for year in range(2019, 2023):                
                year = str(year)
                for part in h5_in[year].keys():
                    for dst in datasets:
                        # import ipdb; ipdb.set_trace()
                        src = h5_in[year][part][dst.name.split('/')[-1]]
                        append_to_dataset(dst, src)


if __name__ == '__main__':
    # merge_all_zones(merged_h5_file='~/flash/data/GVS.h5')
    merge_partitions(zone='37N')