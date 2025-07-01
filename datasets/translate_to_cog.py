import os
import time
import subprocess
import rasterio
from rasterio.io import MemoryFile
import dask
from dask.distributed import Client
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from pathlib import Path
import matplotlib.pyplot as plt
import re


def translate_by_cogger(input_fp, output_fp):
    # check output file's generation time
    cmd = f'stat "%m" {output_fp}'
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    import ipdb; ipdb.set_trace()
    access_time = output.stdout.split(' ')[0]
    modify_time = int(output.stdout)
    print(output.stdout)

    cmd = f"cogger -output {output_fp} {input_fp}"
    start_time = time.time()
    subprocess.run(cmd, shell=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    return end_time - start_time


# def translate_by_gdal(input_fp, output_fp, compression='ZSTD', comp_level=6):
#     if compression in ['JPEG', 'JXL']:
#         comp_params = f"-co COMPRESS={compression} -co QUALITY={comp_level}"
#     else:
#         comp_params = f"-co COMPRESS={compression} -co LEVEL={comp_level}"
#     params = f"-of COG
#                -co NUM_THREADS=8
#                -co BIGTIFF=IF_SAFER
#                -co BLOCKSIZE=1024
#                -co TILED=YES
#                -co TILING_SCHEME=GoogleMapsCompatibleS
#                -co COPY_SRC_OVERVIEWS_ON_WRITE=YES
#                -co COPY_SRC_OVERVIEWS_ON_READ=YES
#                -co COPY_SRC_OVERVIEWS_ON_READ=YES
#                -co COPY_SRC_OVERVIEWS_ON_READ=YES"
#     cmd = f"gdal_translate {params} {comp_params} {input_fp} {output_fp}"
#     start_time = time.time()
#     subprocess.run(cmd, shell=True)
#     end_time = time.time()
#     print(f"Time taken: {end_time - start_time} seconds")
#     return end_time - start_time
    

def compare_compression(input_fp, output_dir, chunk_size=1024):
    """
    Compare the size of a multiband GeoTIFF before and after compression.
    
    Args:
        input_fp (str): Path to input multiband GeoTIFF
        output_dir (str): Directory to save individual COG files
        compression (str): Compression method (default: 'deflate')
        comp_level (int): Compression level (default: 6)
        chunk_size (int): Block size for COG (default: 1024)
    """
    input_fp = Path(input_fp).expanduser()
    output_dir = Path(output_dir).expanduser()
    
    compre_configs = {
        'deflate': range(1, 10),
        'lzma': range(1, 10),
        'lerc': [0],
        'lzw': range(1, 10),
    }
    fig = plt.figure()
    time_taken = {}
    for compression, comp_levels in compre_configs.items():
        time_taken[compression] = []
        dst_profile = cog_profiles.get(compression)
        for comp_level in comp_levels:
            dst_profile.update({
                'blockxsize': chunk_size,
                'blockysize': chunk_size,
                'tiled': True,
                'level': comp_level,
                'interleave': 'band',
                'predictor': 2,
            })
            t0 = time.time()
            output_fp = output_dir / f"{input_fp.stem}_{compression}_{comp_level}_{chunk_size}.tif"
            cog_translate(input_fp, output_fp, dst_profile, in_memory=False, quiet=False, use_cog_driver=True)
            t1 = time.time()
            print(f"Time taken: {t1 - t0} seconds")
            time_taken[compression].append(t1 - t0)

    for compression, time_list in time_taken.items():
        plt.plot(time_list, label=compression, marker='o')
    plt.legend()
    plt.savefig(f"output/compression_time_comparison_{input_fp.stem}.png")
    
    
    
    
    
    # Get base filename without extension

def translate_multiband_to_cogs(input_fp, output_dir, compression='deflate', comp_level=7, chunk_size=512):
    """
    Translate a multiband GeoTIFF to separate COG files using dask for parallelization.
    
    Args:
        input_fp (str): Path to input multiband GeoTIFF
        output_dir (str): Directory to save individual COG files
        compression (str): Compression method (default: 'deflate')
        comp_level (int): Compression level (default: 6)
        chunk_size (int): Block size for COG (default: 1024)
        n_workers (int): Number of parallel workers (default: 4)
    """
    # Create output directory if it doesn't exist
    input_fp = Path(input_fp).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(input_fp).stem
    
    # Read input file to get number of bands and metadata
    with rasterio.open(input_fp) as src:
        n_bands = src.count
        profile = src.profile
    
    # Define COG profile
    dst_profile = cog_profiles.get(compression)
    dst_profile.update({
        'blockxsize': chunk_size,
        'blockysize': chunk_size,
        'tiled': True,
        'level': comp_level,
    })
    
    @dask.delayed
    def process_band(band_idx):
        """Process a single band and convert to COG"""
        output_fp = output_dir / f"{base_name}_band{band_idx+1}.tif"
        
        try:
            # Read the specific band into memory
            with rasterio.open(input_fp) as src:
                # Update profile for single band
                band_profile = profile.copy()
                band_profile.update({
                    'count': 1,
                    'dtype': src.dtypes[band_idx]
                })
                
                # Read band data
                band_data = src.read(band_idx + 1)
                
                # Create memory file for the single band
                with MemoryFile() as memfile:
                    with memfile.open(**band_profile) as dst:
                        dst.write(band_data, 1)
                    
                    # Convert memory file to COG
                    cog_translate(
                        memfile,
                        output_fp,
                        dst_profile,
                        in_memory=True,
                        quiet=True,
                        use_cog_driver=True
                    )
            
            return True
        except Exception as e:
            print(f"Error processing band {band_idx+1}: {str(e)}")
            return False
    
    # Create list of delayed tasks
    tasks = [process_band(i) for i in range(n_bands)]
    print(f"Number of tasks: {len(tasks)}")
    # Execute tasks in parallel
    start_time = time.time()
    results = dask.compute(*tasks)
    end_time = time.time()
    
    
    # Print summary
    successful = sum(results)
    print(f"Processed {successful}/{n_bands} bands successfully")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    return successful == n_bands



def compare_compression_ratio(tif_dir, output_dir):
    tif_dir = Path(tif_dir).expanduser()
    pattern = re.compile(r'(?P<method>[A-Z]+|lzw)_(?P<level>\d+)(?:_(?P<blocksize>\d+))?')

    for file in tif_dir.glob('*.tif'):
        match = pattern.match(file.stem)
        (method, level, blocksize) = match.groups()
        print(method, level, blocksize)
        break
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for compression in ['deflate', 'lzma', 'lerc', 'lzw']:
        for comp_level in range(1, 10):
            pass

if __name__ == "__main__":
    # Setup dask client
    # n_workers = 8
    # client = Client(n_workers=n_workers)
    # Example usage
    # input_fp = "~/data/GVS/Deploy/predictions_2017/32MQE_unnwuvie_DEFLATE_7.tif"
    # output_dir = "~/data/GVS/Deploy/predictions_2017/32MQE"
    # translate_multiband_to_cogs(input_fp, output_dir)
    input_fp = "/home/ksb781/data/GVS/Deploy/predictions_2017/32MQE_xuyou07n_DEFLATE_7.RH100_Q1.tif"
    output_dir = "~/data/GVS/Deploy/predictions_2017/32MQE"
    compare_compression(input_fp, output_dir)