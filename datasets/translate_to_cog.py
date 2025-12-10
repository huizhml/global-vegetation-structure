import os
import time
import subprocess

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


def translate_by_gdal(input_fp, output_fp, compression='ZSTD', comp_level=6):
    if compression in ['JPEG', 'JXL']:
        comp_params = f"-co COMPRESS={compression} -co QUALITY={comp_level}"
    else:
        comp_params = f"-co COMPRESS={compression} -co LEVEL={comp_level}"
    params = f"-of COG
               -co NUM_THREADS=8
               -co BIGTIFF=IF_SAFER
               -co BLOCKSIZE=1024
               -co TILED=YES
               -co TILING_SCHEME=GoogleMapsCompatibleS
               -co COPY_SRC_OVERVIEWS_ON_WRITE=YES
               -co COPY_SRC_OVERVIEWS_ON_READ=YES
               -co COPY_SRC_OVERVIEWS_ON_READ=YES
               -co COPY_SRC_OVERVIEWS_ON_READ=YES"
    cmd = f"gdal_translate {params} {comp_params} {input_fp} {output_fp}"
    start_time = time.time()
    subprocess.run(cmd, shell=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    return end_time - start_time


if __name__ == "__main__":
    input_fp = "/Users/hui/data/GVS/Deploy/predictions_2017/32MQE_2017_10000000000000000000000000000000.tif"
    output_fp = "/Users/hui/data/GVS/Deploy/predictions_2017/32MQE_2017_10000000000000000000000000000000_cog.tif"
    translate_by_cogger(input_fp, output_fp)