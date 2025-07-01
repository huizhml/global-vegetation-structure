"""
gtiff_benchmark.py - Test utility for benchmarking geotiff compression
"""
import os
import argparse
import glob
import configparser
import sys
import subprocess
import tempfile
import time

def parse_config(config_file='config.ini'):
    """
    Parse the configuration file into a more usable dictionary, with all the
    different options as keys, and a list of creation option arguments to pass 
    to gdal_translate as a value.
    """
    config_dict = {}
    
    config = configparser.ConfigParser()
    config.read(config_file)
    config.optionxform = str
    
    for section in config.sections():
        if section.lower() == 'default':
            continue
        
        options = []
        for item in config.items(section):
            options.extend(["-co", "{}={}".format(item[0].upper(), item[1])])
            
        config_dict.update({section: options})

    return config_dict
        
def time_command(cmd=['sleep','1'], rep=1):
    """
    Use python's time module and subprocess to run a command and 
    return the average execution time in seconds.
    """
    times = []
    
    for _ in range(rep):
        start_time = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        end_time = time.time()
        
        if result.returncode != 0:
            raise Exception("Running benchmark failed as command returned a "
                            "non-zero return code. Here are some hints "
                            "from stderr: {}".format(result.stderr))
        
        times.append(end_time - start_time)
    
    # Return average time in seconds
    return sum(times) / len(times)

if __name__ == '__main__':
    
    base_dir = os.path.split(os.path.abspath(__file__))[0]

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GTiff Benchmark")
    parser.add_argument('--config', help='Config file to use', 
                        default='config.ini')
    parser.add_argument('--repetitions', help='Number of reps', default=1)
    parser.add_argument('--input', help='Directory with input files',
                        default=os.path.join(base_dir,'input_rasters'))
    args = parser.parse_args()
    
    # Test that timing works
    print("Testing that timing works...")
    try:
        task_clock = time_command()
        print("Looks good!")
    except Exception as e:
        print("Something went wrong testing the timing command! Without "
              "proper timing this is not going to work...")
        sys.exit(1)
    
    # Parse configuration file and input files
    config = parse_config(config_file=args.config)
    
    # Let's run some tests!
    results = ['test;file;option;time;size;ratio;savings;speed']
    for path in glob.glob(os.path.join(args.input, "*.tif")):
        
        filename = os.path.basename(path)
        
        # Create a temporary directory for all the tests with this file
        with tempfile.TemporaryDirectory(prefix='gtiff-benchmark-') as tmpdir:
            
            # Create the base version of the input file.
            base_file = os.path.join(tmpdir, 'base.tif')
            cmd = ['gdal_translate', '-q', path, base_file, '-co', 'TILED=NO', 
                   '-co', 'COMPRESS=NONE', '-co', 'COPY_SRC_OVERVIEWS=NO']
            subprocess.run(cmd)
            base_file_size = os.stat(base_file).st_size / (1024.0*1024.0)
            
            # Run the tests...
            for option in config.keys():
                # Run the write tests on the base_file, saving the result to
                # the option_file
                print("WRITE test: Running with option '{}' on file '{}'".format(option, filename))
                
                option_file = os.path.join(tmpdir, option+'.tif')
                cmd = ['gdal_translate', '-q', base_file, option_file, *config.get(option)]
                print(" ".join(cmd))
                if "cog" in option:
                    cmd.extend(["-of","COG"])
                if "jpeg" in option: # JPEGSetupEncode seems to support only 8 bit only
                    cmd.extend(["ot","Byte"])
                try:
                    task_clock = time_command(cmd=cmd, rep=args.repetitions)
                    file_size = os.stat(option_file).st_size / (1024.0*1024.0)
                    compression_ratio = base_file_size / file_size
                    savings = 1 - (file_size / base_file_size)
                    speed = (1/task_clock) * base_file_size
                    print("WRITE test: Completed {} repetitions. Average time: {:.2f}s File size: {:.1f}Mb Ratio: {:.2f} Speed: {:.2f}Mb/s".format(args.repetitions, task_clock, file_size, compression_ratio, speed))
                except Exception as e:
                    try: os.remove(option_file)
                    except: pass
                    print("WRITE test: Failed!")
                    task_clock = ''
                    file_size = ''
                    savings = ''
                    compression_ratio = ''
                    speed = ''
                finally:
                    results.append('{};{};{};{};{};{};{};{}'.format('write', filename, option, task_clock, file_size, compression_ratio, savings, speed))
                    
                # Run the read tests on the just created file by decompressing
                # it again.
                print("READ test: Running on file '{}'".format(option+'.tif'))
                read_file_output = os.path.join(tmpdir, 'read.tif')
                cmd = ['gdal_translate', '-q', option_file, read_file_output, 
                       '-co', 'TILED=NO', '-co', 'COMPRESS=NONE']
                       
                try:
                    task_clock = time_command(cmd=cmd, rep=args.repetitions)
                    speed = (1/task_clock) * base_file_size
                    print("READ test: Completed {} repetitions. Average time: {:.2f}s Speed: {:.2f}Mb/s".format(args.repetitions, task_clock, speed))
                except Exception as e:
                    print("READ test: Failed!")
                    task_clock = ''
                    speed = ''
                finally:
                    results.append('{};{};{};{};;;;{}'.format('read', filename, option, task_clock, speed))
                    
    # Write the results to the results file and to stdout
    result_csv = '\n'.join(results)
    with open('results.csv', 'w') as f:
        f.write(result_csv)
    print("===========================================================")
    print("Benchmark complete! Results:")
    print("-----------------------------------------------------------")
    print(result_csv)
    print("===========================================================")