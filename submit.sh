#!/bin/bash

# Check if the configuration file exists
year=$1
config_file="/users/zhanghui/GEDI/$year/download_config.csv"
if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file '$config_file' not found."
    exit 1
fi

# Read the configuration file line by line and submit jobs
read -r header < $config_file
# Process the config file
while IFS=';' read -r ncores level_1 npartitions nparallel MGRS_UTM
do
    mem=$(( ncores * 2 ))
    echo ncores $ncores $mem $npartitions $nparallel $MGRS_UTM
    sbatch --job-name="download_${year}" --ntasks=1 --cpus-per-task=$ncores --mem="${mem}G" download/download_s2.sh $MGRS_UTM $nparallel $year
    sleep 1
done < <(tail -n +2 $config_file)