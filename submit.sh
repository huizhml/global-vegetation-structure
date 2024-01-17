#!/bin/bash

submit_job() {
    local job_name="download_$1"
    local num_cpus="$3"
    local memory="$3"

    # Set the job parameters
    sbatch --job-name="$job_name" \
           --ntasks=1 \
           --cpus-per-task="$num_cpus" \
           --mem="$memory" \
            run.sh $4
}

# Check if the configuration file exists
config_file="/users/zhanghui/GEDI2019/download_config.txt"
if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file '$config_file' not found."
    exit 1
fi

# Read the configuration file line by line and submit jobs
while read -r line; do
    read -a config <<< "$line"
    if [ "${#config[@]}" -eq 4 ]; then
        submit_job "${config[@]}"
    else
        echo "Error: Invalid configuration line: '$line'"
    fi
done < "$config_file"