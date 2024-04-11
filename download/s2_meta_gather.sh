#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
##SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
hostname

source activate mpc

is_comma_separated_list() {
    local var="$1"
    if [[ "$var" == *","* ]]; then
        echo "true"
    else
        echo "false"
    fi
}

get_list() {
    local input_var="$1"
    local year_list

    # Convert array to a comma-separated string if necessary
    if [[ "$(declare -p input_var 2>/dev/null)" =~ "declare -a" ]]; then
        input_var="${input_var[*]}"
    fi

    # Removing potential brackets at the start and end if present
    input_var="${input_var#[}"
    input_var="${input_var%]}"

    if [[ $(is_comma_separated_list "$input_var") == "true" ]]; then
        year_list="${input_var}"
    else
        year_list="${input_var},"
    fi

    echo "$year_list"
}

# Example usage
ID=$SLURM_ARRAY_TASK_ID
# echo 'reading config file ~/GEDI/download_job_hendrix_'${ID}'.txt'
# read years zones merge_zones <  ~/GEDI/download_job_hendrix_${ID}.txt
read zones merge_zones <<< $(sed -n ${ID}p ~/GEDI/slurm_job_config.txt)
years=${2:-[2019,2020,2021,2022]}
echo $years 
echo $zones 
echo $merge_zones

# zones=$1
# merge_zones=${3:-False}
rewrite=${1:-False}

year_list=$(get_list "$years")
echo "years: $year_list"

zone_list=$(get_list "$zones")
echo "zones: $zone_list"

if [[ $merge_zones == "True" ]]; then ## for small zones, processing them together as one big df
    IFS=','
    for year in ${year_list[@]}; do
        echo download zones "[$zones]" $year;
        python -u -m download.s2_meta_gather zone="[$zones]" year=$year rewrite=$rewrite
    done
else ## for large zones, processing them sequentially
    IFS=','
    for zone in ${zone_list[@]}; do
        for year in ${year_list[@]}; do
            echo download zone "$zone" $year;
            python -u -m download.s2_meta_gather zone="$zone" year=$year rewrite=$rewrite
        done
    done
fi

