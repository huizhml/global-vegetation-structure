#!/bin/bash
## SBATCH --account=project_465000894
#SBATCH --partition=small
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
hostname
mkdir -p /scratch/tmp$SLURM_JOB_ID
export SCRATCH=/scratch/tmp$SLURM_JOB_ID
echo $SCRATCH

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
zones=$1
years=$2
rewrite=${3:-False}

year_list=$(get_list "$years")
echo "years: $year_list"

zone_list=$(get_list "$zones")
echo "zones: $zone_list"

IFS=','
# ************************** sync files before exit **********************************
# Function to perform rsync and any additional cleanup before exiting
do_rsync_and_cleanup() {
    echo "Performing rsync and cleanup..."
    for zone in ${zone_list[@]}; do
        # Perform rsync operations
        rsync -a "$SCRATCH/data/GEDI/$zone" "$HOME/data/GEDI/"; #sync flags
        rsync -a "$SCRATCH/data/GEDI/$zone.h5" "$HOME/data/GEDI/"; #sync h5
    done
    rm -rf $SCRATCH
    echo "Rsync and cleanup completed."
    exit 0
}

# Trap various exit signals: SIGHUP, SIGINT, SIGQUIT, SIGTERM
# This ensures do_rsync is called on these signals
trap do_rsync_and_cleanup SIGHUP SIGINT SIGQUIT SIGTERM EXIT

# ************************** sync files every one hour **********************************
while sleep 1h; 
do
    for zone in ${zone_list[@]}; do
        # Perform rsync operations
        rsync -a "$SCRATCH/data/GEDI/$zone" "$HOME/data/GEDI/"; #sync flags
        if [ -f "$SCRATCH/data/GEDI/${zone}.h5" ]; then
            rsync -a "$SCRATCH/data/GEDI/$zone.h5" "$HOME/data/GEDI/"; #sync h5
        fi
    done
done &
LOOPPID=$!

# ************************** copy data **********************************
# Check if rewrite is True
if [[ "$rewrite" == "True" ]]; then
    # Commands to delete some files
    echo "Rewriting, deleting flag files..."
    for zone in ${zone_list[@]}; do
        for year in ${year_list[@]}; do
            rm -f $HOME/GEDI/$zone/$year*
        done
    done
else
    echo "Rewrite is not True, skipping file deletion..."
fi

mkdir $SCRATCH/GEDI
mkdir -p $SCRATCH/data/GEDI


for zone in ${zone_list[@]}; do
    mkdir -p $HOME/data/GEDI/$zone;
    time cp --sparse=always -r $HOME/data/GEDI/$zone $SCRATCH/data/GEDI/; #copy flags
    if [ -f "$HOME/data/GEDI/${zone}.h5" ]; then
        time cp --sparse=always -r "$HOME/data/GEDI/${zone}.h5" "$SCRATCH/data/GEDI/${zone}.h5" # copy h5
        echo "Continue downloading data for ${zone}."
    else
        echo "Downloading data for ${zone} from scratch."
    fi
done

for year in ${year_list[@]}; do
    mkdir $SCRATCH/GEDI/$year
    for zone in ${zone_list[@]}; do
        echo "Copying files for $year $zone ...";
        mkdir $SCRATCH/GEDI/$year/$zone;    
        time cp --sparse=always -r $HOME/GEDI/$year/$zone/partition* $SCRATCH/GEDI/$year/$zone; # copy GEDI data
    done
    echo download zones "$zones" $year;
    python -u -m download.s2 zone="$zones" year=$year rewrite=$rewrite root_dir=$SCRATCH
done

kill $LOOPPID
