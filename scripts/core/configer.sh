# ================================================================
#    Functions to split tiles for further slurm job configuration
# ================================================================
CONFIG_DIR=${HOME}/data/gvs/assets/worklists

split_tiles_by_zone() {
    # Split tiles by zone (first 3 letters).
    # The output is a list of {zone}.txt files, each file contains a list of tiles in the zone.
    local output_dir=$1
    mkdir -p "$output_dir"

    # Clear existing zone files
    rm -f "$output_dir"/*.txt

    # Get all tiles and split by zone
    for tile in $(ls ${HOME}/data/gvs/predictions/2024/original/tiles/geotiff); do
        zone="${tile:0:3}"  # Extract first 3 letters
        echo "$tile" >> "$output_dir/${zone}.txt"
    done

    echo "Done! Tiles grouped by zone in $output_dir"
    ls -lh "$output_dir"

    max=0
    for zone in $(ls $output_dir); do
        echo "Zone: $zone"
        tiles=$(cat $output_dir/$zone)
        echo "Number of tiles: $(wc -l < $output_dir/$zone)"
        echo "--------------------------------"
        if [ $(wc -l < $output_dir/$zone) -gt $max ]; then
            max=$(wc -l < $output_dir/$zone)
        fi
    done
    echo "Max number of tiles: $max"
}

# ================================================================
#    Functions to allocate tasks(tiles/zones) to slurm jobs
# ================================================================

get_n_tiles_per_task() {
    # Get the number of tiles per task.
    # If it is an array job, the number of tiles per task is decided by SLURM_NTASKS and SLURM_ARRAY_TASK_COUNT.
    # If it is not an array job, the number of tiles per task is decided by SLURM_NTASKS.
    local n_tiles_total=${1:-}
    if [[ -z "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
        echo "Not an array job" >&2
        echo $((n_tiles_total / SLURM_NTASKS))
    else
        echo $((n_tiles_total / SLURM_NTASKS / SLURM_ARRAY_TASK_COUNT))
    fi
}

get_start_idx() {
    # TODO: this doesn't work when n_tiles_per_task is equal to total number of tiles
    # Get the start index of the task.
    # If it is an array job, the start index is decided by SLURM_ARRAY_TASK_ID and SLURM_PROCID.
    # If it is not an array job, the start index is decided by SLURM_PROCID.
    local n_tiles_per_task=${1:-}
    if [[ -z "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
        echo "Not an array job" >&2
        start_idx=$((SLURM_PROCID * n_tiles_per_task))
    else
        UNIVERSAL_TASK_ID=$((SLURM_ARRAY_TASK_ID * SLURM_NTASKS + SLURM_PROCID))
        start_idx=$((UNIVERSAL_TASK_ID * n_tiles_per_task))
    fi
    echo $start_idx
}


get_subtask_config_files() {
    # Allocate {zone}.txt files to each task.
    # Tiles are grouped by zones. There are 500+ zones. We split them into chunks of 24 zones per task (~25 tasks on Hendrix).
    # This allocation is mainly for blending. Tiles are ordered in a way to avoid duplicated copying of input files.
    local job_offset=${1:0}
    local n_zones_per_task=${2:-24}
    local task_zones=()

    all_zones=($(ls ${CONFIG_DIR}/tiles_by_mgrs_zone/*.txt | sort))
    n_zones_per_job=$((SLURM_NTASKS * n_zones_per_task))
    job_zones=(${all_zones[@]:job_offset:n_zones_per_job})
    start_idx=$((SLURM_PROCID * n_zones_per_task))
    task_zones=(${job_zones[@]:start_idx:n_zones_per_task})
    echo ${task_zones[@]}
}


# get_subtask_tiles_for_array_jobs() {
#     # Allocate list of tiles to each task in an array job.
#     # The list size for each task is decided by SLURM_NTASKS and SLURM_ARRAY_TASK_COUNT. 
#     # Each job in the array job shares the same universal config file (txt file with tiles), and each job in the array job launches the same number of tasks.
#     # local unfinished_tiles=("$@")
#     local unfinished_tiles_file=${1:-}
#     local task_tiles=()
#     unfinished_tiles=($(cat ${CONFIG_DIR}/$unfinished_tiles_file))
#     n_tiles_total=${#unfinished_tiles[@]}
#     n_tiles_per_task=$(get_n_tiles_per_task $n_tiles_total)
#     UNIVERSAL_TASK_ID=$((SLURM_ARRAY_TASK_ID * SLURM_NTASKS + SLURM_PROCID))
#     start_idx=$((UNIVERSAL_TASK_ID * n_tiles_per_task))
#     task_tiles=(${unfinished_tiles[@]:start_idx:n_tiles_per_task})
#     echo ${task_tiles[@]}
# }

get_subtask_tiles() {
    # Allocate list of tiles to each task in one slurm job.
    # The list size for each task is decided by SLURM_NTASKS. 
    # Each slurm job has its own config file (txt file with tiles), and each slurm job launches its own number of tasks.

    local unfinished_tiles_file=${1:-}
    local task_tiles=()
    
    unfinished_tiles=($(cat ${CONFIG_DIR}/$unfinished_tiles_file))
    n_tiles_total=${#unfinished_tiles[@]}
    n_tiles_per_task=$(get_n_tiles_per_task $n_tiles_total)
    start_idx=$(get_start_idx $n_tiles_per_task)
    task_tiles=(${unfinished_tiles[@]:start_idx:n_tiles_per_task})
    echo ${task_tiles[@]}
}




