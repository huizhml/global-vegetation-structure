#!/bin/bash

year=${1:-2020}
# tiles=$(ls ${HOME}/data/gvs/GEDI_for_correction/partitions_${year}_v1 | sort) # this returns all tiles in a string
mapfile -t tiles < <(ls "${HOME}/data/gvs/GEDI_for_correction/partitions_${year}_v1" | sort)
echo "number of tiles: ${#tiles[@]}"
n_tiles_per_task=${2:-1031}
start_idx=$((SLURM_PROCID * $n_tiles_per_task))
echo "start_idx: $start_idx"
echo "n_tiles_per_task: $n_tiles_per_task"

# check if last task, if so, process the remaining tiles
LAST_TASK_IDX=$(($SLURM_NTASKS - 1))
if [ $SLURM_PROCID -eq $LAST_TASK_IDX ]; then
    for tile in ${tiles[@]:$start_idx}; do # from start_idx to the end
        tile_id="${tile%.*}"
        python -m postprocess.bias_correction tile_id=$tile_id year=$year

    done
else    
    for tile in ${tiles[@]:$start_idx:$n_tiles_per_task}; do
        tile_id="${tile%.*}"
        python -m postprocess.bias_correction tile_id=$tile_id year=$year
    done
fi