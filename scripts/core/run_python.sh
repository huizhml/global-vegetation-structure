#!/bin/bash

FLAG_DIR=${HOME}/data/gvs/state
ROOT_SAVE_DIR=${HOME}/data/gvs/predictions
source scripts/core/utils.sh
source scripts/core/configer.sh


# ================================================================
#    Run Python script
# ================================================================
run_blending() {
    local year=$1
    local tile_id=$2
    local rhs_idx=$3
    local flag_dir=${FLAG_DIR}/${year}/blended
    python -m postprocessing.run run=run_blending \
        run.rhs_idx=$rhs_idx \
        run.year=$year \
        run.tile_id=$tile_id \
        run.flag_dir=$flag_dir \
        run.output_dir=${ROOT_SAVE_DIR}/${year}/blended/tiles
}


run_blending_loop1() {
    # Each task is given a list of tiles
    local year=$1
    local rhs_idx=$2
    local unfinished_tiles_file=$3
    local flag_dir=${FLAG_DIR}/${year}/blended

    task_tiles=($(get_subtask_tiles $unfinished_tiles_file))
    module load lumio-ext-tools/1.0.0
    for tile_id in ${task_tiles[@]}; do
        processed=$(check_if_processed $tile_id $flag_dir/${rhs_idx} true)
        if [ $processed -eq 0 ]; then # this is also checked inside the python script
            echo "Tile $tile_id already processed, skipping"
            continue
        fi
        echo "Processing tile $tile_id, $year, $rhs_idx"
        run_blending $year $tile_id $rhs_idx # this will sync blended tiles to LUMI-O
        zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
        s3cmd setacl --recursive --acl-public s3://${zone}-${year}/${tile_id} # make it public on LUMI-O
        clean_up_local_file ${ROOT_SAVE_DIR}/${year}/blended/tiles $tile_id
    done
}

run_blending_loop2() {
    # Each task read its own list of tiles
    local task_config_files=$1
    local flag_dir=$2
    local year=$3
    for config_file in ${task_config_files[@]}; do
        for tile_id in $(cat $config_file); do
            processed=$(check_if_processed $tile_id ${FLAG_DIR}/${flag_dir})
            if [ $processed -eq 0 ]; then
                echo "Tile $tile_id already processed, skipping"
                continue
            fi
            echo "Processing tile $tile_id"
            run_blending $year $tile_id
        done
    done
}


run_sanity_check() {   
    local run=$1 
    local source_dir=$2
    local target_dir=$3
    
    echo sanity check for two datasets
    python -m postprocessing.run run=$run \
        run.source_dir=$source_dir \
        run.target_dir=$target_dir
}

make_manifest() { 
    local data_dir=$1
    local dataset_name=$2
    local root_note=$3
    python -m postprocessing.run run=make_manifest \
        run.data_dir=$data_dir \
        run.dataset_name=$dataset_name \
        run.root_note=$root_note
}