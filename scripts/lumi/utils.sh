#!/bin/bash

echo_job_info() {
    echo "***************************** JOB INFO *****************************"
    echo "Host: $HOSTNAME"
    echo "Job Name: $SLURM_JOB_NAME"
    echo "Partition: $SLURM_JOB_PARTITION"
    echo "CPUs per task: $SLURM_CPUS_PER_TASK"
    echo "Number of tasks: $SLURM_NTASKS"
    echo "Time requested: $SLURM_TIMELIMIT"
    scontrol show job $SLURM_JOB_ID | grep "TRES="
    echo "********************************************************************"
}

init_env() {
    export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
    export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
    ## Set MIOpen cache to a temporary folder.
    if [ $SLURM_LOCALID -eq 0 ] ; then
        rm -rf $MIOPEN_USER_DB_PATH
        mkdir -p $MIOPEN_USER_DB_PATH
    fi
    source setup_env.sh
}

get_data_root_dir() {
    local use_flash=$1
    if [ "$use_flash" == "True" ]; then
        data_root_dir=${HOME}/flash/data/gvs
    else
        data_root_dir=${HOME}/data/gvs
    fi
    echo "$data_root_dir"
}


get_tile_id() {
    local line_num=$1
    local tile_id_file=$2
    local year=$3

    config_dir=${HOME}/data/GVS/Deploy/slurm_job_files_${year}
    tile_id_file=${config_dir}/deploy_s2_items_${year}_part${part}.txt
    ## Check if processed before submitting job
    # tile_id=$(sed -n "${line_num}p" "$tile_id_file")
    line=$(sed -n "${line_num}p" $tile_id_file)
    IFS=',' read -r tile_id idx <<< "$line"
    echo "Line $line_num: Tile=$tile_id, idx=$idx"
    if [ -z "$idx" ]; then
        echo "idx is null, no meta file"
        meta_file='none'
    else
        idx=$(printf "%d" $idx)
        meta_file=${config_dir}/deploy_s2_items_${year}_part${idx}.parquet
    fi
    echo "Processing tile ID: $tile_id, line $line_num from $tile_id_file"
}


run_inference() {
    local tile_id=$1
    local save_dir=$2
    local meta_file=$3
    local year=$4
    download_data=True


    printf '>%.0s' {1..20}
    echo "Processing tile ID: $tile_id"
    run_id=cg11fpjr
    args=(predict -c config/predict.yaml --model.backbone config/model/xception_mix_order.yaml
            --data.init_args.tile_id $tile_id
            --data.init_args.metadata_file none
            --data.init_args.s2_grid_file ${HOME}/data/gvs/state/s2_tiles_with_growing_months.parquet
            --data.init_args.pred_fp ${data_root_dir}/inputs/inference_${year}
            --data.init_args.prediction_dir ${save_dir}/${tile_id}
            --data.init_args.year $year
            --data.init_args.stream_input True
            --data.init_args.download_data $download_data
            --data.init_args.debug False
            --trainer.logger.init_args.resume False
            --trainer.logger.init_args.offline True
            --trainer.logger.init_args.save_dir /tmp
            --trainer.logger.init_args.id $run_id
    )

    python run.py ${args[@]}

    # Capture the exit status of the command
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Prediction command failed with exit status $exit_status for tile $tile_id"
        rm -rf $save_dir/${tile_id}
        if [ "$use_flash" == "True" ]; then
            echo "Process may be interrupted during streaming, delete input h5 file..."
            rm -f ${h5_file}
        fi
        exit 1
    else
        echo "Prediction command completed successfully"
        echo "Delete input h5 file..."
        rm -f ${h5_file}
        exit 0
    fi
    printf '%.0s-' {1..50}; printf '\n'
}


run_translate() {
    local tile_id=$1
    local pred_dir=$2
    printf '%.0s-' {1..50}; printf '\n'
    echo "Translating predictions for tile $tile_id"
    geotiff_dir=${pred_dir}/geotiff/${tile_id}
    cog_dir=${pred_dir}/cog/${tile_id}
    mkdir -p $cog_dir
    python -m postprocessing.run run=translate_predictions run.src_dir=$geotiff_dir run.dst_dir=$cog_dir
}

sync_to_lumi() {
    local tile_id=$1
    local cog_dir=$2
    local year=$3
    printf '%.0s-' {1..50}; printf '\n'
    echo "Syncing data to LUMI-O for tile $tile_id"
    ## since we'll apply correction and blending to the data, we don't translate the data to cog 
    module load lumio

    lumi_project=465002698
    remote=lumi-${lumi_project}-public:

    # Get bucket name and check if it exists
    zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
    bucket_name=${zone}-${year}
    echo "bucket_name: ${bucket_name}"

    if rclone lsd ${remote} | grep "^.*${bucket_name}.*"; then
        echo "Bucket ${bucket_name} exists"
    else
        echo "Bucket ${bucket_name} does not exist, creating..."
        rclone mkdir ${remote}${bucket_name}
    fi

    SRC=${cog_dir}
    DST=${remote}${bucket_name}/predictions_GTiff_${year}/${tile_id}
    # e.g. remote="lumi-${lumi_project}-private:"  ← note the trailing colon
    rclone sync "$SRC" "$DST" --transfers=16 --checkers=16 --multi-thread-streams=4
    count=$(rclone ls "${DST}" | wc -l)
    echo "Number of files in ${DST}: $count"
    if [ $count -lt 303 ]; then
        echo "Number of files in ${DST} is less than 303, deleting..."
        echo "Error: Number of files in ${DST} is less than 303" >&2
        exit 1
    fi
        rm -rf ${cog_dir}
        printf '%.0s-' {1..50}; printf '\n'
}