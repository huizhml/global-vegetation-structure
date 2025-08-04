#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small-g
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=inference
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu26fl
##SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu08fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch
echo "***************************** JOB INFO *****************************"
echo "Host: $HOSTNAME"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Number of tasks: $SLURM_NTASKS"
echo "Time requested: $SLURM_TIMELIMIT"
scontrol show job $SLURM_JOB_ID | grep "TRES="
echo "********************************************************************"


export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
## Set MIOpen cache to a temporary folder.
if [ $SLURM_LOCALID -eq 0 ] ; then
rm -rf $MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
fi
source setup_env.sh

# TODO: make following a function,
echo "Get tile id"
part=${1:-0}
year=${2:-2020}
hostname=$(hostname)
save_dir=${HOME}/data/GVS/Deploy/predictions_GTiff_${year}
mkdir -p $save_dir

tile_id_file=${HOME}/data/GVS/Deploy/lumi_job_files/deploy_s2_items_${year}_part${part}.txt
echo "Translate tiles from $tile_id_file"
# Get tile ID from line number specified by SLURM array task ID

if [ -z "$line_num" ]; then
    while IFS= read -r tile_id; do
        translate_flag="${HOME}/data/GVS/Deploy/translate_flags_${year}/${tile_id}_done"
        if [ -f "$translate_flag" ]; then
            echo "Translate flag file $translate_flag exists. Skipping tile $tile_id"
            continue
        fi
        # wait for input data being streamed for the first tile
        # Check if the h5 file is being used by another process
        h5_file="${HOME}/data/GVS/Deploy/inference_${year}/${tile_id}.h5"
        while [ ! -f "$h5_file" ]; do
            echo "The h5 file $h5_file does not exist. Waiting for 60 seconds..."
            sleep 60
        done
        stream_flag="${HOME}/data/GVS/Deploy/stream_flags_${year}/${tile_id}_done"
        while [ ! -f "$stream_flag" ]; do
            echo "Tile $tile_id is still in streaming. Waiting for 60 seconds..."
            sleep 60
        done
        echo "Processing tile ID: $tile_id"
        echo "***************************** START INFERENCE *****************************"
        run_id=cg11fpjr
        echo run prediction for model $run_id for tile $tile_id;
        python run.py predict -c config/predict.yaml --model config/model/xception_mix_order.yaml \
                --data.init_args.input_lat_lon True \
                --data.init_args.num_workers 8 \
                --data.init_args.tile_id $tile_id \
                --data.init_args.metadata_file ~/data/GVS/Deploy/lumi_job_files/deploy_s2_items_${year}_part${part}.parquet \
                --data.init_args.pred_fp ~/data/GVS/Deploy/inference_${year} \
                --data.init_args.prediction_dir ~/data/GVS/Deploy/predictions_GTiff_${year}/${tile_id}_GTiff \
                --data.init_args.year $year \
                --data.init_args.batch_size 1 \
                --data.init_args.cache_predictions False \
                --data.init_args.stream_input True \
                --correct_bias True \
                --data.init_args.patch_size 544 \
                --data.init_args.chunk_size 512 \
                --data.init_args.debug False \
                --data.init_args.predict_full_profile True \
                --data.init_args.output_format gtiff \
                --trainer.logger.init_args.resume False \
                --trainer.logger.init_args.offline True \
                --trainer.logger.init_args.id $run_id 

        # Capture the exit status of the command
        exit_status=$?
        if [ $exit_status -ne 0 ]; then
            echo "Prediction command failed with exit status $exit_status for tile $tile_id"
            rm -rf $save_dir/${tile_id}_GTiff
        else
            echo "Prediction command completed successfully"
            touch ${HOME}/data/GVS/Deploy/inference_flags_${year}/${tile_id}_done
            echo "Delete input h5 file..."
            rm -f ${h5_file}
            rm -f ${stream_flag}
        fi
        echo "***************************** END INFERENCE *****************************"
    done < "$tile_id_file"
fi


# echo "***************************** SYNC DATA TO SCRATCH *****************************"
# mkdir -p /scratch/${tile_id}
# t0=$(date +%s)
# data_dir="${HOME}/data/GVS/Deploy/inference_${year}.zarr"
# cd "${data_dir}"
# TAR_FILE="${data_dir}/${tile_id}.tar"
# if [ ! -f "${TAR_FILE}" ]; then
#     echo "Creating ${TAR_FILE}..."
#     tar cf - "${tile_id}" | tee "${TAR_FILE}" | tar xf - -C /scratch/
# else
#     echo "${TAR_FILE} already exists. Skipping creation and extracting..."
#     tar xf "${TAR_FILE}" -C /scratch/
# fi
# cd ~/gvs
# # rsync -a --stats ${HOME}/data/GVS/Deploy/inference_${year}.zarr/${tile_id} /scratch/
# t1=$(date +%s)
# duration=$((t1-t0))
# hours=$((duration / 3600))
# minutes=$(( (duration % 3600) / 60 ))
# seconds=$((duration % 60))
# echo "Time taken to sync data: ${hours}h ${minutes}m ${seconds}s"

