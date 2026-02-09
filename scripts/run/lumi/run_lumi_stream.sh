#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=stream_data
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu26fl
##SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu08fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch
echo "***************************** JOB INFO *****************************"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Number of tasks: $SLURM_NTASKS"
echo "Time requested: $SLURM_TIMELIMIT"
scontrol show job $SLURM_JOB_ID | grep "TRES="
echo "********************************************************************"

part=${SLURM_ARRAY_TASK_ID:-0}
line_num=${1:-0}
year=${2:-2020}
hostname=$(hostname)

use_flash=${3:-False}
echo "use_flash=$use_flash"
if [ "$use_flash" == "True" ]; then
    save_dir=${HOME}/flash/data/GVS/deploy/inference_${year}
    download_data=True # Download data in inference pipeline
else
    save_dir=${HOME}/data/GVS/deploy/inference_${year}
    download_data=False
fi

config_dir=${HOME}/data/GVS/deploy/slurm_job_files
tile_id_file=${config_dir}/deploy_s2_items_${year}_part${part}.txt
tile_id=$(sed -n "${line_num}p" "$tile_id_file")
echo "Processing tile ID: $tile_id, line $line_num from $tile_id_file"


if [ -f "${HOME}/data/GVS/deploy/stream_flags_${year}/${tile_id}_done" ]; then
    echo "Tile $tile_id already processed, skip"
    exit 0
fi

sync_flag_new="${HOME}/data/GVS/deploy/sync_flags_${year}/${tile_id}_best_images_done"
if [ ! -f "$sync_flag_new" ] && [ ! -f "${save_dir}/${tile_id}.h5" ]; then
    echo "***************************** START STREAMING *****************************"
    echo Stream input data for tile $tile_id in year $year;
    python -m download._7_stream_tile metadata_file=${config_dir}/deploy_s2_items_${year}_part${part}.parquet tile_id=$tile_id output_dir=${save_dir} n_iamges_per_tile=20
    exit_status=$?  
    if [ $exit_status -ne 0 ]; then
        echo "Stream command failed with exit status $exit_status"
    else
        echo "Stream command completed successfully"
        touch ${HOME}/data/GVS/deploy/stream_flags_${year}/${tile_id}_best_images_done
    fi
    echo "***************************** END STREAMING *****************************"
fi

# while IFS= read -r tile_id; do
#     sync_flag="${HOME}/data/GVS/deploy/sync_flags_${year}/${tile_id}_done"
#     sync_flag_new="${HOME}/data/GVS/deploy/sync_flags_${year}/${tile_id}_best_images_done"
#     if [ ! -f "$sync_flag" ] && [ ! -f "$sync_flag_new" ] && [ ! -f "${save_dir}/${tile_id}.h5" ]; then
#         echo "***************************** START STREAMING *****************************"
#         echo Stream input data for tile $tile_id in year $year;
#         python -m download._7_stream_tile metadata_file=${config_dir}/deploy_s2_items_${year}_part${part}.parquet tile_id=$tile_id output_dir=${save_dir} n_iamges_per_tile=20
#         exit_status=$?  
#         if [ $exit_status -ne 0 ]; then
#             echo "Stream command failed with exit status $exit_status"
#         else
#             echo "Stream command completed successfully"
#             touch ${HOME}/data/GVS/deploy/stream_flags_${year}/${tile_id}_best_images_done
#         fi
#         echo "***************************** END STREAMING *****************************"
#     fi
# done < "$tile_id_file"  

