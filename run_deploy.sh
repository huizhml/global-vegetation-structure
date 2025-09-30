#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mincpus=4
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24-00:00:00
#SBATCH --job-name=deploy
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu26fl
#SBATCH --exclude hendrixgpu12fl,hendrixgpu11fl,hendrixgpu26fl
echo "***************************** JOB INFO *****************************"
echo "Host: $HOSTNAME"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Number of tasks: $SLURM_NTASKS"
echo "Time requested: $SLURM_TIMELIMIT"
scontrol show job $SLURM_JOB_ID | grep "TRES="
echo "********************************************************************"


# echo "Get tile id"
# part=${1:-0}
# year=${2:-2020}
line_num=${SLURM_ARRAY_TASK_ID:-2}
# tile_id_file=$1
# tile_id_file_num=${2:-0}
year=${1:-2020}


# hostname=$(hostname)
# if [ "$hostname" == "hendrixgpu26fl.unicph.domain" ] || [ "$hostname" == "hendrixgpu01fl.unicph.domain" ]; then
#     save_dir=~/data/GVS/Deploy/predictions_GTiff_${year}
#     echo "Host is $hostname (disk dead). Write to $save_dir"
# else
#     save_dir=/scratch/predictions_${year}
#     echo "Host is $hostname. Write to $save_dir"
# fi
save_dir=~/data/GVS/Deploy/predictions_GTiff_${year}
mkdir -p $save_dir


# **************************************************************
#              Predict one tile in one job
# **************************************************************
# config_dir=${HOME}/data/GVS/Deploy/slurm_job_files_${year}
# tile_id_file=${config_dir}/deploy_s2_items_${year}_part${part}.txt
## Check if processed before submitting job
tile_id_file=${HOME}/data/GVS/Deploy/unfinished_tiles_${year}.txt
line=$(sed -n "${line_num}p" $tile_id_file)
IFS=',' read -r tile_id idx <<< "$line"
echo "Line $line_num: Tile=$tile_id, idx=$idx"
if [ -z "$idx" ]; then
    echo "idx is null, no meta file"
    meta_file='none'
else
    idx=$(printf "%d" $idx)
    meta_file=${HOME}/data/GVS/Deploy/slurm_job_files_${year}/deploy_s2_items_${year}_part${idx}.parquet
fi

echo "Processing tile ID: $tile_id, line $line_num from $tile_id_file"
echo "meta_file: $meta_file"

translate_flag="${HOME}/data/GVS/Deploy/flags_translate_update_${year}/${tile_id}_done"
translate_flag_new="${HOME}/data/GVS/Deploy/flags_inference_update_${year}/${tile_id}_best_images_done"
if [ -f "$translate_flag" ] || [ -f "$translate_flag_new" ]; then
    echo "Translate flag file $translate_flag or $translate_flag_new exists. Skipping tile $tile_id"
    continue
fi
echo "Processing tile ID: $tile_id"
echo "***************************** START INFERENCE *****************************"
run_id=cg11fpjr
echo run prediction for model $run_id for tile $tile_id;
python run.py predict -c config/predict.yaml --model config/model/xception_mix_order.yaml \
        --data.init_args.input_lat_lon True \
        --data.init_args.num_workers 4 \
        --data.init_args.tile_id $tile_id \
        --data.init_args.metadata_file $meta_file \
        --data.init_args.pred_fp ~/data/GVS/Deploy/inference_${year}.zarr \
        --data.init_args.prediction_dir $save_dir/${tile_id}_GTiff \
        --data.init_args.year $year \
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
    touch ${translate_flag_new}
fi
echo "***************************** END INFERENCE *****************************"





# # **************************************************************
# #              Predict multiple tiles in one job
# # **************************************************************
# # tile_id_file=${HOME}/data/GVS/Deploy/s2_deploy_items_${year}_part${part}_unique_images.txt # for 2020
# tile_id_file=${HOME}/data/GVS/Deploy/slurm_job_files_${year}/deploy_s2_items_${year}_part${part}.txt # for 2024

# echo "Translate tiles from $tile_id_file"
# # Get tile ID from line number specified by SLURM array task ID

# if [ -z "$line_num" ]; then
#     echo "No tile ID file provided, using all tiles"
#     while IFS= read -r tile_id; do
#         translate_flag="${HOME}/data/GVS/Deploy/translate_flags_${year}/${tile_id}_done"
#         translate_flag_new="${HOME}/data/GVS/Deploy/inference_flags_${year}/${tile_id}_best_images_done"
#         if [ -f "$translate_flag" ] || [ -f "$translate_flag_new" ]; then
#             echo "Translate flag file $translate_flag or $translate_flag_new exists. Skipping tile $tile_id"
#             continue
#         fi
#         echo "Processing tile ID: $tile_id"
#         echo "***************************** START INFERENCE *****************************"
#         run_id=cg11fpjr
#         echo run prediction for model $run_id for tile $tile_id;
#         python run.py predict -c config/predict.yaml --model config/model/xception_mix_order.yaml \
#                 --data.init_args.input_lat_lon True \
#                 --data.init_args.num_workers 4 \
#                 --data.init_args.tile_id $tile_id \
#                 --data.init_args.metadata_file ${HOME}/data/GVS/Deploy/slurm_job_files_${year}/deploy_s2_items_${year}_part${part}.parquet \
#                 --data.init_args.pred_fp ~/data/GVS/Deploy/inference_${year}.zarr \
#                 --data.init_args.prediction_dir $save_dir/${tile_id}_GTiff \
#                 --data.init_args.year $year \
#                 --correct_bias True \
#                 --data.init_args.patch_size 544 \
#                 --data.init_args.chunk_size 512 \
#                 --data.init_args.debug False \
#                 --data.init_args.predict_full_profile True \
#                 --data.init_args.output_format gtiff \
#                 --trainer.logger.init_args.resume False \
#                 --trainer.logger.init_args.offline True \
#                 --trainer.logger.init_args.id $run_id 

#         # Capture the exit status of the command
#         exit_status=$?
#         if [ $exit_status -ne 0 ]; then
#             echo "Prediction command failed with exit status $exit_status for tile $tile_id"
#             rm -rf $save_dir/${tile_id}_GTiff
#         else
#             echo "Prediction command completed successfully"
#             touch ${translate_flag_new}
#         fi
#         echo "***************************** END INFERENCE *****************************"
#     done < "$tile_id_file"
# fi


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

