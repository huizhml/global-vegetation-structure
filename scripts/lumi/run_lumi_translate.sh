#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --job-name=translate
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
source setup_env.sh

# part=${SLURM_ARRAY_TASK_ID:-0}
# line_num=${1:-0}
line_num=${SLURM_ARRAY_TASK_ID:-0}
tile_id_file=$1
year=${2:-2020}
use_flash=${3:-True}
echo "use_flash=$use_flash"
if [ "$use_flash" == "True" ]; then
    input_dir=${HOME}/flash/data/GVS/deploy/predictions_GTiff_${year}
    # output_dir=${HOME}/flash/data/GVS/deploy/predictions_${year}
else
    input_dir=${HOME}/data/GVS/deploy/predictions_GTiff_${year}
    # output_dir=${HOME}/data/GVS/deploy/predictions_${year}
fi
output_dir=${HOME}/data/GVS/deploy/predictions_${year}

# config_dir=${HOME}/data/GVS/deploy/slurm_job_files
# tile_id_file=${config_dir}/deploy_s2_items_${year}_part${part}.txt
## Check if processed before submitting job
line=$(sed -n "${line_num}p" $tile_id_file)
IFS=',' read -r tile_id idx <<< "$line"
echo "Line $line_num: Tile=$tile_id, idx=$idx"
echo "Processing tile ID: $tile_id, line $line_num from $tile_id_file"



if [ -f "${HOME}/data/GVS/deploy/flags_translate_${year}/${tile_id}_done" ]; then
    echo "Tile $tile_id already translated, skip"
    exit 0
fi

inference_flag_new="${HOME}/data/GVS/deploy/flags_inference_${year}/${tile_id}_best_images_done"
translate_flag_new="${HOME}/data/GVS/deploy/flags_translate_${year}/${tile_id}_best_images_done"
if [ -f "$translate_flag_new" ]; then
    file_count=$(ls ${output_dir}/${tile_id}_cog/*.cog.tif | wc -l)
    if [ $file_count -lt 303 ]; then
        echo "Tile $tile_id has $file_count files, incomplete"
        rm -rf ${output_dir}/${tile_id}_cog
        rm ${translate_flag_new}
    else
        echo "Tile $tile_id already translated, delete input GTiff"
        rm -rf ${input_dir}/${tile_id}_GTiff
        exit 0
    fi
fi

if [ -f "$inference_flag_new" ] && [ ! -f "$translate_flag_new" ]; then
    echo "***************************** START TRANSLATE *****************************"
    echo Translate predictions for tile $tile_id in year $year;
    python -m postprocess.translate \
        src_dir=$input_dir/${tile_id}_GTiff dst_dir=${output_dir}/${tile_id}_cog \
        hydra/job_logging=disabled hydra/hydra_logging=disabled \
        hydra.run.dir=.  hydra.output_subdir=null  hydra.job.chdir=false
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Prediction command failed with exit status $exit_status"
    else
        echo "Prediction command completed successfully"
        rm -rf $input_dir/${tile_id}_GTiff
        touch ${translate_flag_new}
    fi
    echo "***************************** END TRANSLATE *****************************"
fi

# ============================ Sync data to erda ===========================
# NOTE: ERDA only allows 16 open sessions at the same time.
#       So the cog files will be saved to /scratch, and then sync to erda later in another job.


# if [ -f "$translate_flag_new" ]; then # if translate fails, it will try again
#     echo "***************************** START SYNC *****************************"
#     echo "Sync data to erda using sftp..."
#     sftp ucph-erda <<EOF
# mkdir GVS/predictions_${year}
# cd GVS/predictions_${year}
# put -r ${output_dir}/${tile_id}_cog
# EOF
#     exit_status=$?
#     if [ $exit_status -ne 0 ]; then
#         echo "Sync data to erda using sftp failed with exit status $exit_status"
#     else
#         echo "Sync data to erda using sftp completed."
#         echo "Delete local data..."
#         rm -rf ${output_dir}/${tile_id}_cog
#         echo "Delete local data completed."
#         touch ${HOME}/data/GVS/deploy/sync_flags_${year}/${tile_id}_best_images_done
#     fi
#     echo "***************************** END SYNC *****************************"
# fi



# tile_id_file=${HOME}/data/GVS/deploy/lumi_job_files/deploy_s2_items_${year}_part${part}.txt
# echo "Stream tiles listed in $tile_id_file"
# # Get tile ID from line number specified by SLURM array task ID
# while true; do
#     while IFS= read -r tile_id; do
#         inference_flag="${HOME}/data/GVS/deploy/inference_flags_${year}/${tile_id}_done"
#         inference_flag_new="${HOME}/data/GVS/deploy/inference_flags_${year}/${tile_id}_best_images_done"
#         translate_flag="${HOME}/data/GVS/deploy/translate_flags_${year}/${tile_id}_done"
#         translate_flag_new="${HOME}/data/GVS/deploy/translate_flags_${year}/${tile_id}_best_images_done"
#         if [ -f "$inference_flag" || -f "$inference_flag_new" ] && [ ! -f "$translate_flag" && ! -f "$translate_flag_new" ]; then
#             echo "***************************** START TRANSLATE *****************************"
#             echo Translate predictions for tile $tile_id in year $year;
#             python -m postprocess.translate src_dir=$input_dir/${tile_id}_GTiff dst_dir=${output_dir}/${tile_id}_cog
#             exit_status=$?
#             if [ $exit_status -ne 0 ]; then
#                 echo "Prediction command failed with exit status $exit_status"
#             else
#                 echo "Prediction command completed successfully"
#                 rm -rf $input_dir/${tile_id}_GTiff
#                 touch ${translate_flag_new}
#             fi
#             echo "***************************** END TRANSLATE *****************************"
#         fi
#         if [ -f "$translate_flag_new" ]; then # if translate fails, it will try again
#             echo "***************************** START SYNC *****************************"
#             echo "Sync data to erda using sftp..."
#             sftp ucph-erda <<EOF
# mkdir GVS/predictions_${year}
# cd GVS/predictions_${year}
# put -r ${output_dir}/${tile_id}_cog
# EOF
#             exit_status=$?
#             if [ $exit_status -ne 0 ]; then
#                 echo "Sync data to erda using sftp failed with exit status $exit_status"
#             else
#                 echo "Sync data to erda using sftp completed."
#                 echo "Delete local data..."
#                 rm -rf ${output_dir}/${tile_id}_cog
#                 echo "Delete local data completed."
#                 touch ${HOME}/data/GVS/deploy/sync_flags_${year}/${tile_id}_best_images_done
#             fi
#             echo "***************************** END SYNC *****************************"
#         fi
#     done < "$tile_id_file"  
# done
