#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
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

part=${1:-0}
year=${2:-2020}
hostname=$(hostname)
source setup_env.sh
# Define input directory
input_dir="${HOME}/data/GVS/Deploy/predictions_raw_${year}"

tile_id_file=${HOME}/data/GVS/Deploy/lumi_job_files/deploy_s2_items_${year}_part${part}.txt
echo "Stream tiles listed in $tile_id_file"
# Get tile ID from line number specified by SLURM array task ID
while true; do
    while IFS= read -r tile_id; do
        inference_flag="${HOME}/data/GVS/Deploy/inference_flags_${year}/${tile_id}_done"
        translate_flag="${HOME}/data/GVS/Deploy/translate_flags_${year}/${tile_id}_done"
        if [ -f "$inference_flag" ] && [ ! -f "$translate_flag" ]; then
            echo "***************************** START TRANSLATE *****************************"
            echo Translate predictions for tile $tile_id in year $year;
            python -m postprocess.translate src_dir=$input_dir/${tile_id}_GTiff dst_dir=${HOME}/data/GVS/Deploy/predictions_${year}/${tile_id}_cog
            exit_status=$?
            if [ $exit_status -ne 0 ]; then
                echo "Prediction command failed with exit status $exit_status"
            else
                echo "Prediction command completed successfully"
                rm -rf $input_dir/${tile_id}_GTiff
                touch ${translate_flag}
            fi
            echo "***************************** END TRANSLATE *****************************"
        fi
        if [ -f "$translate_flag" ]; then # if translate fails, it will try again
            echo "***************************** START SYNC *****************************"
            echo "Sync data to erda using sftp..."
            sftp ucph-erda <<'EOF'
mkdir predictions_${year}
cd predictions_${year}
put -r ${HOME}/data/GVS/Deploy/predictions_${year}/${tile_id}_cog/
EOF
            exit_status=$?
            if [ $exit_status -ne 0 ]; then
                echo "Sync data to erda using sftp failed with exit status $exit_status"
            else
                echo "Sync data to erda using sftp completed."
                echo "Delete local data..."
                rm -rf ${HOME}/data/GVS/Deploy/predictions_${year}/${tile_id}_cog
                echo "Delete local data completed."
                touch ${HOME}/data/GVS/Deploy/sync_flags_${year}/${tile_id}_done
            fi
            echo "***************************** END SYNC *****************************"
        fi
    done < "$tile_id_file"  
done
