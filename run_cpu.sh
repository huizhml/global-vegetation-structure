#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24-00:00:00
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
if [ "$hostname" == "hendrixgpu26fl.unicph.domain" ]; then
    input_dir=~/data/GVS/Deploy/predictions_GTiff_${year}
    echo "Host is hendrixgpu26fl (disk dead). Translate from $input_dir"
else
    input_dir=/scratch/predictions_${year}
    echo "Host is $hostname. Translate from $input_dir"
fi

input_dir=/scratch/predictions_${year}

tile_id_file=${HOME}/data/GVS/Deploy/deploy_s2_items_${year}_part${part}.txt
echo "Translate tiles from $tile_id_file"
# Get tile ID from line number specified by SLURM array task ID
while true; do
    while IFS= read -r tile_id; do
        inference_flag="${HOME}/data/GVS/Deploy/inference_flags_${year}/${tile_id}_done"
        translate_flag="${HOME}/data/GVS/Deploy/translate_flags_${year}/${tile_id}_done"
        if [ -f "$inference_flag" ] && [ ! -f "$translate_flag" ]; then
            echo "***************************** START INFERENCE *****************************"
            echo Translate predictions for tile $tile_id in year $year;
            python -m postprocess.translate src_dir=$input_dir/${tile_id}_GTiff dst_dir=${HOME}/data/GVS/Deploy/predictions_${year}/${tile_id}_cog
            exit_status=$?
            if [ $exit_status -ne 0 ]; then
                echo "Prediction command failed with exit status $exit_status"
            else
                echo "Prediction command completed successfully"
                rm -rf $input_dir/${tile_id}_GTiff
            fi
            echo "***************************** END INFERENCE *****************************"
            touch ${translate_flag}
        fi
    done < "$tile_id_file"  
done
echo "***************************** END TRANSLATE *****************************"