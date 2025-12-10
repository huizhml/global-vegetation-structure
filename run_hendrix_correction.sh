#!/bin/bash
# SBATCH --partition=ml4good
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=16
# SBATCH --mem=48G
# SBATCH --time=3-00:00:00
# SBATCH --job-name=correction
# SBATCH --output=./logs/%x-%A_%a.out
# SBATCH --mail-type=END,FAIL
# SBATCH --mail-user=huzh@di.ku.dk


job_offset=${1:0}
year=${2:-2020}
# zone=${2:-20M.txt}
n_zones_per_task=24
n_zones=$((SLURM_NTASKS * n_zones_per_task))
all_zones=($(ls ~/data/gvs/deploy/tiles_by_zone_for_postprocess/*.txt | sort))

job_zones=(${all_zones[@]:job_offset:n_zones})
start_idx=$((SLURM_PROCID * n_zones_per_task))
end_idx=$((start_idx + n_zones_per_task - 1))
task_zones=(${job_zones[@]:start_idx:end_idx})


for tile_id_file in ${task_zones[@]}; do

    for tile_id in $(cat $tile_id_file); do
        echo "Processing tile $tile_id"
        python -m postprocess.handle_border_artifacts year=$year \
            corrected_pred_dir=${HOME}/data/gvs/deploy/predictions_corrected_blended_$year \
            correction_stats_dir=${HOME}/data/gvs/deploy/correction/tile_stats_$year \
            task=run_correction_and_blending tile_id=$tile_id
    done
done