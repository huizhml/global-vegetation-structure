#!/bin/bash
#SBATCH --partition=ml4good
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=correction
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk


case $1 in
0)
# ==========================================
#   Run postprocessing with config file
# ==========================================
job_offset=${2:0}
year=${3:-2020}
rhs_idx=${4:-rest_rhs}
# zone=${2:-20M.txt}
n_zones_per_task=24
n_zones=$((SLURM_NTASKS * n_zones_per_task))
all_zones=($(ls ~/data/gvs/deploy/tiles_by_zone_for_postprocess/*.txt | sort))

job_zones=(${all_zones[@]:job_offset:n_zones})
start_idx=$((SLURM_PROCID * n_zones_per_task))
if [ $SLURM_PROCID -eq $((SLURM_NTASKS - 1)) ]; then
    task_zones=(${job_zones[@]:start_idx}) # take the rest of the zones
else
    task_zones=(${job_zones[@]:start_idx:n_zones_per_task})
fi


for tile_id_file in ${task_zones[@]}; do
    for tile_id in $(cat $tile_id_file); do
        echo "Processing tile $tile_id"
        python -m postprocess.handle_border_artifacts year=$year \
            corrected_pred_dir=${HOME}/data/gvs/deploy/predictions_corrected_blended_v1_$year \
            correction_stats_dir=${HOME}/data/gvs/deploy/correction/tile_stats_$year \
            flag_dir=${HOME}/data/gvs/deploy/flags_postprocess_${year}${rhs_idx} \
            task=run_correction_and_blending tile_id=$tile_id rhs_idx=$rhs_idx
    done
done
;;

1)
# ==========================================
#   Run postprocessing with list of tiles
# ==========================================
# unfinished_tiles=(${2:-})
unfinished_tiles=('40QBE') #(${2:-})
year=${3:-2020}
rhs_idx=${4:-rest_rhs}
n_tiles=${#unfinished_tiles[@]}
echo "n_tiles: $n_tiles"
n_tiles_per_task=$((n_tiles / SLURM_NTASKS))
echo "n_tiles_per_task: $n_tiles_per_task"

start_idx=$((SLURM_PROCID * n_tiles_per_task))
if [ $SLURM_PROCID -eq $((SLURM_NTASKS - 1)) ]; then
    task_tiles=(${unfinished_tiles[@]:start_idx}) # take the rest of the tiles
else
    task_tiles=(${unfinished_tiles[@]:start_idx:n_tiles_per_task}) # take the next n_tiles_per_task tiles
fi
echo "task_tiles: ${#task_tiles[@]}"
echo ${task_tiles[@]}
for tile_id in ${task_tiles[@]}; do
    echo "Processing tile $tile_id"
    python -m postprocess.handle_border_artifacts year=$year \
        corrected_pred_dir=${HOME}/data/gvs/deploy/predictions_corrected_blended_v1_$year \
        correction_stats_dir=${HOME}/data/gvs/deploy/correction/tile_stats_$year \
        flag_dir=${HOME}/data/gvs/deploy/flags_postprocess_${year}${rhs_idx} \
        task=run_correction_and_blending tile_id=$tile_id rhs_idx=$rhs_idx
done
;;

*)
echo "Invalid option"
exit 1
;;
esac