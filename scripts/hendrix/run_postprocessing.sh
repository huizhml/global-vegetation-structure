#!/bin/bash
#SBATCH --partition=ml4good
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=mask
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk

source scripts/core/run_python.sh
source scripts/core/configer.sh
source scripts/core/utils.sh

case $1 in
0)
# ---------------------------------------
#    Create distance maps
# ---------------------------------------
python -m postprocessing.run run=create_distance_maps
;;

1)
# ---------------------------------------
#   Split tiles by zone (first 3 letters)
# ---------------------------------------
echo "Splitting tiles by zone (first 3 letters)...";
output_dir="${HOME}/data/gvs/assets/worklists/tiles_by_mgrs_zone"
split_tiles_by_zone $output_dir

;;

3)
# ---------------------------------------
#    Run postprocessing on Hendrix, multitasks, above bash config doesn't matter
# ---------------------------------------
job_offset=${2:0}
rhs_idx=${3:-key_rhs}
n_zones_per_task=24
year=2020
flag_dir=${year}/blended/key_rhs #TODO: make it a parameter
task_config_files=($(get_subtask_config_files $job_offset $rhs_idx $n_zones_per_task))
run_blending_loop2 $task_config_files $flag_dir $year
;;
31)

# ---------------------------------------
#   Run postprocessing with list of tiles, each job has it's own list of tiles
# ---------------------------------------
year=${2:-2020}
unfinished_tiles_file=${3:-tiles_reblend_${year}.txt}
flag_dir=${HOME}/data/gvs/state/${year}/blended/key_rhs #TODO: make it a parameter
task_tiles=($(get_subtask_tiles $unfinished_tiles_file))
run_blending_loop1 $task_tiles $flag_dir $year
;;  
32)

# ---------------------------------------
#   Run postprocessing with list of tiles, all jobs in the array job have the same list of tiles
# ---------------------------------------
echo running array job $SLURM_ARRAY_TASK_ID, task $SLURM_PROCID
year=${2:-2020}
unfinished_tiles_file=${3:-tiles_reblend_${year}_with_neighbors.txt}
flag_dir=${HOME}/data/gvs/state/${year}/blended/key_rhs #TODO: make it a parameter
# task_tiles=($(get_subtask_tiles_for_array_jobs "${unfinished_tiles[@]}"))
task_tiles=($(get_subtask_tiles_for_array_jobs $unfinished_tiles_file))
run_blending_loop1 $task_tiles $flag_dir $year
;;

4)
# ---------------------------------------
#   Mask snow and water predictions for coastal tiles, array task
# ---------------------------------------
year=${2:-2020}
tile_id_file="${HOME}/data/gvs/assets/worklists/tiles_coastal_snow_regions.txt"
n_per_task=554
array_task_id=${SLURM_ARRAY_TASK_ID:-0}
start_idx=$((array_task_id * n_per_task + 1))
end_idx=$((start_idx + n_per_task - 1))
mapfile -t tile_ids < <(sed -n "${start_idx},${end_idx}p" "$tile_id_file")
for tile_id in ${tile_ids[@]}; do
    echo "Processing tile $tile_id"
    python -m postprocessing.run run=mask_snow_water_preds run.tile_id=$tile_id
done

;;
# =======================================
#    Old scripts
# =======================================
6)
# ---------------------------------------
#    Evaluate bias correction performance
# ---------------------------------------
year=2020
split=${2:-cal}
echo "Evaluating bias correction performance for ${split} split..."
input_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_${split}/original_with_sota_chms_ours/${year}
bias_correct_dir=${HOME}/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}
python -m postprocessing.run run=evaluate_bias_correction \
    run.slope_lt20=True \
    run.year=$year \
    run.gedi_chm_ours_dir=${input_dir} \
    run.save_dir=${bias_correct_dir}/figures/${split}_slope_lt20  \
    run.correction_stats_dir=${bias_correct_dir}/stats_with_median_and_trimmed_5_95_by_tile \
;;
7)
# ---------------------------------------
#   get correction stats for all tiles 2020
# ---------------------------------------
year=2020
tiles_per_task=1547
offset=${SLURM_ARRAY_TASK_ID}
# tile_ids=($(grep '^43S' ${HOME}/data/gvs/assets/worklists/tiles_2020.txt))
all_tiles=($(cat ${HOME}/data/gvs/assets/worklists/tiles_valid_for_bc_2020.txt))
start_idx=$((offset * tiles_per_task))
tile_ids=(${all_tiles[@]:start_idx:tiles_per_task})
for tile_id in ${tile_ids[@]}; do
    echo "Processing tile $tile_id"
    python -m postprocess.bias_correction year=$year tile_id=$tile_id \
    task=get_correction_stats \
    +save_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_with_median_and_trimmed_5_95_by_tile
done
;;

*)
echo "Invalid option"
exit 1
;;
esac