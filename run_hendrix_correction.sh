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

2)

# ==========================================
#   get correction stats for all tiles 2020
# ==========================================
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
3)

# ==========================================
#   Check original mosaic
# ==========================================
year=2020

python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +save_dir=${HOME}/data/gvs/predictions/${year}/original/mosaic \
    +bias_dir=null

;;
4)

# ==========================================
#   Check mosaic after bias correction
# ==========================================
year=2020

python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +save_dir=${HOME}/data/gvs/predictions/${year}/original/mosaic \
    +bias_dir=None
;;
5)

# ==========================================
#   Check mosaic after bias correction, bias cutoff = 20
# ==========================================
year=2020
bias_cutoff=10
python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +bias_cutoff=$bias_cutoff \
    +save_dir=${HOME}/data/gvs/predictions/${year}/bias_corrected_slope_lt20_minpoints2000_bias_cutoff${bias_cutoff}/mosaic \
    +bias_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_by_tile
;;

6)

# ==========================================
#   Check mosaic after bias correction, average across RHS
# ==========================================
year=2020
python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +average_across_rhs=True \
    +save_dir=${HOME}/data/gvs/predictions/${year}/bias_corrected_slope_lt20_minpoints2000_average_across_rhs/mosaic \
    +bias_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_by_tile
;;

7)

# ==========================================
#   Check mosaic after bias correction, bias col = mean_bias_trimmed_5_95
# ==========================================
year=2020
python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +bias_col=mean_bias_trimmed_5_95 \
    +save_dir=${HOME}/data/gvs/predictions/${year}/bias_corrected_slope_lt20_minpoints2000_trimmed_5_95/mosaic \
    +bias_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_with_median_and_trimmed_5_95_by_tile
;;

8)

# ==========================================
#   Check mosaic after bias correction, bias col = median_bias
# ==========================================
year=2020
python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +bias_col=median_bias \
    +save_dir=${HOME}/data/gvs/predictions/${year}/bias_corrected_slope_lt20_minpoints2000_median_bias/mosaic \
    +bias_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_with_median_and_trimmed_5_95_by_tile
;;
9)

# ==========================================
#   Plot bias distribution
# ==========================================
year=2020
python -m postprocess.bias_correction year=$year \
    task=plot_bias_distribution \
    +bias_col=median_bias \
    +bias_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_with_median_and_trimmed_5_95_by_tile \
    +save_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/figures
;;


10)
# =======================================
#    Pair predictions with GEDI ref data
# =======================================
split=${2:-cal}
root_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_${split}
echo "Pairing predictions with GEDI ref data..."
python -m postprocess.run run=pair_ours_sota_gedi \
    run.gedi_chm_reference_dir=${root_dir}/original_with_sota_chms/2020 \
    run.save_dir=${root_dir}/original_with_sota_chms_ours/2020 || exit $?

echo sanity check for two datasets
python -m download.run run=check_two_datasets run.source_dir=${root_dir}/original_with_sota_chms/2020 \
    run.target_dir=${root_dir}/original_with_sota_chms_ours/2020 || exit $?

echo make manifest
python -m download.run run=make_manifest run.data_dir=${root_dir}/original_with_sota_chms_ours/2020 \
    run.dataset_name=gedi_${split}_2020_with_sota_chms_ours \
    run.root_note='' || exit $?
;;

11)
# =======================================
#    Run postprocessing on Hendrix, multitasks, above bash config doesn't matter
# =======================================

python -m postprocess.run run=extract_pred
;;
*)
echo "Invalid option"
exit 1
;;
esac
