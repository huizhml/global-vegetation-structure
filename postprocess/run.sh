#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=make_public
#SBATCH --output=/users/zhanghui/scratch/logs/%x-%A_%a.out
#SBATCH --error=/users/zhanghui/scratch/logs/%x-%A_%a.err

case $1 in

1)
# =======================================
#    SPLIT TILES BY ZONE FOR POSTPROCESSING
# =======================================
echo "Splitting tiles by zone (first 3 letters)...";
output_dir="${HOME}/data/gvs/assets/worklists/tiles_by_mgrs_zone"
mkdir -p "$output_dir"

# Clear existing zone files
rm -f "$output_dir"/*.txt

# Get all tiles and split by zone
for tile in $(ls ${HOME}/data/gvs/predictions/2024/original/tiles/geotiff); do
    zone="${tile:0:3}"  # Extract first 3 letters
    echo "$tile" >> "$output_dir/${zone}.txt"
done

echo "Done! Tiles grouped by zone in $output_dir"
ls -lh "$output_dir"

max=0
for zone in $(ls $output_dir); do
    echo "Zone: $zone"
    tiles=$(cat $output_dir/$zone)
    echo "Number of tiles: $(wc -l < $output_dir/$zone)"
    echo "--------------------------------"
    if [ $(wc -l < $output_dir/$zone) -gt $max ]; then
        max=$(wc -l < $output_dir/$zone)
    fi
done
echo "Max number of tiles: $max"
;;

2) 
# =======================================
#    Evaluate bias correction performance
# =======================================
year=2020
split=${2:-cal}
echo "Evaluating bias correction performance for ${split} split..."
root_dir=${HOME}/data/gvs/
python -m postprocess.run run=evaluate_bias_correction \
    run.slope_lt20=True \
    run.year=$year \
    run.gedi_chm_ours_dir=${root_dir}/gedi/veg_sensitivity_gt0p95/subset_${split}/original_with_sota_chms_ours/${year} \
    run.save_dir=${root_dir}/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/figures/${split}_slope_lt20  \
    run.correction_stats_dir=${root_dir}/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_with_median_and_trimmed_5_95_by_tile \
;;
3)
# =======================================
#    Run postprocessing on Hendrix, multitasks, above bash config doesn't matter
# =======================================
job_offset=${2:0}
rhs_idx=${3:-key_rhs}
n_zones_per_task=24
n_zones=$((SLURM_NTASKS * n_zones_per_task))
all_zones=($(ls ${HOME}/data/gvs/assets/worklists/tiles_by_mgrs_zone/*.txt | sort))
job_zones=(${all_zones[@]:job_offset:n_zones})
start_idx=$((SLURM_PROCID * n_zones_per_task))
if [ $SLURM_PROCID -eq $((SLURM_NTASKS - 1)) ]; then
    task_zones=(${job_zones[@]:start_idx}) # take the rest of the zones
else
    task_zones=(${job_zones[@]:start_idx:n_zones_per_task})
fi
year=2020
for zone in ${task_zones[@]}; do
    for tile_id in $(cat $zone); do
        echo "Processing tile $tile_id"
        python -m postprocess.run run=run_blending \
            run.year=$year \
            run.tile_id=$tile_id \
            run.flag_dir=${HOME}/data/gvs/state/${year}/blended/ \
            run.output_dir=${HOME}/data/gvs/predictions/2020/blended/tiles \
            run.total_tiles_file=${HOME}/data/gvs/assets/worklists/total_tiles_2020.txt
    done
done
;;

4)
# =======================================
#    Run postprocessing on Hendrix, multitasks, above bash config doesn't matter
# =======================================
root_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/
python -m postprocess.run run=extract_pred || exit $?
python -m download.run run=check_two_partitioned_datasets \
    run.source_dir=${root_dir}/original/2020/ \
    run.target_dir=${root_dir}/original_with_ours_biome/2020 || exit $?
python -m download.run run=make_manifest \
    run.data_dir=${root_dir}/original_with_ours_biome/2020 \
    run.dataset_name=gedi_cal_2020_with_ours_biome \
    run.root_note=''
;;


*)
echo "Invalid option"
exit 1
;;
esac
