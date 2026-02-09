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
input_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_${split}/original_with_sota_chms_ours/${year}
bias_correct_dir=${HOME}/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}
python -m postprocess.run run=evaluate_bias_correction \
    run.slope_lt20=True \
    run.year=$year \
    run.gedi_chm_ours_dir=${input_dir} \
    run.save_dir=${bias_correct_dir}/figures/${split}_slope_lt20  \
    run.correction_stats_dir=${bias_correct_dir}/stats_with_median_and_trimmed_5_95_by_tile \
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
#   Extract sparse predictions and add biome
# =======================================
year=2020
root_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/${year}
python -m postprocess.run run=extract_pred || exit $?
python -m download.run run=check_two_partitioned_datasets \
    run.source_dir=${root_dir}/original/2020/ \
    run.target_dir=${root_dir}/original_with_ours_biome/2020 || exit $?
python -m download.run run=make_manifest \
    run.data_dir=${root_dir}/original_with_ours_biome/2020 \
    run.dataset_name=gedi_cal_2020_with_ours_biome \
    run.root_note=''
;;

5)
# ===== Repartition Data =====
split=${2:-val}
root_dir=${HOME}/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/
log_file=${root_dir}/index_tables_by_splitted_tile/repartition_${split}.log
{
    echo repartition data for $split
    python -m postprocessing.run run=repartition_data \
        run.parquet_dir=${root_dir}/index_tables/${split} \
        run.save_dir=${root_dir}/index_tables_by_splitted_tile/${split} \
        run.based_on_col=assigned_tile \
        run._target_=datasets.repartition_data.repartition_index_table

    echo check if the repartitioned tiles are all in the $split tiles
    splitted_tiles=($(cat ${root_dir}/tiles_${split}.csv))
    repartitioned_tiles=($(for fp in ${root_dir}/index_tables_by_splitted_tile/${split}/*.parquet; do basename "${fp%.parquet}"; done))
    missing_tiles=($(comm -23 <(printf '%s\n' "${splitted_tiles[@]}" | sort -u) <(printf '%s\n' "${repartitioned_tiles[@]}" | sort -u)))
    extra_tiles=($(comm -23 <(printf '%s\n' "${repartitioned_tiles[@]}" | sort -u) <(printf '%s\n' "${splitted_tiles[@]}" | sort -u)))
    if [ ${#missing_tiles[@]} -gt 0 ]; then
        echo "Missing tiles: ${missing_tiles[@]}"
    fi
    if [ ${#extra_tiles[@]} -gt 0 ]; then
        echo "Extra tiles: ${extra_tiles[@]}"
        exit 1
    fi
    echo all tiles are found
} | tee "${log_file}"


echo sanity check for two partitioned datasets
python -m download.run run=check_two_partitioned_datasets \
    run.source_dir=${root_dir}/index_tables/${split} \
    run.target_dir=${root_dir}/index_tables_by_splitted_tile/${split} \
    run._target_=download.sanity_check.check_total_points_for_two_partitioned_datasets
;;
2)
# ===== Extract GEDI from H5  =====

split=${2:-val}
echo Extract GEDI from H5 for $split
h5_file=${HOME}/data/gvs/datasets/splits/data_${split}.h5
index_table_dir=${HOME}/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/index_tables_by_splitted_tile/${split}
target_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_${split}/original/
for year in 2019 2020 2021 2022; do
    save_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_${split}/original/${year}/
    echo Extract GEDI from H5 for $year
    python -m postprocessing.run run=extract_gedi_from_h5 run.h5_file=$h5_file run.index_table_dir=$index_table_dir run.save_dir=$save_dir run.year=$year || exit $?
done
echo sanity check for two datasets
python -m postprocessing.run run=check_two_partitioned_datasets run.source_dir=$index_table_dir \
    run.target_dir=$target_dir \
    run._target_=tools.sanity_check.check_total_points_for_two_partitioned_data_hiarchy
echo make manifest
for year in 2019 2020 2021 2022; do
    data_dir=$target_dir/$year
    python -m postprocessing.run run=make_manifest run.data_dir=$data_dir run.dataset_name=gedi_${split}_${year} run.root_note=''
done
;;
*)
echo "Invalid option"
exit 1
;;
esac
