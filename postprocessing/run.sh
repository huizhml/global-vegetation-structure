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

run_blending() {
    local year=$1
    local tile_id=$2
    python -m postprocessing.run run=run_blending \
        run.year=$year \
        run.tile_id=$tile_id \
        run.flag_dir=${HOME}/data/gvs/state/${year}/blended/ \
        run.output_dir=${HOME}/data/gvs/predictions/2020/blended/tiles \
        run.total_tiles_file=${HOME}/data/gvs/assets/worklists/total_tiles_2020.txt
}

get_subtask_config_files() {
    # get the list of config files for the current task when running ntasks in one slurm job
    local job_offset=${1:0}
    local rhs_idx=${2:-key_rhs}
    local n_zones_per_task=${3:-24}
    local task_zones=()

    n_zones=$((SLURM_NTASKS * n_zones_per_task))
    all_zones=($(ls ${HOME}/data/gvs/assets/worklists/tiles_by_mgrs_zone/*.txt | sort))
    job_zones=(${all_zones[@]:job_offset:n_zones})
    start_idx=$((SLURM_PROCID * n_zones_per_task))
    if [ $SLURM_PROCID -eq $((SLURM_NTASKS - 1)) ]; then
        task_zones=(${job_zones[@]:start_idx}) # take the rest of the zones
    else
        task_zones=(${job_zones[@]:start_idx:n_zones_per_task})
    fi
    echo ${task_zones[@]}
}

get_subtask_tiles() {
    # get the list of tiles for the current task when running ntasks in one slurm job
    local unfinished_tiles=(${1:-})
    local task_tiles=()
    n_tiles=${#unfinished_tiles[@]}
    n_tiles_per_task=$((n_tiles / SLURM_NTASKS))

    start_idx=$((SLURM_PROCID * n_tiles_per_task))
    if [ $SLURM_PROCID -eq $((SLURM_NTASKS - 1)) ]; then
        task_tiles=(${unfinished_tiles[@]:start_idx}) # take the rest of the tiles
    else
        task_tiles=(${unfinished_tiles[@]:start_idx:n_tiles_per_task}) # take the next n_tiles_per_task tiles
    fi
    echo ${task_tiles[@]}
}

check_if_processed() {
    local tile_id=$1
    local flag_dir=$2
    local rewrite_flag=$3
    if [ ! $rewrite_flag ]; then
        if [ -f "${flag_dir}/${tile_id}_done" ]; then
            return 0
        else
            return 1
        fi
    else
        rm -f "${flag_dir}/${tile_id}_done"
        return 1
    fi
}


case $1 in
0)
# =======================================
#    Create distance maps
# =======================================
python -m postprocessing.run run=create_distance_maps
;;


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
python -m postprocessing.run run=evaluate_bias_correction \
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
year=2020
flag_dir=${HOME}/data/gvs/state/${year}/blended/key_rhs #TODO: make it a parameter
task_config_files=($(get_subtask_config_files $job_offset $rhs_idx $n_zones_per_task))

for config_file in ${task_config_files[@]}; do
    for tile_id in $(cat $config_file); do
        processed=$(check_if_processed $tile_id $flag_dir)
        if [ $processed -eq 0 ]; then
            echo "Tile $tile_id already processed, skipping"
            continue
        fi
        echo "Processing tile $tile_id"
        run_blending $year $tile_id
    done
done
;;
4)

# ==========================================
#   Run postprocessing with list of tiles
# ==========================================
year=${2:-2020}
unfinished_tiles_file=${3:-${HOME}/data/gvs/assets/worklists/tiles_reblend_${year}.txt}
unfinished_tiles=($(cat $unfinished_tiles_file))
flag_dir=${HOME}/data/gvs/state/${year}/blended/key_rhs #TODO: make it a parameter
task_tiles=($(get_subtask_tiles $unfinished_tiles))
for tile_id in ${task_tiles[@]}; do
    processed=$(check_if_processed $tile_id $flag_dir true)
    if [ $processed -eq 0 ]; then # this is also checked inside the python script
        echo "Tile $tile_id already processed, skipping"
        continue
    fi
    echo "Processing tile $tile_id"
    run_blending $year $tile_id
done
;;

5)
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

6)
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
7)
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
