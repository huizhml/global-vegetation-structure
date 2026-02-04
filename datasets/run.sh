#!/bin/bash
##SBATCH --account=project_465000894
#SBATCH --partition=ml4good
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude hendrixgpu06fl
#SBATCH --time=1-23:00:00
#SBATCH --job-name=download
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
hostname


case $1 in
1)
# ===== Repartition Data =====
split=${2:-val}
root_dir=${HOME}/data/gvs/datasets/splits/split_test0.1_cal0.1_val0.1_seed42_v1/
log_file=${root_dir}/index_tables_by_splitted_tile/repartition_${split}.log
{
    echo repartition data for $split
    python -m datasets.run run=repartition_data \
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
    python -m datasets.run run=extract_gedi_from_h5 run.h5_file=$h5_file run.index_table_dir=$index_table_dir run.save_dir=$save_dir run.year=$year || exit $?
done
echo sanity check for two datasets
python -m download.run run=check_two_partitioned_datasets run.source_dir=$index_table_dir \
    run.target_dir=$target_dir \
    run._target_=download.sanity_check.check_total_points_for_two_partitioned_data_hiarchy
echo make manifest
for year in 2019 2020 2021 2022; do
    data_dir=$target_dir/$year
    python -m download.run run=make_manifest run.data_dir=$data_dir run.dataset_name=gedi_${split}_${year} run.root_note=''
done
;;
*)
echo running nothing ;;
esac

echo finished job $id