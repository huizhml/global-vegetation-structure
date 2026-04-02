#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=extract_pred
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

source scripts/core/utils.sh
source scripts/core/run_python.sh

case $1 in
# =======================================
#   Run evaluation
# =======================================
0)
# ---------------------------------------
#   Run evaluation
# ---------------------------------------
echo "Running evaluation"
;;

# =======================================
#    Diversity indices
# =======================================
4)
# ---------------------------------------
#   Calculate diversity indices
# ---------------------------------------
echo "Calculating diversity indices"
bin_width=${2:-1}
gedi_ours_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/original_with_sota_chms_biome_and_ours_full/2020
save_dir=${HOME}/data/gvs/evaluation/with_gedi_on_diversity_indices/indices_by_tile/
python -m evaluation.run run=compute_diversity_indices run.gedi_ours_dir=${gedi_ours_dir} run.save_dir=${save_dir} run.bin_width=${bin_width} || exit $?
run_sanity_check check_two_datasets $gedi_ours_dir $save_dir/bin_width_${bin_width}m/2020 || exit $?
make_manifest $save_dir gedi_test_2020_diversity_indices_bin_width_${bin_width}m ''
;;

41)
# ---------------------------------------
#   Evaluate diversity indices
# ---------------------------------------
echo "Evaluating diversity indices"
indices_dir=${HOME}/data/gvs/evaluation/with_gedi_on_diversity_indices/indices_by_tile
for bin_width in 1 2 3 4 5 6 7 8 9 10; do
    python -m evaluation.run run=evaluate_diversity_indices run.indices_dir=${indices_dir}/bin_width_${bin_width}m/2020 || exit $?
    python -m evaluation.run run=evaluate_diversity_indices run.indices_dir=${indices_dir}/bin_width_${bin_width}m/2020 run.group_by=null || exit $?
done
;;

# =======================================
#    Preparing data for evaluation
# =======================================

5)
# ---------------------------------------
#   Extract sparse predictions and add biome
# ---------------------------------------
year=2020
split=${2:-cal}
root_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_${split}/
save_dir=${root_dir}/original_with_ours_blended_biome/${year}
python -m postprocessing.run run=extract_pred run.save_dir=${save_dir} || exit $?

run_sanity_check check_two_partitioned_datasets $root_dir/original/${year}/ $save_dir || exit $?
make_manifest $save_dir gedi_${split}_${year}_with_ours_blended_biome ''
;;

50)
# ----------------------------------------
#    Evaluate with SOTA: add ours to the table with GEDI and SOTA CHMs, test set
# ----------------------------------------
root_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/
python -m postprocessing.run run=add_biome || exit $?
run_sanity_check check_two_datasets $root_dir/original_with_sota_chms/2020 $root_dir/original_with_sota_chms_biome/2020 || exit $?
make_manifest $root_dir/original_with_sota_chms_biome/2020 gedi_test_2020_with_sota_chms_biome ''
;;


51)
# ----------------------------------------
#    Evaluate with SOTA: add ours to the table with GEDI and SOTA CHMs, test set
# ----------------------------------------
root_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/
python -m postprocessing.run run=add_ours_blended_to_sota_gedi || exit $?
run_sanity_check check_two_datasets $root_dir/original_with_sota_chms/2020 $root_dir/original_with_sota_chms_ours_blended/2020 || exit $?
make_manifest $root_dir/original_with_sota_chms_ours_blended/2020 gedi_test_2020_with_sota_chms_ours_blended ''
;;

52)
# ----------------------------------------
#    Evaluate: add ours to the table with GEDI and SOTA CHMs, test set, full predictions
# ----------------------------------------
split=${2:-test}
root_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_${split}/
gedi_chm_reference_dir=${root_dir}/original_with_sota_chms_biome/2020
save_dir=${root_dir}/original_with_sota_chms_biome_and_ours_full/2020
python -m postprocessing.run run=add_ours_to_sota_gedi run.gedi_chm_reference_dir=${gedi_chm_reference_dir} run.save_dir=${save_dir} || exit $?
run_sanity_check check_two_datasets $root_dir/original_with_sota_chms_biome/2020 $save_dir || exit $?
make_manifest $save_dir gedi_${split}_2020_with_sota_chms_biome_and_ours_full ''
;;


53)
# ---------------------------------------
#    Pair predictions with GEDI ref data
# ---------------------------------------
split=${2:-cal}
root_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_${split}
python -m postprocessing.run run=pair_ours_sota_gedi \
    run.gedi_chm_reference_dir=${root_dir}/original_with_sota_chms/2020 \
    run.save_dir=${root_dir}/original_with_sota_chms_ours/2020 || exit $?

run_sanity_check check_two_datasets $root_dir/original_with_sota_chms/2020 $root_dir/original_with_sota_chms_ours/2020 || exit $?
make_manifest $root_dir/original_with_sota_chms_ours/2020 gedi_${split}_2020_with_sota_chms_ours ''
;;

6)
# ---------------------------------------
#   Repartition Data
# ---------------------------------------
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

run_sanity_check check_two_partitioned_datasets $root_dir/index_tables/${split} $root_dir/index_tables_by_splitted_tile/${split}
;;

7)
# ---------------------------------------
#   Extract GEDI from H5 
# ---------------------------------------

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

run_sanity_check check_two_partitioned_datasets $index_table_dir $target_dir

# make manifest
for year in 2019 2020 2021 2022; do
    make_manifest $target_dir/$year gedi_${split}_${year} ''
done
;;

*)
echo "Invalid option"
exit 1
;;
esac