#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=extract_pred
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

source scripts/core/utils.sh
source scripts/core/run_python.sh

case $1 in
0)
# ---------------------------------------
#   Calculate profile entropy for DK
# ---------------------------------------
echo "Calculating profile entropy for DK"
tile_id_file=${HOME}/data/gvs/assets/worklists/dk.txt
tile_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $tile_id_file)
echo "Processing tile ID: $tile_id"

output_dir=${HOME}/data/gvs/products/profile_entropy/2020/tiles/geotiff
# python -m evaluation.run run=compute_entropy run.output_dir=$output_dir run.tile_id=$tile_id run.year=2020

;;


# =======================================
#   Biome analysis
# =======================================
1)
# ---------------------------------------
#   Extract predictions by tile, for biome analysis
# ---------------------------------------
echo "Extracting predictions by tile"
loc_dir=/projects/dereeco/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome_by_tile/
save_dir=/projects/dereeco/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome_preds
python -m postprocessing.run run=extract_pred \
    run.loc_dir=${loc_dir} \
    run.save_dir=${save_dir}
run_sanity_check check_two_datasets ${loc_dir} ${save_dir} || exit $?

;;

# =======================================
#   Naturalness analysis
# =======================================
2.0)
# ---------------------------------------
#   Partition naturalness locations by tile
# ---------------------------------------
echo "Partitioning naturalness locations by tile"
naturalness_csv=${HOME}/data/gvs/downstream_tasks/naturalness/reference_data_set_updated_val.csv
save_dir=${HOME}/data/gvs/downstream_tasks/naturalness/loc_by_tile_val
python -m evaluation.run run=prepare_naturalness_loc_parquets run.naturalness_csv=${naturalness_csv} run.save_dir=${save_dir} || exit $?
# run_sanity_check check_two_datasets ${gedi_ref_dir} ${save_dir} || exit $?

;;

2.1)
# ---------------------------------------
#   Sample VSM patches
# ---------------------------------------
echo "Sampling VSM patches"
split=${2:-val}
loc_dir=${HOME}/data/gvs/downstream_tasks/naturalness/loc_by_tile_${split}
save_dir=${HOME}/data/gvs/downstream_tasks/naturalness/vsm_patches_ps11_${split}
python -m postprocessing.run run=sample_vsm_patches run.loc_dir=${loc_dir} run.save_dir=${save_dir}  || exit $?
# run_sanity_check check_two_datasets ${gedi_ref_dir} ${save_dir} || exit $?
;;

2.2)
# ---------------------------------------
#   Calculate VSM patch statistics
# ---------------------------------------
echo "Calculating VSM patch statistics"
split=${2:-val}
vsm_patches_dir=${HOME}/data/gvs/downstream_tasks/naturalness/vsm_patches_ps11_${split}
save_dir=${HOME}/data/gvs/downstream_tasks/naturalness/vsm_patch_stats_ps11_${split}
python -m evaluation.run run=cal_vsm_patch_stats run.vsm_patches_dir=${vsm_patches_dir} run.save_dir=${save_dir}  || exit $?
# run_sanity_check check_two_datasets ${gedi_ref_dir} ${save_dir} || exit $?
;;


2.3)
# ---------------------------------------
#   Sample VSM points
# ---------------------------------------
echo "Sampling VSM points"
split=${2:-test}
python -m postprocessing.run run=sample_vsm_points run.split=${split}
# run_sanity_check check_two_datasets ${gedi_ref_dir} ${save_dir} || exit $?
;;

*)
echo "Invalid option"
exit 1
;;
esac