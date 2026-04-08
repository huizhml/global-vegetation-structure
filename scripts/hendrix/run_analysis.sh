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
2)
# ---------------------------------------
#   Sample VSM patches
# ---------------------------------------
echo "Sampling VSM patches"
python -m postprocessing.run run=sample_vsm_patches
# run_sanity_check check_two_datasets ${gedi_ref_dir} ${save_dir} || exit $?

;;

*)
echo "Invalid option"
exit 1
;;
esac