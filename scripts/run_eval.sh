#!/bin/bash
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=visualize
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu26fl

source scripts/core/utils.sh
get_options $1

case $1 in
1)
# ----------------------------------------
#    Evaluate with SOTA: add ours to the table with GEDI and SOTA CHMs, test set
# ----------------------------------------
root_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_test/
python -m postprocessing.run run=add_ours_blended_to_sota_gedi || exit $?
python -m download.run run=check_two_datasets \
    run.source_dir=${root_dir}/original_with_sota_chms/2020 \
    run.target_dir=${root_dir}/original_with_sota_chms_ours_blended/2020 || exit $?
python -m download.run run=make_manifest \
    run.data_dir=${root_dir}/original_with_sota_chms_ours_blended/2020 \
    run.dataset_name=gedi_test_2020_with_sota_chms_ours_blended \
    run.root_note='' || exit $?
;;
*)
echo "Invalid option"
exit 1
;;
esac