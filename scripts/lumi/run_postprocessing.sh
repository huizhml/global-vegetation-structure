#!/bin/bash
#SBATCH --account=project_465002698
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=blending
#SBATCH --output=/users/zhanghui/scratch/logs/%x-%A_%a.out
#SBATCH --error=/users/zhanghui/scratch/logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk

source scripts/core/run_python.sh
source scripts/core/utils.sh
source scripts/lumi/env_init.sh

case $1 in
0)
# ---------------------------------------
#    Run blending
#  NOTE: array job has to start from 0
#       otherwise, the first n_tiles_per_task tiles will be skipped
# ---------------------------------------
rhs_idx=${3:-'all_rhs'}
year=2024
unfinished_tiles_file=total_tiles_${year}.txt
run_blending_loop1 $year $rhs_idx $unfinished_tiles_file
;;


# =======================================
#    Prepare data for postprocessing
# =======================================
1)
# ---------------------------------------
#   Split tiles by zone (first 3 letters)
# ---------------------------------------
;;
*)
echo "Invalid option"
exit 1
;;
esac