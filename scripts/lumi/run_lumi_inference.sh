#!/bin/bash
#SBATCH --account=project_465002698
#SBATCH --partition=small-g
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --job-name=inference
#SBATCH --output=/users/zhanghui/scratch/logs/%x-%A_%a.out
#SBATCH --error=/users/zhanghui/scratch/logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu26fl
##SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu08fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch

source scripts/core/utils.sh
source scripts/lumi/utils.sh
echo_job_info
init_env

case $1 in
0)
# =======================================
#    Predict one tile in one job, get tile_id from txt file
# =======================================
line_num=${SLURM_ARRAY_TASK_ID:-0}
year=${2:-2024}
use_flash=${3:-False}
meta_file=${5:-null}

data_root_dir=$(get_data_root_dir $use_flash)
tile_list_file=${data_root_dir}/assets/worklists/tiles_repredict_${year}.txt
tile_id=$(get_tile_id_from_txt_file $line_num $tile_list_file)
save_dir=${data_root_dir}/predictions/${year}/original/tiles/
mkdir -p $save_dir
run_inference $tile_id $save_dir $meta_file $year || { exit $?; }
run_translate $tile_id $save_dir || { exit $?; }
# sync_to_lumi $tile_id $save_dir $year || { exit $?; }
exit 0
;;
*)
echo "Invalid option"
exit 1
;;
esac
