#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
#SBATCH --ntasks-per-node=60
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G # total memory for all tasks
#SBATCH --time=0-08:00:00
#SBATCH --job-name=make_public
#SBATCH --output=/users/zhanghui/scratch/logs/%x-%A_%a.out
#SBATCH --error=/users/zhanghui/scratch/logs/%x-%A_%a.err

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
## Set MIOpen cache to a temporary folder.
if [ "${SLURM_LOCALID:-0}" -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
# source setup_env.sh
# module load lumio

chmod +x postprocess/run.sh

case $1 in
0)
# ==========================================
#   Run postprocessing with config file
# ==========================================
srun postprocess/run.sh 31
;;
01)
# ==========================================
#   Run postprocessing with list of tiles, all jobs in the array job have the same list of tiles
# ==========================================
srun postprocess/run.sh 32
;;


1)
# ==========================================
#   Calculate size for a single RH - data on LUMI-O
# ==========================================
rh_idx=(0 10 20 25 30 40 50 60 70 75 80 90 95 98 100)
for rhs_idx in ${rh_idx[@]}; do
    srun postprocess/run.sh 6 0 $rhs_idx
    bash postprocess/run.sh 7 $rhs_idx
done
;;

*)
echo "Invalid option"
exit 1
;;
esac
