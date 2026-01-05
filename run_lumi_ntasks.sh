#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
#SBATCH --ntasks-per-node=60
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G # total memory for all tasks
#SBATCH --time=2-00:00:00
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
srun postprocess/run.sh 4
;;
1)
# ==========================================
#   Run postprocessing with list of tiles
# ==========================================
unfinished_tiles=(${2:-})
year=${3:-2020}
srun postprocess/run.sh 1 "${unfinished_tiles[*]}" $year

;;

*)
echo "Invalid option"
exit 1
;;
esac
