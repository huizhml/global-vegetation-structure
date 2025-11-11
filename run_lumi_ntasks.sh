#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=correction
#SBATCH --output=/users/zhanghui/scratch/logs/%x-%A-%t.out
#SBATCH --error=/users/zhanghui/scratch/logs/%x-%A-%t.err

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
## Set MIOpen cache to a temporary folder.
if [ "${SLURM_LOCALID:-0}" -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
source setup_env.sh
module load lumio


year=${1:-2024}
n_tiles_per_task=${2:-386}
chmod +x run_lumi_correction.sh
srun run_lumi_correction.sh $year $n_tiles_per_task