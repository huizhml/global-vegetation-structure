#!/bin/bash
#SBATCH --partition=ml4good
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G # total memory for all tasks
#SBATCH --time=3-00:00:00
#SBATCH --job-name=correction
#SBATCH --output=./logs/%x-%A-%t.out
#SBATCH --nodelist=hendrixgpu26fl
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
## Set MIOpen cache to a temporary folder.
if [ "${SLURM_LOCALID:-0}" -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
# source setup_env.sh
# module load lumio
job_offset=${1:0}
chmod +x run_hendrix_correction.sh
srun run_hendrix_correction.sh $job_offset

