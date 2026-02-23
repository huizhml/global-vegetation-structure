#!/bin/bash
#SBATCH --partition=ml4good
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G # total memory for all tasks for --mem
#SBATCH --time=7-00:00:00
#SBATCH --job-name=blending
#SBATCH --output=./logs/%x-%A-%a.out
##SBATCH --nodelist=hendrixgpu26fl
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

chmod +x postprocessing/run.sh

case $1 in
0)
# ==========================================
#   Run postprocessing with config files
# ==========================================
job_offset=${2:0}
rhs_idx=${3:-key_rhs}
srun postprocessing/run.sh 3 $job_offset $rhs_idx
;;
1)
# ==========================================
#   Run postprocessing with list of tiles
# ==========================================
year=${2:-2020}
srun postprocessing/run.sh 4 $year
;;
2)
# ==========================================
#   Run postprocessing with list of tiles, all jobs in the array job have the same list of tiles
# ==========================================
srun postprocessing/run.sh 32

;;

*)
echo "Invalid option"
exit 1
;;
esac
