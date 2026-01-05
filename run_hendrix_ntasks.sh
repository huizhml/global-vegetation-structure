#!/bin/bash
#SBATCH --partition=ml4good
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G # total memory for all tasks
#SBATCH --time=18-00:00:00
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

chmod +x run_hendrix_correction.sh

case $1 in
0)
# ==========================================
#   Run postprocessing with config file
# ==========================================
job_offset=${2:0}
srun run_hendrix_correction.sh 0 $job_offset
;;
1)
# ==========================================
#   Run postprocessing with list of tiles
# ==========================================
unfinished_tiles=(${2:-})
year=${3:-2020}
srun run_hendrix_correction.sh 1 "${unfinished_tiles[*]}" $year

;;

*)
echo "Invalid option"
exit 1
;;
esac
