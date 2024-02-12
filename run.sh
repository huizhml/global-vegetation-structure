#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --error=error.txt
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/slurm-%A-%a.out
#SBATCH --error=./logs/slurm-%A-%a.err

# export HYDRA_FULL_ERROR=1
# zone=$1
# partition_size=${5:-128K}
# echo $zone $partition_size

export KEY_FILE=keys/venky-key.json
singularity run -B /project/project_465000894,/scratch/project_465000894,/flash/project_465000894 ~/project/pytorch_latest.sif \
            python -u -m download.s2_download_config_gen year=$1