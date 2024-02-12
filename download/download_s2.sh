#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --error=error.txt
#SBATCH --time=3-00:00:00
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

export HYDRA_FULL_ERROR=1
zone=$1
year=$3

n_parallel=${2:-100}
echo download zone $zone n_parallel=$n_parallel
singularity exec -B /project/project_465000894,/scratch/project_465000894,/flash/project_465000894 ~/project/pytorch_latest.sif\
    python -u -m download.s2 zone=$zone n_parallel=$n_parallel year=$year
echo 'done'