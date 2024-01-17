#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small
#SBATCH --error=error.txt
#SBATCH --time=1-00:00:00
#SBATCH --output=./logs/slurm-%A-%a.out
#SBATCH --error=./logs/slurm-%A-%a.err

singularity exec -B /project/project_465000894,/scratch/project_465000894,/flash/project_465000894 ~/project/pytorch_latest.sif\
    python s2_download.py zone=$1 n_parallel=$2