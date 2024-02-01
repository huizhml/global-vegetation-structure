#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=GEDI-download
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

export KEY_FILE=$2
singularity run -B /project/project_465000894,/scratch/project_465000894,/flash/project_465000894 ~/project/pytorch_latest.sif \
            python -u -m download.gedi init.year=$1