#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=GEDI-download
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

singularity run -B /project/project_465000894,/scratch/project_465000894,/flash/project_465000894 ~/project/pytorch_latest.sif python -m -u download.gedi init.year=$1