#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=correction
#SBATCH --output=./logs/%x-%A-%t.out
#SBATCH --error=./logs/%x-%A-%t.err


year=${1:-2020}
n_tiles_per_task=${2:-1031}
chmod +x run_hendrix.sh
srun run_hendrix.sh $year $n_tiles_per_task