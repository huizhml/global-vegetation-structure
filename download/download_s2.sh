#!/bin/bash
#SBATCH --account=project_465000894
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --error=error.txt
#SBATCH --time=1-00:00:00
#SBATCH --output=./logs/slurm-%A-%a.out
#SBATCH --error=./logs/slurm-%A-%a.err

export HYDRA_FULL_ERROR=1
zone=$1
partition_size=${5:-128K}
echo $zone $partition_size

test -f ~/GEDI2019/$zone/partition_0.parquet || { echo 'partition_0.parquet not exist, generating...'; \
singularity exec -B /project/project_465000894,/scratch/project_465000894,/flash/project_465000894 \
~/project/pytorch_latest.sif python -m download.gedi_post_proc zone=$zone partition_size=$partition_size
}

n_parallel=${2:-100}
echo download zone $zone n_parallel=$n_parallel
singularity exec -B /project/project_465000894,/scratch/project_465000894,/flash/project_465000894 ~/project/pytorch_latest.sif\
    python s2_download.py zone=$zone n_parallel=$n_parallel
echo 'done'

singularity run -B /project/project_465000894,/scratch/project_465000894,/flash/project_465000894 ~/project/pytorch_latest.sif python -m download.gedi init.year=$1