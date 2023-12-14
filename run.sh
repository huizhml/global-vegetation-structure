#!/bin/bash
#SBATCH --job-name=download_s2
#SBATCH --error=error.txt
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=./logs/slurm-%A-%a.out
#SBATCH --error=./logs/slurm-%A-%a.err

# python run.py keyFile=private-key.json init.dataFolder=GEDI2019
source activate mpc
python s2_download.py zone=01G partition_size=100

