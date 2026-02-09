#!/bin/bash
##SBATCH --account=project_465000894
#SBATCH --partition=ml4good
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude hendrixgpu06fl
#SBATCH --time=1-23:00:00
#SBATCH --job-name=download
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
hostname


case $1 in
1)
echo ...
*)
echo running nothing ;;
esac

echo finished job $id