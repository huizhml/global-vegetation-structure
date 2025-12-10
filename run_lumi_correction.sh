#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=correction
#SBATCH --output=/users/zhanghui/scratch/logs/%x-%A_%a.out
#SBATCH --error=/users/zhanghui/scratch/logs/%x-%A_%a.err

source setup_env.sh
module load lumio

year=${1:-2024}
zone=${2:-01G.txt}
use_flash=${3:-False} #TODO: not used yet
tile_id_file=${HOME}/data/gvs/deploy/tiles_by_zone_for_postprocess/${zone}

for tile_id in $(cat $tile_id_file); do
    echo "Processing tile $tile_id"
    python -m postprocess.handle_border_artifacts year=$year task=run_correction_and_blending tile_id=$tile_id
done

