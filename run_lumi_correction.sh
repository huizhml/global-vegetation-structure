#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
zone_name=$(basename "${zone}" .txt | tr '[:upper:]' '[:lower:]')
echo "Making bucket public for zone ${zone}..."
module load lumio-ext-tools/1.0.0
s3cmd setacl --recursive --acl-public s3://${zone_name}-${year}/predictions_GTiff_${year}

for tile_id in $(cat $tile_id_file); do
    echo "Processing tile $tile_id"
    python -m postprocess.handle_border_artifacts year=$year task=run_correction_and_blending tile_id=$tile_id use_flash=$use_flash rhs_idx=all_rhs \
        hydra/job_logging=disabled hydra/hydra_logging=disabled \
        hydra.run.dir=.  hydra.output_subdir=null  hydra.job.chdir=false
done

