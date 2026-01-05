#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=make_public
#SBATCH --output=/users/zhanghui/scratch/logs/%x-%A_%a.out
#SBATCH --error=/users/zhanghui/scratch/logs/%x-%A_%a.err

case $1 in

1)
# =======================================
#    SPLIT TILES BY ZONE FOR CORRECTION
# =======================================
echo "Splitting tiles by zone (first 3 letters)...";
output_dir="${HOME}/data/gvs/deploy/tiles_by_zone_for_postprocess"
mkdir -p "$output_dir"

# Clear existing zone files
rm -f "$output_dir"/*.txt

# Get all tiles and split by zone
for tile in $(ls ${HOME}/data/gvs/deploy/predictions_gtiff_2024); do
    zone="${tile:0:3}"  # Extract first 3 letters
    echo "$tile" >> "$output_dir/${zone}.txt"
done

echo "Done! Tiles grouped by zone in $output_dir"
ls -lh "$output_dir"

max=0
for zone in $(ls $output_dir); do
    echo "Zone: $zone"
    tiles=$(cat $output_dir/$zone)
    echo "Number of tiles: $(wc -l < $output_dir/$zone)"
    echo "--------------------------------"
    if [ $(wc -l < $output_dir/$zone) -gt $max ]; then
        max=$(wc -l < $output_dir/$zone)
    fi
done
echo "Max number of tiles: $max"
;;

2)
# =======================================
#    MAKE BUCKET PUBLIC - finished tiles
# =======================================
echo "Making bucket public..."
module load lumio-ext-tools/1.0.0
year=2024
finished_tiles=($(find ${HOME}/data/gvs/deploy/flags_postprocess_${year} -type f -name "*_done" -exec basename {} \; | sed 's/_done//'))
zones=($(for tile in "${finished_tiles[@]}"; do echo "${tile:0:3}" | tr '[:upper:]' '[:lower:]'; done | sort | uniq))
for zone in ${zones[@]}; do
    s3cmd setacl --recursive --acl-public s3://${zone}-${year}
done

;;

3)
# =======================================
#    MAKE BUCKET PUBLIC - all geotiff files
# =======================================
echo "Making bucket public..."
module load lumio-ext-tools/1.0.0
year=2024
zones=($(ls ${HOME}/data/gvs/deploy/tiles_by_zone_for_postprocess))
for zone in ${zones[@]}; do
    zone_name=$(basename "${zone}" .txt | tr '[:upper:]' '[:lower:]')
    echo "Making bucket public for zone: ${zone_name}"
    s3cmd setacl --recursive --acl-public s3://${zone_name}-${year}/predictions_GTiff_${year}
    echo "Done!"
done

;;

4)
# =======================================
#    MAKE BUCKET PUBLIC - all COG files
# =======================================
echo "Making bucket public..."
module load lumio-ext-tools/1.0.0
year=2024
job_offset=${2:0}
n_zones_per_task=10
n_zones=$((SLURM_NTASKS * n_zones_per_task))
all_zones=($(ls ${HOME}/data/gvs/deploy/tiles_by_zone_for_postprocess/*.txt | sort))
job_zones=(${all_zones[@]:$job_offset:n_zones})
start_idx=$((SLURM_PROCID * n_zones_per_task))
if [ $SLURM_PROCID -eq $((SLURM_NTASKS - 1)) ]; then
    task_zones=(${job_zones[@]:start_idx}) # take the rest of the zones
else
    task_zones=(${job_zones[@]:start_idx:n_zones_per_task})
fi
flag_dir=${HOME}/data/gvs/deploy/tiles_by_zone_for_postprocess
for zone in ${task_zones[@]}; do
    zone_name=$(basename "${zone}" .txt | tr '[:upper:]' '[:lower:]')
    tile_ids=$(cat $zone)
    for tile_id in ${tile_ids[@]}; do
        echo "Making bucket public for tile: s3://${zone_name}-${year}/${tile_id}"
        s3cmd setacl --recursive --acl-public s3://${zone_name}-${year}/${tile_id}
        echo "Done!"
    done
done
;;

*)
echo "Invalid option"
exit 1
;;
esac