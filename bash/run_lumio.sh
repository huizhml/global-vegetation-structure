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


5)
# =======================================
#    Calculate VSM size for EU - data on LUMI-O, Hendrix
# =======================================
echo "Calculating VSM size for EU - data on LUMI-O..."
module load rclone
year=2024
eu_tiles=($(ls ${HOME}/data/gvs/deploy/eu_results/tiles_warp_RH98/*.tif | sed 's/.*\///' | sed 's/_resampled\.tif//'))
total_size=0
for tile in ${eu_tiles[@]}; do
    zone=$(echo ${tile:0:3} | tr '[:upper:]' '[:lower:]')
    bucket_name=${zone}-${year}
    size=$(rclone size lumi-465001846-private:${bucket_name}/${tile} | awk -F '[()]' '/Byte\)/{print $(NF-1)}' | awk '{print $1}')
    total_size=$((total_size + size))
    echo "Size of tile ${tile}: $size"
done
echo "Total size of EU tiles: $total_size"
echo "Total size of EU tiles: $(echo "scale=2; $total_size / 1024^3" | bc) GB"
;;

51)
# =======================================
#    Calculate VSM size for the whole globe (2024) - data on LUMI-O, Hendrix
# =======================================
echo "Calculating VSM size for the whole globe (2024) - data on LUMI-O..."
module load rclone
year=2024
tiles=($(cat /projects/dereeco/data/gvs/assets/worklists/total_tiles_2024.txt))
total_size=0
for tile in ${tiles[@]}; do
    zone=$(echo ${tile:0:3} | tr '[:upper:]' '[:lower:]')
    bucket_name=${zone}-${year}
    size=$(rclone size lumi-465001846-private:${bucket_name}/${tile} | awk -F '[()]' '/Byte\)/{print $(NF-1)}' | awk '{print $1}')
    total_size=$((total_size + size))
    echo "Size of tile ${tile}: $size"
done
echo "Total size of tiles: $total_size"
echo "Total size of tiles: $(echo "scale=2; $total_size / 1024^3" | bc) GB"
;;

6)
# =======================================
#    Calculate size for a single RH - data on LUMI-O
# =======================================
echo "Calculating size for a single RH - data on LUMI-O..."
module load lumio-ext-tools/1.0.0
year=2024
job_offset=${2:0}
rhs_idx=${3:-98}
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
total_size=0
for zone in ${task_zones[@]}; do
    zone_name=$(basename "${zone}" .txt | tr '[:upper:]' '[:lower:]')
    tile_ids=$(cat $zone)
    for tile_id in ${tile_ids[@]}; do
        size_q0=$(rclone size lumi-465001846-private:${zone_name}-${year}/${tile_id}/RH${rhs_idx}_Q0.tif | awk -F '[()]' '/Byte\)/{print $(NF-1)}' | awk '{print $1}')
        size_q1=$(rclone size lumi-465001846-private:${zone_name}-${year}/${tile_id}/RH${rhs_idx}_Q1.tif | awk -F '[()]' '/Byte\)/{print $(NF-1)}' | awk '{print $1}')
        size_q2=$(rclone size lumi-465001846-private:${zone_name}-${year}/${tile_id}/RH${rhs_idx}_Q2.tif | awk -F '[()]' '/Byte\)/{print $(NF-1)}' | awk '{print $1}')
        total_size=$((total_size + size_q0 + size_q1 + size_q2))
    done
done
file_name=${HOME}/data/gvs/deploy/data_size_tmp/data_size_rh${rhs_idx}_${year}_part${SLURM_PROCID}.txt
echo "Zones: ${task_zones[@]}" >> "$file_name"
echo "Total size: $(echo "scale=2; $total_size / 1024^3" | bc) GB" >> "$file_name"
;;

7)
# =======================================
#    Aggregate size for a single RH - data on LUMI-O
# =======================================
echo "Aggregating size for a single RH - data on LUMI-O..."
module load lumio-ext-tools/1.0.0
year=2024
rhs_idx=${2:-98}
all_files=($(ls ${HOME}/data/gvs/deploy/data_size_tmp/data_size_rh${rhs_idx}_${year}_part*.txt))
total_size=0
for file in ${all_files[@]}; do
    # Extract the numeric part before "GB", and convert to integer bytes (multiply by 1024^3), accumulate as integer
    size_gb=$(cat $file | grep "Total size:" | awk '{print $3}')
    size_bytes=$(echo "$size_gb * 1024 * 1024 * 1024" | bc | awk '{printf("%d\n",$1)}')
    total_size=$((total_size + size_bytes))
done
file_name=${HOME}/data/gvs/deploy/data_size_rh${rhs_idx}_${year}.txt
echo "Total size: $(echo "scale=2; $total_size / 1024^3" | bc) GB" >> "$file_name"
;;

*)
echo "Invalid option"
exit 1
;;
esac