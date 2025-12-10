#!/bin/bash



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

esac