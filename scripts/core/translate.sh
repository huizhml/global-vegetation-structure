#!/bin/bash

source scripts/core/utils.sh



case $1 in
0)
# =======================================
#    Translate all tif files in a directory to COG, save to the same directory
# =======================================
tif_dir=${HOME}/data/gvs/evaluation/with_airborne_lidar/ALS_MaxGEDIFootprint_GSD10m
warn "Don't run this repeatedly, it doesn't differentiate .cog.tif and .tif"
tif_files=$(find $tif_dir -type f -name "*.tif")
for tif_file in $tif_files; do
    translate_to_cog $tif_file
done
;;
01)
# =======================================
#    Translate all tif files in a directory to COG, save to the specified directory
# =======================================
tif_dir=${HOME}/data/gvs/products/diversity_indices/2020/tiles/geotiff
save_dir="${tif_dir%/geotiff}/cog"
mkdir -p $save_dir
tif_files=$(find $tif_dir -type f -name "*.tif")
for tif_file in $tif_files; do
    translate_to_cog $tif_file $save_dir
done
;;

*)
    echo "Invalid option"
    exit 1
    ;;
esac
