#!/bin/bash

source scripts/core/utils.sh



case $1 in
0)
# =======================================
#    Translate ALS data to COG
# =======================================
tif_dir=${HOME}/data/gvs/evaluation/with_airborne_lidar/ALS_MaxGEDIFootprint_GSD10m
tif_files=$(find $tif_dir -type f -name "*.tif")
for tif_file in $tif_files; do
    translate_to_cog $tif_file
done
;;
01)
# =======================================
#    Translate LVIS data to COG
# =======================================
tif_dir=${HOME}/data/gvs/evaluation/with_airborne_lidar/LVIS_RH98_GSD10m
tif_files=$(find $tif_dir -type f -name "*.tif")
for tif_file in $tif_files; do
    translate_to_cog $tif_file
done
;;

*)
    echo "Invalid option"
    exit 1
    ;;
esac
