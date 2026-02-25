#!/bin/bash
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=visualize
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu26fl

source scripts/core/utils.sh
get_options $1


case $1 in
1)
# ----------------------------------------
#    Resample and mosaic the predictions
# ----------------------------------------
python -m visualization.run run=resample_and_mosaic run.rh_idx=${2:-98} run.q_idx=${3:-1}
;;
1.1)
# ----------------------------------------
#    Create a global mosaic of the prediction intervals
# ----------------------------------------
python -m visualization.run run=create_global_diff_mosaic \
          run.rh_idx=${2:-98} \
          run.left_q_idx=${3:-0} \
          run.right_q_idx=${4:-2}
;;

2)
# ----------------------------------------
#    Calculate the difference between two tif files
# ----------------------------------------
data_root_dir=${HOME}/data/gvs/predictions/2020/blended/mosaic/
gdal_calc.py -A ${data_root_dir}/${2}.tif -B ${data_root_dir}/${3}.tif --outfile=${data_root_dir}/${4}.tif \
    --calc="A-B" --format=GTiff \
    --co="TILED=YES" \
    --co="COPY_SRC_OVERVIEWS=YES" \
    --co="COMPRESS=LERC_ZSTD" \
    --type='Int16' \
    --NoDataValue=32767 \
    --overwrite

gdal_translate ${data_root_dir}/${4}.tif ${data_root_dir}/${4}.cog.tif \
    -of COG \
    -co COMPRESS=LERC_ZSTD \
    -co MAX_Z_ERROR=0
    
rm ${data_root_dir}/${4}.tif
;;

*)
echo "Invalid option"
exit 1
;;
esac