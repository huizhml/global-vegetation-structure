#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=extract_pred
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

source scripts/core/utils.sh
source scripts/core/run_python.sh


case $1 in
0)
# ---------------------------------------
#   Check original mosaic
# ---------------------------------------
echo "Checking original mosaic"
;;

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
#    Calculate the skewness of the upper span(Q0-Q1) and lower span(Q1-Q2)
# ----------------------------------------
data_root_dir=${HOME}/data/gvs/predictions/2020/blended/mosaic/
a_filename_pattern=global_mosaic_2020_RH${2}_Q0-Q1.cog.tif
b_filename_pattern=global_mosaic_2020_RH${2}_Q1-Q2.cog.tif
out_filename=global_mosaic_2020_RH${2}_Qskewness
gdal_calc.py -A ${data_root_dir}/${a_filename_pattern} -B ${data_root_dir}/${b_filename_pattern} --outfile=${data_root_dir}/${out_filename}.tif \
    --calc="A-B" --format=GTiff \
    --co="TILED=YES" \
    --co="COPY_SRC_OVERVIEWS=YES" \
    --co="COMPRESS=LERC_ZSTD" \
    --type='Int16' \
    --NoDataValue=32767 \
    --overwrite

gdal_translate ${data_root_dir}/${out_filename}.tif ${data_root_dir}/${out_filename}.cog.tif \
    -of COG \
    -co COMPRESS=LERC_ZSTD \
    -co MAX_Z_ERROR=0
    
rm ${data_root_dir}/${out_filename}.tif
;;

4)

# ---------------------------------------
#   Check mosaic after bias correction
# ---------------------------------------
year=2020

python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +save_dir=${HOME}/data/gvs/predictions/${year}/original/mosaic \
    +bias_dir=None
;;
5)

# ---------------------------------------
#   Check mosaic after bias correction, bias cutoff = 20
# ---------------------------------------
year=2020
bias_cutoff=10
python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +bias_cutoff=$bias_cutoff \
    +save_dir=${HOME}/data/gvs/predictions/${year}/bias_corrected_slope_lt20_minpoints2000_bias_cutoff${bias_cutoff}/mosaic \
    +bias_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_by_tile
;;

6)

# ---------------------------------------
#   Check mosaic after bias correction, average across RHS
# ---------------------------------------
year=2020
python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +average_across_rhs=True \
    +save_dir=${HOME}/data/gvs/predictions/${year}/bias_corrected_slope_lt20_minpoints2000_average_across_rhs/mosaic \
    +bias_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_by_tile
;;

7)

# ---------------------------------------
#   Check mosaic after bias correction, bias col = mean_bias_trimmed_5_95
# ---------------------------------------
year=2020
python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +bias_col=mean_bias_trimmed_5_95 \
    +save_dir=${HOME}/data/gvs/predictions/${year}/bias_corrected_slope_lt20_minpoints2000_trimmed_5_95/mosaic \
    +bias_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_with_median_and_trimmed_5_95_by_tile
;;

8)

# ---------------------------------------
#   Check mosaic after bias correction, bias col = median_bias
# ---------------------------------------
year=2020
python -m visualization.create_global_view year=$year  \
    task=check_mosaic_after_bias_correction \
    +bias_col=median_bias \
    +save_dir=${HOME}/data/gvs/predictions/${year}/bias_corrected_slope_lt20_minpoints2000_median_bias/mosaic \
    +bias_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_with_median_and_trimmed_5_95_by_tile
;;
9)

# ---------------------------------------
#   Plot bias distribution
# ---------------------------------------
year=2020
python -m postprocess.bias_correction year=$year \
    task=plot_bias_distribution \
    +bias_col=median_bias \
    +bias_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/stats_with_median_and_trimmed_5_95_by_tile \
    +save_dir=${HOME}/data/gvs/assets/bias_correction_stats/slope_lt20_minpoints2000/${year}/figures
;;

*)
echo "Invalid option"
exit 1
;;
esac
