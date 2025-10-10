

# **************************************************
# Why using LUMI-O
# 1. We found tile-level systematic bias in GVSM predictions, a post-processing step (linear correction/bias correction & blending) is needed to correct it.
# 2. The overall speed on LUMI (predict on flash/translate to scratch/sync to erda) is serverly limited by the number of parallel sessions on ERDA. (16 written on the docs, but tested max 10)
# 3. LUMI-O offers large storage quota, (150 TB by default, up to 30PB?)

# LUMI-O storage structure
# - bucket (max 1k)
#   - object (max 500k)
# **************************************************

# module load lumio

# check if the bucket exists


lumi_project=465001846 
remote=lumi-${lumi_project}-private:

tile_id=18MVC
year=2024
zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
bucket_name=${zone}-${year}
echo "bucket_name: ${bucket_name}"

if rclone lsd ${remote} | grep "^.*${bucket_name}.*"; then
    echo "Bucket ${bucket_name} exists"
else
    echo "Bucket ${bucket_name} does not exist, creating..."
    rclone mkdir ${remote}${bucket_name}
fi


# rclone sync ${HOME}/flash/data/GVS/deploy/predictions_GTiff_${year}/${tile_id}_GTiff ${remote}${bucket_name}/predictions_GTiff_${year}/${tile_id}
echo copy files
SRC=${HOME}/flash/data/GVS/deploy/predictions_GTiff_${year}/${tile_id}_GTiff
DST=${remote}${bucket_name}/predictions_GTiff_${year}/${tile_id}
# e.g. remote="lumi-${lumi_project}-private:"  ← note the trailing colon
rclone sync "$SRC" "$DST"  --modify-window 2s  # --local-no-check-updated
# rclone sync ${HOME}/data/GVS/deploy/predictions_${year}/${tile_id}_cog ${remote}${bucket_name}/predictions_${year}/${tile_id}_cog --local-no-check-updated
count=$(rclone ls "${DST}" | wc -l)
echo "Number of files in ${DST}: $count"