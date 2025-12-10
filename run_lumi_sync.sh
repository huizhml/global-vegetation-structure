#!/bin/bash
#SBATCH --account=project_465001846
#SBATCH --partition=small
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=sync
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

year=2024

case $1 in
3)
# =======================================
#    SYNC TILES FROM LUMI-O TO FLASH
# =======================================
tile_id=$2
zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
bucket_name=${zone}-${year}
lumi_project=465001846
remote=lumi-${lumi_project}-private:
dst_dir=${HOME}/data/gvs/deploy/predictions_gtiff_${year}/${tile_id}
mkdir -p ${dst_dir}
rclone sync ${remote}${bucket_name}/predictions_GTiff_${year}/${tile_id} ${HOME}/data/gvs/deploy/predictions_gtiff_${year}/${tile_id} --transfers=16 --checkers=16 --multi-thread-streams=4
;;
2)
# =======================================
#    CHECK PREDICTION INTEGRITY
# =======================================
flaged_tiles=$(find ${HOME}/data/GVS/Deploy/flags_inference_${year} -type f -name "*_best_images_done" -exec basename {} \; | sed 's/_best_images_done//')
module load lumio
lumi_project=465001846
remote=lumi-${lumi_project}-private:
for tile in $flaged_tiles; do
    zone=$(echo ${tile:0:3} | tr '[:upper:]' '[:lower:]')
    bucket_name=${zone}-${year}
    count=$(rclone ls "${remote}${bucket_name}/predictions_GTiff_${year}/${tile}" | wc -l)
    if [ $count -lt 303 ]; then
        echo "Tile ${tile} has $count predictions, delete flag..."
        # rm ${HOME}/data/GVS/Deploy/flags_inference_${year}/${tile}_best_images_done
    else
        # check the smallest prediction size, if less than 200 MB, delete flag
        smallest_prediction_size=$(rclone ls "${remote}${bucket_name}/predictions_GTiff_${year}/${tile}" | awk '{print $1}' | sort -n | head -n 1)
        if [ $smallest_prediction_size -lt 209715200 ]; then
            echo "Tile ${tile} has smallest prediction size $smallest_prediction_size, delete flag..."
            # rm ${HOME}/data/GVS/Deploy/flags_inference_${year}/${tile}_best_images_done
        fi
    fi
done
;;
1)
# =======================================
#    SYNC TILES TO LUMI-O
# =======================================
# **************************************************
# Why using LUMI-O
# 1. We found tile-level systematic bias in GVSM predictions, a post-processing step (linear correction/bias correction & blending) is needed to correct it.
# 2. The overall speed on LUMI (predict on flash/translate to scratch/sync to erda) is serverly limited by the number of parallel sessions on ERDA. (16 written on the docs, but tested max 10)
# 3. LUMI-O offers large storage quota, (150 TB by default, up to 30PB?)

# LUMI-O storage structure
# - bucket (max 1k)
#   - object (max 500k)
# **************************************************
module load lumio
lumi_project=465001846
remote=lumi-${lumi_project}-private:
gtiff_dir=${HOME}/flash/data/GVS/Deploy/predictions_GTiff_${year}
tiles=$(find $gtiff_dir -type d -name "*_GTiff" -exec basename {} \; | sed 's/_GTiff//')
echo "Number of tiles: ${#tiles[@]}"
echo "Tiles: ${tiles[@]}"

# clean LUMI-O empty buckets
# buckets=$(rclone lsd ${remote} | awk '{print $5}')
# for bucket in $buckets; do
#     year=$(echo $bucket | grep -oE '[0-9]{4}$')
#     empty_bucket=$(rclone lsd ${remote}${bucket}/predictions_GTiff_${year}/ | wc -l)
#     echo "Bucket ${bucket} has $empty_bucket predictions"
#     if [ $empty_bucket -eq 0 ]; then
#         echo "Bucket ${bucket} is empty, deleting..."
#         echo ${remote}${bucket}
#         rclone rmdir ${remote}${bucket}
#     fi
    
# done
year=2020
for tile_id in $tiles; do
    # check if the local GTiff files are completed in writing, older than 2 hours
    SRC=${gtiff_dir}/${tile_id}_GTiff
    # if [ $(find ${SRC} -type f -mmin -180 | wc -l) -gt 0 ]; then
    #     echo "Tile ${tile_id} till has files newer than 3 hours, skipping..."
    #     continue
    # else
    #     echo "Tile ${tile_id} is completed in writing, syncing..."
    # fi
    # check if there are 303 predictions for tile_id in the lumio bucket
    zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
    bucket_name=${zone}-${year}
    DST=${remote}${bucket_name}/predictions_GTiff_${year}/${tile_id}
    if rclone lsd ${remote} | grep "^.*${bucket_name}.*"; then
        echo "Bucket ${bucket_name} exists"
    else
        echo "Bucket ${bucket_name} does not exist, creating..."
        rclone mkdir ${remote}${bucket_name}
    fi

    while true; do
        count=$(rclone ls "${DST}" | wc -l)
        if [ $count -lt 303 ]; then
            echo "Tile ${tile_id} has $count predictions, syncing..."
            rclone sync "$SRC" "$DST"  --modify-window 2h --transfers=16 --checkers=16 --multi-thread-streams=4
        else
            echo "Tile ${tile_id} has $count predictions, done, removing local GTiff..."
            rm -rf ${SRC}
            break
        fi
        sleep 10
    done
done
;;
0)
# =======================================
#    SYNC TILES TO ERDA
# =======================================

input_dir=${HOME}/data/GVS/deploy/predictions_${year}

MAX_ERDA_SESSIONS=10

check_unsynced_tiles() {
    local sync_dir="${HOME}/data/GVS/deploy/flags_sync_${year}"
    local translate_dir="${HOME}/data/GVS/deploy/flags_translate_${year}"

    translated_tiles=$(find "$translate_dir" -name "*_best_images_done" -exec basename {} \; | sed 's/_best_images_done//')
    synced_tiles=$(find "$sync_dir" -name "*_best_images_done" -exec basename {} \; | sed 's/_best_images_done//')
    unsynced_tiles=$(comm -23 <(echo "$translated_tiles" | sort) <(echo "$synced_tiles" | sort))

    echo "$unsynced_tiles"
}



sync_tile() {
    local tile_id=$1
    local lock_file="${HOME}/data/GVS/deploy/sync_flags_${year}/${tile_id}.lock"
    local translate_flag_new="${HOME}/data/GVS/deploy/translate_flags_${year}/${tile_id}_best_images_done"
    local predicted_cogs_count

    predicted_cogs_count=$(find "${input_dir}/${tile_id}_cog" -type f 2>/dev/null | wc -l)

    if [ -f "$translate_flag_new" ]; then
        if [ "$predicted_cogs_count" -lt 303 ]; then
            echo "[${tile_id}] Translation corrupted, deleting incomplete COG files..."
            # rm -rf "${input_dir}/${tile_id}_cog"
            return
        fi

        echo "[${tile_id}] Syncing data to ERDA..."
        remote_count=$(sftp -q ucph-erda <<EOF | wc -l
cd GVS/predictions_${year}
put -r ${input_dir}/${tile_id}_cog
cd ${tile_id}_cog
ls -1
EOF
)
        sftp_status=$?
        if [ $sftp_status -eq 0 ] && [ "$remote_count" -ge 303 ]; then
            echo "[${tile_id}] Remote verification passed (${remote_count}/${predicted_cogs_count}). Cleaning up..."
            rm -rf "${input_dir}/${tile_id}_cog"
            touch "${HOME}/data/GVS/deploy/sync_flags_${year}/${tile_id}_best_images_done"
        else
            echo "[${tile_id}] Sync failed (sftp exit ${sftp_status}). Remote count: ${remote_count}"
        fi
    fi

}


while true; do
    unsynced_tiles=$(check_unsynced_tiles)
    if [ -z "$unsynced_tiles" ]; then
        echo "No tiles to sync. Sleeping..."
        sleep 10
        continue
    fi

    echo "Found unsynced tiles:"
    echo "$unsynced_tiles"

    job_count=0
    for tile_id in $unsynced_tiles; do
        sync_tile "$tile_id" &
        ((job_count++))

        if [ "$job_count" -ge "$MAX_ERDA_SESSIONS" ]; then
            wait -n  # Wait for one job to finish before launching more
            ((job_count--))
        fi
    done
    wait
    sleep 5
done
;;
esac