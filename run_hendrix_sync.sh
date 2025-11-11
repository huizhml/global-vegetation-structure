#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=sync
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

MAX_ERDA_SESSIONS=10


check_missing_tiles_on_hendrix() {
    local year=$1
    local translate_dir="${HOME}/data/GVS/deploy/translate_flags_${year}"
    local prediction_dir="${HOME}/data/GVS/deploy/predictions_${year}"

    all_tiles=$(find "$translate_dir" -name "*_done" -exec basename {} \; | sed -E 's/(_best_images_done|_done)//')
    existing_tiles=$(find "$prediction_dir" -name "*_cog" -exec basename {} \; | sed -E 's/(_cog)//')
    missing_tiles=$(comm -23 <(echo "$all_tiles" | sort) <(echo "$existing_tiles" | sort))
    echo "$missing_tiles"
}

check_unsynced_tiles() {
    local year=$1
    local sync_dir="${HOME}/data/GVS/deploy/sync_flags_${year}"
    local translate_dir="${HOME}/data/GVS/deploy/translate_flags_${year}"

    translated_tiles=$(find "$translate_dir" -name "*_best_images_done" -exec basename {} \; | sed 's/_best_images_done//')
    synced_tiles=$(find "$sync_dir" -name "*_best_images_done" -exec basename {} \; | sed 's/_best_images_done//')
    unsynced_tiles=$(comm -23 <(echo "$translated_tiles" | sort) <(echo "$synced_tiles" | sort))

    echo "$unsynced_tiles"
}

sync_tile_from_erda() {
    local tile_id=$1
    local year=$2
    local existing_dir="${HOME}/data/GVS/deploy/predictions_${year}/${tile_id}_cog"
    if [ -d "$existing_dir" ]; then
        count=$(ls ${existing_dir}/*.cog.tif | wc -l)
        if [ $count -gt 303 ]; then
            echo "[${tile_id}] Existing directory has $count files, complete, skipping..."
            return
        fi
    fi
    echo "[${tile_id}] Syncing data from ERDA..."
    sftp -q ucph-erda <<EOF | wc -l
cd GVS/predictions_${year}
get -r ${tile_id}_cog ${HOME}/data/GVS/deploy/predictions_${year}/
EOF
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "[${tile_id}] Sync failed (sftp exit ${exit_status})"
fi
count=$(ls ${existing_dir}/*.cog.tif | wc -l)
echo "[${tile_id}] Sync completed (${count} files)"

}


sync_tile_to_erda() {
    local tile_id=$1
    local year=$2
    local input_dir="${HOME}/data/gvs/deploy/predictions_${year}"
    local predicted_cogs_count

    predicted_cogs_count=$(find "${input_dir}/${tile_id}" -type f 2>/dev/null | wc -l)

    if [ "$predicted_cogs_count" -lt 303 ]; then
        echo "[${tile_id}] Translation corrupted, skipping sync..."
        return
    fi

    echo "[${tile_id}] Syncing data to ERDA..."
    remote_count=$(sftp -q ucph-erda <<EOF | wc -l
cd gvsm/predictions_gtiff_${year}
put -r ${input_dir}/${tile_id}
cd ${tile_id}
ls -1
EOF
)
    sftp_status=$?
    if [ $sftp_status -eq 0 ] && [ "$remote_count" -ge 303 ]; then
        echo "[${tile_id}] Remote verification passed (${remote_count}/${predicted_cogs_count}). Create sync flag..."
        touch "${HOME}/data/gvs/deploy/sync_flags_${year}/${tile_id}_best_images_done"
    else
        echo "[${tile_id}] Sync failed (sftp exit ${sftp_status}). Remote count: ${remote_count}"
    fi
}

case $1 in
0)
# =======================================
#    SYNC ALL TILES FROM ERDA
# SBATCH --cpus-per-task=8
# SBATCH --mem=4G
# =======================================
echo "Syncing all tiles from ERDA"
year=2024
job_count=0
tile_ids=$(check_missing_tiles_on_hendrix $year)
for tile_id in ${tile_ids[@]}; do

    sync_tile_from_erda $tile_id $year &
    ((job_count++))

    if [ "$job_count" -ge "$MAX_ERDA_SESSIONS" ]; then
        wait -n  # Wait for one job to finish before launching more
        ((job_count--))
    fi
done
wait
sleep 5
;;

1)
# =======================================
#    SYNC SPECIFIC TILES FROM ERDA
# =======================================
echo "Syncing specific tiles from ERDA"
tile_ids=(32VNK 32VNJ 32VMJ 32VPJ 32VMH 32VNH 32VPH 32UMG 32UNG 32UPG 33UUB 33VUC 32UMF 32UNF 32UPF 38KMC 35NQB 51UWB 32PLT)
year=2024
job_count=0
for tile_id in ${tile_ids[@]}; do

    sync_tile_from_erda $tile_id $year &
    ((job_count++))

    if [ "$job_count" -ge "$MAX_ERDA_SESSIONS" ]; then
        wait -n  # Wait for one job to finish before launching more
        ((job_count--))
    fi
done
wait
sleep 5
;;
        
2)
# =======================================
#    SYNC SPECIFIC TILES TO ERDA
# =======================================
echo "Syncing specific tiles to ERDA"
tile_ids=(32MRE)
year=2020
job_count=0
start_time=$(date +%s)
for tile_id in ${tile_ids[@]}; do
    echo "Syncing tile $tile_id to ERDA"
    sync_tile_to_erda $tile_id $year &
    ((job_count++))

    if [ "$job_count" -ge "$MAX_ERDA_SESSIONS" ]; then
        wait -n  # Wait for one job to finish before launching more
        ((job_count--))
    fi
done
wait
end_time=$(date +%s)
echo "Syncing specific tiles to ERDA took $((end_time - start_time)) seconds"
sleep 5
;;
3)
# =======================================
#    SYNC ALL TILES TO ERDA
# =======================================
echo "Syncing all tiles to ERDA"
year=$2
while true; do
    unsynced_tiles=$(check_unsynced_tiles $year)
    if [ -z "$unsynced_tiles" ]; then
        echo "No tiles to sync. Sleeping..."
        sleep 10
        continue
    fi

    echo "Found unsynced tiles:"
    echo "$unsynced_tiles"

    job_count=0
    for tile_id in $unsynced_tiles; do
        sync_tile_to_erda $tile_id $year &
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
4)
# =======================================
#    SYNC TILES FROM LUMI-O
# =======================================
echo "Syncing tiles from LUMI-O"
module load rclone
year=2020
lumi_project=465001846
remote=lumi-${lumi_project}-private:
buckets=$(rclone lsd ${remote} | grep '\-2020$' | awk '{print $5}')
echo "Number of buckets: ${#buckets[@]}"
for zone in ${buckets[@]}; do
    tile_ids=$(rclone lsd ${remote}/${zone}/predictions_GTiff_${year} | awk '{print $5}')
    for tile_id in ${tile_ids[@]}; do
        count=$(rclone ls ${remote}/${zone}/predictions_GTiff_${year}/${tile_id} | wc -l)
        if [ $count -lt 303 ]; then
            echo "[${tile_id}] Source directory has $count files, incomplete, skipping..."
            continue
        fi
        while true; do

            count=$(ls ${HOME}/data/gvs/deploy/predictions_GTiff_${year}/${tile_id}_GTiff/*.tif | wc -l)
            if [ $count -ge 303 ]; then
                echo "[${tile_id}] Destination directory has $count files, done, deleting source directory..."
                rclone delete ${remote}/${zone}/predictions_GTiff_${year}/${tile_id}
                touch ${HOME}/data/gvs/deploy/flags_inference_${year}/${tile_id}_best_images_done
                break
            else
                echo "[${tile_id}] Copying data from LUMI-O to local..."
                rclone copy --transfers=16 --multi-thread-streams=4 ${remote}/${zone}/predictions_GTiff_${year}/${tile_id} ${HOME}/data/gvs/deploy/predictions_GTiff_${year}/${tile_id}_GTiff
            fi

        done
    done
done
;;
5)
# =======================================
#    SYNC SPECIFIC TILES TO LUMI-O
# =======================================
echo "Syncing specific tiles to LUMI-O"
tile_ids=(13XEG)
year=2020
lumi_project=465001846
remote=lumi-${lumi_project}-private:
start_time=$(date +%s)
for tile_id in ${tile_ids[@]}; do
    zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
    bucket_name=${zone}-${year}
    echo "Syncing tile $tile_id to LUMI-O bucket $bucket_name"
    rclone mkdir ${remote}${bucket_name}
    rclone copy --transfers=8 --checkers=8 --multi-thread-streams=4 \
        ${HOME}/data/gvs/deploy/predictions_gtiff_${year}/${tile_id} ${remote}${bucket_name}/predictions_GTiff_${year}/${tile_id}
done
end_time=$(date +%s)
echo "Syncing specific tiles to LUMI-O took $((end_time - start_time)) seconds"
sleep 5
;;
6)
# =======================================
#    SYNC SPECIFIC TILES FROM LUMI-O
# =======================================
echo "Syncing specific tiles from LUMI-O"
tile_ids=(13XEG)
year=2020
lumi_project=465001846
remote=lumi-${lumi_project}-private:
start_time=$(date +%s)
for tile_id in ${tile_ids[@]}; do
    zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
    bucket_name=${zone}-${year}
    echo "Syncing tile $tile_id from LUMI-O bucket $bucket_name"
    rclone copy --transfers=8 --checkers=8 --multi-thread-streams=4 \
        ${remote}${bucket_name}/predictions_GTiff_${year}/${tile_id} ${HOME}/data/gvs/deploy/predictions_gtiff_${year}/${tile_id} 
done
end_time=$(date +%s)
echo "Syncing specific tiles from LUMI-O took $((end_time - start_time)) seconds"
sleep 5
;;
esac





