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
input_dir=${HOME}/data/GVS/Deploy/predictions_${year}

MAX_ERDA_SESSIONS=10

check_unsynced_tiles() {
    local sync_dir="${HOME}/data/GVS/Deploy/sync_flags_${year}"
    local translate_dir="${HOME}/data/GVS/Deploy/translate_flags_${year}"

    translated_tiles=$(find "$translate_dir" -name "*_best_images_done" -exec basename {} \; | sed 's/_best_images_done//')
    synced_tiles=$(find "$sync_dir" -name "*_best_images_done" -exec basename {} \; | sed 's/_best_images_done//')
    unsynced_tiles=$(comm -23 <(echo "$translated_tiles" | sort) <(echo "$synced_tiles" | sort))

    echo "$unsynced_tiles"
}



sync_tile() {
    local tile_id=$1
    local lock_file="${HOME}/data/GVS/Deploy/sync_flags_${year}/${tile_id}.lock"
    local translate_flag_new="${HOME}/data/GVS/Deploy/translate_flags_${year}/${tile_id}_best_images_done"
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
            touch "${HOME}/data/GVS/Deploy/sync_flags_${year}/${tile_id}_best_images_done"
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



