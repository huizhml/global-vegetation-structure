#!bin/bash
sync_tile_from_erda() {
    # TODO: use rclone
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
