#!/bin/bash


lumi_project=465001846
remote=lumi-${lumi_project}-private:


sync_from_lumi_o() {
    local tile_id=$1
    local year=$2
    # sync data from lumi-o
    zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
    bucket_name=${zone}-${year}
    dst_dir=${HOME}/flash/data/gvs/deploy/predictions_gtiff_${year}/${tile_id}
    start_time=$(date +%s)
    mkdir -p "${dst_dir}"
    # sleep $((RANDOM % 6))
    rclone copy ${remote}${bucket_name}/predictions_GTiff_${year}/${tile_id} ${dst_dir} \
        --filter '+ *Q1*.tif' --filter '- **' \
        --transfers=8 --checkers=8 --multi-thread-streams=2

    end_time=$(date +%s)
    echo "rclone copy elapsed: $((end_time - start_time)) seconds"

    # Rename files from *_Q1_uncompressed.tif to *_Q1.tif
    renamed_count=0
    while IFS= read -r -d '' src; do
        dst="${src%_uncompressed.tif}.tif"
        mv -f "$src" "$dst"
        renamed_count=$((renamed_count+1))
    done < <(find "${dst_dir}" -type f -name '*_Q1_uncompressed.tif' -print0)
    end_time=$(date +%s)
    echo "rename elapsed: $((end_time - start_time)) seconds"
}




year=${1:-2024}
n_tiles_per_task=${2:-386}
mapfile -t tiles < <(ls "${HOME}/data/gvs/GEDI_for_correction/partitions_${year}_v1" | sort)
echo "number of tiles: ${#tiles[@]}"

start_idx=$((SLURM_PROCID * $n_tiles_per_task))
echo "start_idx: $start_idx"

# check if last task, if so, process the remaining tiles
LAST_TASK_IDX=$(($SLURM_NTASKS - 1))
if [ $SLURM_PROCID -eq $LAST_TASK_IDX ]; then
    for tile in ${tiles[@]:$start_idx}; do # from start_idx to the end
        tile_id="${tile%.*}"
        sync_from_lumi_o ${tile_id} ${year}
        python -m postprocess.bias_correction \
            year=${year} \
            tile_id=${tile_id} \
            stac_collection_dir=${HOME}/data/gvs/deploy/gvsm_stac_catalog/vsm_local \
            ref_data_dir=${HOME}/data/gvs/GEDI_for_correction/partitions_${year}_v1 \
            root_save_dir=${HOME}/data/gvs/deploy/correction
        rm ${HOME}/flash/data/gvs/deploy/predictions_gtiff_${year}/${tile_id}/*
    done
else
    for tile in ${tiles[@]:$start_idx:$n_tiles_per_task}; do
        tile_id="${tile%.*}"
        sync_from_lumi_o ${tile_id} ${year}
        python -m postprocess.bias_correction \
            year=${year} \
            tile_id=${tile_id} \
            stac_collection_dir=${HOME}/data/gvs/deploy/gvsm_stac_catalog/vsm_local \
            ref_data_dir=${HOME}/data/gvs/GEDI_for_correction/partitions_${year}_v1 \
            root_save_dir=${HOME}/data/gvs/deploy/correction
        rm ${HOME}/flash/data/gvs/deploy/predictions_gtiff_${year}/${tile_id}/*
    done
fi
