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

source scripts/core/sync.sh
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
                rclone copy --transfers=16 --multi-thread-streams=4 ${remote}/${zone}/predictions_GTiff_${year}/${tile_id} ${HOME}/data/gvs/deploy/predictions_gtiff_${year}/${tile_id}
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
tile_ids=(19NBF)
year=2024
lumi_project=465001846
remote=lumi-${lumi_project}-private:
start_time=$(date +%s)
for tile_id in ${tile_ids[@]}; do
    zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
    bucket_name=${zone}-${year}
    echo "Syncing tile $tile_id from LUMI-O bucket $bucket_name"
    rclone copy --transfers=8 --checkers=8 --multi-thread-streams=2 \
        ${remote}${bucket_name}/predictions_GTiff_${year}/${tile_id} ${HOME}/data/gvs/deploy/predictions_gtiff_${year}/${tile_id} 
done
end_time=$(date +%s)
echo "Syncing specific tiles from LUMI-O took $((end_time - start_time)) seconds"
sleep 5
;;

7)
# =======================================
#    Compare tar and gdal merge&compress
# =======================================
echo "Testing compression before copying"
tile_id=47RLJ
year=2020
cog_dir="${HOME}/data/gvs/deploy/predictions_${year}/${tile_id}"
gtiff_dir="${HOME}/data/gvs/deploy/predictions_gtiff_${year}/${tile_id}"
dst_dir="${HOME}/data/gvs/deploy/predictions_zip_${year}/${tile_id}"

# archive with tar
start_time=$(date +%s)
cd ${cog_dir}
tar -cvf ${dst_dir}.tar .
end_time=$(date +%s)
echo "Archiving took $((end_time - start_time)) seconds, size: $(du -sh ${dst_dir}.tar | awk '{print $1}')"


# Unarchiving
start_time=$(date +%s)
cd ${HOME}
mkdir -p ${dst_dir}/unarchived
tar -xvf ${dst_dir}.tar -C ${dst_dir}/unarchived/
end_time=$(date +%s)
echo "Unarchiving took $((end_time - start_time)) seconds"

# compression with gdal merge&compress
start_time=$(date +%s)
gdalbuildvrt -separate ${dst_dir}.vrt ${gtiff_dir}/*.tif # order?
echo 'translating to one file'
gdal_translate ${dst_dir}.vrt ${dst_dir}.tif \
  -co COMPRESS=ZSTD \
  -co ZSTD_LEVEL=3 \
  -co TILED=YES \
  -co BLOCKXSIZE=1024 \
  -co BLOCKYSIZE=1024 \
  -co NUM_THREADS=16 \
  -co PREDICTOR=2 \
  -co INTERLEAVE=BAND \
  -co BIGTIFF=IF_SAFER

# # TODO: set band names
# # gdal_edit.py -metadatafile band_names.txt ${dst_dir}.tif
end_time=$(date +%s)
echo "Gdal merge&compress took $((end_time - start_time)) seconds, size: $(du -sh ${dst_dir}.tif | awk '{print $1}')"


# compress with tar zstd 3
start_time=$(date +%s)
cd ${gtiff_dir}
tar -I 'pzstd -p 16 -3' -cvf ${dst_dir}.tar.gz .
end_time=$(date +%s)
echo "Compressing with tar zstd 3 took $((end_time - start_time)) seconds, size: $(du -sh ${dst_dir}.tar.gz | awk '{print $1}')"

# Uncompressing with tar zstd 3
start_time=$(date +%s)
mkdir -p ${dst_dir}/decompressed
tar -I 'pzstd -p 16 -3' -xvf ${dst_dir}.tar.gz -C ${dst_dir}/decompressed/
end_time=$(date +%s)
echo "Uncompressing with tar zstd 3 took $((end_time - start_time)) seconds"
;;


8) 
# =======================================
#    Compare 
# =======================================
echo "Testing transfer speed after compression"
tile_id=47RLJ
year=2020
big_tif_dir="${HOME}/data/gvs/deploy/predictions_zip_${year}/${tile_id}/test"
# for i in {1..10}; do
#   cp ${big_tif_file} "${big_tif_file}_duplicate_$i.tif"
# done

module load rclone
time_start=$(date +%s)
lumi_project=465001846
remote=lumi-${lumi_project}-private:
rclone copy ${big_tif_dir} ${remote}/dummy-bucket/ \
    --transfers=8 --checkers=8 --multi-thread-streams=4 -P
    # --s3-chunk-size 100M

time_end=$(date +%s)
echo "Transfer took $((time_end - time_start)) seconds"

;;

9) 
# =======================================
#    GDAL pipe 
# =======================================
lumi_project=465001846
remote=lumi-${lumi_project}-private:

tile_id=47RLJ
year=2020
cog_dir="${HOME}/data/gvs/deploy/predictions_${year}/${tile_id}"
gtiff_dir="${HOME}/data/gvs/deploy/predictions_gtiff_${year}/${tile_id}"
dst_dir="${HOME}/data/gvs/deploy/predictions_zip_${year}/${tile_id}"


# compression with gdal merge&compress
start_time=$(date +%s)
gdalbuildvrt -separate ${dst_dir}.vrt ${gtiff_dir}/*.tif # order?
echo 'translating to one file'
gdal_translate ${dst_dir}.vrt /vsistdout/ \
  -co COMPRESS=ZSTD \
  -co ZSTD_LEVEL=3 \
  -co TILED=YES \
  -co BLOCKXSIZE=1024 \
  -co BLOCKYSIZE=1024 \
  -co NUM_THREADS=16 \
  -co PREDICTOR=2 \
  -co INTERLEAVE=BAND \
  -co BIGTIFF=IF_SAFER | rclone rcat ${remote}/dummy-bucket/test.tif  --ignore-checksum --transfers=1 --multi-thread-streams=16 -P
;;

esac





