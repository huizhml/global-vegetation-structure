
tile_id=${2:-'22UUL'}
rh_idx=${3:-98}
year=${4:-2024}
q_idx=${5:-1}

case $1 in
00)
# =======================================
#    DOWNLOAD One RH PREDICTION (COG) FROM Hendrix
# =======================================
dst_dir=${HOME}/GVS/predictions_${year}/${tile_id}_cog
mkdir -p $dst_dir
src_dir=/home/ksb781/data/gvs/deploy/predictions_${year}/${tile_id}_cog
host=ksb781@hendrixgate02fl
scp -r $host:$src_dir/RH${rh_idx}_Q${q_idx}.cog.tif $dst_dir/
        ;;
01)
# =======================================
#    DOWNLOAD All RH PREDICTIONS (COGs) FROM Hendrix
# =======================================
dst_dir=${HOME}/GVS/predictions_${year}/${tile_id}_cog
mkdir -p $dst_dir
src_dir=/home/ksb781/data/gvs/deploy/predictions_${year}/${tile_id}_cog
host=ksb781@hendrixgate02fl
rsync -avz --progress "$host:$src_dir/RH*_Q${q_idx}.cog.tif" $dst_dir/
;;
10)
# =======================================
#    DOWNLOAD ONE RH PREDICTION (uncompressed GTiff) FROM Hendrix
# =======================================
dst_dir=${HOME}/GVS/predictions_${year}/${tile_id}_GTiff
mkdir -p $dst_dir
src_dir=/home/ksb781/data/gvs/deploy/predictions_GTiff_${year}/${tile_id}_GTiff
host=ksb781@hendrixgate02fl
scp -r $host:$src_dir/RH${rh_idx}_Q${q_idx}_uncompressed.tif $dst_dir/
        ;;

02)
# =======================================
#    DOWNLOAD ONE RH PREDICTIONS (uncompressed GTiffs) FROM LUMIO
# =======================================
tile_id=${2:-'22UUL'}
rh_idx=${3:-98}
q_idx=${4:-1}
year=${5:-2024}
zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
bucket_name=${zone}-${year}
dst_dir=${HOME}/gvsm/predictions_${year}/${tile_id}_GTiff
mkdir -p $dst_dir
src_dir=lumi-465001846-private:${bucket_name}/predictions_GTiff_${year}/${tile_id}/RH${rh_idx}_Q${q_idx}_uncompressed.tif
rclone copy ${src_dir} ${dst_dir} --transfers=16 --checkers=16 --multi-thread-streams=4
;;

03)
# =======================================
#    DOWNLOAD ONE CORRECTED RH PREDICTIONS (cog) FROM LUMIO
# =======================================
lumi_project=465001846
remote=lumi-${lumi_project}-private:
zone=$(echo ${tile_id:0:3} | tr '[:upper:]' '[:lower:]')
bucket_name=${zone}-${year}
remote=${remote}${bucket_name}/${tile_id}/RH${rh_idx}_Q${q_idx}.tif
dst_dir=${HOME}/gvsm/predictions_${year}/${tile_id}_cog/
mkdir -p $dst_dir
echo $remote
echo $dst_dir
rclone copy ${remote} ${dst_dir} --transfers=16 --checkers=16 --multi-thread-streams=4
;;

04)
# =======================================
#    DOWNLOAD ONE CORRECTED RH PREDICTIONS (geotiff) FROM Hendrix for a list of tiles
# =======================================
rh_idx=25
for tile_id in $(cat temp.txt); do
    dst_dir=${HOME}/gvsm/predictions_${year}/${tile_id}_corrected
    mkdir -p $dst_dir
    src_dir=/home/ksb781/data/gvs/deploy/predictions_corrected_blended_v1_${year}/${tile_id}
    host=ksb781@hendrixgate02fl
    scp -r $host:$src_dir/RH${rh_idx}_Q${q_idx}.tif $dst_dir/
done
;;

esac