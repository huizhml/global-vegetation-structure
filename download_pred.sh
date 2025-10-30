
tile_id=${2:-'22UUL'}
rh_idx=${3:-98}
year=${4:-2020}
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
#    DOWNLOAD PREDICTIONS (uncompressed GTiffs) FROM Hendrix
# =======================================
dst_dir=${HOME}/GVS/predictions_${year}/${tile_id}_GTiff
mkdir -p $dst_dir
src_dir=/home/ksb781/data/gvs/deploy/predictions_GTiff_${year}/${tile_id}_GTiff
host=ksb781@hendrixgate02fl
scp -r $host:$src_dir/RH${rh_idx}_Q${q_idx}_uncompressed.tif $dst_dir/
        ;;
esac