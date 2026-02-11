case $1 in
0)
# =======================================
#    UPLOAD data to ERDA
# =======================================
echo "Uploading data to ERDA"
python -m deploy.erda upload_data_to_erda $2
;;
1)
# =======================================
#    DOWNLOAD data from ERDA
# =======================================
echo "Downloading data from ERDA"
python -m deploy.erda download_data_from_erda $2
;;
3)
# =======================================
#    DELETE data on ERDA
# =======================================
echo "Deleting data on ERDA"
module load rclone
rclone delete ucph-erda:GVS/predictions_${year}/${tile_id}
;;

*)
echo "Invalid option"
exit 1
;;
esac