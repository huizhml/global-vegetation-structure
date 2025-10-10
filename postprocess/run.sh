
PX=10    # pixel size X
PY=10  
tile_ids=(20LPP 20LQP 20LPQ 20LQQ)
year=2020
EPSG=32720
NODATA=32767
T=64
mkdir -p tmp/valid tmp/dist tmp/alpha tmp/warped tmp/alpha_w tmp/contrib
for tile_id in ${tile_ids[@]}; do
    file_path=${HOME}/GVS/Prediction${year}/${tile_id}_RH98_Q1.cog.tif
    echo "Processing tile ID: $file_path"
    gdal_calc.py -A "${file_path}" --calc="A!=${NODATA}" --NoDataValue=0 --outfile="tmp/valid/${tile_id}_valid.tif"

    # 1b) distance (in pixels) to outside (edge/NoData)
    gdal_calc.py -A "tmp/valid/${tile_id}_valid.tif" --calc="1-A" --type=Byte --outfile="tmp/${tile_id}_outside.tif" --overwrite
    gdal_proximity.py "tmp/${tile_id}_outside.tif" "tmp/dist/${tile_id}_dist.tif" -distunits PIXEL -q

    # 1c) linear ramp: alpha = min(1, dist/T) * valid   → Float32 in [0,1]
    gdal_calc.py -A "tmp/dist/${tile_id}_dist.tif" -B "tmp/valid/${tile_id}_valid.tif" \
        --calc="minimum(1, A/$T) * B" --type=Float32 \
        --outfile="tmp/alpha/${tile_id}_alpha_f32.tif" --overwrite
done


for tile_id in ${tile_ids[@]}; do
    file_path=${HOME}/GVS/Prediction${year}/${tile_id}_RH98_Q1.cog.tif
    gdalwarp -overwrite -r bilinear -multi -wo NUM_THREADS=ALL_CPUS \
        -t_srs EPSG:$EPSG -tr $PX $PY -tap \
        -srcnodata $NODATA -dstnodata $NODATA \
        "${file_path}" "tmp/warped/${tile_id}.tif"

    gdalwarp -overwrite -r bilinear -multi -wo NUM_THREADS=ALL_CPUS \
        -t_srs EPSG:$EPSG -tr $PX $PY -tap \
        -srcnodata 0 -dstnodata 0 \
        "tmp/alpha/${tile_id}_alpha_f32.tif" "tmp/alpha_w/${tile_id}_alpha.tif"
done


first_num=1
first_den=1
for f in tmp/warped/*.tif; do
  base=$(basename "$f" .tif)
  a="tmp/alpha_w/${base}_alpha.tif"

  # contribution = raster * alpha
  gdal_calc.py -A "$f" -B "$a" --calc="A*B" --type=Float32 \
    --outfile="tmp/contrib/${base}_contrib.tif" --overwrite

  # rolling sums (no limit on number of tiles)
  if [ $first_num -eq 1 ]; then
    gdal_translate "tmp/contrib/${base}_contrib.tif" NUM.tif
    gdal_translate "$a" DEN.tif
    first_num=0; first_den=0
  else
    gdal_calc.py -A NUM.tif -B "tmp/contrib/${base}_contrib.tif" \
      --calc="A+B" --type=Float32 --outfile=NUM.tif --overwrite
    gdal_calc.py -A DEN.tif -B "$a" \
      --calc="A+B" --type=Float32 --outfile=DEN.tif --overwrite
  fi
done

# final mosaic: safe divide
gdal_calc.py -A NUM.tif -B DEN.tif \
  --calc="A/(B+1e-6)" --type=Float32 --NoDataValue=$NODATA \
  --outfile=mosaic_blend.tif --overwrite