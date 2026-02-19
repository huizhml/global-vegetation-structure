# 1. Create a secure temporary file
tile_id=$1
year=${2:-2020}
q_idx=${3:-1}
rh_idx=${4:-*}
pred_root_dir=${HOME}/data/gvs/predictions/${year}/blended/tiles
pred_dir=${pred_root_dir}/geotiff
tif_dir=${pred_dir}/${tile_id}
output_file=${pred_root_dir}/multibands/${tile_id}_RH_profile_Q${q_idx}.tif
mkdir -p ${pred_root_dir}/multibands || exit 1
temp_list=$(mktemp)

# 2. Find and sort your files into that temp file
ls ${tif_dir}/RH${rh_idx}_Q${q_idx}.tif | sort -V > "$temp_list" || exit 1

# 3. Build the VRT using the temp list
# Using -separate to ensure each file becomes its own band
vrt_file=${pred_root_dir}/multibands/${tile_id}_stacked.vrt
gdalbuildvrt -separate -input_file_list "$temp_list" ${vrt_file} || exit 1

# 4. Inject filenames as Band Descriptions into the VRT
while read -r filename; do
    bandname=$(basename "$filename" .tif)
    # We use sed to insert the Description tag before the SourceFilename tag
    sed -i "/$(basename "$filename")/i \    <Description>$bandname</Description>" ${vrt_file}
done < "$temp_list"

# 5. Convert to final GeoTIFF with LZW compression
gdal_translate ${vrt_file} ${output_file} -co COMPRESS=LZW || exit 1

# 6. Cleanup: Remove the temp file and the intermediate VRT
rm "$temp_list" ${vrt_file}

echo "Processing complete. File created: final_stacked.tif"