get_options() {
    # Print the available options for bash script.
    if [[ "$1" == "list" || -z "$1" ]]; then
        echo "--- Available Options ---"
        # 1. Finds the case (e.g., 00)
        # 2. Skips the first comment line (# ====)
        # 3. Prints the second comment line
        awk '/^[[:space:]]*[0-9a-zA-Z._-]+\)/ && !/case/ {
            case_val = $0;
            getline; # skip the # === line
            getline; # get the description line
            print case_val " " $0;
        }' "$0" | sed 's/)//g'
        exit 0
    fi
}


check_if_processed() {
    local tile_id=$1
    local flag_dir=$2
    local rewrite_flag=$3
    if [ ! $rewrite_flag ]; then
        if [ -f "${flag_dir}/${tile_id}_done" ]; then
            echo 0
        else
            echo 1
        fi
    else
        rm -f "${flag_dir}/${tile_id}_done"
        echo 1
    fi
}


get_tile_id_from_txt_file() {
    # Get the tile_id from the txt file at the given line number.
    local line_num=$1
    local tile_id_file=$2

    tile_id=$(sed -n "${line_num}p" $tile_id_file)
    echo "$tile_id"
}

clean_up_local_file() {
    local save_dir=$1
    local tile_id=$2
    if [[ -n "$tile_id" ]]; then # tile_id should be non-empty
        if [ -d "${save_dir}/geotiff/${tile_id}" ]; then
            rm -rf "${save_dir}/geotiff/${tile_id}"
        fi
        if [ -d "${save_dir}/cog/${tile_id}" ]; then
            rm -rf "${save_dir}/cog/${tile_id}"
        fi
    fi
}


translate_to_cog() {
    local tif_file=$1
    local cog_file=${tif_file%.tif}.cog.tif
    gdal_translate $tif_file $cog_file -of COG -co COMPRESS=ZSTD -co MAX_Z_ERROR=0
}