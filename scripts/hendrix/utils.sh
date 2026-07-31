#!/bin/bash

greet() {
    echo "Hello, $1! Welcome to the script."
}


run_inference() {
    local tile_id=$1
    local stream_input=$2
    local save_dir=$3
    local meta_file=$4
    local year=$5

    printf '>%.0s' {1..10}
    echo "Processing tile ID: $tile_id"
    run_id=cg11fpjr
    echo run prediction for model $run_id for tile $tile_id;
    args=(
        predict -c config/predict.yaml --model.backbone config/model/xception_mix_order.yaml
        --data.init_args.tile_id $tile_id
        --data.init_args.metadata_file $meta_file
        --data.init_args.s2_grid_file ${HOME}/data/gvs/state/s2_tiles_with_growing_months.parquet
        --data.init_args.pred_fp ${HOME}/data/gvs/inputs/inference_${year}
        --data.init_args.prediction_dir ${save_dir}/${tile_id}
        --data.init_args.year $year
        --data.init_args.debug False 
        --trainer.logger.init_args.resume False 
        --trainer.logger.init_args.offline True 
        --trainer.logger.init_args.save_dir /tmp 
        --trainer.logger.init_args.id $run_id
    )
    if [ "$stream_input" == "True" ]; then
        echo "Stream input"
        args+=(--data.init_args.stream_input True 
                --data.init_args.download_data True 
        )
    fi

    python run.py ${args[@]}
    # Capture the exit status of the command
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Prediction command failed with exit status $exit_status for tile $tile_id"
        rm -rf $save_dir/${tile_id}
    else
        echo "Prediction command completed successfully"
        touch ${inference_flag}
    fi
    printf '>%.0s' {1..10}

}

# GeoTIFF -> COG for one tile.
#   run_translate <TILE_ID> [YEAR] [VARIANT] [PROFILE]
# VARIANT is the product flavour under products/vsm/<year>: `original` or
# `masked`. It reads <variant>/tiles/geotiff/<tile> and writes
# <variant>/tiles/cog/<tile>.
#
# PROFILE must stay LERC_ZSTD to match the COGs already in
# original/tiles/cog (LERC_ZSTD, 1024 blocks, 5 NEAREST overviews, lossless
# because rio-cogeo is called with MAX_Z_ERROR=0). The op's own default is
# plain ZSTD, which produces ~40% larger files -- that is how the one
# already-converted masked tile (60WVT) ended up different from every
# original/ COG, so pass it explicitly rather than relying on the default.
run_translate() {
    local tile_id=$1
    local year=${2:-2020}
    local variant=${3:-original}
    local profile=${4:-LERC_ZSTD}

    printf '>%.0s' {1..10}
    echo "Translating $variant predictions for tile $tile_id in year $year"
    local base=${HOME}/data/gvs/products/vsm/${year}/${variant}/tiles
    local scr_dir=${base}/geotiff/${tile_id}
    local dst_dir=${base}/cog/${tile_id}
    if [ -d $scr_dir ]; then
        echo "Source directory $scr_dir exists"
    else
        echo "Source directory $scr_dir does not exist"
        return 1
    fi

    # Skip tiles a previous array task already finished. The conversion is
    # 1.7M files across 5674 tiles, so re-running the whole array to pick up
    # stragglers has to be cheap.
    local n_src=$(ls -1 ${scr_dir}/*.tif 2>/dev/null | wc -l)
    local n_dst=$(ls -1 ${dst_dir}/*.tif 2>/dev/null | wc -l)
    if [ "$n_dst" -ge "$n_src" ] && [ "$n_src" -gt 0 ]; then
        echo "Tile $tile_id already has $n_dst/$n_src COGs, skipping"
        printf '>%.0s' {1..10}
        return 0
    fi

    # SLURM inherits the submitting shell's env (--export=ALL), so submitting
    # without an activated env leaves $CONDA_PREFIX empty and python resolves to
    # the system 3.6 -- 5674 array tasks all failing the same way.
    if [ -z "${CONDA_PREFIX}" ]; then
        echo "CONDA_PREFIX is empty: activate the env before submitting (conda activate inference)" >&2
        return 1
    fi

    # Non-interactive shells do not get the env's activate hooks, and without
    # PROJ_DATA the PROJ database cannot be opened -- EPSG:32660 then fails to
    # resolve and the CRS written into the COG is degraded. Silent, and it
    # would hit every file in the array.
    export PROJ_DATA=${PROJ_DATA:-${CONDA_PREFIX}/share/proj}

    python -m postprocessing.run run=translate_predictions \
        run.src_dir=$scr_dir run.dst_dir=$dst_dir run.profile=$profile
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Translation command failed with exit status $exit_status for tile $tile_id"
    else
        echo "Translation command completed successfully"
        # Only submit.sh sets translate_flag; bare `touch` errors without it.
        [ -n "${translate_flag}" ] && touch "${translate_flag}"
    fi
    printf '>%.0s' {1..10}
    return $exit_status
}

read_line_from_txt() {
    local txt_file=$1
    local line_num=$2
    local line=$(sed -n "${line_num}p" $txt_file)
    echo $line
}

read_line_from_csv() {
    local csv_file=$1
    local line_num=$2

    tile_id=$(awk -F',' -v n="$line_num" 'NR==n {print $2}' "$csv_file")
    # get last year
    # year=$(awk -F',' -v n="$line_num" 'NR==n {print $4}' "$csv_file" | sed 's/.*[–-]//') # matching last year in a range
    year=$(awk -F',' -v n="$line_num" 'NR==n {print $4}' "$csv_file" | sed 's/[–\-].*//') # matching first year in a range
    year=$(printf '%s' "$year" | tr -d '\r' | xargs)
    if [[ "$year" =~ ^[0-9]+$ ]] && [ "$year" -le 2016 ]; then
        year=2017
    fi
    echo $tile_id $year
}