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

run_translate() {
    local tile_id=$1
    local year=$2

    printf '>%.0s' {1..10}
    echo "Translating predictions for tile $tile_id in year $year"
    scr_dir=${HOME}/data/gvs/predictions/${year}/original/tiles/geotiff/${tile_id}
    dst_dir=${HOME}/data/gvs/predictions/${year}/original/tiles/cog/${tile_id}
    python -m postprocessing.run run=translate_predictions run.src_dir=$scr_dir run.dst_dir=$dst_dir
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Translation command failed with exit status $exit_status for tile $tile_id"
    else
        echo "Translation command completed successfully"
        touch ${translate_flag}
    fi
    printf '>%.0s' {1..10}
}

