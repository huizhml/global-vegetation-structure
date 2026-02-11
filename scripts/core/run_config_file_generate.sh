case $1 in 
0)
    # ================================================================
    #    Update predicted_not_ordered_tiles_2020.txt
    #    Add finished tiles to the start of the file
    # ================================================================
    year=2020
    tiles_file="${HOME}/data/gvs/deploy/predicted_not_ordered_tiles_${year}.txt"
    tiles=$(tail -n +2 $tiles_file) # header line: Name,meta_file_idx_2020
    echo "Num tiles: ${#tiles[@]}"

    unfinished_tiles=()
    finished_tiles=()
    for line in $tiles; do
        IFS=',' read -r tile_id idx <<< "$line"
        new_flag_file="${HOME}/data/gvs/deploy/inference_flags_${year}/${tile_id}_best_images_done"
        if [ ! -f "$new_flag_file" ]; then
            unfinished_tiles+=("$tile_id,$idx")
        else
            finished_tiles+=("$tile_id,$idx")
        fi
    done
    echo "Num unfinished tiles: ${#unfinished_tiles[@]}"
    echo "Num finished tiles: ${#finished_tiles[@]}"
    # Save finished tiles to a text file
    finished_tiles_file="${HOME}/data/gvs/deploy/predicted_not_ordered_tiles_${year}_v1.txt"
    echo "Name,meta_file_idx_${year}" > "$finished_tiles_file"
    printf "%s\n" "${finished_tiles[@]}" >> "$finished_tiles_file"
    printf "%s\n" "${unfinished_tiles[@]}" >> "$finished_tiles_file"
    # echo "Unfinished tiles: ${unfinished_tiles[@]}"
    ;;

1)
    # ================================================================
    #    Check unfinished tiles
    # ================================================================
    year=2020
    config_dir="${HOME}/data/GVS/Deploy/slurm_job_files_${year}"
    files=$(find $config_dir -type f -name "deploy_s2_items_${year}_part*.txt")
    total_num_tiles=0
    for file in $files; do
        tiles=$(tail -n +2 $file | cut -d',' -f1)
        mapfile -t tiles_array < <(echo "$tiles")
        num_tiles="${#tiles_array[@]}"
        total_num_tiles=$((total_num_tiles + num_tiles))
        echo "Num tiles in $file: $num_tiles"
    done
    echo "Total num tiles: $total_num_tiles"
    ;;

2) 
    # ================================================================
    #    sync deploy status
    #    e.g, remove flags if tile is not actually predicted
    # ================================================================
    year=2020
    inference_flag_dir="${HOME}/data/gvs/deploy/inference_flags_${year}"
    translated_flag_dir="${HOME}/data/gvs/deploy/translate_flags_${year}"
    prediction_dir="${HOME}/data/gvs/deploy/predictions_${year}"

    # Get tile ids (first 5 characters of file name) of both inference and translate tiles
    inference_tiles=$(find "$inference_flag_dir" -name "*_done" | sed 's|^.*/||' | cut -c1-5)
    translate_tiles=$(find "$translated_flag_dir" -name "*_done" | sed 's|^.*/||' | cut -c1-5)

    # Get the union set of inference_tiles and translate_tiles
    mapfile -t tiles < <( (echo "$inference_tiles"; echo "$translate_tiles") | sort | uniq )
    
    echo "Tiles: ${#tiles[@]}"

    for tile in "${tiles[@]}"; do
        # Correct translate flags
        # Check if there are 303 files under $prediction_dir/$tile
        num_files=$(find "${prediction_dir}/${tile}_cog" -type f | wc -l)
        if [ "$num_files" -ne 303 ]; then
            echo "Warning: $tile does not have 303 files under ${prediction_dir}/${tile} (found $num_files)."
            # Optionally, handle the anomaly here
        fi
    done

esac