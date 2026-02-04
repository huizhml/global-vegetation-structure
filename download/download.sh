#!/bin/bash
##SBATCH --account=project_465000894
#SBATCH --partition=ml4good
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude hendrixgpu06fl
#SBATCH --time=1-23:00:00
#SBATCH --job-name=download
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
hostname

is_comma_separated_list() {
    local var="$1"
    if [[ "$var" == *","* ]]; then
        echo "true"
    else
        echo "false"
    fi
}

get_list() {
    local input_var="$1"
    local year_list

    # Convert array to a comma-separated string if necessary
    if [[ "$(declare -p input_var 2>/dev/null)" =~ "declare -a" ]]; then
        input_var="${input_var[*]}"
    fi

    # Removing potential brackets at the start and end if present
    input_var="${input_var#[}"
    input_var="${input_var%]}"

    if [[ $(is_comma_separated_list "$input_var") == "true" ]]; then
        year_list="${input_var}"
    else
        year_list="${input_var},"
    fi

    echo "$year_list"
}

# Example usage
ID=$SLURM_ARRAY_TASK_ID
# echo 'reading config file ~/GEDI/download_job_hendrix_'${ID}'.txt'
# read years zones merge_zones <  ~/GEDI/download_job_hendrix_${ID}.txt
offset=${2:-0}
idx=$((ID+offset))
read zones merge_zones <<< $(sed -n ${idx}p ~/gvs/download/correct_job_config.txt)
years=${2:-[2019,2020,2021,2022]}
echo $zones 
echo $merge_zones

# zones=$1
# years=$2
# merge_zones=${3:-False}
# rewrite=${4:-False}
rewrite=False
zone_list=$(get_list "$zones")
echo "zones: $zone_list"

case $1 in
    1)
    echo "sample GEDI points"
    python -m download._1_gedi year=$3 key_file=$4
    ;;
    2)
    echo gather S2 metadata
    python -m download._2_s2_meta_gather zone="[$zone_list]" merge_zones=$merge_zones rewrite=$rewrite
    ;;
    3) echo find best S2 images
    python -m download._3_find_best_s2 zone="[$zone_list]" rewrite=$rewrite merge_zones=$merge_zones
    ;;
    4)
    echo download S2 images
    python -m download._4_download zone="[$zone_list]"
    ;;
    5) echo download deploy data
    python -m download._5_download_inference job_id=$ID year=2024
    ;;
    6)
    echo sync data from hendrix to LUMI
    TARGET_HOST="zhanghui@lumi.csc.fi:/users/zhanghui/data/train_subsets/"
    TARGET_DIR="/users/zhanghui/data/train_subsets"
    SOURCE_DIR="/home/ksb781/data/GVS/train_subsets"
    # SUBSETS=($(ls "$SOURCE_DIR"))

    sync_subset() {
        local subset_number="$1"
        echo "Syncing $subset_number..."
        echo "${SOURCE_DIR}/train${subset_number}_filtered_v1.beton"
        echo "${TARGET_HOST}/train${subset_number}_filtered_v1.beton"
        rsync -avz --progress "${SOURCE_DIR}/train${subset_number}_filtered_v1.beton" "${TARGET_HOST}"
        # echo "Finished syncing $subset."
        sleep 1
    }
    # Run rsync in parallel for all subsets
    subset_numbers=($(seq 0 19))
    for subset in "${subset_numbers[@]}"; do
        sync_subset "$subset" &
    done

    # Wait for all background jobs to finish
    wait

    echo "All subsets have been synced."


    # sed -n "${START_LINE},${END_LINE}p" "$CONFIG_FILE" | while IFS= read -r line; do
    #     # Split the line into individual folder paths
    #     IFS=',' read -ra folders <<< "$line"
    #     for folder in "${folders[@]}"; do
    #         for year in 2019 2020 2021 2022; do
    #             if [ -d "$SOURCE_DIR/$year/$folder" ]; then
    #                 echo "Syncing $folder to $TARGET_HOST:$TARGET_DIR"
    #                 rsync -avz --progress "$folder" "$TARGET_HOST:$TARGET_DIR"
    #             else
    #                 echo "Warning: $folder does not exist. Skipping."
    #             fi
    #         done

    #     done
    # done

    # echo "Sync complete!"
        ;;
    7)
    echo download S2 images for downstream task, conda env is py3
    python -m download._6_download_downstream_task_data job_id=$ID
    ;;
    8)
    echo download S2 images for specified tiles
    python -m download._5_download_inference job_id=0 year=$year specified_tiles_file=download/evaluation_tiles.txt task=download_by_api_query
    ;;
    9)
    echo download GEDI points for GVS correction for year 2024
    python -m download._1_gedi task=download_gedi_for_gvs_correction \
            correction_number_per_tile=4000 \
            year=2024 \
            save_dir=${HOME}/data/gvs/GEDI_for_correction/partitions_2024_v1 \
            exclude_used_gedi_points=True \
            used_gedi_points_dir=${HOME}/data/GVS/fitting_data_coord_partitions
    ;;
    10)
    echo download all valid GEDI points for 2020
    python -m download._1_gedi task=download_all_valid year=2020
    ;;
    11)
    echo add slope column to GEDI points for 2020
    python -m download._1_gedi task=add_slope year=2020
    ;;
    12)
    echo check slope distribution
    python -m download._1_gedi task=check_slope_distribution year=2020
    ;;
    13)
    echo download SOTA CHM data for 2020 for val data
    python -m download._7_download_sota_chm \
    location_files=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/subset_4k/2020/*.parquet \
    output_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_cal/subset_4k_with_sota_chm/2020/
    ;;

    14)
    split=${2:-cal}
    root_dir=${HOME}/data/gvs/gedi/veg_sensitivity_gt0p95/subset_${split}
    echo download SOTA CHM data for 2020 for $split data
    python -m download.run run=download_sota_chms \
        run.location_files=${root_dir}/original/2020/*.parquet \
        run.save_dir=${root_dir}/original_with_sota_chms/2020 || exit $?
    echo sanity check for two datasets
    python -m download.run run=check_two_datasets \
        run.source_dir=${root_dir}/original/2020/ \
        run.target_dir=${root_dir}/original_with_sota_chms/2020/ || exit $?

    echo make manifest
    python -m download.run run=make_manifest \
        run.data_dir=${root_dir}/original_with_sota_chms/2020/ \
        run.dataset_name=gedi_${split}_2020_with_sota_chms \
        run.root_note='' 
    ;;
esac

# python -m download.correct_order zone="[$zone_list]" merge_zones=$merge_zones


# if [[ $merge_zones == "True" ]]; then ## for small zones, processing them together as one big df
#     IFS=','
#     for year in ${year_list[@]}; do
#         echo download zones "[$zones]" $year;
#         python -u -m download.s2_download zone="[$zones]" year=$year rewrite=$rewrite
#     done
# else ## for large zones, processing them sequentially
#     IFS=','
#     for zone in ${zone_list[@]}; do
#         for year in ${year_list[@]}; do
#             echo download zone "$zone" $year;
#             python -u -m download.s2_download zone="$zone" year=$year rewrite=$rewrite
#         done
#     done
# fi

