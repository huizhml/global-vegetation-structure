#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mincpus=4
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=deploy
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu26fl
#SBATCH --exclude hendrixgpu12fl,hendrixgpu11fl,hendrixgpu26fl
echo "***************************** JOB INFO *****************************"
echo "Host: $HOSTNAME"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Number of tasks: $SLURM_NTASKS"
echo "Time requested: $SLURM_TIMELIMIT"
# scontrol show job $SLURM_JOB_ID | grep "TRES="
echo "********************************************************************"

source scripts/hendrix/utils.sh
root_data_dir=/projects/dereeco/data/gvs

# **************************************************************
#    Named modes: sbatch scripts/hendrix/run_deploy.sh <mode> [args]
# **************************************************************

# Time series: predict every growing-season image (cloud cover below
# max_cloud_cover) of one tile separately, no yearly median.
# Variables and threshold: config/predict_timeseries.yaml
#   sbatch scripts/hendrix/run_deploy.sh predict_timeseries <tile_id> [year]
# Inputs go to their own folder, so they never mix with the yearly 20-image
# h5 files of the same tile.
# Result: ~/data/gvs/deploy/timeseries/<year>/<tile_id>/RH25_Q1.tif,
#         RH98_Q1.tif (one band per image) + images.csv
predict_timeseries() {
    local tile_id=${1:?usage: run_deploy.sh predict_timeseries <tile_id> [year]}
    local year=${2:-2020}
    local save_dir=${HOME}/data/gvs/deploy/timeseries/${year}
    mkdir -p $save_dir
    run_inference $tile_id True $save_dir none $year \
        -c config/predict_timeseries.yaml \
        --data.init_args.pred_fp ${HOME}/data/gvs/deploy/inputs/timeseries_${year}
}

# Reproducibility check: predict one tile with a reference version of the code
# and with the current code in this folder, then check the yearly GeoTIFFs
# match file by file.
#   sbatch --time=02:00:00 scripts/hendrix/run_deploy.sh check_reproducibility <tile_id> [year] [ref] [quick]
# quick=True (default) predicts only the top-left 5x5 patches (debug mode, the
# same in old and new code) and compares that 2560x2560 px region.
# quick=False predicts and compares the whole tile (~1 h more).
# Use a tile whose input h5 already exists, e.g. 32UNG, to skip the download.
#
# This folder is synced by SFTP, not git, so the reference code has to be
# uploaded first as a snapshot of the commit. On the laptop, in the repo:
#   git archive <ref> | ssh ksb781@hendrixgate02fl "mkdir -p ~/gvs_versions/<ref> && tar -x -C ~/gvs_versions/<ref>"
#
# Result: ~/data/gvs/deploy/reproducibility/<ref>_vs_<new_label>/<tile_id>/
#         per_file_diff.csv + check.log. Predictions are deleted if reproduced.
check_reproducibility() {
    local tile_id=${1:?usage: run_deploy.sh check_reproducibility <tile_id> [year] [ref] [quick]}
    local year=${2:-2020}
    local ref=${3:-74692d2} # last commit before the "all images in one sample" refactor
    local quick=${4:-True}
    local new=synced_$(date +%Y%m%d)
    [ "$quick" == "True" ] && new=${new}_quick
    local out_dir=${HOME}/data/gvs/deploy/reproducibility/${ref}_vs_${new}/${tile_id}
    local repo_dir=$(pwd)
    local ref_code=${HOME}/gvs_versions/${ref}

    if [ ! -f $ref_code/run.py ]; then
        echo "Reference code not found at $ref_code. Upload it from the laptop first:" >&2
        echo "  git archive $ref | ssh ksb781@hendrixgate02fl \"mkdir -p ~/gvs_versions/$ref && tar -x -C ~/gvs_versions/$ref\"" >&2
        return 1
    fi
    # run.py downloads the checkpoint into ./artifacts; reuse this folder's copy
    [ -d $repo_dir/artifacts ] && [ ! -e $ref_code/artifacts ] && ln -s $repo_dir/artifacts $ref_code/artifacts
    mkdir -p $ref_code/logs $out_dir/ref $out_dir/new

    local predict_args=() check_args=()
    if [ "$quick" == "True" ]; then
        predict_args=(--data.init_args.debug True)
        check_args=("run.region=[0,0,2560,2560]") # 5x5 patches x 512 px (patch_size 544 - 2 x border 16)
    fi
    # Streaming input (~/data/gvs/deploy/inputs/inference_<year>/<tile>.h5, downloaded if
    # missing), as in the production runs. The first run downloads, the second reuses it.
    (cd $ref_code && run_inference $tile_id True $out_dir/ref none $year "${predict_args[@]}")
    run_inference $tile_id True $out_dir/new none $year "${predict_args[@]}"

    python -m tools.run run=check_reproducibility run.tile_id=$tile_id run.save_dir=$out_dir \
        run.ref_label=$ref run.new_label=$new "${check_args[@]}" 2>&1 | tee $out_dir/check.log

    # A full prediction set is ~73 GB (303 uncompressed GTiffs); keep them only if
    # they differ, for debugging. per_file_diff.csv and check.log always stay.
    if grep -q -- '-> REPRODUCED' $out_dir/check.log; then
        echo "Reproduced, deleting $out_dir/ref and $out_dir/new"
        rm -rf $out_dir/ref $out_dir/new
    else
        echo "Not reproduced (or check failed), keeping predictions in $out_dir for debugging"
    fi
}

# Yearly (median) prediction for a worklist of tile-years, one per array task.
# Worklist: CSV with header tile_id,year (python -m download.run run=dk_tile_years / fi_tile_years)
#   sbatch --array=1-<n_rows> scripts/hendrix/run_deploy.sh predict_worklist <worklist>
# Progress: <worklist without .csv>_state/predict, key <year>_<tile_id>
#   python -m tools.run run=state_summary run.path=<worklist without .csv>_state/predict
# Never overwrites: a tile-year with GeoTIFFs or COGs that this state knows
# nothing about is an existing product and is skipped. A tile-year left as
# "started" whose job is no longer in squeue was killed: its outputs are removed
# and it is redone, so resubmitting the whole array only redoes what is not done.
# One still running in another job is skipped.
# Input h5 files stay in ~/data/gvs/deploy/inputs/inference_<year>/.
# Then convert to COG: sbatch --array=1-<n_rows> scripts/hendrix/run_cpu.sh translate_worklist <worklist>
predict_worklist() {
    local worklist=${1:?usage: run_deploy.sh predict_worklist <worklist.csv>}
    local line_num=$(( ${SLURM_ARRAY_TASK_ID:-1} + 1 )) # line 1 is the header
    local tile_id year
    IFS=, read -r tile_id year <<< "$(sed -n "${line_num}p" $worklist | tr -d '\r')"
    [ -z "$tile_id" ] && { echo "No row $line_num in $worklist" >&2; return 1; }
    echo "Tile $tile_id, year $year (row $line_num of $worklist)"

    local state_dir=${worklist%.*}_state/predict key=${year}_${tile_id}
    local status=$(state_status $state_dir $key | cut -f1)
    if [ "$status" == done ]; then
        echo "Already done in $state_dir, skipping"
        return 0
    fi
    local tiles_dir=${HOME}/data/gvs/products/vsm/${year}/original/tiles
    local n_geotiff=$(( $(ls ${tiles_dir}/geotiff/${tile_id}/*.tif 2>/dev/null | wc -l) ))
    local n_cog=$(( $(ls ${tiles_dir}/cog/${tile_id}/*.tif 2>/dev/null | wc -l) ))
    if [ "$n_cog" -gt 0 ] || { [ "$n_geotiff" -gt 0 ] && [ -z "$status" ]; }; then
        echo "Existing product ($n_geotiff GeoTIFFs, $n_cog COGs in ${tiles_dir}), not overwriting"
        state_mark $state_dir $key done "existing product, not predicted ($n_geotiff GeoTIFFs, $n_cog COGs)"
        return 0
    fi
    if [ "$status" == started ]; then
        # running in another job (array resubmitted too early) or killed
        local prev_job=$(state_status $state_dir $key | cut -f2 | grep -o 'job [0-9]*' | awk '{print $2}')
        if [ -n "$prev_job" ] && [ "$prev_job" != "${SLURM_JOB_ID:-}" ]; then
            job_alive "$prev_job"
            case $? in
                0) echo "Still running in job $prev_job, skipping"; return 0 ;;
                2) echo "squeue failed, cannot tell whether job $prev_job still runs; skipping" >&2; return 1 ;;
            esac
        fi
    fi
    if [ "$n_geotiff" -gt 0 ]; then
        echo "Removing $n_geotiff GeoTIFFs left by an earlier attempt ($(state_status $state_dir $key))"
        rm -rf ${tiles_dir}/geotiff/${tile_id}
    fi

    mkdir -p ${tiles_dir}/geotiff
    # Stays "started" if the job is killed, so the next attempt knows the outputs
    # are ours; a finished run overwrites it with done or failed.
    # scancel and the time limit send SIGTERM to the whole job before SIGKILL. The
    # trap keeps this shell alive until the outcome is recorded; it stays set until
    # then, since the shell's own SIGTERM can arrive after the children are gone.
    # SIGKILL, OOM or a node failure leave "started", which state_summary reports
    # as interrupted once the job is no longer in squeue.
    local sigterm=""
    trap 'sigterm="SIGTERM (scancel or time limit)"' TERM
    state_mark $state_dir $key started "job ${SLURM_JOB_ID:-local}"
    local run_log=$(mktemp) t0=$SECONDS rc
    run_inference $tile_id True ${tiles_dir}/geotiff none $year 2>&1 | tee $run_log
    rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ] && [ -z "$sigterm" ]; then
        state_mark $state_dir $key done "$(( SECONDS - t0 ))s"
    else
        [ -z "$sigterm" ] && [ $rc -gt 128 ] && sigterm="killed by signal $(( rc - 128 ))"
        # e.g. "RuntimeError: 32UNG has no images in growing months, ..."
        state_mark $state_dir $key failed "rc=$rc $sigterm $(grep -o '[A-Za-z]*Error: .*' $run_log | tail -1)"
    fi
    trap - TERM
    rm -f $run_log
    [ -n "$sigterm" ] && return 143
    return $rc
}

case $1 in
1) 
# **************************************************************
#    Predict one tile in one job, get tile_id from txt file
# **************************************************************
line_num=${SLURM_ARRAY_TASK_ID:-2}
stream_input=${2:-False}
year=${4:-2020}
repredict_tiles=${4:-True}
save_dir=${root_data_dir}/products/vsm/${year}/original/tiles/geotiff
mkdir -p $save_dir

tile_id_file=${3:-${root_data_dir}/assets/worklists/tiles_duplicated.txt}
tile_id_name=$(read_line_from_txt $tile_id_file $line_num)
meta_file=none
echo "Processing tile ID: $tile_id, line $line_num from $tile_id_file"
echo "meta_file: $meta_file"

run_inference $tile_id $stream_input $save_dir $meta_file $year

;;
2)
# **************************************************************
#    Predict one tile in one job, get tile_id from csv file
# **************************************************************
line_num=${SLURM_ARRAY_TASK_ID:-2}
stream_input=${2:-True}
repredict_tiles=${3:-False}
lvis_year=${4:-2023}
# tile_id_file=${root_data_dir}/evaluation/with_airborne_lidar/meta_als.csv
tile_id_file=${root_data_dir}/evaluation/with_airborne_lidar/meta_lvis${lvis_year}_profile.csv

read tile_id year <<< $(read_line_from_csv "$tile_id_file" "$line_num")

echo "Processing tile ID: $tile_id, year: $year"
save_dir=${root_data_dir}/products/vsm/${year}/original/tiles/geotiff
mkdir -p $save_dir

meta_file=none
echo "Processing tile ID: $tile_id, line $line_num from $tile_id_file"
echo "meta_file: $meta_file"

run_inference $tile_id $stream_input $save_dir $meta_file $year

;;
predict_timeseries|check_reproducibility|predict_worklist)
"$@"
;;
*)
echo "Invalid option"
exit 1
;;
esac

# echo "***************************** SYNC DATA TO SCRATCH *****************************"
# mkdir -p /scratch/${tile_id}
# t0=$(date +%s)
# data_dir="${HOME}/data/gvs/deploy/inference_${year}.zarr"
# cd "${data_dir}"
# TAR_FILE="${data_dir}/${tile_id}.tar"
# if [ ! -f "${TAR_FILE}" ]; then
#     echo "Creating ${TAR_FILE}..."
#     tar cf - "${tile_id}" | tee "${TAR_FILE}" | tar xf - -C /scratch/
# else
#     echo "${TAR_FILE} already exists. Skipping creation and extracting..."
#     tar xf "${TAR_FILE}" -C /scratch/
# fi
# cd ~/gvs
# # rsync -a --stats ${HOME}/data/gvs/deploy/inference_${year}.zarr/${tile_id} /scratch/
# t1=$(date +%s)
# duration=$((t1-t0))
# hours=$((duration / 3600))
# minutes=$(( (duration % 3600) / 60 ))
# seconds=$((duration % 60))
# echo "Time taken to sync data: ${hours}h ${minutes}m ${seconds}s"

