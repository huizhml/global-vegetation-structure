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
3)
# **************************************************************
#    Time series: predict every growing-season image (cloud cover below
#    max_cloud_cover) of one tile separately, no yearly median.
#    Variables and threshold: config/predict_timeseries.yaml
#    sbatch scripts/hendrix/run_deploy.sh 3 <tile_id> [year]
#
#    Inputs go to their own folder, so they never mix with the yearly
#    20-image h5 files of the same tile.
#    Result: ~/data/gvs/deploy/timeseries/<year>/<tile_id>/RH25_Q1.tif,
#            RH98_Q1.tif (one band per image) + images.csv
# **************************************************************
tile_id=${2:?usage: run_deploy.sh 3 <tile_id> [year]}
year=${3:-2020}
save_dir=${HOME}/data/gvs/deploy/timeseries/${year}
mkdir -p $save_dir

run_inference $tile_id True $save_dir none $year \
    -c config/predict_timeseries.yaml \
    --data.init_args.pred_fp ${HOME}/data/gvs/deploy/inputs/timeseries_${year}

;;
4)
# **************************************************************
#    Reproducibility check: predict one tile with a reference version of the
#    code and with the current code in this folder, then check the yearly
#    GeoTIFFs match file by file.
#    sbatch --time=02:00:00 scripts/hendrix/run_deploy.sh 4 <tile_id> [year] [ref] [quick]
#
#    quick=True (default) predicts only the top-left 5x5 patches (debug mode,
#    the same in old and new code) and compares that 2560x2560 px region.
#    quick=False predicts and compares the whole tile (~1 h more).
#    Use a tile whose input h5 already exists, e.g. 32UNG, to skip the download.
#
#    This folder is synced by SFTP, not git, so the reference code has to be
#    uploaded first as a snapshot of the commit. On the laptop, in the repo:
#      git archive <ref> | ssh ksb781@hendrixgate02fl "mkdir -p ~/gvs_versions/<ref> && tar -x -C ~/gvs_versions/<ref>"
#
#    Result: ~/data/gvs/deploy/reproducibility/<ref>_vs_<new_label>/<tile_id>/
#            per_file_diff.csv + check.log. Predictions are deleted if reproduced.
# **************************************************************
tile_id=${2:?usage: run_deploy.sh 4 <tile_id> [year] [ref] [quick]}
year=${3:-2020}
ref=${4:-74692d2} # last commit before the "all images in one sample" refactor
quick=${5:-True}
new=synced_$(date +%Y%m%d)
[ "$quick" == "True" ] && new=${new}_quick
out_dir=${HOME}/data/gvs/deploy/reproducibility/${ref}_vs_${new}/${tile_id}
repo_dir=$(pwd)
ref_code=${HOME}/gvs_versions/${ref}

if [ ! -f $ref_code/run.py ]; then
    echo "Reference code not found at $ref_code. Upload it from the laptop first:" >&2
    echo "  git archive $ref | ssh ksb781@hendrixgate02fl \"mkdir -p ~/gvs_versions/$ref && tar -x -C ~/gvs_versions/$ref\"" >&2
    exit 1
fi
# run.py downloads the checkpoint into ./artifacts; reuse this folder's copy
[ -d $repo_dir/artifacts ] && [ ! -e $ref_code/artifacts ] && ln -s $repo_dir/artifacts $ref_code/artifacts
mkdir -p $ref_code/logs $out_dir/ref $out_dir/new

if [ "$quick" == "True" ]; then
    predict_args=(--data.init_args.debug True)
    check_args=("run.region=[0,0,2560,2560]") # 5x5 patches x 512 px (patch_size 544 - 2 x border 16)
else
    predict_args=()
    check_args=()
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

