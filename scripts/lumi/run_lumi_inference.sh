#!/bin/bash
#SBATCH --account=project_465002698
#SBATCH --partition=small-g
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --job-name=inference
#SBATCH --output=/users/zhanghui/scratch/logs/%x-%A_%a.out
#SBATCH --error=/users/zhanghui/scratch/logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu26fl
##SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu08fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch

source scripts/core/utils.sh
source scripts/lumi/utils.sh
echo_job_info
init_env

case $1 in
0)
# =======================================
#    Predict one tile in one job, get tile_id from txt file
# =======================================
line_num=${SLURM_ARRAY_TASK_ID:-0}
year=${2:-2024}
use_flash=${3:-False}
meta_file=${5:-null}

data_root_dir=$(get_data_root_dir $use_flash)
tile_list_file=${data_root_dir}/assets/worklists/tiles_repredict_${year}.txt
tile_id=$(get_tile_id_from_txt_file $line_num $tile_list_file)
save_dir=${data_root_dir}/predictions/${year}/original/tiles/
mkdir -p $save_dir
run_inference $tile_id $save_dir $meta_file $year || { exit $?; }
run_translate $tile_id $save_dir || { exit $?; }
# sync_to_lumi $tile_id $save_dir $year || { exit $?; }
exit 0
;;
predict_tiles)
# =======================================
#    One tile per array task (line SLURM_ARRAY_TASK_ID of tiles_file): predict
#    Q0/Q1/Q2 of all RHs, translate to COG, delete the GeoTIFFs once all 303 COGs
#    exist. Output <save_dir>/cog/<tile>/; streamed inputs in <h5_root>/inputs/.
#    Used for the stability test (tools/vsm_stability.py), e.g.
#    sbatch --account=project_465003168 --array=1-39 scripts/lumi/run_lumi_inference.sh predict_tiles \
#        tiles_file=$S/vsm_stability_state/tiles_2024.txt save_dir=$S/vsm_stability/2024/new_prediction \
#        h5_root=$S/vsm_stability_state     (S=/scratch/project_465003168/hui/gvs)
#    quantiles=1 converts only the median (RH*_Q1, 101 COGs); the model still
#    predicts all three.
# =======================================
# Args: tiles_file, save_dir, h5_root (required), year (default: 2024),
#       quantiles (digits of Q0/Q1/Q2 to keep as COG, default 012)
parse_kv_args "${@:2}" || exit $?
: "${tiles_file:?}" "${save_dir:?}" "${h5_root:?}"
year=${year:-2024}
quantiles=${quantiles:-012}
cog_pattern="RH*_Q[${quantiles}].tif"
n_expected=$(( 101 * ${#quantiles} ))
tile_id=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" $tiles_file | tr -d '[:space:]')
[ -n "$tile_id" ] || { echo "no tile at line ${SLURM_ARRAY_TASK_ID:-1} of $tiles_file" >&2; exit 1; }
export S2_GRID_FILE=${S2_GRID_FILE:-/scratch/project_465003168/hui/gvs/01_derived_ancillary/growing_months/s2_tiles_with_growing_months.parquet}
data_root_dir=$h5_root          # run_inference streams inputs to $data_root_dir/inputs/inference_<year>
mkdir -p $save_dir/geotiff/$tile_id $data_root_dir/inputs/inference_${year}
if [ $(ls $save_dir/cog/$tile_id/$cog_pattern 2>/dev/null | wc -l) -eq $n_expected ]; then
    echo "$tile_id already done"; exit 0
fi
run_inference $tile_id $save_dir none $year || exit $?
run_translate $tile_id $save_dir "$cog_pattern" || exit $?
n=$(ls $save_dir/cog/$tile_id/$cog_pattern 2>/dev/null | wc -l)
if [ $n -eq $n_expected ]; then
    rm -rf $save_dir/geotiff/$tile_id
    echo "$tile_id: $n_expected COGs, GeoTIFFs deleted"
else
    echo "$tile_id: only $n of $n_expected COGs, GeoTIFFs kept" >&2; exit 1
fi
exit 0
;;
predict_sample_worklist)
# =======================================
#    Multi-year stability test (tools/vsm_stability.py): line SLURM_ARRAY_TASK_ID+1 of
#    worklist (tsv tile, year) -> points of the tile-year (fixed pixels + GEDI shots,
#    run=write_stability_points) -> predict keeping only these points, no GeoTIFF
#    (S2DatasetStream sample_points_file) -> <save_dir>/point_samples/<year>/<tile>.parquet
#    (all 303 bands, kept) -> pixel_values / gedi_values (run=sample_stability_tile).
#    Progress: TaskState <state_dir>, key <year>_<tile>; done keys are skipped;
#    "no images" is recorded as done.
#    S=/scratch/project_465003168/hui/gvs; W=$S/vsm_stability_state/2017_2026
#    sbatch --account=project_465003168 --time=4:00:00 --array=1-N%65 scripts/lumi/run_lumi_inference.sh \
#        predict_sample_worklist worklist=$W/worklist.tsv save_dir=$S/vsm_stability/2017_2026 \
#        h5_root=$S/vsm_stability_state state_dir=$W/predict_sample
#    Optional per_image_vars='[RH98_Q1]': also every image's value at the points
#    -> point_samples/<year>/<tile>_per_image.parquet
# =======================================
parse_kv_args "${@:2}" || exit $?
: "${worklist:?}" "${save_dir:?}" "${h5_root:?}" "${state_dir:?}"
line=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-1} + 1 ))p" $worklist)
tile_id=$(echo "$line" | cut -f1 | tr -d '[:space:]')
year=$(echo "$line" | cut -f2 | tr -d '[:space:]')
[ -n "$tile_id" ] && [ -n "$year" ] || { echo "no tile/year at task ${SLURM_ARRAY_TASK_ID:-1} of $worklist" >&2; exit 1; }
key=${year}_${tile_id}
mark() {   # mark <started|done|failed> [info]
    python -c "import sys; from tools.task_state import TaskState as T; s=T(sys.argv[1]);
getattr(s, {'started':'mark_started','done':'mark_done','failed':'mark_failed'}[sys.argv[3]])(sys.argv[2], sys.argv[4])" \
        "$state_dir" "$key" "$1" "${2-job ${SLURM_JOB_ID:-local}}"
}
if python -c "import sys; from tools.task_state import TaskState as T; sys.exit(0 if T(sys.argv[1]).is_done(sys.argv[2], refresh=True) else 1)" "$state_dir" "$key"; then
    echo "$key already done"; exit 0
fi
mark started ""   # TaskState adds the job id itself
export S2_GRID_FILE=${S2_GRID_FILE:-/scratch/project_465003168/hui/gvs/01_derived_ancillary/growing_months/s2_tiles_with_growing_months.parquet}
tmp_dir=$h5_root/tmp_${year}_${tile_id}          # inputs + point file, deleted at the end
data_root_dir=$tmp_dir                           # run_inference streams to $data_root_dir/inputs/inference_<year>
samples=$save_dir/point_samples/$year/$tile_id.parquet
mkdir -p $tmp_dir/inputs/inference_${year} $tmp_dir/prediction/geotiff/$tile_id   # run.py needs the parent of prediction_dir
trap 'mark failed "SIGTERM (job ${SLURM_JOB_ID:-local})"; rm -rf $tmp_dir; exit 143' TERM
python -m tools.run run=write_stability_points run.tile=$tile_id run.year=$year run.out_file=$tmp_dir/points.parquet \
    || { mark failed "points (job ${SLURM_JOB_ID:-local})"; rm -rf $tmp_dir; exit 1; }
log=$tmp_dir/predict.log
run_inference $tile_id $tmp_dir/prediction none $year \
    --data.init_args.sample_points_file $tmp_dir/points.parquet \
    --data.init_args.sample_output_file $samples \
    ${per_image_vars:+--data.init_args.sample_per_image_vars "$per_image_vars"} 2>&1 | tee $log
status=${PIPESTATUS[0]}
if [ $status -ne 0 ]; then
    if grep -q "has no images" $log; then
        mark done "no images: $(grep -m1 'has no images' $log | cut -c1-200)"; rm -rf $tmp_dir; exit 0
    fi
    mark failed "predict exit $status (job ${SLURM_JOB_ID:-local})"; rm -rf $tmp_dir; exit $status
fi
rm -rf $tmp_dir
python -m tools.run run=sample_stability_tile run.tile=$tile_id run.year=$year run.samples_file=$samples run.out_dir=$save_dir \
    || { mark failed "sample (job ${SLURM_JOB_ID:-local})"; exit 1; }
mark done "predicted, points only (job ${SLURM_JOB_ID:-local})"
exit 0
;;
*)
echo "Invalid option"
exit 1
;;
esac
