#!/bin/bash
#SBATCH --partition=ml4good
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cp_points
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

# Points for conformal calibration (tools/cp_points.py), as the 2020 subset_cal /
# subset_test: GEDI shots of the model's cal/test tiles, as many per tile as in 2020,
# with the year's Q0/Q1/Q2 at every shot (Q1 from the local COGs, Q0/Q2 from LUMI-O).
#
#   sbatch --mem=96G scripts/hendrix/cp_points.sh points year=2024 workers=16   (all three steps)
#   sbatch scripts/hendrix/cp_points.sh select year=2024
#   sbatch scripts/hendrix/cp_points.sh add_q1 year=2024
#   sbatch scripts/hendrix/cp_points.sh add_q02 year=2024
#   sbatch scripts/hendrix/cp_points.sh combine split=cal
#   sbatch --mem=128G scripts/hendrix/cp_points.sh run_cp config=config/cp/2020.yaml
#   sbatch --array=1-10 scripts/hendrix/cp_points.sh apply year=2020
#   sbatch scripts/hendrix/cp_points.sh check year=2020 tiles=01GEL,32UNG
#   source-coop login --duration 12h && sbatch --time=11:00:00 scripts/hendrix/cp_points.sh upload year=2020
#   sbatch scripts/hendrix/cp_points.sh prune year=2020 dry_run=false

source scripts/core/utils.sh
export PYTHONUNBUFFERED=1

# GDAL reads /vsis3/ with the keys of the rclone remote copy_from_lumio.sh uses
lumio_s3_env() {
    local remote="lumi-${LUMI_PROJECT:-465002698}-private"
    local conf=${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}
    _key() { awk -v s="[$remote]" -v k="$1" '$0==s {f=1; next} /^\[/ {f=0} f && $1==k {print $3}' "$conf"; }
    AWS_ACCESS_KEY_ID=$(_key access_key_id)
    AWS_SECRET_ACCESS_KEY=$(_key secret_access_key)
    [ -n "$AWS_ACCESS_KEY_ID" ] || { echo "no access_key_id for [$remote] in $conf" >&2; return 1; }
    export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
}

case $1 in
points)
# ---------------------------------------
#   select, add_q1 and add_q02 one after the other (each resumable).
# ---------------------------------------
# Args: as select / add_q1 / add_q02
bash $0 select "${@:2}" && bash $0 add_q1 "${@:2}" && bash $0 add_q02 "${@:2}"

;;
select)
# ---------------------------------------
#   GEDI shots of the model's cal/test tiles + biome, no predictions:
#   ~/data/gvs/conformal/<year>/model_split_points/gedi/<split>/<tile>.parquet
# ---------------------------------------
# Args: year (default: 2024), workers (default: 8), tiles_file
parse_kv_args "${@:2}" || exit $?
save_dir=$HOME/data/gvs/conformal/${year:-2024}/model_split_points/gedi

python -m tools.run run=select_cp_points run.year=${year:-2024} run.save_dir=$save_dir \
    run.workers=${workers:-8} ${tiles_file:+run.tiles_file=$tiles_file} || exit $?
python -m tools.run run=state_summary run.path=${save_dir}_state

;;
add_q1|add_q02)
# ---------------------------------------
#   add_q1:  Q1 at every shot of select's output       -> model_split_points/with_q1
#   add_q02: Q0/Q2 at every shot of add_q1's output     -> model_split_points/with_q012
#   Local COGs first, else LUMI-O (Q0/Q2 of 2024 are only there).
# ---------------------------------------
# Args: year (default: 2024), workers (default: 8), threads (default: 16), tiles_file
parse_kv_args "${@:2}" || exit $?
lumio_s3_env || exit 1
root=$HOME/data/gvs/conformal/${year:-2024}/model_split_points
if [ $1 = add_q1 ]; then src=gedi dst=with_q1 q='[1]'; else src=with_q1 dst=with_q012 q='[0,2]'; fi

python -m tools.run run=add_cp_predictions run.year=${year:-2024} \
    run.points_dir=$root/$src run.save_dir=$root/$dst "run.q_idxs=$q" \
    run.workers=${workers:-8} run.threads=${threads:-16} ${tiles_file:+run.tiles_file=$tiles_file} || exit $?
python -m tools.run run=state_summary run.path=$root/${dst}_state

;;
combine)
# ---------------------------------------
#   ~/data/gvs/conformal/<year>/model_split_points/<split>.parquet (all shots, as 2020).
# ---------------------------------------
# Args: split (cal|test, required), year (default: 2024), stage (default: with_q012)
parse_kv_args "${@:2}" || exit $?
: "${split:?split=cal|test is required}"
root=$HOME/data/gvs/conformal/${year:-2024}/model_split_points

python -m tools.run run=combine_cp_points run.split=$split \
    run.save_dir=$root/${stage:-with_q012} run.out_file=$root/$split.parquet

;;
run_cp)
# ---------------------------------------
#   Fit per-biome offsets on the cal points, evaluate on the test points, plot
#   (scripts/cp/run_cp.py). Output in the config's output_dir. Loads the whole
#   cal/test tables: 101 RHs need ~128G.
# ---------------------------------------
# Args: config (required), extra (passed on, e.g. "--load_corrections" or "--plot_only")
parse_kv_args "${@:2}" || exit $?
: "${config:?config=config/cp/<year>.yaml is required}"

python scripts/cp/run_cp.py --config_path $config $extra

;;
apply)
# ---------------------------------------
#   Calibrated Q0/Q2 COGs (tools/cp_apply.py) for the tiles of the release
#   manifest, in ~/data/gvs/products/vsm/<year>/calibrated/tiles/cog/<TILE>/,
#   progress in .../cog_state. Array tasks split the tile list between them.
# ---------------------------------------
# Args: year (default: 2020), tiles_file, save_dir (e.g. for a test run), workers (default: 8),
#       n_chunks (default: the array size; chunk = array task id). Give n_chunks explicitly
#       to split finer next to a running coarser array without overlap, e.g. next to
#       --array=1-8 (chunk 1 of 8 running): --array=2-8,10-16,...,58-64 n_chunks=64
parse_kv_args "${@:2}" || exit $?
year=${year:-2020}
save_dir=${save_dir:-$HOME/data/gvs/products/vsm/$year/calibrated/tiles/cog}

python -m tools.run run=apply_cp_offsets run.year=$year run.save_dir=$save_dir \
    run.workers=${workers:-8} \
    run.chunk=${chunk:-${SLURM_ARRAY_TASK_ID:-1}} run.n_chunks=${n_chunks:-${SLURM_ARRAY_TASK_COUNT:-1}} \
    ${tiles_file:+run.tiles_file=$tiles_file}
python -m tools.run run=state_summary run.path=${save_dir}_state

;;
apply_packed)
# ---------------------------------------
#   `apply` for a list of chunks inside ONE job (the QoS limits jobs per user, not
#   CPUs): `parallel` chunks at a time, the next starts when one finishes.
#   Request cpus = parallel x workers. One log per chunk in logs/.
#   sbatch --cpus-per-task=21 --mem=80G --job-name=cp_apply_2020 scripts/hendrix/cp_points.sh \
#       apply_packed year=2020 chunks=2,3,4 parallel=3 workers=7 n_chunks=64
# ---------------------------------------
# Args: chunks (required, comma separated), parallel (default: 3), n_chunks (required),
#       plus the args of apply (year, workers, save_dir, tiles_file)
parse_kv_args "${@:2}" || exit $?
: "${chunks:?chunks=<c1,c2,...> is required}" "${n_chunks:?n_chunks is required}"
pass=(); for arg in "${@:2}"; do [[ $arg == chunks=* || $arg == parallel=* ]] || pass+=("$arg"); done
for c in ${chunks//,/ }; do
    while (( $(jobs -rp | wc -l) >= ${parallel:-3} )); do wait -n; done
    echo "$(date -u +%FT%TZ) start chunk $c/$n_chunks"
    bash $0 apply "${pass[@]}" chunk=$c > logs/${SLURM_JOB_NAME:-cp_apply}-${SLURM_JOB_ID:-0}_c$c.out 2>&1 &
done
wait
echo "$(date -u +%FT%TZ) all chunks finished"

;;
check)
# ---------------------------------------
#   Written tiles vs the test points in them (tools/cp_apply.py check_cp_tiles).
# ---------------------------------------
# Args: tiles (comma separated, required), year (default: 2020), save_dir, save_file
parse_kv_args "${@:2}" || exit $?
: "${tiles:?tiles=<TILE>,<TILE> is required}"
year=${year:-2020}

python -m tools.run run=check_cp_tiles run.year=$year "run.tiles=[$tiles]" \
    ${save_dir:+run.save_dir=$save_dir} ${save_file:+run.save_file=$save_file}

;;
upload)
# ---------------------------------------
#   Calibrated Q0/Q2 of finished tiles -> Source Coop gvsm/<year>/<TILE>/ (sc_upload,
#   kind=local). Only tiles not yet fully uploaded are listed. The source.coop
#   token lasts 12 h from `source-coop login` (login node, browser), so submit
#   right after logging in; sc_upload stops cleanly before it expires and a
#   resubmit continues from <year>/calibrated/source_coop_upload.tsv (one row per
#   file: key, status, time, bytes; no per-file flag files).
# ---------------------------------------
# Args: year (default: 2020), workers (default: 128, the benchmarked knee)
parse_kv_args "${@:2}" || exit $?
year=${year:-2020}
cal_dir=$HOME/data/gvs/products/vsm/$year/calibrated
[ -n "${CONDA_PREFIX:-}" ] || { echo "CONDA_PREFIX is empty: activate the env before submitting" >&2; exit 1; }

python -m tools.run run=cp_upload_tiles run.year=$year || exit $?
if [ ! -s $cal_dir/upload_tiles.txt ]; then
    echo "nothing to upload"
    exit 0
fi
python -m deploy.run run=sc_upload \
    run.src.kind=local run.src.dir=$cal_dir/tiles/cog \
    run.src.tiles=$cal_dir/upload_tiles.txt "run.src.glob='**/RH*_Q[02].tif'" \
    run.dst.prefix=gvsm/$year/ run.state_prefix=$cal_dir/source_coop_upload run.state_format=tsv \
    run.workers=${workers:-128}

;;
prune)
# ---------------------------------------
#   Delete local calibrated tiles that are completely on Source Coop
#   (all 202 files done in source_coop_upload.tsv). Dry run unless dry_run=false.
# ---------------------------------------
# Args: year (default: 2020), dry_run (default: true)
parse_kv_args "${@:2}" || exit $?

python -m tools.run run=prune_uploaded_tiles run.year=${year:-2020} run.dry_run=${dry_run:-true}

;;
*)
echo "Unknown task: $1" >&2
exit 1
;;
esac
