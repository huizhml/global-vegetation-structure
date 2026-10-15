#!/bin/bash
#SBATCH --partition=ml4good
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cp_points
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

# Points for conformal calibration (tools/cp_points.py): GEDI shots paired with
# the year's Q0/Q1/Q2, split into cal/test by tile. Q0/Q2 are read from LUMI-O.
#
#   sbatch --array=1-4 scripts/hendrix/cp_points.sh sample year=2024
#   bash scripts/hendrix/cp_points.sh sample year=2024 tiles_file=<one tile per line> workers=1
#   sbatch scripts/hendrix/cp_points.sh combine split=cal
#   python -m tools.run run=state_summary run.path=~/data/gvs/conformal/2024/points_state

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
sample)
# ---------------------------------------
#   One parquet per tile in ~/data/gvs/conformal/<year>/points/<split>/,
#   progress in .../points_state. Array tasks split the tile list between them.
# ---------------------------------------
# Args: year (default: 2024), tiles_file, save_dir (e.g. for a test run),
#       workers (default: 8), threads (default: 16)
parse_kv_args "${@:2}" || exit $?
lumio_s3_env || exit 1
save_dir=${save_dir:-$HOME/data/gvs/conformal/${year:-2024}/points}

python -m tools.run run=sample_cp_points run.year=${year:-2024} run.save_dir=$save_dir \
    run.workers=${workers:-8} run.threads=${threads:-16} \
    run.chunk=${SLURM_ARRAY_TASK_ID:-1} run.n_chunks=${SLURM_ARRAY_TASK_COUNT:-1} \
    ${tiles_file:+run.tiles_file=$tiles_file}
python -m tools.run run=state_summary run.path=${save_dir}_state

;;
combine)
# ---------------------------------------
#   ~/data/gvs/conformal/<year>/points/<split>.parquet, capped per biome.
# ---------------------------------------
# Args: split (cal|test, required), year (default: 2024), max_per_biome
parse_kv_args "${@:2}" || exit $?
: "${split:?split=cal|test is required}"

python -m tools.run run=combine_cp_points run.split=$split \
    run.save_dir=~/data/gvs/conformal/${year:-2024}/points \
    ${max_per_biome:+run.max_per_biome=$max_per_biome}

;;
*)
echo "Unknown task: $1" >&2
exit 1
;;
esac
