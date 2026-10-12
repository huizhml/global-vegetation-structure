#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=download_gedi
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

# Dispatch named tasks. Pass the task name as $1; per-task args use key=value
# (parsed by parse_kv_args in scripts/core/utils.sh).
#
#   sbatch scripts/hendrix/download_gedi.sh download year=2020
#   sbatch scripts/hendrix/download_gedi.sh add_mgrs parq_dir=<.../all_valid/2019>
#   sbatch scripts/hendrix/download_gedi.sh to_s2_tiles year=2019

source scripts/core/utils.sh
# print() to a SLURM log file is block-buffered; without this the progress
# lines show up minutes late (or only at exit)
export PYTHONUNBUFFERED=1

case $1 in
download)
# ---------------------------------------
#   Download all valid GEDI L2A orbits for each S2 tile's growing season.
#   Writes to ~/data/gvs/gedi/veg_sensitivity_all/all_valid/<year>/
#   (sensitivity_threshold defaults to null — no sensitivity filter).
#   Per-tile outcomes go to <save_dir>_state (tools/task_state.py).
# ---------------------------------------
# Args: year (default: 2020), save_dir (optional, overrides the config path)
parse_kv_args "${@:2}" || exit $?
# year=${year:-2020}

python -m download.run run=download_all_valid_gedi run.year=$year run.key_file=$key_file \
    ${save_dir:+run.save_dir=$save_dir}

;;
add_mgrs)
# ---------------------------------------
#   Add the mgrs_tile column to already downloaded tile parquets (in place,
#   files that have it are skipped) and write a per-tile summary to
#   <parq_dir>_state/mgrs_summary_<time>.tsv. Do not run while `download`
#   writes into the same dir: it writes parquets non-atomically.
# ---------------------------------------
# Args: parq_dir (required)
parse_kv_args "${@:2}" || exit $?
: "${parq_dir:?parq_dir=<dir of tile parquets> is required}"

python -m tools.run run=add_mgrs_tile run.parq_dir=$parq_dir run.workers=${SLURM_CPUS_PER_TASK:-16}

;;
to_s2_tiles)
# ---------------------------------------
#   Regroup the legacy <zone>/<orbit>.parquet download (GEDI_high_sens) into
#   one parquet per S2 tile (tools/gedi_to_s2_tiles.py). Resumable.
# ---------------------------------------
# Args: year (default: 2019), workers (default: 8; each holds a whole zone in memory)
parse_kv_args "${@:2}" || exit $?
year=${year:-2019}

python -m tools.run run=gedi_to_s2_tiles run.year=$year run.workers=${workers:-8}

;;
*)
echo "Unknown task: $1" >&2
exit 1
;;
esac
