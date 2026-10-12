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
#   <parq_dir>_state/mgrs_summary_<time>.tsv. Run it after `download` has
#   finished: tiles written meanwhile are missed and the summary is partial
#   (downloads from before the atomic write could also be read half-written).
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
upload_erda)
# ---------------------------------------
#   Copy downloaded all_valid years to ERDA without the all_valid level:
#     <src>/all_valid/<year>/*.parquet  ->  <dst>/<year>/
#     <src>/all_valid/<year>_state/     ->  <dst>/<year>_state/  (snapshot, mgrs summaries,
#                                          upload record uploads.tsv + erda_<time>.log)
#   `rclone copy`, never `sync`: no data on ERDA is deleted. Reruns only send
#   what is missing or changed, then `rclone check` compares sizes per file.
#   The only deletion: *.partial in <dst>/<year>, temp files of an upload that
#   was killed mid-transfer (rclone renames them when a file is complete).
#   So never run two uploads of the same year at once.
#   ERDA allows ~10 parallel sftp sessions, hence transfers + checkers <= 10.
# ---------------------------------------
# Args: years (comma-separated, required: only years whose download is complete,
#       check with state_summary first), src, dst, dry_run (any value)
parse_kv_args "${@:2}" || exit $?
: "${years:?years=<comma-separated years whose download is complete> is required}"
src=${src:-/projects/dereeco/data/gvs/modeling/gedi/veg_sensitivity_all}
dst=${dst:-ucph-erda:GEDI/veg_sensitivity_all}
opts=(--transfers 6 --checkers 4 --stats 2m --stats-one-line ${dry_run:+--dry-run})

for y in ${years//,/ }; do
    echo "===== $y: $src/all_valid/$y -> $dst/$y"
    [ -d "$src/all_valid/$y" ] || { echo "$y: $src/all_valid/$y not found, skipped"; continue; }
    state="$src/all_valid/${y}_state"
    if [ -n "$dry_run" ]; then
        rclone copy "$src/all_valid/$y" "$dst/$y" --include '*.parquet' --include run.log "${opts[@]}"
        rclone delete "$dst/$y" --include '*.partial' --dry-run
        continue
    fi

    # Upload record, kept in the year's state dir and copied to ERDA with it:
    #   uploads.tsv            one line per run; columns are named to read on their own
    #   erda_<time>.log        rclone's per-file log of that run
    mkdir -p "$state"
    t=$(date -u +%Y%m%dT%H%M%SZ)
    rclone copy "$src/all_valid/$y" "$dst/$y" --include '*.parquet' --include run.log "${opts[@]}" \
        --log-file "$state/erda_$t.log" --log-level INFO
    rclone delete "$dst/$y" --include '*.partial' --log-file "$state/erda_$t.log" --log-level INFO
    if rclone check "$src/all_valid/$y" "$dst/$y" --one-way --size-only --include '*.parquet' \
            --checkers 8 --log-file "$state/erda_$t.log" --log-level INFO; then
        same=yes; echo "$y: all parquets on ERDA"
    else
        same="no, see erda_$t.log"; echo "$y: CHECK FAILED, rerun to resend (details in $state/erda_$t.log)"
    fi
    n_local=$(find "$src/all_valid/$y" -maxdepth 1 -name '*.parquet' | wc -l)
    read -r n bytes < <(rclone size "$dst/$y" --include '*.parquet' --json \
        | python -c 'import json, sys; d = json.load(sys.stdin); print(d["count"], d["bytes"])')
    [ -f "$state/uploads.tsv" ] || printf '%s\n' "#started (UTC)	destination	parquet files local	parquet files at destination	parquet bytes at destination	sizes match local	slurm job id" > "$state/uploads.tsv"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$dst/$y" "$n_local" "$n" "$bytes" "$same" "${SLURM_JOB_ID:-local}" >> "$state/uploads.tsv"

    # years downloaded before the TaskState existed have no log/ or snapshot
    if [ -d "$state/log" ] || [ -f "$state/snapshot.tsv" ]; then
        # merge the per-process logs so snapshot.tsv holds every tile
        python -m tools.run run=state_compact run.path="$state"
        python -m tools.run run=state_summary run.path="$state"
    fi
    rclone copy "$state" "$dst/${y}_state" --include snapshot.tsv --include expected.tsv --include 'mgrs_summary_*.tsv' \
        --include uploads.tsv --include 'erda_*.log' "${opts[@]}"
done

;;
*)
echo "Unknown task: $1" >&2
exit 1
;;
esac
