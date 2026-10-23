#!/bin/bash
#SBATCH --account=project_465003168
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=download_gedi
#SBATCH --output=/users/zhanghui/scratch/logs/%x-%j.out
#SBATCH --error=/users/zhanghui/scratch/logs/%x-%j.err

# GEDI all-valid download on LUMI (download_all_valid in download/products/gedi.py).
# ~/data/gvs on LUMI is read-only, so inputs and outputs live under $gvs_root,
# in the same layout as ~/data/gvs on Hendrix. Run from the repo root (~/gvs):
#
#   sbatch --time=01:00:00 --mem=16G --cpus-per-task=2 scripts/lumi/download_gedi.sh download year=2020 dry_run=1
#   sbatch scripts/lumi/download_gedi.sh download year=2020 only_planned=1
#
# 2026-10 rework of the Hendrix years (2019-2022, 2024), in this order:
#   1. download ... only_planned=1     repair: orbits the old downloads missed
#   2. pull_erda years=...             all other tiles from ERDA into <year>_state/erda_copy/
#   3. organize years=...              move them in place, list float-id tiles
#   4. redownload year=Y key_file=...  tiles with rounded ids, whole, pft 1-9,11
#   5. add_pft year=Y key_file=...     pft 9 and 11 for every other tile
# One GEE project per running job (keys/private-key, ee-default, gvsm), since
# concurrency and quota are per project.
#
# A repair of a year downloaded on Hendrix runs on copies: its <year>_state and
# the parquets of the tiles in the dry run's <year>_state/fetch_plan.tsv. only_planned
# keeps the run to those tiles; any other tile would look empty here and be
# marked "no valid shots" in the state.
#
# Args: year (required), prior_year (default 2022), key_file (default in the
#       config), dry_run (any value: only print what it would fetch and write
#       <year>_state/fetch_plan.tsv), only_planned (any value: only the tiles of
#       <year>_state/fetch_plan.tsv)

source scripts/core/utils.sh
source scripts/lumi/env_init.sh
# print() to a SLURM log file is block-buffered otherwise
export PYTHONUNBUFFERED=1
gvs_root=/scratch/project_465003168/hui/gvs

download() {
    local year prior_year key_file dry_run only_planned tiles_file
    parse_kv_args "$@" || exit $?
    : "${year:?year=<year> is required}"
    prior_year=${prior_year:-2022}
    local gedi=$gvs_root/gedi
    local tiles_arg=
    if [ -n "$only_planned" ]; then
        local plan=$gedi/veg_sensitivity_all/${year}_state/fetch_plan.tsv
        [ -f "$plan" ] || { echo "no $plan: run with dry_run=1 first" >&2; exit 1; }
        tiles_arg="run.tiles=[$(tail -n +2 "$plan" | cut -f1 | paste -sd, -)]"
        echo "only the $(tail -n +2 "$plan" | wc -l) tiles of $plan"
    fi
    python -m download.run run=download_all_valid_gedi run.year=$year \
        run.save_dir=$gedi/veg_sensitivity_all/$year \
        run.state_dir=$gedi/veg_sensitivity_all/${year}_state \
        run.gedi_table_index_file=$gedi/l2a_orbit_table/l2a_orbit_table_$year.parquet \
        run.s2_grid_file=$gvs_root/01_derived_ancillary/growing_months/s2_tiles_with_growing_months.parquet \
        run.prior_year=$prior_year \
        run.prior_state_dir=$gedi/veg_sensitivity_all/${prior_year}_state \
        run.prior_index_file=$gedi/l2a_orbit_table/l2a_orbit_table_$prior_year.parquet \
        ${key_file:+run.key_file=$key_file} ${dry_run:+run.dry_run=true} ${tiles_arg:+"$tiles_arg"} \
        ${tiles_file:+run.tiles_file=$tiles_file}
}

pull_erda() {
    # Copy downloaded years from ERDA to <year>_state/erda_copy/
    # (a staging copy: tiles repaired here must not be overwritten by it).
    # ERDA allows ~10 sftp sessions, hence transfers + checkers = 10.
    # Args: years (comma-separated, required)
    local years
    parse_kv_args "$@" || exit $?
    : "${years:?years=<comma-separated years> is required}"
    module load lumio-ext-tools/1.1.0
    # an sftp remote from the ssh config entry ucph-erda, without a config file
    export RCLONE_CONFIG_ERDA_TYPE=sftp RCLONE_CONFIG_ERDA_HOST=io.erda.dk \
        RCLONE_CONFIG_ERDA_USER=huzh@di.ku.dk RCLONE_CONFIG_ERDA_KEY_FILE=$HOME/.ssh/id_rsa_erda
    local y src dst
    for y in ${years//,/ }; do
        src=erda:GEDI/veg_sensitivity_all/$y
        dst=$gvs_root/gedi/veg_sensitivity_all/${y}_state/erda_copy
        echo "===== $y: $src -> $dst"
        rclone copy "$src" "$dst" --include '*.parquet' --transfers 8 --checkers 2 \
            --stats 10m --stats-one-line
        if rclone check "$src" "$dst" --one-way --size-only --include '*.parquet' --checkers 8; then
            echo "$y: all parquets copied"
        else
            echo "$y: CHECK FAILED, rerun to copy what is missing"
        fi
    done
}

organize() {
    # Move the ERDA copies (<year>_state/erda_copy/) into the year dirs, except
    # the tiles repaired here (<year>_state/fetch_plan.tsv), and list the tiles
    # with float64 shot_number in <year>_state/float_id_tiles.txt.
    # Args: years (comma-separated, required), pull_log (optional: the
    #       pull_erda log; wait until it says the year is copied, fail if its
    #       check failed, so jobs can be queued behind a running pull)
    local years y pull_log
    parse_kv_args "$@" || exit $?
    : "${years:?years=<comma-separated years> is required}"
    for y in ${years//,/ }; do
        if [ -n "$pull_log" ]; then
            until grep -qE "^$y: (all parquets copied|CHECK FAILED)" "$pull_log"; do sleep 120; done
            grep -q "^$y: all parquets copied" "$pull_log" || { echo "$y: ERDA copy check failed" >&2; exit 1; }
        fi
        python -m download.run run=organize_erda_copy_gedi run.year=$y \
            run.save_dir=$gvs_root/gedi/veg_sensitivity_all/$y
    done
}

redownload() {
    # Download the tiles of tiles_file again whole, with the current pft
    # classes (e.g. the tiles whose ids were rounded): their parquets go to
    # <year>_state/replaced/ and their state is reset, once (a rerun only resumes).
    # Args: year (required), tiles_file (default <year>_state/float_id_tiles.txt), key_file
    local year tiles_file key_file
    parse_kv_args "$@" || exit $?
    : "${year:?year=<year> is required}"
    local all=$gvs_root/gedi/veg_sensitivity_all
    tiles_file=${tiles_file:-$all/${year}_state/float_id_tiles.txt}
    if [ ! -f "$all/${year}_state/replaced/.reset_done" ]; then
        python -m download.run run=reset_gedi_tiles run.year=$year run.save_dir=$all/$year \
            run.tiles_file=$tiles_file || exit $?
        touch "$all/${year}_state/replaced/.reset_done"
    fi
    download year=$year tiles_file=$tiles_file ${key_file:+key_file=$key_file}
}

add_pft() {
    # Append pft 9 and 11 to every tile of a year downloaded with 1-8
    # (add_pft_classes), skipping the tiles redownloaded whole.
    # Args: year (required), key_file, exclude_tiles_file (default
    #       <year>_state/float_id_tiles.txt if it exists)
    local year key_file exclude_tiles_file
    parse_kv_args "$@" || exit $?
    : "${year:?year=<year> is required}"
    local gedi=$gvs_root/gedi all=$gvs_root/gedi/veg_sensitivity_all
    [ -z "$exclude_tiles_file" ] && [ -s "$all/${year}_state/float_id_tiles.txt" ] && exclude_tiles_file=$all/${year}_state/float_id_tiles.txt
    python -m download.run run=add_pft_gedi run.year=$year \
        run.save_dir=$all/$year \
        run.gedi_table_index_file=$gedi/l2a_orbit_table/l2a_orbit_table_$year.parquet \
        run.s2_grid_file=$gvs_root/01_derived_ancillary/growing_months/s2_tiles_with_growing_months.parquet \
        ${key_file:+run.key_file=$key_file} ${exclude_tiles_file:+run.exclude_tiles_file=$exclude_tiles_file}
}

push_erda() {
    # Copy finished years to ERDA (GEDI/veg_sensitivity_all/<year>): parquets
    # (changed ones overwritten, new ones added), then, only if the size check
    # passes, delete parquets ERDA has and this copy not (tiles left without
    # valid shots), then the state files. Record in <year>_state/uploads.tsv.
    # Args: years (comma-separated, required)
    local years y
    parse_kv_args "$@" || exit $?
    : "${years:?years=<comma-separated years> is required}"
    module load lumio-ext-tools/1.1.0
    # ERDA refused all connections for a while twice on 2026-10-22, each time
    # just as a new year's copy opened its connections: few connections, a
    # cap on them, timeouts so a dead one is dropped, and a wait for ERDA
    export RCLONE_CONFIG_ERDA_TYPE=sftp RCLONE_CONFIG_ERDA_HOST=io.erda.dk \
        RCLONE_CONFIG_ERDA_USER=huzh@di.ku.dk RCLONE_CONFIG_ERDA_KEY_FILE=$HOME/.ssh/id_rsa_erda \
        RCLONE_CONFIG_ERDA_CONNECTIONS=6
    local all=$gvs_root/gedi/veg_sensitivity_all dst_root=erda:GEDI/veg_sensitivity_all
    local opts=(--transfers 4 --checkers 2 --stats 10m --stats-one-line --contimeout 60s --timeout 10m
                --retries 10 --low-level-retries 20)
    erda_ready() {      # up to an hour for ERDA to take a connection
        local i
        for i in $(seq 30); do
            rclone lsf "$dst_root" --max-depth 1 >/dev/null 2>&1 && return 0
            echo "ERDA not reachable, retry in 2 min ($i/30)"; sleep 120
        done
        return 1
    }
    for y in ${years//,/ }; do
        local src=$all/$y dst=$dst_root/$y state=$all/${y}_state t=$(date -u +%Y%m%dT%H%M%SZ)
        echo "===== $y: $src -> $dst"
        python -m download.run run=finalize_gedi_states run.year=$y \
            run.s2_grid_file=$gvs_root/01_derived_ancillary/growing_months/s2_tiles_with_growing_months.parquet \
            "run.state_dirs=[$state,$state/add_pft]" || exit $?
        local log=$state/erda_$t.log ok= attempt
        for attempt in 1 2 3; do
            erda_ready || { echo "$y: ERDA not reachable for an hour, stopping"; exit 1; }
            rclone copy "$src" "$dst" --include '*.parquet' "${opts[@]}" --log-file "$log" --log-level INFO
            # temp files of transfers that were cut off
            rclone delete "$dst" --include '*.partial' --log-file "$log" --log-level INFO
            if rclone check "$src" "$dst" --one-way --size-only --include '*.parquet' --checkers 4 \
                    --log-file "$log" --log-level INFO; then
                ok=1; break
            fi
            echo "$y: check failed (attempt $attempt of 3), retry in 5 min"; sleep 300
        done
        if [ -z "$ok" ]; then
            echo "$y: CHECK FAILED, nothing deleted (details in $log); rerun to resend"
            continue
        fi
        # parquets only on ERDA: tiles without valid shots now
        comm -23 <(rclone lsf "$dst" --include '*.parquet' | sort) <(ls "$src" | grep '\.parquet$' | sort) > "$state/erda_stale_$t.txt"
        while read -r f; do
            echo "$y: deleting $dst/$f (not in $src)"
            rclone deletefile "$dst/$f" --log-file "$log" --log-level INFO
        done < "$state/erda_stale_$t.txt"
        local n_local n bytes
        n_local=$(ls "$src" | grep -c '\.parquet$')
        read -r n bytes < <(rclone size "$dst" --include '*.parquet' --json \
            | python -c 'import json, sys; d = json.load(sys.stdin); print(d["count"], d["bytes"])')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$dst" "$n_local" "$n" "$bytes" "yes" "${SLURM_JOB_ID:-local}" >> "$state/uploads.tsv"
        # snapshot.tsv / expected.tsv also match add_pft/ (the pft 9, 11 state)
        # (excludes first: the first matching rule wins)
        rclone copy "$state" "$dst_root/${y}_state" --exclude '{erda_copy,replaced,old_*}/**' \
            --include snapshot.tsv --include expected.tsv \
            --include uploads.tsv --include 'erda_*.log' --include 'erda_stale_*.txt' \
            --include 'shot_fields_fill_*.tsv' --include table_fields.tsv --include 'mgrs_summary_*.tsv' \
            --include fetch_plan.tsv --include float_id_tiles.txt "${opts[@]}"
        echo "$y: done ($n parquets on ERDA, $(numfmt --to=iec $bytes))"
    done
    rclone copyto ~/gvs/docs/data/gedi_veg_sensitivity_all.md "$dst_root/README.md"
}

case $1 in
download|pull_erda|organize|redownload|add_pft|push_erda)
    "$1" "${@:2}"
    ;;
*)
    echo "Unknown task: $1" >&2
    exit 1
    ;;
esac
