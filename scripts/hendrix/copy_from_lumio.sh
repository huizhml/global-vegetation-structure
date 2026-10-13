#!/bin/bash
#SBATCH --partition=ml4good
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=copy_q1
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
# =============================================================================
#  Copy prediction tiles from LUMI-O  ->  Hendrix.
#
#  Source on LUMI-O (current convention, lumi-${LUMI_PROJECT}-private):
#    ${bucket}/${TILE_ID}      (postprocessed COGs)
#  bucket = "${tile_id[:3] lowercase}-${year}", e.g. 32MRE/2024 -> 32m-2024.
#
#  Destination on hendrix follows the standard products layout:
#    ${ROOT_DATA_DIR}/products/${PRODUCT}/${YEAR}/${PRODUCT_VERSION}/tiles/${PRODUCT_FORMAT}/${TILE_ID}
#  defaults: PRODUCT=vsm, PRODUCT_VERSION=bias_corrected, PRODUCT_FORMAT=cog.
#  Override DST_BASE if you need to drop into some other root.
#
#  Subcommands:
#     copy       <TILE_ID> [YEAR]       copy one tile with the fixed knobs below
#     copy-list  <FILE>    [YEAR]       copy every tile id (one per line) in FILE
#     size       <FILE>    [YEAR]       sum the source bytes the filter selects
#     benchmark  <TILE_ID> [YEAR]       sweep --transfers x --checkers, write CSV
#
#  ... and the bulk entrypoints, which default INCLUDE to the median quantile
#  (BULK_INCLUDE, RH*_Q1.tif) and record per-tile progress in STATE_DIR:
#     slurm      [YEAR] [WORKLIST]      sbatch entry: srun over --ntasks tasks
#     chunk      [YEAR] [WORKLIST]      worker for one chunk (called by `slurm`)
#     audit      [YEAR] [WORKLIST]      tiles still missing / short on hendrix
#
#  The worklist is split the way postprocessing/core/translate.py splits it, so
#  `--ntasks=N` inside a job array partitions it with no bookkeeping:
#      n_chunks = SLURM_ARRAY_TASK_COUNT * SLURM_NTASKS
#      chunk    = (SLURM_ARRAY_TASK_ID-1) * SLURM_NTASKS + SLURM_PROCID + 1
#  Outside SLURM that is 1/1, i.e. the whole worklist in one process.
#
#  INCLUDE selects which assets to move; it is applied to the copy, to the
#  `rclone check` that verifies it, and to `size`, so a filtered copy verifies
#  against the same subset instead of reporting every skipped file as missing.
#  Default '**' is every file, i.e. the previous behaviour.
#
#  Bulk runs are resumable: a tile is marked done only after its copy verified
#  with `rclone check --include ... --one-way`, so re-submitting the same job
#  picks up exactly what did not finish. Progress is a tools/task_state.py
#  state dir (a few append-only logs + snapshot.tsv instead of one flag file
#  per tile), one per year and asset selection:
#      ${ROOT_DATA_DIR}/sync/copy_from_lumio/<year>_q1    INCLUDE='RH*_Q1.tif'
#      ${ROOT_DATA_DIR}/sync/copy_from_lumio/<year>_all   INCLUDE='**'
#  Inspect / re-copy one tile:
#      python -m tools.run run=state_summary run.path=<STATE_DIR>
#      printf '%s\treset\t%s\t\n' 32MRE "$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)" \
#          >> <STATE_DIR>/log/manual.tsv
#
#  Override behaviour via env vars (all optional):
#     LUMI_PROJECT          (default 465002698)
#     PRODUCT               (default vsm)
#     PRODUCT_VERSION       (default bias_corrected)
#     PRODUCT_FORMAT        (default cog)
#     DST_BASE              destination root, overrides the default pattern above
#     INCLUDE               rclone --include pattern (default '**' = everything);
#                           'RH*_Q1.tif' is the median quantile only (101 files)
#     TRANSFERS / CHECKERS / MULTI_THREAD_STREAMS   knobs for `copy` / `copy-list`
#     TRANSFERS_LIST / CHECKERS_LIST                grids for `benchmark`
#     BULK_INCLUDE          what `slurm`/`chunk`/`audit` copy when INCLUDE is
#                           unset (default 'RH*_Q1.tif')
#     N_EXPECTED            files per tile INCLUDE should select; default 101
#                           for 'RH*_Q1.tif', else unknown (only LUMI-O vs
#                           hendrix are compared)
#     STATE_DIR             per-tile progress; the bulk entrypoints default it
#                           as above, empty (the default for `copy` /
#                           `copy-list`) means no tracking, always re-copy
#     ROOT_DATA_DIR         (default ${HOME}/data/gvs)
#     VERIFY=1|0            verify after a copy (default 1): LUMI-O has the
#                           expected file count, both sides match in file
#                           count and bytes, and `rclone check` passes
#     DELETE_SOURCE=1|0     delete the INCLUDE-selected files on LUMI-O after a
#                           verified copy (default 0). Needs VERIFY=1 and
#                           STATE_DIR; deletions are recorded in
#                           STATE_DIR/src_deleted, and a re-run re-verifies and
#                           deletes tiles copied earlier but not yet deleted
#     KEEP_LAST=1|0         (benchmark only) keep the last copy on disk
#
#  Examples:
#     bash scripts/hendrix/copy_from_lumio.sh copy 32MRE 2024
#     PRODUCT_VERSION=original bash scripts/hendrix/copy_from_lumio.sh copy 32MRE 2024
#     bash scripts/hendrix/copy_from_lumio.sh copy-list tiles.txt 2024
#     INCLUDE='RH*_Q1.tif' bash scripts/hendrix/copy_from_lumio.sh copy-list tiles.txt 2024
#     INCLUDE='RH*_Q1.tif' bash scripts/hendrix/copy_from_lumio.sh size tiles.txt 2024
#     bash scripts/hendrix/copy_from_lumio.sh benchmark 32MRE 2024
#
#  Bulk median-only run (the current job):
#     # 1. see the volume first: listings only, no data moved
#     INCLUDE='RH*_Q1.tif' bash scripts/hendrix/copy_from_lumio.sh size <worklist> 2024
#     # 2. one node, 4 concurrent tiles
#     sbatch scripts/hendrix/copy_from_lumio.sh slurm 2024
#     # 3. wider: 10 array tasks x 4 = 40 chunks
#     sbatch --array=1-10 scripts/hendrix/copy_from_lumio.sh slurm 2024
#     # 4. what is left after a timeout, then resubmit with the list it writes
#     bash scripts/hendrix/copy_from_lumio.sh audit 2024
#     # 5. same, and free LUMI-O as tiles verify (also cleans up tiles copied
#     #    by earlier runs)
#     DELETE_SOURCE=1 sbatch --array=1-10 scripts/hendrix/copy_from_lumio.sh slurm 2024
#
#  This file is safe to `source`: the CLI dispatcher only runs when executed
#  directly, so other scripts can pull in the functions below.
# =============================================================================
set -uo pipefail

# ----------------------------- shared config --------------------------------
ROOT_DATA_DIR=${ROOT_DATA_DIR:-${HOME}/data/gvs}
LUMI_PROJECT=${LUMI_PROJECT:-465002698}
REMOTE="lumi-${LUMI_PROJECT}-private:"
PRODUCT=${PRODUCT:-vsm}
PRODUCT_VERSION=${PRODUCT_VERSION:-bias_corrected}
PRODUCT_FORMAT=${PRODUCT_FORMAT:-cog}

# Which assets to move. '**' is every file at any depth; 'RH*_Q1.tif' is the
# median quantile only. Kept as a single always-set pattern rather than an
# optional array so every rclone call below can quote it under `set -u`.
# Remember whether the caller set it, so the bulk entrypoints below can fall
# back to BULK_INCLUDE without overriding an explicit choice.
_INCLUDE_FROM_ENV=${INCLUDE+1}
INCLUDE=${INCLUDE:-**}
BULK_INCLUDE=${BULK_INCLUDE:-RH*_Q1.tif}
# Files per tile INCLUDE should select; empty = 101 for RH*_Q1.tif, else
# unknown (see _n_expected).
N_EXPECTED=${N_EXPECTED:-}

# Per-tile progress (tools/task_state.py format). Set to make re-runs cheap: a
# tile marked done is skipped without touching the network. Empty = no
# tracking, always re-copy (rclone still skips files that already match, but
# it has to list them).
STATE_DIR=${STATE_DIR:-}

# Copy knobs (used by rclone_copy_tile).
TRANSFERS=${TRANSFERS:-16}
CHECKERS=${CHECKERS:-16}
MULTI_THREAD_STREAMS=${MULTI_THREAD_STREAMS:-4}
VERIFY=${VERIFY:-1}
# 1 = delete a tile's INCLUDE-selected files on LUMI-O once the hendrix copy
# verified. Opt-in, because it cannot be undone.
DELETE_SOURCE=${DELETE_SOURCE:-0}

# Benchmark grids (used by benchmark_rclone_copy).
TRANSFERS_LIST=${TRANSFERS_LIST:-"4 8 16 32"}
CHECKERS_LIST=${CHECKERS_LIST:-"8 16 32"}
KEEP_LAST=${KEEP_LAST:-0}
EXPECTED_FILES=${EXPECTED_FILES:-0}

# Load rclone on hendrix; ignore if `module` isn't present (e.g. local dev).
module load rclone 2>/dev/null || true

# ----------------------------- helpers --------------------------------------
_require_rclone() {
    command -v rclone >/dev/null \
        || { echo "ERROR: rclone not found (try: module load rclone)"; return 1; }
}

# ----------------------------- progress state -------------------------------
# Same on-disk format as tools/task_state.py, so `state_summary`, `compact` and
# Python readers see what these functions write. Every process appends to its
# own log, so parallel tasks never write the same file.
#
# Two states per run: STATE_DIR records verified copies, and
# STATE_DIR/src_deleted (a state dir of its own) records tiles whose LUMI-O
# files were deleted after that, so a deletion that failed half-way is retried
# on the next run instead of being skipped with the copy.

# Usage: _state_mark <DIR> <KEY> <done|failed|reset> [INFO]
_state_mark() {
    local dir=$1 key=$2 status=$3 info=${4:-}
    [ -n "${dir}" ] || return 0
    local job=${SLURM_JOB_ID:-local}
    [ -n "${SLURM_PROCID:-}" ] && job="${job}.${SLURM_PROCID}"
    info=$(printf '%s' "${info}" | tr '\t\n\r' '   ' | cut -c1-500)
    # UTC, fixed width, so string order is time order (see task_state.py)
    local now
    now=$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)
    [[ "${now}" == *N* ]] && now=$(date -u +%Y-%m-%dT%H:%M:%S.000000Z)   # BSD date
    mkdir -p "${dir}/log"
    local log="${dir}/log/$(hostname -s)_$$_${job}.tsv"
    # new log: same header as snapshot.tsv (TaskState writes it too)
    [ -s "${log}" ] || printf '#key\tstatus\ttime (UTC)\tinfo\n' >> "${log}"
    printf '%s\t%s\t%s\t%s\n' "${key}" "${status}" "${now}" "${info}" >> "${log}"
}

# Usage: _state_done_keys <DIR>   -> keys whose newest status is done
# Same rule as TaskState.load(): newest line per key wins; '#' lines (header,
# old '#offset' bookkeeping) and lines cut short by a crash are skipped. Times
# compare as strings: ISO times order correctly, and older Unix-seconds times
# ("17...") sort before every ISO time ("20..."), which they all predate.
_state_done_keys() {
    local dir=$1 files=() f
    for f in "${dir}/snapshot.tsv" "${dir}"/log/*.tsv; do
        [ -f "${f}" ] && files+=("${f}")
    done
    [ "${#files[@]}" -eq 0 ] && return 0
    awk -F'\t' 'NF == 4 && $1 !~ /^#/ && ($2 == "done" || $2 == "failed" || $2 == "reset") {
            if (!($1 in t) || ($3 "") >= t[$1]) { t[$1] = $3 ""; s[$1] = $2 }
        }
        END { for (k in s) if (s[k] == "done") print k }' "${files[@]}"
}

# Load the done sets into files once, so a list of thousands of tiles does not
# re-read the logs per tile. Tiles finished by other tasks after this point are
# not seen, which is fine: each task copies its own disjoint chunk.
_COPIED_FILE="${TMPDIR:-/tmp}/copy_from_lumio_copied_$$.txt"
_DELETED_FILE="${TMPDIR:-/tmp}/copy_from_lumio_deleted_$$.txt"
_STATE_LOADED=0
_state_load() {
    _STATE_LOADED=1
    : > "${_COPIED_FILE}"
    : > "${_DELETED_FILE}"
    [ -n "${STATE_DIR}" ] || return 0
    _state_done_keys "${STATE_DIR}" > "${_COPIED_FILE}"
    _state_done_keys "${STATE_DIR}/src_deleted" > "${_DELETED_FILE}"
}

# Usage: _state_has <copied|deleted> <KEY>
_state_has() {
    [ -n "${STATE_DIR}" ] || return 1
    [ "${_STATE_LOADED}" = 1 ] || _state_load
    local f=${_COPIED_FILE}
    [ "$1" = deleted ] && f=${_DELETED_FILE}
    grep -qxF -- "$2" "${f}"
}

# Resolve the rclone source URL for one tile (COG layout: ${bucket}/${tile_id}).
_lumio_src() {
    local tile_id=$1 year=$2
    local zone bucket
    zone=$(echo "${tile_id:0:3}" | tr '[:upper:]' '[:lower:]')
    bucket="${zone}-${year}"
    echo "${REMOTE}${bucket}/${tile_id}"
}

# Default destination root if DST_BASE isn't set.
_dst_base_default() {
    local year=$1
    echo "${ROOT_DATA_DIR}/products/${PRODUCT}/${year}/${PRODUCT_VERSION}/tiles/${PRODUCT_FORMAT}"
}

# ----------------------------- verification ---------------------------------
# Files per tile that INCLUDE should select: N_EXPECTED if set, else 101 for
# the median quantile (RH0..RH100), else 0 = unknown, only compared side to side.
_n_expected() {
    if [ -n "${N_EXPECTED}" ]; then echo "${N_EXPECTED}"; return; fi
    case "${INCLUDE}" in
        'RH*_Q1.tif') echo 101 ;;
        *)            echo 0 ;;
    esac
}

# Usage: _count_bytes <PATH>   -> "<files> <bytes>" that INCLUDE selects
_count_bytes() {
    local json c b
    json=$(rclone size "$1" --include "${INCLUDE}" --json 2>/dev/null) || return 1
    c=$(grep -o '"count":[0-9]*' <<< "${json}" | cut -d: -f2)
    b=$(grep -o '"bytes":[0-9]*' <<< "${json}" | cut -d: -f2)
    [ -n "${c}" ] && [ -n "${b}" ] || return 1
    echo "${c} ${b}"
}

# Usage: _verify_copy <SRC> <DST>   (sets _WHY on failure, _VERIFIED on success)
# Passes only when LUMI-O has files, as many as expected when that is known,
# both sides hold the same number of files and bytes, and `rclone check` finds
# every LUMI-O file on hendrix with the same size and, where both sides have
# one, the same hash. A tile missing on LUMI-O no longer passes as "0 == 0".
_verify_copy() {
    local src=$1 dst=$2 s d sc sb dc db n_exp
    s=$(_count_bytes "${src}") || { _WHY="cannot list LUMI-O"; return 1; }
    d=$(_count_bytes "${dst}") || { _WHY="cannot list hendrix"; return 1; }
    read -r sc sb <<< "${s}"
    read -r dc db <<< "${d}"
    n_exp=$(_n_expected)
    if [ "${sc}" -eq 0 ]; then
        _WHY="no files on LUMI-O"; return 1
    fi
    if [ "${n_exp}" -gt 0 ] && [ "${sc}" -ne "${n_exp}" ]; then
        _WHY="LUMI-O has ${sc} files, expected ${n_exp}"; return 1
    fi
    if [ "${sc}" -ne "${dc}" ] || [ "${sb}" -ne "${db}" ]; then
        _WHY="files/bytes differ: LUMI-O ${sc}/${sb}, hendrix ${dc}/${db}"; return 1
    fi
    if ! rclone check "${src}" "${dst}" --include "${INCLUDE}" --one-way >/dev/null 2>&1; then
        _WHY="rclone check mismatch"; return 1
    fi
    _VERIFIED="${dc} files, $(_human_bytes "${db}")"
}

# Usage: _human_bytes <BYTES>   -> e.g. "76.2 MiB"
_human_bytes() {
    awk -v b="$1" 'BEGIN { split("B KiB MiB GiB TiB", u); i = 1
        while (b >= 1024 && i < 5) { b /= 1024; i++ }
        printf (i == 1 ? "%d %s" : "%.1f %s"), b, u[i] }'
}

# Usage: _verify_resume <SRC> <DST>   (sets _WHY on failure)
# For a tile verified on an earlier run whose LUMI-O side was not (fully)
# deleted: LUMI-O may already be partly empty, so the side-by-side totals no
# longer apply. Require the hendrix copy to be complete and every file still
# on LUMI-O to match it.
_verify_resume() {
    local src=$1 dst=$2 d dc db n_exp
    d=$(_count_bytes "${dst}") || { _WHY="cannot list hendrix"; return 1; }
    read -r dc db <<< "${d}"
    n_exp=$(_n_expected)
    if [ "${dc}" -eq 0 ]; then
        _WHY="no files on hendrix"; return 1
    fi
    if [ "${n_exp}" -gt 0 ] && [ "${dc}" -ne "${n_exp}" ]; then
        _WHY="hendrix has ${dc} files, expected ${n_exp}"; return 1
    fi
    if ! rclone check "${src}" "${dst}" --include "${INCLUDE}" --one-way >/dev/null 2>&1; then
        _WHY="rclone check mismatch"; return 1
    fi
}

# Usage: _delete_src <TILE_ID> <SRC>
# Deletes only what INCLUDE selects (e.g. the Q1 files, not Q0/Q2), records
# the result in STATE_DIR/src_deleted, and checks nothing selected is left.
_delete_src() {
    local tile_id=$1 src=$2 left why=""
    if ! rclone delete "${src}" --include "${INCLUDE}" --checkers="${CHECKERS}"; then
        why="rclone delete failed"
    elif ! left=$(_count_bytes "${src}"); then
        why="cannot list LUMI-O after delete"
    elif [ "${left%% *}" -ne 0 ]; then
        why="${left%% *} files still on LUMI-O after delete"
    fi
    if [ -n "${why}" ]; then
        echo "[${tile_id}] LUMI-O delete FAILED: ${why}"
        _state_mark "${STATE_DIR}/src_deleted" "${tile_id}" failed "${why}"
        return 1
    fi
    echo "[${tile_id}] deleted on LUMI-O (include='${INCLUDE}')"
    _state_mark "${STATE_DIR}/src_deleted" "${tile_id}" done
}

# Deleting from LUMI-O cannot be undone, so it needs both a verified copy and
# a state dir that remembers which tiles are already gone.
_check_delete_config() {
    [ "${DELETE_SOURCE}" = 1 ] || return 0
    if [ "${VERIFY}" != 1 ]; then
        echo "ERROR: DELETE_SOURCE=1 needs VERIFY=1" >&2; return 1
    fi
    if [ -z "${STATE_DIR}" ]; then
        echo "ERROR: DELETE_SOURCE=1 needs STATE_DIR" >&2; return 1
    fi
}

# ----------------------------- copy one tile --------------------------------
# Usage: rclone_copy_tile <TILE_ID> [YEAR]
rclone_copy_tile() {
    local tile_id=${1:?Usage: rclone_copy_tile <TILE_ID> [YEAR]}
    local year=${2:-2024}
    _require_rclone || return 1
    _check_delete_config || return 1

    local src dst
    src=$(_lumio_src "${tile_id}" "${year}") || return 1
    local dst_base=${DST_BASE:-$(_dst_base_default "${year}")}
    dst="${dst_base}/${tile_id}"

    if _state_has copied "${tile_id}"; then
        if [ "${DELETE_SOURCE}" != 1 ] || _state_has deleted "${tile_id}"; then
            echo "[${tile_id}] already done (${STATE_DIR}), skipping"
            return 0
        fi
        # Copied on an earlier run, LUMI-O side still there: check the copy
        # again before deleting, since hendrix may have changed since.
        if ! _verify_resume "${src}" "${dst}"; then
            echo "[${tile_id}] NOT deleting on LUMI-O: ${_WHY}"
            _state_mark "${STATE_DIR}/src_deleted" "${tile_id}" failed "${_WHY}"
            return 2
        fi
        _delete_src "${tile_id}" "${src}" || return 2
        return 0
    fi

    mkdir -p "${dst}"
    echo "[${tile_id}] ${src}  ->  ${dst}   (include='${INCLUDE}')"
    local start end elapsed
    start=$(date +%s)
    rclone copy "${src}" "${dst}" \
        --include "${INCLUDE}" \
        --transfers="${TRANSFERS}" --checkers="${CHECKERS}" \
        --multi-thread-streams="${MULTI_THREAD_STREAMS}" \
        --stats=30s --stats-one-line
    local rc=$?
    end=$(date +%s)
    elapsed=$((end - start))

    if [ "${rc}" -ne 0 ]; then
        echo "[${tile_id}] rclone copy failed (rc=${rc}) after ${elapsed}s"
        _state_mark "${STATE_DIR}" "${tile_id}" failed "rclone copy rc=${rc}"
        return "${rc}"
    fi

    _VERIFIED="not verified"
    if [ "${VERIFY}" = "1" ]; then
        if _verify_copy "${src}" "${dst}"; then
            echo "[${tile_id}] OK  (${elapsed}s, ${_VERIFIED})"
        else
            echo "[${tile_id}] MISMATCH after copy (${elapsed}s): ${_WHY}"
            _state_mark "${STATE_DIR}" "${tile_id}" failed "${_WHY}"
            return 2
        fi
    else
        echo "[${tile_id}] copied in ${elapsed}s (verify skipped)"
    fi

    # Only mark a tile that copied (and, when VERIFY=1, verified) cleanly, so a
    # re-run picks up exactly the tiles that did not make it.
    _state_mark "${STATE_DIR}" "${tile_id}" done "${elapsed}s, ${_VERIFIED}"

    if [ "${DELETE_SOURCE}" = 1 ]; then
        _delete_src "${tile_id}" "${src}" || return 2
    fi
    return 0
}

# Usage: rclone_copy_tile_list <FILE> [YEAR]
# FILE: one tile_id per line; '#' and blank lines ignored.
rclone_copy_tile_list() {
    local list_file=${1:?Usage: rclone_copy_tile_list <FILE> [YEAR]}
    local year=${2:-2024}
    [ -r "${list_file}" ] || { echo "ERROR: cannot read ${list_file}"; return 1; }
    _check_delete_config || return 1
    _state_load

    local total ok fail tile
    total=$(grep -cvE '^\s*(#|$)' "${list_file}" || true)
    ok=0
    fail=0
    local idx=0
    while IFS= read -r tile; do
        [[ "${tile}" =~ ^[[:space:]]*(#|$) ]] && continue
        tile=${tile//[[:space:]]/}
        idx=$((idx + 1))
        echo "------------------------------------------------------------------"
        echo "[${idx}/${total}] ${tile}"
        if rclone_copy_tile "${tile}" "${year}"; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi
    done < "${list_file}"
    echo "=================================================================="
    echo "Done. ok=${ok} fail=${fail} total=${total}"
    [ "${fail}" -eq 0 ]
}

# ----------------------------- size estimate --------------------------------
# Usage: rclone_size_tile_list <FILE> [YEAR]
# Sums the source bytes/files INCLUDE selects, so the disk cost of a bulk copy
# is known before it starts. One `rclone size` per tile: cheap (a listing), but
# on a few thousand tiles it is still minutes, not seconds.
rclone_size_tile_list() {
    local list_file=${1:?Usage: rclone_size_tile_list <FILE> [YEAR]}
    local year=${2:-2024}
    _require_rclone || return 1
    [ -r "${list_file}" ] || { echo "ERROR: cannot read ${list_file}"; return 1; }

    local total_bytes=0 total_files=0 tiles=0 tile src json bytes count
    while IFS= read -r tile; do
        [[ "${tile}" =~ ^[[:space:]]*(#|$) ]] && continue
        tile=${tile//[[:space:]]/}
        src=$(_lumio_src "${tile}" "${year}")
        json=$(rclone size "${src}" --include "${INCLUDE}" --json 2>/dev/null)
        bytes=$(echo "${json}" | grep -o '"bytes":[0-9]*' | cut -d: -f2)
        count=$(echo "${json}" | grep -o '"count":[0-9]*' | cut -d: -f2)
        bytes=${bytes:-0}
        count=${count:-0}
        total_bytes=$((total_bytes + bytes))
        total_files=$((total_files + count))
        tiles=$((tiles + 1))
        printf '%-8s %6s files  %8.2f GiB\n' "${tile}" "${count}" \
            "$(awk "BEGIN{printf \"%.2f\", ${bytes}/1073741824}")"
    done < "${list_file}"

    echo "=================================================================="
    echo "include='${INCLUDE}'  year=${year}  tiles=${tiles}"
    echo "files : ${total_files}"
    echo "size  : $(awk "BEGIN{printf \"%.2f\", ${total_bytes}/1073741824}") GiB" \
         "($(awk "BEGIN{printf \"%.2f\", ${total_bytes}/1099511627776}") TiB)"
}

# ----------------------------- benchmark ------------------------------------
# Usage: benchmark_rclone_copy <TILE_ID> [YEAR]
# Sweeps TRANSFERS_LIST x CHECKERS_LIST for one tile, writes CSV, prints
# fastest-first table at the end. Re-copies the whole tile per combination,
# so wall-clock cost = (#combinations) x (tile size).
benchmark_rclone_copy() {
    local tile_id=${1:?Usage: benchmark_rclone_copy <TILE_ID> [YEAR]}
    local year=${2:-2024}
    _require_rclone || return 1

    local src dst_base dst
    src=$(_lumio_src "${tile_id}" "${year}") || return 1
    dst_base=${DST_BASE:-${HOME}/data/gvs/products/${PRODUCT}/${year}/${PRODUCT_VERSION}/rclone_bench/${PRODUCT_FORMAT}}
    dst="${dst_base}/${tile_id}"
    mkdir -p "${dst_base}"

    local results_csv
    results_csv="${dst_base}/bench_${tile_id}_$(date +%Y%m%d_%H%M%S).csv"

    echo "=================================================================="
    echo " rclone copy benchmark   LUMI-O -> Hendrix"
    echo "   source : ${src}"
    echo "   dest   : ${dst}"
    echo "   results: ${results_csv}"
    echo "=================================================================="

    echo "Probing source size/count ..."
    local size_json src_bytes src_count src_gib
    size_json=$(rclone size "${src}" --include "${INCLUDE}" --json 2>/dev/null)
    src_bytes=$(echo "${size_json}" | grep -o '"bytes":[0-9]*' | cut -d: -f2)
    src_count=$(echo "${size_json}" | grep -o '"count":[0-9]*' | cut -d: -f2)
    if [ -z "${src_bytes:-}" ] || [ "${src_bytes}" = "0" ]; then
        echo "ERROR: could not read source (empty/inaccessible): ${src}"
        return 1
    fi
    src_gib=$(awk "BEGIN{printf \"%.2f\", ${src_bytes}/1073741824}")

    local n_transfers n_checkers n_runs total_gib expected_show
    n_transfers=$(echo ${TRANSFERS_LIST} | wc -w)
    n_checkers=$(echo ${CHECKERS_LIST} | wc -w)
    n_runs=$(( n_transfers * n_checkers ))
    total_gib=$(awk "BEGIN{printf \"%.1f\", ${src_bytes}*${n_runs}/1073741824}")
    expected_show=${EXPECTED_FILES}
    [ "${expected_show}" = "0" ] && expected_show=${src_count}

    echo "   files       : ${src_count} (expecting ${expected_show})"
    echo "   size        : ${src_gib} GiB"
    echo "   sweep       : transfers={${TRANSFERS_LIST}}  checkers={${CHECKERS_LIST}}"
    echo "   runs        : ${n_runs}  ->  ~${total_gib} GiB will be moved in total"
    echo "------------------------------------------------------------------"

    local free_gib floor_gib
    free_gib=$(df -BG --output=avail "${dst_base}" 2>/dev/null | tail -1 | tr -dc '0-9')
    floor_gib=$(awk "BEGIN{print int(${src_gib}+1)}")
    if [ -n "${free_gib:-}" ] && [ "${free_gib}" -lt "${floor_gib}" ]; then
        echo "WARNING: only ${free_gib} GiB free on $(df -h "${dst_base}" | tail -1 | awk '{print $6}')"
        echo "         one copy needs ~${src_gib} GiB."
    fi

    echo "transfers,checkers,multi_thread_streams,wall_s,throughput_MBps,files,integrity_ok" > "${results_csv}"

    local run_idx=0 t c start end rc elapsed mbps got integrity
    for t in ${TRANSFERS_LIST}; do
      for c in ${CHECKERS_LIST}; do
        run_idx=$((run_idx + 1))
        echo
        echo ">>> [${run_idx}/${n_runs}] --transfers=${t} --checkers=${c} --multi-thread-streams=${MULTI_THREAD_STREAMS}"

        rm -rf "${dst}"
        mkdir -p "${dst}"

        start=$(date +%s.%N)
        rclone copy "${src}" "${dst}" \
            --include "${INCLUDE}" \
            --transfers="${t}" --checkers="${c}" \
            --multi-thread-streams="${MULTI_THREAD_STREAMS}" \
            --stats=30s --stats-one-line
        rc=$?
        end=$(date +%s.%N)

        elapsed=$(awk "BEGIN{printf \"%.2f\", ${end}-${start}}")
        mbps=$(awk "BEGIN{printf \"%.1f\", ${src_bytes}/1000000/(${end}-${start})}")
        got=$(find "${dst}" -type f | wc -l | tr -d ' ')

        integrity="skipped"
        if [ "${rc}" -ne 0 ]; then
            integrity="copy_failed(rc=${rc})"
        elif [ "${VERIFY}" = "1" ]; then
            if rclone check "${src}" "${dst}" --include "${INCLUDE}" --one-way \
                    >/dev/null 2>&1; then
                integrity="ok"
            else
                integrity="MISMATCH"
            fi
        fi

        echo "    -> ${elapsed}s | ${mbps} MB/s | files=${got}/${expected_show} | integrity=${integrity}"
        echo "${t},${c},${MULTI_THREAD_STREAMS},${elapsed},${mbps},${got},${integrity}" >> "${results_csv}"

        if [ "${KEEP_LAST}" = "1" ] && [ "${run_idx}" -eq "${n_runs}" ]; then
            echo "    (keeping final copy at ${dst})"
        else
            rm -rf "${dst}"
        fi
      done
    done

    echo
    echo "=================================================================="
    echo " RESULTS (fastest first)            tile=${tile_id}  ${src_gib} GiB"
    echo "=================================================================="
    printf "%-10s %-9s %-9s %-12s %-9s %-12s\n" "transfers" "checkers" "wall_s" "MB/s" "files" "integrity"
    tail -n +2 "${results_csv}" | sort -t, -k4 -n | \
      awk -F, '{printf "%-10s %-9s %-9s %-12s %-9s %-12s\n",$1,$2,$4,$5,$6,$7}'
    echo "------------------------------------------------------------------"
    echo "CSV: ${results_csv}"
}

# ----------------------------- bulk / SLURM ---------------------------------
# Default state dir: one per year and per asset selection, so a Q1-only pass
# is never mistaken for a full copy later.
_state_dir_default() {
    local year=$1 tag
    case "${INCLUDE}" in
        'RH*_Q1.tif') tag=q1 ;;
        '**')         tag=all ;;
        *) echo "ERROR: no default STATE_DIR for INCLUDE='${INCLUDE}', set STATE_DIR" >&2
           return 1 ;;
    esac
    echo "${ROOT_DATA_DIR}/sync/copy_from_lumio/${year}_${tag}"
}

# Shared setup for `slurm` / `chunk` / `audit`: median-only unless the caller
# said otherwise, plus the state dir that makes a re-submit cheap. Exported so
# the srun'd children inherit the resolved values (and, because INCLUDE is then
# set in their env, keep them instead of re-defaulting).
_bulk_setup() {
    local year=$1
    [ -z "${_INCLUDE_FROM_ENV}" ] && INCLUDE=${BULK_INCLUDE}
    STATE_DIR=${STATE_DIR:-$(_state_dir_default "${year}")} || return 1
    DST_BASE=${DST_BASE:-$(_dst_base_default "${year}")}
    export INCLUDE STATE_DIR DST_BASE

    # Flags from before the state dir are not read any more.
    local legacy="${ROOT_DATA_DIR}/state/${year}/copy_from_lumio_q1"
    if [ "${INCLUDE}" = 'RH*_Q1.tif' ] && [ -d "${legacy}" ] \
            && [ ! -e "${STATE_DIR}/snapshot.tsv" ]; then
        echo "WARNING: old flags in ${legacy} are ignored. Import them first:" >&2
        echo "  python -m tools.run run=import_flags run.flag_dir=${legacy} run.path=${STATE_DIR}" >&2
    fi
}

# Default worklist for a year, if the caller did not name one.
_bulk_worklist() {
    echo "${ROOT_DATA_DIR}/assets/worklists/total_tiles_${1}.txt"
}

# Write this task's slice of the worklist to a file and echo its path. Same
# integer slicing as translate.py: chunk boundaries are total*i/n_chunks.
_chunk_file() {
    local list=$1
    [ -r "${list}" ] || { echo "ERROR: cannot read worklist ${list}" >&2; return 1; }

    local ntasks=${SLURM_NTASKS:-1}
    local procid=${SLURM_PROCID:-0}
    local array_id=${SLURM_ARRAY_TASK_ID:-1}
    local array_n=${SLURM_ARRAY_TASK_COUNT:-1}
    local chunk=$(( (array_id - 1) * ntasks + procid + 1 ))
    local n_chunks=$(( array_n * ntasks ))

    local total start end
    total=$(grep -cvE '^\s*(#|$)' "${list}")
    start=$(( total * (chunk - 1) / n_chunks ))
    end=$(( total * chunk / n_chunks ))

    local out="${STATE_DIR}/chunks/chunk_${n_chunks}_${chunk}.txt"
    mkdir -p "$(dirname "${out}")"
    grep -vE '^\s*(#|$)' "${list}" | sed -n "$((start + 1)),${end}p" > "${out}"

    echo "chunk ${chunk}/${n_chunks} on $(hostname -s): tiles $((start + 1))-${end}" \
         "of ${total}  (${list})" >&2
    echo "${out}"
}

# Usage: copy_bulk_slurm [YEAR] [WORKLIST]   (the `sbatch <this file> slurm` entry)
copy_bulk_slurm() {
    local year=${1:-2024}
    local worklist=${2:-$(_bulk_worklist "${year}")}
    _bulk_setup "${year}" || return 1

    # $0 under sbatch is the node-local spool copy, which other nodes in a
    # multi-node allocation cannot read; the repo path lives on shared storage.
    local self="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/hendrix/copy_from_lumio.sh"

    echo "***************************** JOB INFO *****************************"
    echo "Job          : ${SLURM_JOB_NAME:-local} ${SLURM_JOB_ID:-} ${SLURM_ARRAY_TASK_ID:-}"
    echo "Tasks        : ${SLURM_NTASKS:-1} x ${SLURM_CPUS_PER_TASK:-?} cpus"
    echo "Worklist     : ${worklist}"
    echo "Include      : ${INCLUDE}"
    echo "Destination  : ${DST_BASE}"
    echo "State        : ${STATE_DIR}"
    echo "Delete LUMI-O: ${DELETE_SOURCE}"
    echo "********************************************************************"
    mkdir -p "${STATE_DIR}" "${DST_BASE}"
    _check_delete_config || return 1
    srun bash "${self}" chunk "${year}" "${worklist}"
}

# Usage: copy_bulk_chunk [YEAR] [WORKLIST]   (one task's share; called by srun)
copy_bulk_chunk() {
    local year=${1:-2024}
    local worklist=${2:-$(_bulk_worklist "${year}")}
    _bulk_setup "${year}" || return 1

    local chunk_file
    chunk_file=$(_chunk_file "${worklist}") || return 1
    rclone_copy_tile_list "${chunk_file}" "${year}"
}

# Usage: audit_bulk [YEAR] [WORKLIST]
# A done mark alone is not proof: the destination has to hold the expected
# files too, so a tile whose files went missing after it was marked shows up.
audit_bulk() {
    local year=${1:-2024}
    local worklist=${2:-$(_bulk_worklist "${year}")}
    _bulk_setup "${year}" || return 1
    [ -r "${worklist}" ] || { echo "ERROR: cannot read ${worklist}"; return 1; }
    mkdir -p "${STATE_DIR}"
    _state_load

    local missing_file="${STATE_DIR}/missing_$(date +%Y%m%d_%H%M%S).txt"
    : > "${missing_file}"
    local n_exp n_total=0 n_done=0 n_short=0 n_absent=0 n_deleted=0 tile n_have
    n_exp=$(_n_expected)
    while IFS= read -r tile; do
        [[ "${tile}" =~ ^[[:space:]]*(#|$) ]] && continue
        tile=${tile//[[:space:]]/}
        n_total=$((n_total + 1))
        n_have=$(ls -1 "${DST_BASE}/${tile}"/${INCLUDE} 2>/dev/null | wc -l | tr -d ' ')
        if [ "${n_have}" -gt 0 ] && [ "${n_have}" -ge "${n_exp}" ] \
                && _state_has copied "${tile}"; then
            n_done=$((n_done + 1))
            _state_has deleted "${tile}" && n_deleted=$((n_deleted + 1))
        elif [ "${n_have}" -eq 0 ]; then
            n_absent=$((n_absent + 1)); echo "${tile}" >> "${missing_file}"
        else
            n_short=$((n_short + 1)); echo "${tile}" >> "${missing_file}"
            if [ "${n_have}" -ge "${n_exp}" ]; then
                echo "UNMARKED ${tile}: ${n_have} files but no verified copy recorded"
            else
                echo "SHORT ${tile}: ${n_have}/${n_exp}"
            fi
        fi
    done < "${worklist}"

    echo "=================================================================="
    echo "include='${INCLUDE}'  total=${n_total} done=${n_done}" \
         "short=${n_short} absent=${n_absent}  (LUMI-O deleted: ${n_deleted})"
    echo "Re-run with:  sbatch scripts/hendrix/copy_from_lumio.sh slurm ${year} ${missing_file}"
    echo "Missing list: ${missing_file}"
    [ "$((n_short + n_absent))" -eq 0 ]
}

# ----------------------------- CLI dispatcher -------------------------------
# Only run when executed directly; safe to `source` from other scripts.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    trap 'rm -f "${_COPIED_FILE}" "${_DELETED_FILE}"' EXIT
    cmd=${1:-}
    case "${cmd}" in
        copy)
            shift; rclone_copy_tile "$@" ;;
        copy-list)
            shift; rclone_copy_tile_list "$@" ;;
        size)
            shift; rclone_size_tile_list "$@" ;;
        benchmark)
            shift; benchmark_rclone_copy "$@" ;;
        slurm)
            shift; copy_bulk_slurm "$@" ;;
        chunk)
            shift; copy_bulk_chunk "$@" ;;
        audit)
            shift; audit_bulk "$@" ;;
        ""|-h|--help)
            # Print the header block, whatever length it has grown to.
            awk 'NR>1 && /^#SBATCH/ {next} NR>1 && /^#/ {print; next} NR>1 {exit}' \
                "${BASH_SOURCE[0]}" ;;
        *)
            echo "ERROR: unknown subcommand '${cmd}'." \
                 "Try: copy | copy-list | size | benchmark | slurm | chunk | audit" >&2
            exit 1 ;;
    esac
fi
