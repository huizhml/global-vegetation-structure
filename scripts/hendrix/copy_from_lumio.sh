#!/bin/bash
# =============================================================================
#  Copy prediction tiles from LUMI-O  ->  Hendrix.
#
#  Source on LUMI-O (current convention, lumi-${LUMI_PROJECT}-private):
#    ${bucket}/${TILE_ID}      (postprocessed COGs)
#  bucket = "${tile_id[:3] lowercase}-${year}", e.g. 32MRE/2024 -> 32m-2024.
#
#  Destination on hendrix follows the standard products layout:
#    ${HOME}/data/gvs/products/${PRODUCT}/${YEAR}/${PRODUCT_VERSION}/tiles/${PRODUCT_FORMAT}/${TILE_ID}
#  defaults: PRODUCT=vsm, PRODUCT_VERSION=bias_corrected, PRODUCT_FORMAT=cog.
#  Override DST_BASE if you need to drop into some other root.
#
#  Subcommands:
#     copy       <TILE_ID> [YEAR]       copy one tile with the fixed knobs below
#     copy-list  <FILE>    [YEAR]       copy every tile id (one per line) in FILE
#     benchmark  <TILE_ID> [YEAR]       sweep --transfers x --checkers, write CSV
#
#  Override behaviour via env vars (all optional):
#     LUMI_PROJECT          (default 465002698)
#     PRODUCT               (default vsm)
#     PRODUCT_VERSION       (default bias_corrected)
#     PRODUCT_FORMAT        (default cog)
#     DST_BASE              destination root, overrides the default pattern above
#     TRANSFERS / CHECKERS / MULTI_THREAD_STREAMS   knobs for `copy` / `copy-list`
#     TRANSFERS_LIST / CHECKERS_LIST                grids for `benchmark`
#     VERIFY=1|0            run `rclone check` after a copy (default 1)
#     KEEP_LAST=1|0         (benchmark only) keep the last copy on disk
#
#  Examples:
#     bash scripts/hendrix/copy_from_lumio.sh copy 32MRE 2024
#     PRODUCT_VERSION=original bash scripts/hendrix/copy_from_lumio.sh copy 32MRE 2024
#     bash scripts/hendrix/copy_from_lumio.sh copy-list tiles.txt 2024
#     bash scripts/hendrix/copy_from_lumio.sh benchmark 32MRE 2024
#
#  This file is safe to `source`: the CLI dispatcher only runs when executed
#  directly, so other scripts can pull in the two functions below.
# =============================================================================
set -uo pipefail

# ----------------------------- shared config --------------------------------
LUMI_PROJECT=${LUMI_PROJECT:-465002698}
REMOTE="lumi-${LUMI_PROJECT}-private:"
PRODUCT=${PRODUCT:-vsm}
PRODUCT_VERSION=${PRODUCT_VERSION:-bias_corrected}
PRODUCT_FORMAT=${PRODUCT_FORMAT:-cog}

# Copy knobs (used by rclone_copy_tile).
TRANSFERS=${TRANSFERS:-16}
CHECKERS=${CHECKERS:-16}
MULTI_THREAD_STREAMS=${MULTI_THREAD_STREAMS:-4}
VERIFY=${VERIFY:-1}

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
    echo "${HOME}/data/gvs/products/${PRODUCT}/${year}/${PRODUCT_VERSION}/tiles/${PRODUCT_FORMAT}"
}

# ----------------------------- copy one tile --------------------------------
# Usage: rclone_copy_tile <TILE_ID> [YEAR]
rclone_copy_tile() {
    local tile_id=${1:?Usage: rclone_copy_tile <TILE_ID> [YEAR]}
    local year=${2:-2024}
    _require_rclone || return 1

    local src dst
    src=$(_lumio_src "${tile_id}" "${year}") || return 1
    local dst_base=${DST_BASE:-$(_dst_base_default "${year}")}
    dst="${dst_base}/${tile_id}"
    mkdir -p "${dst}"

    echo "[${tile_id}] ${src}  ->  ${dst}"
    local start end elapsed
    start=$(date +%s)
    rclone copy "${src}" "${dst}" \
        --transfers="${TRANSFERS}" --checkers="${CHECKERS}" \
        --multi-thread-streams="${MULTI_THREAD_STREAMS}" \
        --stats=30s --stats-one-line
    local rc=$?
    end=$(date +%s)
    elapsed=$((end - start))

    if [ "${rc}" -ne 0 ]; then
        echo "[${tile_id}] rclone copy failed (rc=${rc}) after ${elapsed}s"
        return "${rc}"
    fi

    if [ "${VERIFY}" = "1" ]; then
        if rclone check "${src}" "${dst}" --one-way >/dev/null 2>&1; then
            echo "[${tile_id}] OK  (${elapsed}s)"
        else
            echo "[${tile_id}] MISMATCH after copy (${elapsed}s) — rerun this tile"
            return 2
        fi
    else
        echo "[${tile_id}] copied in ${elapsed}s (verify skipped)"
    fi
}

# Usage: rclone_copy_tile_list <FILE> [YEAR]
# FILE: one tile_id per line; '#' and blank lines ignored.
rclone_copy_tile_list() {
    local list_file=${1:?Usage: rclone_copy_tile_list <FILE> [YEAR]}
    local year=${2:-2024}
    [ -r "${list_file}" ] || { echo "ERROR: cannot read ${list_file}"; return 1; }

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
    size_json=$(rclone size "${src}" --json 2>/dev/null)
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
            if rclone check "${src}" "${dst}" --one-way >/dev/null 2>&1; then
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

# ----------------------------- CLI dispatcher -------------------------------
# Only run when executed directly; safe to `source` from other scripts.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    cmd=${1:-}
    case "${cmd}" in
        copy)
            shift; rclone_copy_tile "$@" ;;
        copy-list)
            shift; rclone_copy_tile_list "$@" ;;
        benchmark)
            shift; benchmark_rclone_copy "$@" ;;
        ""|-h|--help)
            sed -n '2,32p' "${BASH_SOURCE[0]}" ;;
        *)
            echo "ERROR: unknown subcommand '${cmd}'. Try: copy | copy-list | benchmark" >&2
            exit 1 ;;
    esac
fi
