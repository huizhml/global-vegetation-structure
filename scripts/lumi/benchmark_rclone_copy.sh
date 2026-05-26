#!/bin/bash
# =============================================================================
#  Benchmark rclone copy of ONE prediction tile: LUMI-O  ->  LUMI /project
#
#  Sweeps --transfers x --checkers, measures wall time + throughput, and
#  verifies integrity (file count + `rclone check`) after each run.
#
#  A tile = 303 files, ~243 MB each (~72 GiB total). Each combination
#  re-downloads the whole tile from scratch, so total data moved =
#  (#transfers values) x (#checkers values) x ~72 GiB. LUMI-O <-> LUMI is
#  intra-datacentre, so this is fast and incurs no egress cost.
#
#  Usage:
#     bash benchmark_rclone_copy.sh <TILE_ID> [YEAR]
#     # override grids / destination via env vars, e.g.:
#     TRANSFERS_LIST="8 16 32" CHECKERS_LIST="16 32" \
#         bash benchmark_rclone_copy.sh N00E010 2024
#
#  Recommended to run on a compute node (login nodes throttle I/O), e.g.:
#     srun -A project_465002698 -p small -c 32 --mem=16G -t 00:30:00 \
#          bash benchmark_rclone_copy.sh N00E010
# =============================================================================
set -uo pipefail

# ----------------------------- configuration --------------------------------
TILE_ID=${1:?Usage: $0 <TILE_ID> [YEAR]   (e.g. $0 32MRE 2024)}
YEAR=${2:-2024}
LUMI_PROJECT=${LUMI_PROJECT:-465002698}
REMOTE="lumi-${LUMI_PROJECT}-private:"

# Destination root on LUMI project storage. EDIT/override to match your setup.
# (LUMI persistent project storage is /projappl/project_<id>; large scratch is
#  /scratch/project_<id>. Pick whichever you meant by "/project".)
DST_BASE=${DST_BASE:-/scratch/project_${LUMI_PROJECT}/${USER}/rclone_bench}

# Parameter grids to sweep (space-separated; override via env).
TRANSFERS_LIST=${TRANSFERS_LIST:-"4 8 16 32"}
CHECKERS_LIST=${CHECKERS_LIST:-"8 16 32"}

# Fixed rclone knobs (the existing sync scripts use these).
MULTI_THREAD_STREAMS=${MULTI_THREAD_STREAMS:-4}
EXPECTED_FILES=${EXPECTED_FILES:-303}
VERIFY=${VERIFY:-1}          # 1 = run `rclone check` (hash compare) after copy
KEEP_LAST=${KEEP_LAST:-0}    # 1 = keep the final copy on disk instead of deleting

# ----------------------------- derived paths --------------------------------
zone=$(echo "${TILE_ID:0:3}" | tr '[:upper:]' '[:lower:]')
BUCKET="${zone}-${YEAR}"
SRC="${REMOTE}${BUCKET}/predictions_GTiff_${YEAR}/${TILE_ID}"
DST="${DST_BASE}/${TILE_ID}"
RESULTS_CSV="${DST_BASE}/bench_${TILE_ID}_$(date +%Y%m%d_%H%M%S).csv"

# rclone is provided either by the `rclone` module or by `lumio`.
module load rclone 2>/dev/null || module load lumio 2>/dev/null || true
command -v rclone >/dev/null || { echo "ERROR: rclone not found (load the rclone/lumio module)"; exit 1; }

mkdir -p "${DST_BASE}"

# ----------------------------- pre-flight -----------------------------------
echo "=================================================================="
echo " rclone copy benchmark"
echo "   source : ${SRC}"
echo "   dest   : ${DST}"
echo "   results: ${RESULTS_CSV}"
echo "=================================================================="

echo "Probing source size/count ..."
size_json=$(rclone size "${SRC}" --json 2>/dev/null)
src_bytes=$(echo "${size_json}" | grep -o '"bytes":[0-9]*' | cut -d: -f2)
src_count=$(echo "${size_json}" | grep -o '"count":[0-9]*' | cut -d: -f2)
if [ -z "${src_bytes:-}" ] || [ "${src_bytes}" = "0" ]; then
    echo "ERROR: could not read source (empty/inaccessible): ${SRC}"
    exit 1
fi
src_gib=$(awk "BEGIN{printf \"%.2f\", ${src_bytes}/1073741824}")
n_transfers=$(echo ${TRANSFERS_LIST} | wc -w)
n_checkers=$(echo ${CHECKERS_LIST} | wc -w)
n_runs=$(( n_transfers * n_checkers ))
total_gib=$(awk "BEGIN{printf \"%.1f\", ${src_bytes}*${n_runs}/1073741824}")

echo "   files       : ${src_count} (expecting ${EXPECTED_FILES})"
echo "   size        : ${src_gib} GiB"
echo "   sweep       : transfers={${TRANSFERS_LIST}}  checkers={${CHECKERS_LIST}}"
echo "   runs        : ${n_runs}  ->  ~${total_gib} GiB will be moved in total"
echo "------------------------------------------------------------------"

echo "transfers,checkers,multi_thread_streams,wall_s,throughput_MBps,files,integrity_ok" > "${RESULTS_CSV}"

# ----------------------------- benchmark loop -------------------------------
run_idx=0
for t in ${TRANSFERS_LIST}; do
  for c in ${CHECKERS_LIST}; do
    run_idx=$((run_idx + 1))
    echo
    echo ">>> [${run_idx}/${n_runs}] --transfers=${t} --checkers=${c} --multi-thread-streams=${MULTI_THREAD_STREAMS}"

    rm -rf "${DST}"
    mkdir -p "${DST}"

    start=$(date +%s.%N)
    rclone copy "${SRC}" "${DST}" \
        --transfers="${t}" --checkers="${c}" \
        --multi-thread-streams="${MULTI_THREAD_STREAMS}" \
        --stats=30s --stats-one-line
    rc=$?
    end=$(date +%s.%N)

    elapsed=$(awk "BEGIN{printf \"%.2f\", ${end}-${start}}")
    mbps=$(awk "BEGIN{printf \"%.1f\", ${src_bytes}/1000000/(${end}-${start})}")
    got=$(find "${DST}" -type f | wc -l | tr -d ' ')

    # integrity check
    integrity="skipped"
    if [ "${rc}" -ne 0 ]; then
        integrity="copy_failed(rc=${rc})"
    elif [ "${VERIFY}" = "1" ]; then
        if rclone check "${SRC}" "${DST}" --one-way >/dev/null 2>&1; then
            integrity="ok"
        else
            integrity="MISMATCH"
        fi
    fi

    echo "    -> ${elapsed}s | ${mbps} MB/s | files=${got}/${EXPECTED_FILES} | integrity=${integrity}"
    echo "${t},${c},${MULTI_THREAD_STREAMS},${elapsed},${mbps},${got},${integrity}" >> "${RESULTS_CSV}"

    # free disk before the next run (keep only the very last copy if requested)
    if [ "${KEEP_LAST}" = "1" ] && [ "${run_idx}" -eq "${n_runs}" ]; then
        echo "    (keeping final copy at ${DST})"
    else
        rm -rf "${DST}"
    fi
  done
done

# ----------------------------- summary --------------------------------------
echo
echo "=================================================================="
echo " RESULTS (fastest first)            tile=${TILE_ID}  ${src_gib} GiB"
echo "=================================================================="
printf "%-10s %-9s %-9s %-12s %-9s %-12s\n" "transfers" "checkers" "wall_s" "MB/s" "files" "integrity"
tail -n +2 "${RESULTS_CSV}" | sort -t, -k4 -n | \
  awk -F, '{printf "%-10s %-9s %-9s %-12s %-9s %-12s\n",$1,$2,$4,$5,$6,$7}'
echo "------------------------------------------------------------------"
echo "CSV: ${RESULTS_CSV}"
