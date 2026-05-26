#!/bin/bash
# =============================================================================
#  Benchmark s5cmd upload of ONE prediction tile: Hendrix  ->  CloudFerro
#
#  Sibling of:
#    - scripts/lumi/benchmark_rclone_copy.sh   (LUMI-O  -> LUMI /scratch)
#    - scripts/hendrix/benchmark_rclone_copy.sh (LUMI-O -> Hendrix)
#
#  Sweeps s5cmd --numworkers x cp --concurrency for a tile's local COG
#  directory, measures wall time + throughput, and verifies that the
#  destination object count + total size matches the source after each run.
#
#  Source : ${SRC_BASE}/${TILE_ID}                       (local on hendrix)
#  Dest   : s3://vsm-data-public/${YEAR}/${TILE_ID}      (on CloudFerro)
#
#  Usage:
#     bash benchmark_s5cmd_upload.sh <TILE_ID> [YEAR]
#     # override grids / paths via env vars, e.g.:
#     NUMWORKERS_LIST="64 256" CONCURRENCY_LIST="5 10 20" \
#         bash benchmark_s5cmd_upload.sh 32MRE 2024
#
#  Each run uploads the whole tile, then deletes it from the bucket before
#  the next run. Total data uploaded = (#numworkers) x (#concurrency) x size.
# =============================================================================
set -uo pipefail

# ----------------------------- configuration --------------------------------
TILE_ID=${1:?Usage: $0 <TILE_ID> [YEAR]   (e.g. $0 32MRE 2024)}
YEAR=${2:-2020}

# Local source (the COG layout produced by postprocessing).
SRC_BASE=${SRC_BASE:-${HOME}/data/gvs/products/vsm/${YEAR}/blended/tiles/cog}
SRC="${SRC_BASE}/${TILE_ID}"

# CloudFerro endpoint + credentials (matches scripts/core/transfer_data.sh).

CF_ENDPOINT=https://s3.waw4-1.cloudferro.com
CF_CREDS=${CF_CREDS:-${HOME}/.config/s5cmd/s5cmd.cfg}

# Destination on CloudFerro: shared bucket, year/tile prefix.
DST_BUCKET=${DST_BUCKET:-vsm-data-public}
DST_PREFIX=${DST_PREFIX:-${YEAR}/${TILE_ID}}
DST="s3://${DST_BUCKET}/${DST_PREFIX}"

# Parameter grids to sweep (space-separated; override via env).
#   numworkers : global s5cmd worker pool (default 256)
#   concurrency: parallel parts per file inside `cp` (default 5)
NUMWORKERS_LIST=${NUMWORKERS_LIST:-"16 64 256"}
CONCURRENCY_LIST=${CONCURRENCY_LIST:-"2 5 10 20"}

# Fixed knobs.
PART_SIZE_MB=${PART_SIZE_MB:-50}      # multipart part size for `cp`
VERIFY=${VERIFY:-1}                   # 1 = compare object count + bytes after upload
KEEP_LAST=${KEEP_LAST:-0}             # 1 = leave the final upload on CloudFerro

# Results CSV stays local on hendrix.
RESULTS_DIR=${RESULTS_DIR:-${HOME}/data/gvs/products/vsm/${YEAR}/s5cmd_bench}
mkdir -p "${RESULTS_DIR}"
RESULTS_CSV="${RESULTS_DIR}/bench_${TILE_ID}_$(date +%Y%m%d_%H%M%S).csv"

# ----------------------------- pre-flight -----------------------------------
command -v s5cmd >/dev/null || { echo "ERROR: s5cmd not found on PATH"; exit 1; }
[ -r "${CF_CREDS}" ] || { echo "ERROR: credentials file not readable: ${CF_CREDS}"; exit 1; }
[ -d "${SRC}" ] || { echo "ERROR: source dir does not exist: ${SRC}"; exit 1; }

S5="s5cmd --credentials-file ${CF_CREDS} --endpoint-url ${CF_ENDPOINT}"

echo "=================================================================="
echo " s5cmd upload benchmark   Hendrix -> CloudFerro"
echo "   source  : ${SRC}"
echo "   dest    : ${DST}"
echo "   endpoint: ${CF_ENDPOINT}"
echo "   results : ${RESULTS_CSV}"
echo "=================================================================="

# Sum file sizes only (du -sb includes directory inodes, which have no S3 analogue).
src_bytes=$(find "${SRC}" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
src_count=$(find "${SRC}" -type f | wc -l | tr -d ' ')
[ "${src_bytes:-0}" = "0" ] && { echo "ERROR: source is empty"; exit 1; }

src_gib=$(awk "BEGIN{printf \"%.2f\", ${src_bytes}/1073741824}")
n_w=$(echo ${NUMWORKERS_LIST} | wc -w)
n_c=$(echo ${CONCURRENCY_LIST} | wc -w)
n_runs=$(( n_w * n_c ))
total_gib=$(awk "BEGIN{printf \"%.1f\", ${src_bytes}*${n_runs}/1073741824}")

echo "   files       : ${src_count}"
echo "   size        : ${src_gib} GiB"
echo "   sweep       : numworkers={${NUMWORKERS_LIST}}  concurrency={${CONCURRENCY_LIST}}  part_size=${PART_SIZE_MB}MB"
echo "   runs        : ${n_runs}  ->  ~${total_gib} GiB will be uploaded in total"
echo "------------------------------------------------------------------"

echo "numworkers,concurrency,part_size_MB,wall_s,throughput_MBps,files,integrity_ok" > "${RESULTS_CSV}"

# ----------------------------- benchmark loop -------------------------------
run_idx=0
for w in ${NUMWORKERS_LIST}; do
  for c in ${CONCURRENCY_LIST}; do
    run_idx=$((run_idx + 1))
    echo
    echo ">>> [${run_idx}/${n_runs}] --numworkers=${w}  cp --concurrency=${c} --part-size=${PART_SIZE_MB}"

    # Clean destination prefix so this run is a real fresh upload.
    ${S5} rm "${DST}/*" >/dev/null 2>&1 || true

    start=$(date +%s.%N)
    ${S5} --numworkers="${w}" \
        cp --concurrency="${c}" --part-size="${PART_SIZE_MB}" \
        "${SRC}/*" "${DST}/"
    rc=$?
    end=$(date +%s.%N)

    elapsed=$(awk "BEGIN{printf \"%.2f\", ${end}-${start}}")
    mbps=$(awk "BEGIN{printf \"%.1f\", ${src_bytes}/1000000/(${end}-${start})}")

    integrity="skipped"
    got="?"
    if [ "${rc}" -ne 0 ]; then
        integrity="upload_failed(rc=${rc})"
    elif [ "${VERIFY}" = "1" ]; then
        # Bytes from `s5cmd du` (first numeric token on the line),
        # object count from `s5cmd ls` (one line per object).
        dst_bytes=$(${S5} du "${DST}/*" 2>/dev/null | grep -oE '[0-9]+' | head -1)
        dst_count=$(${S5} ls "${DST}/*" 2>/dev/null | wc -l | tr -d ' ')
        got=${dst_count:-0}
        if [ "${dst_bytes:-0}" = "${src_bytes}" ] && [ "${dst_count:-0}" = "${src_count}" ]; then
            integrity="ok"
        else
            integrity="MISMATCH(${dst_bytes:-?}B/${dst_count:-?}f vs ${src_bytes}B/${src_count}f)"
        fi
    fi

    echo "    -> ${elapsed}s | ${mbps} MB/s | files=${got}/${src_count} | integrity=${integrity}"
    echo "${w},${c},${PART_SIZE_MB},${elapsed},${mbps},${got},${integrity}" >> "${RESULTS_CSV}"
  done
done

# Optional final cleanup (keep last run's data if requested).
if [ "${KEEP_LAST}" = "1" ]; then
    echo "    (keeping final upload at ${DST})"
else
    ${S5} rm "${DST}/*" >/dev/null 2>&1 || true
fi

# ----------------------------- summary --------------------------------------
echo
echo "=================================================================="
echo " RESULTS (fastest first)            tile=${TILE_ID}  ${src_gib} GiB"
echo "=================================================================="
printf "%-11s %-12s %-9s %-12s %-9s %-12s\n" "numworkers" "concurrency" "wall_s" "MB/s" "files" "integrity"
tail -n +2 "${RESULTS_CSV}" | sort -t, -k4 -n | \
  awk -F, '{printf "%-11s %-12s %-9s %-12s %-9s %-12s\n",$1,$2,$4,$5,$6,$7}'
echo "------------------------------------------------------------------"
echo "CSV: ${RESULTS_CSV}"
