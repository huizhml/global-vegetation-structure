#!/bin/bash
# =============================================================================
#  Clean redundant raw predictions on LUMI-O
#
#  For every bucket named {zone}-{year} (e.g. 60g-2024) on the private LUMI-O
#  remote, this verifies the *corrected* per-tile output and, ONLY when that
#  output is complete and valid, deletes the now-redundant raw predictions:
#
#      lumi-465002698-private:60g-2024/60GTA                         <- VERIFY
#      lumi-465002698-private:60g-2024/predictions_GTiff_2024/60GTA  <- DELETE
#
#  A tile passes (and its predictions become deletable) when:
#    1. {bucket}/{tile} holds exactly EXPECTED_TIF (303) .tif files,
#    2. none of those .tif files is zero-byte, and
#    3. (deep check) every .tif opens cleanly with `gdalinfo` over /vsis3
#       (validates the GeoTIFF header via HTTP range reads -- no full download).
#       NOTE: header-only -- it does not decompress all pixels, so a
#       truncated-yet-openable file is not caught.
#
#  SAFETY: dry-run by default. Nothing is deleted unless you pass --apply.
#          If the deep check is requested but GDAL / credentials are not
#          available, the script aborts rather than deleting on a weaker check.
#
#  Usage:
#     bash clean_lumio_predictions.sh                    # dry-run, all buckets
#     bash clean_lumio_predictions.sh --apply            # actually delete
#     YEAR=2024 bash clean_lumio_predictions.sh          # only *-2024 buckets
#     bash clean_lumio_predictions.sh --apply 60g-2024   # one bucket only
#     DEEP_CHECK=0 bash clean_lumio_predictions.sh       # metadata check only
#
#  Tunables (env): LUMI_PROJECT EXPECTED_TIF DEEP_CHECK DEEP_JOBS YEAR
#
#  Run on a compute node with the rclone/lumio + a GDAL module loaded, e.g.:
#     srun -A project_465002698 -p small -c 16 --mem=8G -t 02:00:00 \
#          bash clean_lumio_predictions.sh
# =============================================================================
set -uo pipefail

# ----------------------------- argument parsing -----------------------------
APPLY=0
BUCKET_FILTER=()
for arg in "$@"; do
  case "$arg" in
    --apply)    APPLY=1 ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    -*)         echo "Unknown option: $arg" >&2; exit 2 ;;
    *)          BUCKET_FILTER+=("$arg") ;;
  esac
done

# ----------------------------- configuration --------------------------------
LUMI_PROJECT=${LUMI_PROJECT:-465002698}
REMOTE_NAME="lumi-${LUMI_PROJECT}-private"
REMOTE="${REMOTE_NAME}:"

EXPECTED_TIF=${EXPECTED_TIF:-303}        # expected .tif count per tile
DEEP_CHECK=${DEEP_CHECK:-1}              # 1 = also gdalinfo each tif over /vsis3
DEEP_JOBS=${DEEP_JOBS:-8}               # parallel gdalinfo workers
YEAR=${YEAR:-}                           # optional: restrict to buckets ending -YEAR

RESULTS_CSV=${RESULTS_CSV:-clean_lumio_$(date +%Y%m%d_%H%M%S).csv}

# rclone is provided either by the `rclone` module or by `lumio`.
module load rclone 2>/dev/null || module load lumio 2>/dev/null || true
command -v rclone >/dev/null || { echo "ERROR: rclone not found (load the rclone/lumio module)"; exit 1; }

# ----------------------- GDAL /vsis3 credential setup -----------------------
# Re-uses the S3 access key/secret/endpoint already stored in the rclone remote
# (LUMI-O is Ceph S3 and rclone keeps these in plaintext) so GDAL can read the
# objects directly via /vsis3 with cheap HTTP range requests.
setup_vsis3_env() {
    local cfg ak sk ep
    cfg=$(rclone config show "${REMOTE_NAME}" 2>/dev/null) \
        || { echo "ERROR: cannot read rclone config for ${REMOTE_NAME}"; return 1; }
    ak=$(printf '%s\n' "$cfg" | sed -n 's/^[[:space:]]*access_key_id[[:space:]]*=[[:space:]]*//p'     | head -1)
    sk=$(printf '%s\n' "$cfg" | sed -n 's/^[[:space:]]*secret_access_key[[:space:]]*=[[:space:]]*//p' | head -1)
    ep=$(printf '%s\n' "$cfg" | sed -n 's/^[[:space:]]*endpoint[[:space:]]*=[[:space:]]*//p'          | head -1)
    [ -n "$ak" ] && [ -n "$sk" ] && [ -n "$ep" ] \
        || { echo "ERROR: missing access_key_id / secret_access_key / endpoint in rclone config for ${REMOTE_NAME}"; return 1; }

    ep=${ep#http://}; ep=${ep#https://}; ep=${ep%/}     # GDAL wants host[:port], no scheme
    export AWS_ACCESS_KEY_ID="$ak"
    export AWS_SECRET_ACCESS_KEY="$sk"
    export AWS_S3_ENDPOINT="$ep"
    export AWS_HTTPS=YES
    export AWS_VIRTUAL_HOSTING=FALSE                    # Ceph radosgw -> path-style URLs
    export AWS_REGION=${AWS_REGION:-us-east-1}          # ignored by Ceph but GDAL wants one
    export AWS_DEFAULT_REGION="$AWS_REGION"
    # Performance: never list the whole bucket when opening a single object.
    export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
    export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif
    export VSI_CACHE=TRUE
}

if [ "$DEEP_CHECK" = "1" ]; then
    command -v gdalinfo >/dev/null \
        || { echo "ERROR: DEEP_CHECK=1 but gdalinfo not found. Load a GDAL module, or rerun with DEEP_CHECK=0."; exit 1; }
    setup_vsis3_env \
        || { echo "ERROR: could not configure GDAL /vsis3 access. Rerun with DEEP_CHECK=0 to skip the header check."; exit 1; }
fi

# Returns (on stdout) the relative path of every .tif that FAILS to open with
# gdalinfo over /vsis3. Empty output => all tifs have a readable header.
deep_check_failures() {
    local bucket=$1 tile=$2 relpaths=$3
    export VSIS3_BUCKET="$bucket" VSIS3_TILE="$tile"
    printf '%s\n' "$relpaths" | xargs -r -P "$DEEP_JOBS" -I{} bash -c '
        gdalinfo "/vsis3/${VSIS3_BUCKET}/${VSIS3_TILE}/$1" >/dev/null 2>&1 || printf "%s\n" "$1"
    ' _ {}
}

# ----------------------------- discover buckets -----------------------------
mapfile -t all_buckets < <(rclone lsf "${REMOTE}" --dirs-only 2>/dev/null | sed 's:/$::')

buckets=()
for b in "${all_buckets[@]}"; do
    [[ "$b" =~ ^[a-z0-9]+-[0-9]{4}$ ]] || continue                 # {zone}-{year}
    [ -n "$YEAR" ] && [ "${b##*-}" != "$YEAR" ] && continue        # YEAR filter
    if [ "${#BUCKET_FILTER[@]}" -gt 0 ]; then                      # explicit bucket(s)
        local_match=0
        for f in "${BUCKET_FILTER[@]}"; do [ "$f" = "$b" ] && local_match=1; done
        [ "$local_match" -eq 1 ] || continue
    fi
    buckets+=("$b")
done

# ----------------------------- banner ---------------------------------------
echo "=================================================================="
echo " LUMI-O prediction cleanup"
echo "   remote     : ${REMOTE}"
echo "   mode       : $([ "$APPLY" -eq 1 ] && echo 'APPLY (will DELETE)' || echo 'DRY-RUN (no deletions)')"
echo "   buckets    : ${#buckets[@]} matched${YEAR:+ (year=$YEAR)}"
echo "   pass rule  : ${EXPECTED_TIF} tifs, none empty$([ "$DEEP_CHECK" = 1 ] && echo ", gdalinfo OK")"
echo "   results    : ${RESULTS_CSV}"
echo "=================================================================="
[ "${#buckets[@]}" -eq 0 ] && { echo "No matching buckets. Nothing to do."; exit 0; }

echo "bucket,tile,tif_count,min_tif_kb,deep_failures,decision,freed_gib" > "${RESULTS_CSV}"

# ----------------------------- main loop ------------------------------------
n_pass=0 n_skip=0 n_del=0 n_delfail=0
total_freed_bytes=0

for bucket in "${buckets[@]}"; do
    year=${bucket##*-}
    preds_root="${REMOTE}${bucket}/predictions_GTiff_${year}"

    # Deletion candidates = tiles that actually HAVE a predictions folder.
    mapfile -t cand_tiles < <(rclone lsf "${preds_root}/" --dirs-only 2>/dev/null | sed 's:/$::')
    echo
    echo ">>> bucket ${bucket}: ${#cand_tiles[@]} tile(s) with predictions to consider"
    [ "${#cand_tiles[@]}" -eq 0 ] && continue

    for tile in "${cand_tiles[@]}"; do
        verify_path="${REMOTE}${bucket}/${tile}"
        preds_path="${preds_root}/${tile}"

        # --- (1) list the corrected output's .tif files (size + relpath) ---
        listing=$(rclone ls "${verify_path}" --include "*.tif" --fast-list 2>/dev/null)
        count=$(printf '%s\n' "$listing" | grep -c '[^[:space:]]')

        decision="" reason="" deep_fail="-" min_kb="-" freed_gib="0"

        if [ "$count" -eq 0 ]; then
            decision="KEEP"; reason="no corrected output at ${bucket}/${tile}"
        elif [ "$count" -ne "$EXPECTED_TIF" ]; then
            decision="KEEP"; reason="tif count ${count} != ${EXPECTED_TIF}"
        else
            # --- (2) reject empty (zero-byte) tifs ---
            min_size=$(printf '%s\n' "$listing" | awk 'NF{print $1}' | sort -n | head -1)
            min_kb=$(( ${min_size:-0} / 1024 ))
            empty=$(printf '%s\n' "$listing" | awk 'NF && $1==0{c++} END{print c+0}')
            if [ "${min_size:-0}" -le 0 ]; then
                decision="KEEP"; reason="${empty} zero-byte tif(s)"
            elif [ "$DEEP_CHECK" = "1" ]; then
                # --- (3) deep check: gdalinfo every tif header over /vsis3 ---
                relpaths=$(printf '%s\n' "$listing" | awk 'NF{ $1=""; sub(/^[[:space:]]+/,""); print }')
                fails=$(deep_check_failures "$bucket" "$tile" "$relpaths")
                deep_fail=$(printf '%s\n' "$fails" | grep -c '[^[:space:]]')
                if [ "$deep_fail" -gt 0 ]; then
                    decision="KEEP"; reason="${deep_fail} tif(s) failed gdalinfo"
                else
                    decision="PASS"
                fi
            else
                deep_fail="skipped"; decision="PASS"
            fi
        fi

        # --- act on the decision ---
        if [ "$decision" = "PASS" ]; then
            n_pass=$((n_pass + 1))
            # size of the predictions folder we are about to free
            freed_bytes=$(rclone size "${preds_path}" --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)
            freed_bytes=${freed_bytes:-0}
            freed_gib=$(awk "BEGIN{printf \"%.2f\", ${freed_bytes}/1073741824}")

            if [ "$APPLY" -eq 1 ]; then
                if rclone purge "${preds_path}"; then
                    decision="DELETED"; n_del=$((n_del + 1)); total_freed_bytes=$((total_freed_bytes + freed_bytes))
                else
                    decision="DELETE_FAILED"; n_delfail=$((n_delfail + 1))
                fi
            else
                decision="WOULD_DELETE"; total_freed_bytes=$((total_freed_bytes + freed_bytes))
            fi
            echo "    [${decision}] ${tile}: ${count} tifs ok, frees ${freed_gib} GiB  ->  ${preds_path}"
        else
            n_skip=$((n_skip + 1))
            echo "    [KEEP]  ${tile}: ${reason}"
        fi

        echo "${bucket},${tile},${count},${min_kb},${deep_fail},${decision},${freed_gib}" >> "${RESULTS_CSV}"
    done
done

# ----------------------------- summary --------------------------------------
total_freed_gib=$(awk "BEGIN{printf \"%.1f\", ${total_freed_bytes}/1073741824}")
echo
echo "=================================================================="
echo " SUMMARY"
echo "   passed check : ${n_pass}"
echo "   kept (skip)  : ${n_skip}"
if [ "$APPLY" -eq 1 ]; then
    echo "   deleted      : ${n_del}   (failed: ${n_delfail})"
    echo "   freed        : ${total_freed_gib} GiB"
else
    echo "   would delete : ${n_pass}   ->  would free ${total_freed_gib} GiB"
    echo "   (dry-run: re-run with --apply to actually delete)"
fi
echo "   CSV          : ${RESULTS_CSV}"
echo "=================================================================="
