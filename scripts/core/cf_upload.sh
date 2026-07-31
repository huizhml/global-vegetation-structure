#!/bin/bash
# =============================================================================
#  Upload VSM prediction COGs from local (Hendrix)  ->  CloudFerro.
#
#  Production counterpart of scripts/hendrix/benchmark_s5cmd_upload.sh:
#  same `s5cmd cp` transport with tuned knobs, but it NEVER deletes the
#  destination and verifies (object count + total bytes) after every upload.
#
#  Two entry points:
#    cf_upload_tile <TILE_ID> [YEAR]   upload ALL RHs of one tile
#    cf_upload_rh   <RH>      [YEAR]    upload ONE RH across ALL tiles
#
#  Local source layout (produced by postprocessing):
#    ${CF_LOCAL_BASE}/${YEAR}/blended/tiles/cog/${TILE_ID}/RH{n}_Q{q}.tif
#  Destination layout on CloudFerro:
#    s3://${CF_BUCKET}/${YEAR}/${TILE_ID}/RH{n}_Q{q}.tif
#
#  Usage (source the file, then call the functions):
#      source scripts/core/cf_upload.sh
#      cf_upload_tile 32MRE 2024
#      cf_upload_rh   98    2024
#
#  Or run directly as a CLI:
#      bash scripts/core/cf_upload.sh tile 32MRE 2024
#      bash scripts/core/cf_upload.sh rh   98    2024
#
#  Config via env vars (shared defaults come from cloudferro_utils.sh):
#      CF_ENDPOINT / CF_CREDS / CF_BUCKET / CF_LOCAL_BASE
#      CF_NUMWORKERS   default 16   (global s5cmd worker pool)
#      CF_CONCURRENCY  default 10    (parallel parts per file for `cp`)
#      CF_PART_SIZE_MB default 50    (multipart part size, MB)
# =============================================================================

# Reuse the CloudFerro helpers (_cf_s5, CF_* defaults, cf_verify_tile).
_CF_UPLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/core/cloudferro_utils.sh
source "${_CF_UPLOAD_DIR}/cloudferro_utils.sh"

# Transfer tuning (override via env). Defaults mirror the benchmark's grid.
CF_NUMWORKERS=${CF_NUMWORKERS:-16}
CF_CONCURRENCY=${CF_CONCURRENCY:-10}
CF_PART_SIZE_MB=${CF_PART_SIZE_MB:-50}

# ---- internals ------------------------------------------------------------

_cf_cog_base() {
    # Local COG root for a given year.
    local year=$1
    echo "${CF_LOCAL_BASE}/${year}/blended/tiles/cog"
}

_cf_cp() {
    # `s5cmd cp` with the tuned knobs. Args: <src_glob> <dst_url>.
    local src=$1 dst=$2
    _cf_s5 --numworkers="${CF_NUMWORKERS}" \
        cp --concurrency="${CF_CONCURRENCY}" --part-size="${CF_PART_SIZE_MB}" \
        "${src}" "${dst}"
}

_cf_local_stats() {
    # Print "<total_bytes> <file_count>" for files in <dir> matching <name_glob>.
    local dir=$1 name=$2
    local bytes count
    bytes=$(find "${dir}" -maxdepth 1 -type f -name "${name}" -printf '%s\n' 2>/dev/null \
            | awk '{s+=$1} END{print s+0}')
    count=$(find "${dir}" -maxdepth 1 -type f -name "${name}" 2>/dev/null | wc -l | tr -d ' ')
    echo "${bytes:-0} ${count:-0}"
}

_cf_remote_stats() {
    # Print "<total_bytes> <object_count>" for remote objects under <url_glob>.
    local glob=$1
    local bytes count
    bytes=$(_cf_s5 du "${glob}" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    count=$(_cf_s5 ls "${glob}" 2>/dev/null | wc -l | tr -d ' ')
    echo "${bytes:-0} ${count:-0}"
}

_cf_verify_subset() {
    # Compare a local file subset against its remote prefix on (count, bytes).
    # Args: <local_dir> <name_glob> <remote_url_glob> <label>
    local dir=$1 name=$2 rglob=$3 label=$4
    local src dst
    read -r s_bytes s_count < <(_cf_local_stats "${dir}" "${name}")
    read -r d_bytes d_count < <(_cf_remote_stats "${rglob}")
    if [ "${d_bytes}" = "${s_bytes}" ] && [ "${d_count}" = "${s_count}" ]; then
        echo "    OK  ${label}: ${d_count} files, ${d_bytes} bytes"
        return 0
    fi
    echo "    MISMATCH ${label}: local ${s_count}f/${s_bytes}B vs remote ${d_count}f/${d_bytes}B"
    return 1
}

# ---- public: upload one tile (all RHs) ------------------------------------

cf_upload_tile() {
    # Upload every COG of ONE tile to s3://${CF_BUCKET}/${YEAR}/${TILE_ID}/.
    # Usage: cf_upload_tile <TILE_ID> [YEAR]
    local tile=${1:?Usage: cf_upload_tile <TILE_ID> [YEAR]}
    local year=${2:-2024}

    local cog_base tile_dir url
    cog_base=$(_cf_cog_base "${year}")
    tile_dir="${cog_base}/${tile}"
    url="s3://${CF_BUCKET}/${year}/${tile}"

    if [ ! -d "${tile_dir}" ]; then
        echo "ERROR: local tile dir does not exist: ${tile_dir}"
        return 1
    fi
    read -r s_bytes s_count < <(_cf_local_stats "${tile_dir}" '*.tif')
    if [ "${s_count}" = "0" ]; then
        echo "ERROR: no *.tif under ${tile_dir}"
        return 1
    fi

    echo ">>> upload tile ${tile} (${year}): ${s_count} files, ${s_bytes} bytes"
    echo "    ${tile_dir}  ->  ${url}/"
    _cf_cp "${tile_dir}/*.tif" "${url}/" || { echo "ERROR: s5cmd cp failed for ${tile}"; return 1; }

    # Whole-tile verify reuses the shared helper.
    cf_verify_tile "${tile}" "${year}" "${tile_dir}"
}

# ---- public: upload one RH (all tiles) ------------------------------------

cf_upload_rh() {
    # Upload ONE RH across ALL local tiles for a year. The RH argument is the
    # number that appears after "RH" in the filenames (e.g. 98 -> RH98_Q*.tif),
    # so it works regardless of whether that number is a band index or metric.
    # Usage: cf_upload_rh <RH> [YEAR]
    local rh=${1:?Usage: cf_upload_rh <RH> [YEAR]}
    local year=${2:-2024}
    local name="RH${rh}_Q*.tif"

    local cog_base
    cog_base=$(_cf_cog_base "${year}")
    if [ ! -d "${cog_base}" ]; then
        echo "ERROR: local COG base does not exist: ${cog_base}"
        return 1
    fi

    echo ">>> upload RH${rh} (${year}) across all tiles under ${cog_base}"

    local n_ok=0 n_fail=0 n_skip=0 rc_all=0
    local tile_dir tile count url
    for tile_dir in "${cog_base}"/*/; do
        [ -d "${tile_dir}" ] || continue
        tile=$(basename "${tile_dir}")
        count=$(find "${tile_dir}" -maxdepth 1 -type f -name "${name}" 2>/dev/null | wc -l | tr -d ' ')
        if [ "${count}" = "0" ]; then
            n_skip=$((n_skip + 1))
            continue
        fi
        url="s3://${CF_BUCKET}/${year}/${tile}"
        echo "  - ${tile}: ${count} file(s) matching ${name}"
        if _cf_cp "${tile_dir}${name}" "${url}/"; then
            if _cf_verify_subset "${tile_dir%/}" "${name}" "${url}/RH${rh}_Q*" "${tile}/RH${rh}"; then
                n_ok=$((n_ok + 1))
            else
                n_fail=$((n_fail + 1)); rc_all=1
            fi
        else
            echo "    ERROR: s5cmd cp failed for ${tile}"
            n_fail=$((n_fail + 1)); rc_all=1
        fi
    done

    echo "------------------------------------------------------------------"
    echo "RH${rh} (${year}): ${n_ok} ok, ${n_fail} failed, ${n_skip} skipped (no RH${rh} files)"
    return ${rc_all}
}

# ---- CLI dispatch (only when executed, not when sourced) ------------------

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    cmd=${1:-}
    shift || true
    case "${cmd}" in
        tile) cf_upload_tile "$@" ;;
        rh)   cf_upload_rh   "$@" ;;
        *)
            echo "Usage:"
            echo "  bash $0 tile <TILE_ID> [YEAR]   # all RHs of one tile"
            echo "  bash $0 rh   <RH>      [YEAR]    # one RH across all tiles"
            exit 2
            ;;
    esac
fi
