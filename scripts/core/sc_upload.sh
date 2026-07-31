#!/bin/bash
# =============================================================================
#  Upload VSM prediction COGs  lumi-o  ->  Source Cooperative (source.coop).
#
#  Source Cooperative counterpart of cf_upload.sh. TWO-HOP transport, because
#  source.coop's gateway rejects rclone's writes but accepts the AWS CLI:
#      rclone copy  lumi-o tile -> local scratch   (rclone reads lumi-o fine)
#      aws s3 sync  scratch     -> source.coop      (aws cli writes fine, auto-
#                                                    refreshes creds via profile)
#      rm -rf       scratch
#  A tile is ~75 MB, so only one tile sits on disk at a time. `aws s3 sync` is
#  RESUMABLE (skips already-uploaded objects); before staging we also skip a
#  tile outright if the destination already matches the source (count+bytes),
#  so restarts don't re-download finished tiles.
#
#  Two entry points:
#    sc_upload_tile <TILE_ID> [YEAR]   upload ALL RHs of one tile
#    sc_upload_rh   <RH>      [YEAR]    upload ONE RH across ALL tiles
#
#  Source layout on lumi-o (bucket per zone-band + year; zone-band derived from
#  the tile id, e.g. 60UUD -> 60u):
#    ${SC_SRC_REMOTE}:<zoneband>-<YEAR>/<TILE_ID>/RH{n}_Q{q}.tif
#  Destination layout on source.coop:
#    ${SC_DEST_ROOT}/<YEAR>/<TILE_ID>/RH{n}_Q{q}.tif
#
#  Usage (source the file, then call the functions):
#      source scripts/core/sc_upload.sh
#      sc_check_creds                 # confirm aws profile reaches source.coop
#      sc_upload_tile 60UUD 2024
#      sc_upload_rh   98    2024
#
#  Or run directly as a CLI:
#      bash scripts/core/sc_upload.sh check
#      bash scripts/core/sc_upload.sh tile 60UUD 2024
#      bash scripts/core/sc_upload.sh rh   98    2024
#
#  Full multi-day run (resumable — re-run any time):
#      for t in $(sc_list_src_tiles 2024); do sc_upload_tile "$t" 2024; done
#
#  Config via env vars (shared defaults come from source_coop_utils.sh):
#      SC_RCLONE / SC_AWS / SC_AWS_PROFILE / SC_SRC_REMOTE / SC_DEST_ROOT
#      SC_ENDPOINT / SC_REGION / SC_STAGE_DIR
#      SC_TRANSFERS   default 16   (rclone parallel downloads from lumi-o)
#      SC_AWS_CONC    default 16   (aws s3 max_concurrent_requests on upload)
#      SC_QUIET       default ""   (set to 1 to reduce tool output)
# =============================================================================

# Reuse the source.coop helpers (path builders, _sc_aws, sc_verify_tile, ...).
_SC_UPLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/core/source_coop_utils.sh
source "${_SC_UPLOAD_DIR}/source_coop_utils.sh"

SC_TRANSFERS=${SC_TRANSFERS:-16}
SC_AWS_CONC=${SC_AWS_CONC:-16}
SC_QUIET=${SC_QUIET:-}

# ---- internals ------------------------------------------------------------

_sc_download() {
    # rclone copy <src> <stage_dir> restricted to a filename glob (lumi-o read).
    local src=$1 stage=$2 include=$3
    local args=(copy "${src}" "${stage}"
        --transfers "${SC_TRANSFERS}" --checkers "${SC_TRANSFERS}" --fast-list)
    [ -n "${include}" ] && args+=(--include "${include}")
    [ -n "${SC_QUIET}" ] && args+=(--stats-one-line --stats 60s)
    "${SC_RCLONE}" "${args[@]}"
}

_sc_sync_up() {
    # aws s3 sync <stage_dir> <dest_url>/ (source.coop write, resumable).
    local stage=$1 url=$2
    local extra=()
    [ -n "${SC_QUIET}" ] && extra+=(--only-show-errors)
    AWS_MAX_CONCURRENT_REQUESTS="${SC_AWS_CONC}" \
        _sc_aws s3 sync "${stage}" "${url%/}/" "${extra[@]}"
}

_sc_stage_upload_verify() {
    # Stage one tile's files (matching <include>), upload, verify, clean up.
    # Args: <tile> <year> <include_glob> <label>
    # Returns: 0 ok, 2 skipped (already complete), 1 failed.
    local tile=$1 year=$2 include=$3 label=$4
    local src url stage
    src="$(_sc_src_tile "${tile}" "${year}")"
    url="$(_sc_dest_url "${tile}" "${year}")"
    stage="${SC_STAGE_DIR}/${tile}"

    local s_bytes s_count d_bytes d_count
    read -r s_bytes s_count < <(_sc_rclone_size "${src}" "${include}")
    if [ "${s_count}" = "0" ]; then
        return 3   # nothing to upload for this tile/glob
    fi

    # Skip if the destination already matches the source (resumable restart).
    read -r d_bytes d_count < <(_sc_remote_stats "${url}" "${include}")
    if [ "${d_count}" = "${s_count}" ] && [ "${d_bytes}" = "${s_bytes}" ]; then
        echo "    SKIP ${label}: already complete (${d_count} files, ${d_bytes} bytes)"
        return 2
    fi

    mkdir -p "${stage}"
    if ! _sc_download "${src}" "${stage}" "${include}"; then
        echo "    ERROR ${label}: rclone download from lumi-o failed" >&2
        rm -rf "${stage}"; return 1
    fi
    if ! _sc_sync_up "${stage}" "${url}"; then
        echo "    ERROR ${label}: aws s3 sync to source.coop failed" >&2
        rm -rf "${stage}"; return 1
    fi

    # Verify remote vs the staged local files, then clean up.
    read -r d_bytes d_count < <(_sc_remote_stats "${url}" "${include}")
    local l_bytes l_count
    read -r l_bytes l_count < <(_sc_local_stats "${stage}" "${include}")
    rm -rf "${stage}"

    if [ "${d_count}" = "${l_count}" ] && [ "${d_bytes}" = "${l_bytes}" ]; then
        echo "    OK  ${label}: ${d_count} files, ${d_bytes} bytes"
        return 0
    fi
    echo "    MISMATCH ${label}: staged ${l_count}f/${l_bytes}B vs dest ${d_count}f/${d_bytes}B" >&2
    return 1
}

# ---- public: upload one tile (all RHs) ------------------------------------

sc_upload_tile() {
    # Upload every COG of ONE tile to ${SC_DEST_ROOT}/${YEAR}/${TILE}/.
    # Usage: sc_upload_tile <TILE_ID> [YEAR]
    _sc_check_src_remote || return 1
    _sc_check_dest_root  || return 1
    local tile=${1:?Usage: sc_upload_tile <TILE_ID> [YEAR]}
    local year=${2:-2024}

    echo ">>> upload tile ${tile} (${year})"
    echo "    $(_sc_src_tile "${tile}" "${year}")  ->  $(_sc_dest_url "${tile}" "${year}")/"
    _sc_stage_upload_verify "${tile}" "${year}" '*.tif' "${tile}"
    local rc=$?
    [ ${rc} = 3 ] && { echo "    ERROR: no *.tif for tile ${tile}" >&2; return 1; }
    return ${rc}
}

# ---- public: upload one RH (all tiles) ------------------------------------

sc_upload_rh() {
    # Upload ONE RH across ALL lumi-o tiles for a year. The RH argument is the
    # number after "RH" in the filenames (e.g. 98 -> RH98_Q*.tif).
    # Usage: sc_upload_rh <RH> [YEAR]
    _sc_check_src_remote || return 1
    _sc_check_dest_root  || return 1
    local rh=${1:?Usage: sc_upload_rh <RH> [YEAR]}
    local year=${2:-2024}
    local name="RH${rh}_Q*.tif"

    echo ">>> upload RH${rh} (${year}) across all tiles on ${SC_SRC_REMOTE}"
    local tiles; tiles=$(sc_list_src_tiles "${year}")
    if [ -z "${tiles}" ]; then
        echo "ERROR: no tiles found for ${year} on ${SC_SRC_REMOTE}" >&2
        return 1
    fi

    local n_ok=0 n_fail=0 n_skip=0 n_none=0 rc_all=0 tile rc
    while IFS= read -r tile; do
        [ -n "${tile}" ] || continue
        echo "  - ${tile}:"
        _sc_stage_upload_verify "${tile}" "${year}" "${name}" "${tile}/RH${rh}"
        rc=$?
        case ${rc} in
            0) n_ok=$((n_ok+1)) ;;
            2) n_skip=$((n_skip+1)) ;;
            3) n_none=$((n_none+1)) ;;
            *) n_fail=$((n_fail+1)); rc_all=1 ;;
        esac
    done <<<"${tiles}"

    echo "------------------------------------------------------------------"
    echo "RH${rh} (${year}): ${n_ok} ok, ${n_skip} already-done, ${n_fail} failed, ${n_none} without RH${rh}"
    return ${rc_all}
}

# ---- CLI dispatch (only when executed, not when sourced) ------------------

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    cmd=${1:-}
    shift || true
    case "${cmd}" in
        tile)    sc_upload_tile "$@" ;;
        rh)      sc_upload_rh   "$@" ;;
        check)   sc_check_creds ;;
        tiles)   sc_list_src_tiles "$@" ;;
        buckets) sc_list_src_buckets "$@" ;;
        *)
            echo "Usage:"
            echo "  bash $0 check                   # verify aws profile reaches source.coop"
            echo "  bash $0 buckets [YEAR]           # list source zone-band buckets on lumi-o"
            echo "  bash $0 tiles [YEAR]             # list source tiles on lumi-o"
            echo "  bash $0 tile <TILE_ID> [YEAR]    # all RHs of one tile"
            echo "  bash $0 rh   <RH>      [YEAR]    # one RH across all tiles"
            exit 2
            ;;
    esac
fi
