#!/bin/bash
# =============================================================================
#  Download specific per-tile RH-band TIFs from hendrix to local.
#
#  Runs ON the LOCAL machine (your laptop). Hendrix is reached via the SSH
#  config alias `hendrix1:` (set up in ~/.ssh/config), same as bulk_download.sh.
#
#  Path mapping:
#     remote   hendrix1:~/data/gvs/products/vsm/{YEAR}/{VERSION}/tiles/cog/{TILE}/RH{RH}_Q{Q}.tif
#     local    ${LOCAL_ROOT}/{YEAR}/RH{RH}/{TILE}.tif
#
#  Example:
#     remote   hendrix1:~/data/gvs/products/vsm/2020/original/tiles/cog/32UMF/RH98_Q1.tif
#     local    ~/gvsm/products/diversity_indices/2020/RH98/32UMF.tif
#
#  Subcommands:
#     download       <TILE>     one tile
#     download-list  <FILE>     tile_ids from FILE (one per line; '#' / blank skipped)
#
#  Per-run knobs (env vars — set any subset; unset ones use defaults):
#     YEAR            (default 2020)
#     VERSION         (default original)
#     RH              (default 98)
#     Q               (default 1)
#     HENDRIX_ALIAS   SSH alias (default hendrix1)
#     HENDRIX_BASE    remote base, relative to the hendrix user's $HOME
#                     (default data/gvs/products/vsm)
#     LOCAL_ROOT      local destination root
#                     (default ${HOME}/gvsm/products/diversity_indices)
#     PARALLEL        concurrent rsync transfers (default 8)
#
#  Examples:
#     bash scripts/hendrix/download_from_hendrix.sh download 32UMF
#     # → ~/gvsm/products/diversity_indices/2020/RH98/32UMF.tif
#
#     bash scripts/hendrix/download_from_hendrix.sh download-list dk.txt
#
#     # Different year / version / band / quartile (any subset of overrides):
#     YEAR=2024 VERSION=bias_corrected RH=25 \
#         bash scripts/hendrix/download_from_hendrix.sh download-list dk.txt
#
#     # Loop over several bands for one tile list:
#     for rh in 98 75 50 25; do
#         RH=${rh} bash scripts/hendrix/download_from_hendrix.sh download-list dk.txt
#     done
#
#  Safe to `source`: the CLI dispatcher at the bottom only fires when this
#  file is executed directly.
# =============================================================================
set -uo pipefail

# ----------------------------- shared config --------------------------------
HENDRIX_ALIAS=${HENDRIX_ALIAS:-hendrix1}
# Relative to the hendrix user's $HOME — no leading slash, no tilde. rsync/ssh
# resolves relative remote paths from $HOME on the remote side. Using ~ here
# would get tilde-expanded on the LOCAL machine (wrong $HOME).
HENDRIX_BASE=${HENDRIX_BASE:-data/gvs/products/vsm}
LOCAL_ROOT=${LOCAL_ROOT:-${HOME}/gvsm/products/diversity_indices}
PARALLEL=${PARALLEL:-8}

# Per-run knobs — set via env on the command line, e.g.
#   YEAR=2024 RH=25 bash download_from_hendrix.sh download-list dk.txt
# Exported so the xargs worker subshell inherits them.
: "${YEAR:=2020}"
: "${VERSION:=original}"
: "${RH:=98}"
: "${Q:=1}"
export YEAR VERSION RH Q HENDRIX_ALIAS HENDRIX_BASE LOCAL_ROOT

# ----------------------------- helpers --------------------------------------
_require_rsync() {
    command -v rsync >/dev/null \
        || { echo "ERROR: rsync not found" >&2; return 1; }
}

# Build the remote source path. Kept as a function so the worklist worker
# (run via xargs -I {}) and the single-tile path can share one definition.
_remote_path() {
    local tile_id=$1 year=$2 version=$3 rh=$4 q=$5
    echo "${HENDRIX_ALIAS}:${HENDRIX_BASE}/${year}/${version}/tiles/cog/${tile_id}/RH${rh}_Q${q}.tif"
}

_local_path() {
    local tile_id=$1 year=$2 rh=$3
    echo "${LOCAL_ROOT}/${year}/RH${rh}/${tile_id}.tif"
}

# Worker for the batch path. Exported so xargs -I {} bash -c can call it
# without inlining the body (macOS BSD xargs has a ~256-byte cap on the
# fixed/template portion of the command, which inline rsync logic blows
# right past). Reads params from env vars set by the caller.
_download_one_tile_worker() {
    local tile_id=$1
    local src="${HENDRIX_ALIAS}:${HENDRIX_BASE}/${YEAR}/${VERSION}/tiles/cog/${tile_id}/RH${RH}_Q${Q}.tif"
    local dst="${LOCAL_ROOT}/${YEAR}/RH${RH}/${tile_id}.tif"
    if [ -e "${dst}" ]; then
        echo "[skip] ${tile_id}"
    elif rsync -az --partial "${src}" "${dst}" 2>/dev/null; then
        echo "[ok  ] ${tile_id}"
    else
        echo "[FAIL] ${tile_id}" >&2
    fi
}
export -f _download_one_tile_worker

# ----------------------------- one tile -------------------------------------
# Usage: rsync_tile_from_hendrix <TILE>
# Reads YEAR / VERSION / RH / Q from env (set defaults at file top).
rsync_tile_from_hendrix() {
    local tile_id=${1:?Usage: rsync_tile_from_hendrix <TILE>   (use env: YEAR / VERSION / RH / Q)}
    _require_rsync || return 1

    local src dst dst_dir
    src=$(_remote_path "${tile_id}" "${YEAR}" "${VERSION}" "${RH}" "${Q}")
    dst=$(_local_path "${tile_id}" "${YEAR}" "${RH}")
    dst_dir=$(dirname "${dst}")
    mkdir -p "${dst_dir}"

    if [ -e "${dst}" ]; then
        echo "[skip] ${tile_id}  (already at ${dst})"
        return 0
    fi

    # -a: archive, -z: compress, --partial: keep partial on interrupt so resumable
    if rsync -az --partial "${src}" "${dst}" 2>/dev/null; then
        echo "[ok  ] ${tile_id}  ->  ${dst}"
        return 0
    else
        # Distinguish "remote file missing" from real failures via a follow-up
        # ssh -q test: cheaper than a verbose rsync rerun.
        if ! ssh -q "${HENDRIX_ALIAS}" "test -f ${HENDRIX_BASE}/${YEAR}/${VERSION}/tiles/cog/${tile_id}/RH${RH}_Q${Q}.tif" 2>/dev/null; then
            echo "[miss] ${tile_id}  (remote file not found)" >&2
            return 3
        fi
        echo "[FAIL] ${tile_id}  (rsync failed)" >&2
        return 1
    fi
}

# ----------------------------- batch ----------------------------------------
# Usage: rsync_tile_list_from_hendrix <FILE>
# Reads YEAR / VERSION / RH / Q from env (set defaults at file top).
rsync_tile_list_from_hendrix() {
    local list_file=${1:?Usage: rsync_tile_list_from_hendrix <FILE>   (use env: YEAR / VERSION / RH / Q)}
    _require_rsync || return 1
    [ -r "${list_file}" ] || { echo "ERROR: cannot read ${list_file}" >&2; return 1; }

    local total
    total=$(grep -cvE '^\s*(#|$)' "${list_file}" || true)
    [ "${total}" -gt 0 ] || { echo "ERROR: ${list_file} has no tile ids" >&2; return 1; }

    local dst_dir="${LOCAL_ROOT}/${YEAR}/RH${RH}"
    mkdir -p "${dst_dir}"

    echo "=================================================================="
    echo "  ${total} tiles  ->  ${dst_dir}/{tile_id}.tif"
    echo "  source : ${HENDRIX_ALIAS}:${HENDRIX_BASE}/${YEAR}/${VERSION}/tiles/cog/{tile_id}/RH${RH}_Q${Q}.tif"
    echo "  parallel: ${PARALLEL}"
    echo "=================================================================="

    local t0
    t0=$(date +%s)

    # All knobs (YEAR / VERSION / RH / Q / HENDRIX_* / LOCAL_ROOT) are exported
    # at the top of the file, so the xargs subshell inherits them. The xargs
    # template stays tiny — macOS BSD xargs caps fixed-template size at ~256
    # bytes and chokes ("command line cannot be assembled, too long") otherwise.
    grep -vE '^\s*(#|$)' "${list_file}" \
        | tr -d '[:blank:]' \
        | grep -v '^$' \
        | xargs -P "${PARALLEL}" -I {} bash -c '_download_one_tile_worker "$@"' _ {}

    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))

    # Count outcomes from the dst dir state (cheap and accurate).
    local got missing
    got=$(find "${dst_dir}" -maxdepth 1 -name '*.tif' | wc -l | tr -d ' ')
    missing=$((total - got))

    echo "=================================================================="
    echo "Done in ${mins}m${secs}s. local files: ${got}/${total}  (missing/failed: ${missing})"
}

# ----------------------------- CLI dispatcher -------------------------------
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    cmd=${1:-}
    case "${cmd}" in
        download)
            shift
            [ "$#" -gt 1 ] && \
                echo "WARN: extra positional args after <TILE> are ignored; use env vars (e.g. YEAR=2024)." >&2
            rsync_tile_from_hendrix "$1" ;;
        download-list)
            shift
            [ "$#" -gt 1 ] && \
                echo "WARN: extra positional args after <FILE> are ignored; use env vars (e.g. YEAR=2024)." >&2
            rsync_tile_list_from_hendrix "$1" ;;
        ""|-h|--help)
            sed -n '2,55p' "${BASH_SOURCE[0]}" ;;
        *)
            echo "ERROR: unknown subcommand '${cmd}'. Try: download | download-list" >&2
            exit 1 ;;
    esac
fi
