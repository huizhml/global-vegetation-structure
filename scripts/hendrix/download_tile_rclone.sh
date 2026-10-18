#!/bin/bash
# =============================================================================
#  Download tile predictions from hendrix to a local / external drive with
#  rclone (remote `hendrix:` = sftp to hendrixgate, see `rclone config`).
#  Runs on the LOCAL machine.
#
#  Each tile-year lands flat in {DEST_ROOT}/{YEAR}/{VERSION}/{TILE}/RH*_Q*.tif
#  (303 files), whether the source was cog/ or geotiff/.
#
#  Modes:
#     download_dir  <TILE> <YEAR> [VERSION]     (default VERSION=original)
#         hendrix:data/gvs/products/vsm/{YEAR}/{VERSION}/tiles/{cog|geotiff}/{TILE}/
#         first of cog/{TILE}, cog/{TILE}_cog (2024 masked, *.cog.tif names kept),
#         geotiff/{TILE} that holds all 303 files.
#     download_manifest <YEAR> <TILE>...        (VERSION = $RELEASE, default released_v0)
#         the files of these tiles per the release manifest (one row per tile),
#         hendrix:data/gvs/products/vsm/{YEAR}/{RELEASE}.tsv (tools/release_manifest.py,
#         `sbatch scripts/hendrix/run_cpu.sh release_manifest <YEAR>`). Each
#         quantile has its own dir (2020: Q1 in cog/, Q0/Q2 in geotiff/), one
#         rclone call each. The manifest is cached as {DEST_ROOT}/{YEAR}/{RELEASE}.tsv,
#         the index of where each tile came from.
#
#  Env knobs:
#     DEST_ROOT   (default "/Volumes/One Touch/VSM_hui/")
#     RELEASE     (default released_v0, download_manifest only)
#     INCLUDE     rclone filter for a subset, e.g. INCLUDE='RH98_Q*.tif' (download_dir only)
#     DRY_RUN=1   only list what would be copied
#
#  Examples:
#     bash scripts/hendrix/download_tile_rclone.sh download_dir 32UNG 2019
#     bash scripts/hendrix/download_tile_rclone.sh download_manifest 2020 32UPG 32UPF 33VUC 32UNG
#
#  Re-running is safe: rclone copy skips files that are already complete and
#  verifies the md5 of every file it transfers (md5sum run on hendrix). The
#  closing rclone check compares sizes only, to catch missing files without
#  re-hashing ~72 GB per GeoTIFF tile-year on every run.
#
#  Speed: over the VPN the total tops out at ~10 MB/s whatever the number of
#  streams (one stream ~8 MB/s; measured 2026-10-16). Do NOT raise the sftp chunk
#  size: hendrix serves at most 64k per read and rclone 1.61 then truncates
#  files (128k -> half, 255k -> a quarter; copy catches it as "sizes differ").
#  A COG is ~7x smaller than the GeoTIFF of the same tile-year.
# =============================================================================
set -euo pipefail

REMOTE=${REMOTE:-hendrix:data/gvs}
DEST_ROOT=${DEST_ROOT:-"/Volumes/One Touch/VSM_hui/"}
N_FILES=303  # RH0..RH100 x Q0..Q2

_check_dest() {
    # Without this rclone would silently create the path on the internal disk
    # when the external drive is not mounted.
    local volume=$(echo "$DEST_ROOT" | cut -d/ -f1-3)
    if [[ $DEST_ROOT == /Volumes/* ]] && [ ! -d "$volume" ]; then
        echo "ERROR: $volume is not mounted" >&2
        return 1
    fi
}

_copy_args() {
    copy_args=(--transfers 4 --checkers 8 --stats-one-line --stats 30s -v)
    [ "${DRY_RUN:-0}" = 1 ] && copy_args+=(--dry-run)
    return 0
}

download_dir() {
    local tile_id=${1:?usage: download_dir <TILE> <YEAR> [VERSION]} year=${2:?year} version=${3:-original}
    _check_dest
    local base=$REMOTE/products/vsm/$year/$version/tiles sub="" n
    # cog/{TILE}_cog/RH*_Q*.cog.tif is the layout of 2024 masked
    for s in cog/$tile_id cog/${tile_id}_cog geotiff/$tile_id; do
        n=$( (rclone lsf --include '*.tif' $base/$s 2>/dev/null || true) | wc -l | tr -d ' ')
        [ "$n" -ge $N_FILES ] && { sub=$s; break; }
    done
    [ -z "$sub" ] && { echo "[miss] $year $version $tile_id: no cog/ or geotiff/ with $N_FILES files" >&2; return 3; }

    local src=$base/$sub dst="$DEST_ROOT/$year/$version/$tile_id"
    local filter=(--include "${INCLUDE:-*.tif}")
    _copy_args
    echo "== $src -> $dst"
    rclone copy "$src" "$dst" "${filter[@]}" "${copy_args[@]}"
    [ "${DRY_RUN:-0}" = 1 ] || rclone check "$src" "$dst" "${filter[@]}" --one-way --size-only
}

download_manifest() {
    local year=${1:?usage: download_manifest <YEAR> <TILE>...}; shift
    [ $# -gt 0 ] || { echo "usage: download_manifest <YEAR> <TILE>..." >&2; return 1; }
    _check_dest
    local release=${RELEASE:-released_v0}
    local manifest="$DEST_ROOT/$year/$release.tsv"
    # Skipped by rclone when unchanged, so this costs a stat on every run.
    rclone copyto "$REMOTE/products/vsm/$year/$release.tsv" "$manifest"

    _copy_args
    local list=$(mktemp) tile row dst q dir suffix n_files n_missing rc=0
    for tile in "$@"; do
        # tile_id q0_root q1_root q2_root file_suffix n_files size_bytes
        row=$(awk -F'\t' -v t="$tile" 'NR > 1 && $1 == t' "$manifest")
        [ -z "$row" ] && { echo "[miss] $tile not in $manifest" >&2; rc=3; continue; }
        suffix=$(echo "$row" | cut -f5)
        n_files=$(echo "$row" | cut -f6)
        dst="$DEST_ROOT/$year/$release/$tile"
        echo "== $year $release $tile: $n_files files on hendrix -> $dst"
        [ "$n_files" -lt $N_FILES ] && echo "[incomplete] $tile: only $n_files of $N_FILES files on hendrix" >&2 && rc=1
        for q in 0 1 2; do
            dir=$(echo "$row" | cut -f$((q + 2)))/$tile
            for rh in $(seq 0 100); do echo "RH${rh}_Q${q}${suffix}"; done > "$list"
            echo "   Q$q: $dir"
            rclone copy "$REMOTE/products/vsm/$dir" "$dst" --files-from "$list" "${copy_args[@]}" || rc=1
            [ "${DRY_RUN:-0}" = 1 ] || rclone check "$REMOTE/products/vsm/$dir" "$dst" \
                --files-from "$list" --one-way --size-only || rc=1
            # copy and check both skip a listed file that is missing on hendrix
            [ "${DRY_RUN:-0}" = 1 ] && continue
            n_missing=$(while read -r f; do [ -f "$dst/$f" ] || echo "$f"; done < "$list" | wc -l | tr -d ' ')
            [ "$n_missing" -gt 0 ] && { echo "[incomplete] $tile Q$q: $n_missing files not downloaded" >&2; rc=1; }
        done
    done
    rm -f "$list"
    return $rc
}

case ${1:-} in
download_dir|download_manifest)
    "$@" ;;
*)
    awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"
    exit 1 ;;
esac
