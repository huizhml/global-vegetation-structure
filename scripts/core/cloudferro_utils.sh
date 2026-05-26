#!/bin/bash
# =============================================================================
#  CloudFerro / s5cmd helper functions.
#
#  Source this file, then call the cf_* functions:
#      source scripts/core/cloudferro_utils.sh
#      cf_ls 2020/32MRE
#      cf_du 2020
#      cf_verify_tile 32MRE 2020
#
#  Configure via env vars (sensible defaults):
#      CF_ENDPOINT   default https://s3.waw4-1.cloudferro.com
#      CF_CREDS      default ~/.config/s5cmd/s5cmd.cfg
#      CF_BUCKET     default vsm-data-public
#      CF_LOCAL_BASE default $HOME/data/gvs/products/vsm   (for diff helpers)
# =============================================================================

CF_ENDPOINT=${CF_ENDPOINT:-https://s3.waw4-1.cloudferro.com}
CF_CREDS=${CF_CREDS:-${HOME}/.config/s5cmd/s5cmd.cfg}
CF_BUCKET=${CF_BUCKET:-vsm-data-public}
CF_LOCAL_BASE=${CF_LOCAL_BASE:-${HOME}/data/gvs/products/vsm}

# ---- internals ------------------------------------------------------------

_cf_s5() {
    # Run s5cmd with the configured endpoint + credentials.
    s5cmd --credentials-file "${CF_CREDS}" --endpoint-url "${CF_ENDPOINT}" "$@"
}

_cf_url() {
    # Resolve a key (or full s3:// URL) to s3://bucket/key.
    local key=$1
    case "${key}" in
        s3://*) echo "${key}" ;;
        *)      echo "s3://${CF_BUCKET}/${key#/}" ;;
    esac
}

_cf_http() {
    # Public HTTPS URL for the same key.
    local key=$1
    case "${key}" in
        s3://*) key=${key#s3://*/} ;;
    esac
    echo "${CF_ENDPOINT}/${CF_BUCKET}/${key#/}"
}

# ---- list / size ----------------------------------------------------------

cf_ls() {
    # List objects under a prefix. Pass "" to list the bucket root.
    local key=${1:-}
    local url=$(_cf_url "${key}")
    _cf_s5 ls "${url%/}/*"
}

cf_du() {
    # Total bytes + object count under a prefix.
    local key=${1:-}
    local url=$(_cf_url "${key}")
    _cf_s5 du "${url%/}/*"
}

cf_stat() {
    # Metadata (size, etag, content-type) for one object.
    local key=${1:?Usage: cf_stat <key>}
    local url=$(_cf_url "${key}")
    _cf_s5 cat --print-stat "${url}" >/dev/null
}

# ---- verify ---------------------------------------------------------------

cf_verify_tile() {
    # Compare local tile dir vs CloudFerro prefix on (file count, total bytes).
    # Usage: cf_verify_tile <TILE_ID> [YEAR] [LOCAL_DIR]
    local tile=${1:?Usage: cf_verify_tile <TILE_ID> [YEAR] [LOCAL_DIR]}
    local year=${2:-2024}
    local local_dir=${3:-${CF_LOCAL_BASE}/${year}/blended/tiles/cog/${tile}}
    local url="s3://${CF_BUCKET}/${year}/${tile}"

    if [ ! -d "${local_dir}" ]; then
        echo "ERROR: local dir does not exist: ${local_dir}"
        return 1
    fi

    local src_bytes src_count dst_bytes dst_count
    src_bytes=$(find "${local_dir}" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
    src_count=$(find "${local_dir}" -type f | wc -l | tr -d ' ')
    dst_bytes=$(_cf_s5 du "${url}/*" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    dst_count=$(_cf_s5 ls "${url}/*" 2>/dev/null | wc -l | tr -d ' ')

    echo "local : ${src_count} files, ${src_bytes} bytes   (${local_dir})"
    echo "remote: ${dst_count:-0} files, ${dst_bytes:-0} bytes   (${url})"
    if [ "${dst_bytes:-0}" = "${src_bytes}" ] && [ "${dst_count:-0}" = "${src_count}" ]; then
        echo "OK: counts + bytes match"
        return 0
    else
        echo "MISMATCH"
        return 1
    fi
}

cf_diff_file() {
    # Download one object to a temp file and byte-compare with a local file.
    # Usage: cf_diff_file <remote_key> <local_path>
    local key=${1:?Usage: cf_diff_file <remote_key> <local_path>}
    local local_path=${2:?Usage: cf_diff_file <remote_key> <local_path>}
    local url=$(_cf_url "${key}")
    local tmp=$(mktemp)
    _cf_s5 cp "${url}" "${tmp}" >/dev/null
    if cmp -s "${tmp}" "${local_path}"; then
        echo "OK: ${url} matches ${local_path}"
        rm -f "${tmp}"
        return 0
    else
        echo "DIFFERS: ${url} vs ${local_path}"
        ls -l "${tmp}" "${local_path}"
        rm -f "${tmp}"
        return 1
    fi
}

# ---- public-URL helpers (no creds needed) ---------------------------------

cf_url() {
    # Print the public HTTPS URL for a key.
    _cf_http "${1:?Usage: cf_url <key>}"
}

cf_head() {
    # HEAD request via public HTTPS (no creds). Confirms reachability + headers.
    curl -sI "$(_cf_http "${1:?Usage: cf_head <key>}")"
}

cf_xml_ls() {
    # Public XML listing of a prefix (no creds). Useful for sanity-checking ACLs.
    local prefix=${1:-}
    curl -s "${CF_ENDPOINT}/${CF_BUCKET}/?prefix=${prefix}" | xmllint --format - 2>/dev/null || \
    curl -s "${CF_ENDPOINT}/${CF_BUCKET}/?prefix=${prefix}"
}

cf_gdalinfo() {
    # `gdalinfo` on a remote COG via /vsicurl/. Confirms the file is reachable
    # and well-formed without downloading it.
    local key=${1:?Usage: cf_gdalinfo <key>}
    gdalinfo "/vsicurl/$(_cf_http "${key}")"
}
