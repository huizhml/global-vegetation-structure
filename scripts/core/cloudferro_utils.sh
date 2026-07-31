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
# CloudFerro (Ceph RGW) rejects s5cmd's GetBucketLocation region probe with a
# 403; pinning the region makes s5cmd skip that probe. The request-id suffix
# ("...-default") shows the endpoint's region is "default".
CF_REGION=${CF_REGION:-waw4-1}

# ---- internals ------------------------------------------------------------

_cf_s5() {
    # Run s5cmd with the configured endpoint + credentials + region.
    AWS_REGION="${CF_REGION}" \
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

cf_overview() {
    # Summarize what lives on CloudFerro, grouped one level down, with a
    # per-group file count + total size and a grand total. Drills down as you
    # pass more of the path:
    #     cf_overview              -> per-year   totals (bucket root)
    #     cf_overview 2024         -> per-tile   totals under 2024/
    #     cf_overview 2024 32MRE   -> per-RH     totals under 2024/32MRE/
    local year=${1:-}
    local tile=${2:-}

    local prefix mode
    if   [ -z "${year}" ]; then prefix="s3://${CF_BUCKET}";              mode="year"
    elif [ -z "${tile}" ]; then prefix="s3://${CF_BUCKET}/${year}";      mode="tile"
    else                        prefix="s3://${CF_BUCKET}/${year}/${tile}"; mode="rh"
    fi

    echo "# ${prefix}/*   (grouped by ${mode})"
    # s5cmd ls output for objects: "DATE TIME SIZE KEY". The wildcard recurses,
    # so KEY is the full path within the bucket. DIR lines have no numeric size.
    _cf_s5 ls "${prefix}/*" 2>/dev/null | awk -v mode="${mode}" '
        {
            size = $(NF-1); key = $NF
            if (size !~ /^[0-9]+$/) next          # skip DIR / malformed lines
            n = split(key, seg, "/")
            if      (mode == "year") g = seg[1]
            else if (mode == "tile") g = seg[2]
            else {                                 # rh: "RH<num>" from filename
                if (match(seg[n], /^RH[0-9]+/)) g = substr(seg[n], RSTART, RLENGTH)
                else                            g = seg[n]
            }
            if (g == "") next
            cnt[g]++; bytes[g] += size; tot++; totb += size
        }
        END {
            for (g in cnt)
                printf "%-16s %8d files  %12d B  %8.2f GiB\n",
                       g, cnt[g], bytes[g], bytes[g]/1073741824 | "sort"
            close("sort")
            if (tot == 0) { print "(no objects found)"; exit }
            printf "%-16s %8d files  %12d B  %8.2f GiB\n",
                   "TOTAL", tot, totb, totb/1073741824
        }
    '
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
