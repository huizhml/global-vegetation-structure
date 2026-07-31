#!/bin/bash
# =============================================================================
#  Source Cooperative (source.coop) upload helpers — lumi-o -> source.coop.
#
#  Counterpart of cloudferro_utils.sh. Source data lives on LUMI object storage
#  (lumi-o); the destination is Source Cooperative. They are two different S3
#  services with different credentials, and — critically — source.coop's gateway
#  REJECTS rclone's requests (endpoint parsing, HEAD 403, SignatureDoesNotMatch,
#  and unsigned x-amz-meta-mtime headers, in turn), while the AWS CLI works and
#  is source.coop's officially recommended tool. So the transport is TWO-HOP:
#
#      rclone copy   lumi-o tile  ->  local scratch      (rclone reads lumi-o fine)
#      aws s3 sync   scratch      ->  source.coop        (aws cli writes fine)
#      rm -rf        scratch
#
#  A tile is ~75 MB, so staging is trivial (one tile on disk at a time).
#  `aws s3 sync` is resumable (skips already-uploaded).
#
#  CREDENTIALS DO NOT AUTO-REFRESH. The profile's credential_process runs
#  `source-coop creds`, which only PRINTS the token cached by the last login —
#  it never mints a new one. (Proof: every `aws` invocation is a fresh process
#  and re-runs credential_process, yet `aws` still starts failing at expiry.)
#  Log in with `source-coop login --duration 12h` and the run must fit inside
#  that window; see ONE-TIME SETUP below.
#
#  Source this file, then call the sc_* functions:
#      source scripts/core/source_coop_utils.sh
#      sc_check_creds
#      sc_verify_tile 60UUD 2024
#
#  ---------------------------------------------------------------------------
#  ONE-TIME SETUP
#  ---------------------------------------------------------------------------
#    0. source-coop CLI. The prebuilt Linux binaries need glibc >= 2.35, so on
#       an older host (hendrix is RHEL 8 / glibc 2.28) the installer refuses and
#       you build it: rustup + `cargo install --path .` from the release
#       source.tar.gz, with the Linux `keyring` backend dropped from Cargo.toml
#       (no libdbus headers) AND cache.rs forced to the file cache — with no
#       keyring backend the crate defaults to an in-memory MOCK store whose
#       writes silently succeed, so the token would not survive the process.
#    1. source-coop login --duration 12h   # 12 h is the max; the token is
#                                          # cached under ~/.cache/source-coop/
#       ON A LOGIN NODE, PASS -v. login serves the OAuth callback on
#       127.0.0.1:<port> and shells out to a browser; it only prints the URL if
#       that shell-out FAILS. xdg-open exists on the cluster and exits 0 without
#       doing anything, so plain `login` just hangs with no link. `-v` prints the
#       authorization URL unconditionally, before the browser attempt:
#           ssh -L 8765:127.0.0.1:8765 hendrix2      # NOT the round-robin alias
#           source-coop login -v --port 8765 --duration 12h
#       then open the printed URL in the local browser. The tunnel must reach the
#       SAME node the login runs on — a second ssh through a round-robin gate
#       alias can land elsewhere and the callback is delivered to the wrong host.
#    2. ~/.aws/config profile (once):
#         [profile source-coop]
#         credential_process = source-coop creds
#         endpoint_url       = https://data.source.coop
#         region             = us-west-2
#       endpoint_url is REQUIRED — without it the aws CLI talks to real AWS S3
#       and returns InvalidAccessKeyId, which looks like a credential problem
#       but isn't. (Profile-level endpoint_url needs aws-cli >= 2.13; on older
#       versions pass --endpoint-url, or go through _sc_aws below.)
#    3. lumio-conf / rclone config         # ensure the lumi-o rclone remote works
#
#  The token is valid for at most 12 h FROM LOGIN — not from job start. A SLURM
#  job that queues for 10 h has 2 h left when it finally runs, so log in right
#  before sbatch, not the night before. Past expiry every write 403s; re-run
#  `source-coop login --duration 12h` and restart — the uploads are resumable.
#  How much is left:
#      source-coop creds | grep Expiration
#  The Python path checks this before it starts moving bytes; see
#  deploy/source_coop/upload.py (min_cred_minutes).
#
#  ---------------------------------------------------------------------------
#  LAYOUT
#  ---------------------------------------------------------------------------
#  Source on lumi-o — ONE bucket per UTM zone-band + year, tiles directly under
#  it, RH COGs directly under each tile:
#      ${SC_SRC_REMOTE}:<zoneband>-<year>/<tile>/RH{n}_Q{q}.tif
#      e.g. lumi-465002698-private:60u-2024/60UUD/RH98_Q50.tif
#  The <zoneband> is DERIVED from the tile id (leading digits + band letter,
#  lowercased): 60UUD -> 60u, 32MRE -> 32m.
#
#  Destination on source.coop — organised by version / year / tile:
#      ${SC_DEST_ROOT}/<year>/<tile>/RH{n}_Q{q}.tif
#
#  ---------------------------------------------------------------------------
#  Configure via env vars (sensible defaults):
#      SC_RCLONE      rclone binary for reading lumi-o     default rclone
#      SC_AWS         aws cli binary for writing source.coop  default aws
#      SC_AWS_PROFILE aws profile for source.coop          default source-coop
#      SC_SRC_REMOTE  lumi-o rclone remote (NO bucket)     default lumi-465002698-private
#      SC_DEST_ROOT   source.coop product root             default s3://geoai-ucph/gvsm/
#      SC_ENDPOINT    source.coop endpoint                 default https://data.source.coop
#      SC_REGION      default us-west-2
#      SC_STAGE_DIR   local scratch for staging            default /tmp/sc_stage
# =============================================================================

SC_RCLONE=${SC_RCLONE:-rclone}
SC_AWS=${SC_AWS:-aws}
SC_AWS_PROFILE=${SC_AWS_PROFILE:-source-coop}
SC_SRC_REMOTE=${SC_SRC_REMOTE:-lumi-465002698-private}
SC_DEST_ROOT=${SC_DEST_ROOT:-s3://geoai-ucph/gvsm/}
SC_ENDPOINT=${SC_ENDPOINT:-https://data.source.coop}
SC_REGION=${SC_REGION:-us-west-2}
SC_STAGE_DIR=${SC_STAGE_DIR:-/tmp/sc_stage}

# ---- config guards --------------------------------------------------------

_sc_check_src_remote() {
    case "${SC_SRC_REMOTE}" in
        *REPLACE-ME*|"")
            echo "ERROR: SC_SRC_REMOTE is not configured." >&2
            echo "  export SC_SRC_REMOTE=<rclone-remote>   (the lumi-o remote, no bucket)" >&2
            return 1 ;;
    esac
    return 0
}

_sc_check_dest_root() {
    case "${SC_DEST_ROOT}" in
        *REPLACE-ME*|""|s3://)
            echo "ERROR: SC_DEST_ROOT is not configured." >&2
            echo "  export SC_DEST_ROOT=s3://<account>/<repository>   (from your product's Upload Data page)" >&2
            return 1 ;;
    esac
    return 0
}

# ---- source (lumi-o) path builders ----------------------------------------

_sc_src_zoneband() {
    # Derive the lumi-o zone-band bucket prefix from a tile id.
    #   60UUD -> 60u   32MRE -> 32m   (leading digits + one band letter, lower)
    echo "$1" | grep -oE '^[0-9]+[A-Za-z]' | tr 'A-Z' 'a-z'
}

_sc_src_bucket() {
    # lumi-o bucket for a tile + year:  60UUD 2024 -> 60u-2024
    local tile=$1 year=$2
    echo "$(_sc_src_zoneband "${tile}")-${year}"
}

_sc_src_tile() {
    # Full rclone source path for a tile:
    #   60UUD 2024 -> lumi-465002698-private:60u-2024/60UUD
    local tile=$1 year=$2
    echo "${SC_SRC_REMOTE}:$(_sc_src_bucket "${tile}" "${year}")/${tile}"
}

# ---- destination (source.coop) path builders ------------------------------

_sc_dest_bucket() {
    # Just the bucket component of SC_DEST_ROOT (for the auth probe).
    local bp=${SC_DEST_ROOT#s3://}
    echo "s3://${bp%%/*}"
}

_sc_dest_key() {
    # Destination key under the product root for a year/tile.
    #   60UUD 2024 -> 2024/60UUD
    local tile=$1 year=$2
    echo "${year}/${tile}"
}

_sc_dest_url() {
    # Full s3:// destination URL for a tile (no trailing slash).
    local tile=$1 year=$2
    echo "${SC_DEST_ROOT%/}/$(_sc_dest_key "${tile}" "${year}")"
}

# ---- aws cli wrapper (source.coop side) -----------------------------------

_sc_aws() {
    # Run the AWS CLI against source.coop with the configured profile/endpoint.
    local ep=()
    [ -n "${SC_ENDPOINT}" ] && ep=(--endpoint-url "${SC_ENDPOINT}")
    "${SC_AWS}" --profile "${SC_AWS_PROFILE}" --region "${SC_REGION}" "${ep[@]}" "$@"
}

# ---- stats helpers --------------------------------------------------------

_sc_glob_awk() {
    # Shared awk program body: sums bytes+count of `aws s3 ls --recursive`
    # ("date time size key") lines whose basename matches the glob `pat`.
    cat <<'AWK'
    function glob2re(g,   r,i,c) {
        r=""; for (i=1;i<=length(g);i++){ c=substr(g,i,1)
            if (c=="*") r=r".*"; else if (c=="?") r=r".";
            else if (c ~ /[.^$+(){}|\[\]\\]/) r=r"\\"c; else r=r c }
        return "^" r "$" }
    BEGIN { re = glob2re(pat) }
    { size=$3; key=$4
      if (size !~ /^[0-9]+$/) next
      n=split(key, seg, "/"); base=seg[n]
      if (base ~ re) { c++; b+=size } }
    END { printf "%d %d\n", b+0, c+0 }
AWK
}

_sc_rclone_size() {
    # Print "<bytes> <count>" for an rclone (lumi-o) target, optional glob.
    local target=$1 include=${2:-}
    local args=(size --json --fast-list "${target}")
    [ -n "${include}" ] && args+=(--include "${include}")
    local out; out=$("${SC_RCLONE}" "${args[@]}" 2>/dev/null) || { echo "0 0"; return 0; }
    if command -v jq >/dev/null 2>&1; then
        printf '%s' "${out}" | jq -r '"\(.bytes) \(.count)"' 2>/dev/null || echo "0 0"
    else
        printf '%s' "${out}" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("bytes",0),d.get("count",0))' 2>/dev/null || echo "0 0"
    fi
}

_sc_remote_stats() {
    # Print "<bytes> <count>" for source.coop objects under a dest URL, matching
    # an optional filename glob. Empty/absent prefix -> "0 0".
    local url=$1 name=${2:-'*'}
    _sc_aws s3 ls "${url%/}/" --recursive 2>/dev/null \
        | awk -v pat="${name}" "$(_sc_glob_awk)"
}

_sc_local_stats() {
    # Print "<bytes> <count>" for staged local files in <dir> matching a glob.
    local dir=$1 name=${2:-'*'}
    local bytes count
    bytes=$(find "${dir}" -type f -name "${name}" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
    count=$(find "${dir}" -type f -name "${name}" 2>/dev/null | wc -l | tr -d ' ')
    echo "${bytes:-0} ${count:-0}"
}

# ---- credentials / reachability -------------------------------------------

sc_check_creds() {
    # Confirm the AWS profile can reach source.coop, and show token expiry.
    echo ">>> checking source.coop reachability (aws profile: ${SC_AWS_PROFILE})"
    _sc_check_dest_root || return 1

    if command -v source-coop >/dev/null 2>&1; then
        local exp
        exp=$(source-coop creds 2>/dev/null | grep -oE '"[Ee]xpiration"[^,]*' | grep -oE '[0-9T:.+-]{10,}' | head -1)
        [ -n "${exp}" ] && echo "    temp-cred Expiration: ${exp}"
    fi

    # Probe the BUCKET ROOT (lists cleanly); an empty product prefix can 404.
    local bucket; bucket=$(_sc_dest_bucket)
    if _sc_aws s3 ls "${bucket}" >/dev/null 2>&1; then
        echo "    OK: aws reached ${bucket}/ via profile ${SC_AWS_PROFILE}"
        return 0
    fi
    echo "    FAILED: aws could not list ${bucket}/" >&2
    echo "    -> re-run 'source-coop login', or check the '${SC_AWS_PROFILE}' aws profile." >&2
    return 1
}

# ---- source enumeration ---------------------------------------------------

sc_list_src_buckets() {
    # lumi-o buckets for a year:  <zoneband>-<year>  (e.g. 60u-2024, 32m-2024).
    _sc_check_src_remote || return 1
    local year=${1:-2024}
    "${SC_RCLONE}" lsd "${SC_SRC_REMOTE}:" 2>/dev/null \
        | awk '{print $NF}' | grep -E "^[0-9]+[a-z]-${year}$" | sort
}

sc_list_src_tiles() {
    # All tile ids present on lumi-o for a year (across every zone-band bucket).
    _sc_check_src_remote || return 1
    local year=${1:-2024}
    local bucket
    while IFS= read -r bucket; do
        [ -n "${bucket}" ] || continue
        "${SC_RCLONE}" lsf --dirs-only "${SC_SRC_REMOTE}:${bucket}" 2>/dev/null | sed 's:/*$::'
    done < <(sc_list_src_buckets "${year}")
}

# ---- verify ---------------------------------------------------------------

sc_verify_tile() {
    # Compare source (lumi-o) tile vs destination (source.coop) prefix on
    # (object count, total bytes). Usage: sc_verify_tile <TILE_ID> [YEAR]
    _sc_check_src_remote || return 1
    _sc_check_dest_root  || return 1
    local tile=${1:?Usage: sc_verify_tile <TILE_ID> [YEAR]}
    local year=${2:-2024}
    local src; src="$(_sc_src_tile "${tile}" "${year}")"
    local url; url="$(_sc_dest_url "${tile}" "${year}")"

    local s_bytes s_count d_bytes d_count
    read -r s_bytes s_count < <(_sc_rclone_size "${src}" '*.tif')
    read -r d_bytes d_count < <(_sc_remote_stats "${url}" '*.tif')

    echo "source: ${s_count} files, ${s_bytes} bytes   (${src})"
    echo "dest  : ${d_count} files, ${d_bytes} bytes   (${url})"
    if [ "${d_bytes}" = "${s_bytes}" ] && [ "${d_count}" = "${s_count}" ]; then
        echo "OK: counts + bytes match"
        return 0
    fi
    echo "MISMATCH"
    return 1
}
