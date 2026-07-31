#!/bin/bash
# =============================================================================
#  AWS CLI v2 installer — the tool that writes to source.coop / CloudFerro.
#
#  Two hosts, same bundle, different prefix: LUMI installs into the shared
#  project dir so every node and every job sees the same binary; hendrix has no
#  such shared dir, so it goes under $HOME/.local (already on PATH via .bashrc).
#
#  Source this file and call a function, or run it directly:
#      bash scripts/core/install_aws.sh lumi
#      bash scripts/core/install_aws.sh hendrix
#
#  Both are idempotent — re-running upgrades in place (--update).
#
#  The official bundle ships its own Python and runs on RHEL 8 / glibc 2.28
#  (verified: aws-cli/2.36.13 ... exe/x86_64.rhel.8), so no version pinning is
#  needed. This is NOT the case for the source-coop CLI, whose prebuilt binaries
#  need glibc >= 2.35 and must be built from source on hendrix — see the
#  ONE-TIME SETUP header in source_coop_utils.sh.
#
#  aws-cli >= 2.13 matters: profile-level `endpoint_url` (used by the
#  source-coop profile) is ignored by older versions.
# =============================================================================

AWS_LUMI_PROJECT=${AWS_LUMI_PROJECT:-/project/project_465002698}

# ---- shared bundle install -------------------------------------------------

_aws_install_bundle() {
    # Download the official AWS CLI v2 bundle and install it into
    # <install_dir> with launchers in <bin_dir>. Upgrades an existing install.
    local install_dir=$1 bin_dir=$2
    local tmp cmd

    for cmd in curl unzip; do
        command -v "${cmd}" >/dev/null 2>&1 || {
            echo "ERROR: ${cmd} is required but not on PATH" >&2; return 1; }
    done

    mkdir -p "${bin_dir}" || return 1
    tmp=$(mktemp -d /tmp/awscli.XXXXXX) || return 1

    echo ">>> downloading AWS CLI v2 bundle"
    if ! curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
            -o "${tmp}/awscliv2.zip"; then
        echo "ERROR: download failed" >&2
        rm -rf "${tmp}"
        return 1
    fi

    unzip -q "${tmp}/awscliv2.zip" -d "${tmp}" || { rm -rf "${tmp}"; return 1; }

    # The installer refuses to write over an existing install unless told to.
    local update=()
    [ -d "${install_dir}/v2" ] && update=(--update)

    echo ">>> installing to ${install_dir} (bin: ${bin_dir})"
    "${tmp}/aws/install" --install-dir "${install_dir}" --bin-dir "${bin_dir}" \
        "${update[@]}" || { rm -rf "${tmp}"; return 1; }

    rm -rf "${tmp}"

    export PATH="${bin_dir}:${PATH}"
    "${bin_dir}/aws" --version
}

# ---- per-host entry points -------------------------------------------------

install_aws_on_lumi() {
    # LUMI: shared project software dir, visible from compute nodes.
    #   $PROJECT/software/aws-cli  +  $PROJECT/software/bin/aws
    local project=${1:-${AWS_LUMI_PROJECT}}

    if [ ! -d "${project}" ]; then
        echo "ERROR: project dir not found: ${project}" >&2
        echo "  pass it explicitly:  install_aws_on_lumi /project/project_XXXXXXXXX" >&2
        return 1
    fi

    _aws_install_bundle "${project}/software/aws-cli" "${project}/software/bin" || return 1

    echo
    echo "Add to your LUMI shell rc / job script:"
    echo "    export PATH=\"${project}/software/bin:\$PATH\""
}

install_aws_on_hendrix() {
    # hendrix (RHEL 8): no shared project dir, so install per-user under
    # $HOME/.local. ~/.local/bin is already prepended to PATH by the stock
    # .bashrc, so nothing to configure afterwards.
    local prefix=${1:-${HOME}/.local}

    _aws_install_bundle "${prefix}/aws-cli" "${prefix}/bin" || return 1

    echo
    case ":${PATH}:" in
        *":${prefix}/bin:"*) ;;
        *) echo "NOTE: ${prefix}/bin is not on PATH — add to ~/.bashrc:"
           echo "    export PATH=\"${prefix}/bin:\$PATH\"" ;;
    esac
}

# ---- direct invocation -----------------------------------------------------

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        lumi)    install_aws_on_lumi "${2:-}" ;;
        hendrix) install_aws_on_hendrix "${2:-}" ;;
        *)
            echo "Usage: bash $0 {lumi|hendrix} [prefix]" >&2
            echo "  lumi     -> \$PROJECT/software/{aws-cli,bin}   (default project: ${AWS_LUMI_PROJECT})" >&2
            echo "  hendrix  -> \$HOME/.local/{aws-cli,bin}" >&2
            exit 2 ;;
    esac
fi
