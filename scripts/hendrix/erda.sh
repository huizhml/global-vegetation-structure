#!/bin/bash
# =============================================================================
#  ERDA SFTP I/O  —  copy (non-destructive) and delete/purge (destructive).
#
#  ERDA remote (must exist in `rclone config`):
#     [ucph-erda]
#     type = sftp
#     host = io.erda.dk
#     ...
#
#  Path convention (copy only — delete/purge take ERDA paths directly):
#     local  ${LOCAL_ROOT}/<rel>            (LOCAL_ROOT defaults to ${HOME}/data/gvs)
#     erda   ucph-erda:${ERDA_ROOT}/<rel>   (ERDA_ROOT defaults to GVS)
#
#  Anything under ~/data/gvs/ mirrors 1:1 under ucph-erda:GVS/. Paths outside
#  ${LOCAL_ROOT} fall back to ${ERDA_ROOT}/<basename> with a warning; pass an
#  explicit destination as the 2nd arg to override.
#
#  Subcommands:
#     # ─── non-destructive ────────────────────────────────────────────
#     copy         <LOCAL>     [ERDA_REL]   copy one file/dir up
#     copy-list    <FILE>                   copy each local path in FILE
#                                           (auto-derive ERDA destination)
#
#     # ─── DESTRUCTIVE — read the safety section below ────────────────
#     delete       <ERDA_PATH>              purge one path  (probes first;
#                                           per-path TSV log; resume-friendly)
#     delete-list  <FILE>                   iterate `delete` over FILE
#     purge        <ERDA_PATH>              one-shot recursive purge of a
#                                           parent tree. rclone owns the log
#                                           (`--log-file` INFO level);
#                                           tuned for long-running sbatch jobs.
#
#  All deletes use `rclone purge` internally — removes files AND the
#  containing directory. For deleting only matching files (keep dir),
#  use `rclone delete` directly.
#
#  Safety knobs (destructive subcommands only — env vars):
#     YES=1                Skip the interactive "type DELETE to confirm"
#                          prompt. Required for SLURM / non-tty execution.
#     DRY_RUN=1            Print what would happen, do nothing. Always honour
#                          this before destructive runs.
#     PREVIEW_N=5          How many head/tail paths to show in the confirm
#                          summary (default 5).
#
#  rclone tuning (env vars, apply to all subcommands):
#     REMOTE                  rclone remote name      (default ucph-erda)
#     TRANSFERS / CHECKERS    parallel ops            (defaults 8 / 8 —
#                             keeps total SSH sessions ≤ ERDA's ~10-session cap)
#     MULTI_THREAD_STREAMS    per-file streaming      (default 2; copy only)
#     RETRIES                 top-level retries       (default 3)
#     RETRIES_SLEEP           sleep between retries   (default 30s)
#     LOW_LEVEL_RETRIES       SFTP read/write retries (default 10; `purge` uses 15)
#     VERIFY=1|0              `rclone check --size-only` after copy (default 1)
#                             — ERDA SFTP disables md5/sha1, so size+mtime is the best
#     LOCAL_ROOT              local prefix to strip   (default ${HOME}/data/gvs)
#     ERDA_ROOT               erda prefix to prepend  (default GVS)
#     LOG_FILE                delete/delete-list per-path TSV log
#                             (default ./logs/erda_delete_<timestamp>.log)
#
#  Examples:
#     # ── Copy ───────────────────────────────────────────────────────
#     bash scripts/hendrix/erda.sh copy \
#         ~/data/gvs/products/diversity_indices/2020/masked/tiles/cog
#     # → ucph-erda:GVS/products/diversity_indices/2020/masked/tiles/cog/
#
#     bash scripts/hendrix/erda.sh copy-list paths.txt
#
#     # ── Delete (per-path, list-driven; resume-friendly) ────────────
#     # Always dry-run first:
#     DRY_RUN=1 bash scripts/hendrix/erda.sh delete-list paths.txt
#     # Real:
#     bash scripts/hendrix/erda.sh delete-list paths.txt
#     # Generate paths.txt example (per-tile raw predictions):
#     for t in $(cat tiles.txt); do
#         echo "GVS/predictions_2020/${t}"
#     done > paths.txt
#
#     # ── Purge (one big tree, sbatch-friendly) ──────────────────────
#     DRY_RUN=1 bash scripts/hendrix/erda.sh purge GVS/predictions_2020
#     sbatch --time=7-00:00:00 --mem=4G --cpus-per-task=2 --partition=ml4good \
#            --job-name=erda_purge --output=./logs/%x_%j.out \
#            --wrap="YES=1 bash scripts/hendrix/erda.sh purge \
#                    GVS/predictions_2020"
#
#  Safe to `source`: the CLI dispatcher at the bottom only runs when this file
#  is executed directly. Other scripts can `source` it to reuse the functions:
#     rclone_copy_to_erda / rclone_copy_to_erda_list /
#     rclone_delete_on_erda / rclone_delete_on_erda_list /
#     rclone_purge_dir_on_erda
# =============================================================================
set -uo pipefail

# ----------------------------- shared config --------------------------------
REMOTE=${REMOTE:-ucph-erda}
LOCAL_ROOT=${LOCAL_ROOT:-${HOME}/data/gvs}
ERDA_ROOT=${ERDA_ROOT:-GVS}

TRANSFERS=${TRANSFERS:-8}
CHECKERS=${CHECKERS:-8}
MULTI_THREAD_STREAMS=${MULTI_THREAD_STREAMS:-2}
RETRIES=${RETRIES:-3}
RETRIES_SLEEP=${RETRIES_SLEEP:-30s}
LOW_LEVEL_RETRIES=${LOW_LEVEL_RETRIES:-10}
VERIFY=${VERIFY:-1}
PREVIEW_N=${PREVIEW_N:-5}
LOG_FILE=${LOG_FILE:-./logs/erda_delete_$(date +%Y%m%d_%H%M%S).log}

# Hendrix module env; harmless on local dev where `module` isn't a thing.
module load rclone 2>/dev/null || true

# ----------------------------- helpers --------------------------------------
_require_rclone() {
    command -v rclone >/dev/null \
        || { echo "ERROR: rclone not found (try: module load rclone)" >&2; return 1; }
}

# Auto-derive an ERDA destination path from a local path:
#   ${LOCAL_ROOT}/foo/bar   ->   ${ERDA_ROOT}/foo/bar
# Paths outside ${LOCAL_ROOT} fall back to ${ERDA_ROOT}/<basename>.
_erda_dest_for() {
    local local_path=$1
    local abs
    abs=$(realpath -m "${local_path}")
    local prefix="${LOCAL_ROOT%/}/"
    case "${abs}" in
        ${prefix}*) echo "${ERDA_ROOT}/${abs#${prefix}}" ;;
        *)
            echo "WARN: ${local_path} is outside ${LOCAL_ROOT}; using basename" >&2
            echo "${ERDA_ROOT}/$(basename "${abs}")" ;;
    esac
}

# Gate for destructive subcommands. Honours DRY_RUN / YES / tty / typed
# "DELETE" confirmation. Returns 1 on abort, 0 to proceed.
_confirm_destructive() {
    local what=$1 count=$2
    shift 2
    local previews=("$@")

    echo "=================================================================="
    echo "  ABOUT TO PURGE  ${count} ${what} ON ucph-erda (${REMOTE})"
    echo "=================================================================="
    if [ "${count}" -gt 0 ]; then
        echo "  preview (first / last ${PREVIEW_N}):"
        for p in "${previews[@]}"; do
            echo "    ${REMOTE}:${p}"
        done
    fi
    echo "------------------------------------------------------------------"
    echo "  log    : ${LOG_FILE}"
    echo "  DRY_RUN: ${DRY_RUN:-0}"
    echo "  YES    : ${YES:-0}"
    echo "------------------------------------------------------------------"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN=1 set — no deletes will be executed."
        return 0
    fi

    if [ "${YES:-0}" = "1" ]; then
        echo "YES=1 set — proceeding without interactive confirm."
        return 0
    fi

    if [ ! -t 0 ]; then
        echo "ERROR: stdin is not a tty and YES=1 is not set. Refusing to delete." >&2
        return 1
    fi

    # Belt-and-braces: word "DELETE" (not y/yes) to slow muscle memory.
    read -r -p "Type DELETE (all caps) to proceed: " reply
    if [ "${reply}" != "DELETE" ]; then
        echo "Aborted." >&2
        return 1
    fi
}

# ============================================================================
#  non-destructive — copy
# ============================================================================

# Usage: rclone_copy_to_erda <LOCAL> [ERDA_REL]
#   LOCAL    file or directory on hendrix
#   ERDA_REL path under the ERDA remote root (no leading slash); auto-derived if omitted
rclone_copy_to_erda() {
    local local_path=${1:?Usage: rclone_copy_to_erda <LOCAL> [ERDA_REL]}
    local erda_rel=${2:-}
    _require_rclone || return 1

    [ -e "${local_path}" ] || { echo "ERROR: ${local_path} not found" >&2; return 1; }

    if [ -z "${erda_rel}" ]; then
        erda_rel=$(_erda_dest_for "${local_path}")
    fi
    local dest="${REMOTE}:${erda_rel}"

    echo "[$(basename "${local_path}")] ${local_path}  ->  ${dest}"
    local start end elapsed
    start=$(date +%s)
    rclone copy "${local_path}" "${dest}" \
        --transfers="${TRANSFERS}" --checkers="${CHECKERS}" \
        --multi-thread-streams="${MULTI_THREAD_STREAMS}" \
        --retries="${RETRIES}" --retries-sleep="${RETRIES_SLEEP}" \
        --low-level-retries="${LOW_LEVEL_RETRIES}" \
        --stats=30s --stats-one-line
    local rc=$?
    end=$(date +%s)
    elapsed=$((end - start))

    if [ "${rc}" -ne 0 ]; then
        echo "[$(basename "${local_path}")] rclone copy failed (rc=${rc}) after ${elapsed}s" >&2
        return "${rc}"
    fi

    if [ "${VERIFY}" = "1" ]; then
        # ERDA SFTP has md5/sha1 disabled; --size-only is the best available.
        if rclone check "${local_path}" "${dest}" --size-only --one-way >/dev/null 2>&1; then
            echo "[$(basename "${local_path}")] OK  (${elapsed}s)"
        else
            echo "[$(basename "${local_path}")] size mismatch after copy (${elapsed}s) — rerun" >&2
            return 2
        fi
    else
        echo "[$(basename "${local_path}")] copied in ${elapsed}s (verify skipped)"
    fi
}

# Usage: rclone_copy_to_erda_list <FILE>
# FILE: one local path per line; '#' and blank lines ignored.
rclone_copy_to_erda_list() {
    local list_file=${1:?Usage: rclone_copy_to_erda_list <FILE>}
    [ -r "${list_file}" ] || { echo "ERROR: cannot read ${list_file}" >&2; return 1; }

    local total ok fail line idx=0
    total=$(grep -cvE '^\s*(#|$)' "${list_file}" || true)
    ok=0
    fail=0
    while IFS= read -r line; do
        [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
        line=${line#"${line%%[![:space:]]*}"}   # ltrim
        line=${line%"${line##*[![:space:]]}"}   # rtrim
        idx=$((idx + 1))
        echo "------------------------------------------------------------------"
        echo "[${idx}/${total}] ${line}"
        if rclone_copy_to_erda "${line}"; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi
    done < "${list_file}"
    echo "=================================================================="
    echo "Done. ok=${ok} fail=${fail} total=${total}"
    [ "${fail}" -eq 0 ]
}

# ============================================================================
#  DESTRUCTIVE — delete (per-path) and purge (whole tree)
# ============================================================================

# Usage: rclone_delete_on_erda <ERDA_PATH>
#   ERDA_PATH: path under the remote root, no leading slash, e.g. "GVS/foo/bar"
#
# Uses `rclone purge` (removes files AND the containing directory). Wraps in
# an lsf existence probe so missing paths log "skip" rather than burning the
# rclone retry budget on a known-absent target.
rclone_delete_on_erda() {
    local erda_path=${1:?Usage: rclone_delete_on_erda <ERDA_PATH>}
    _require_rclone || return 1

    local dest="${REMOTE}:${erda_path}"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN: rclone purge ${dest}"
        return 0
    fi

    # Existence check — `rclone purge` on a missing path returns rc=3
    # ("directory not found"); `lsf --max-depth 1` is much cheaper than
    # cycling purge through its full retry budget.
    if ! rclone lsf --max-depth 1 "${dest}" >/dev/null 2>&1; then
        printf '%s\t%s\t%s\n' "$(date -Iseconds)" "skip"  "${erda_path}" >> "${LOG_FILE}"
        echo "[skip ] ${dest}  (not found)"
        return 0
    fi

    local start end elapsed
    start=$(date +%s)
    rclone purge "${dest}" \
        --transfers="${TRANSFERS}" \
        --retries="${RETRIES}" --retries-sleep="${RETRIES_SLEEP}" \
        --low-level-retries="${LOW_LEVEL_RETRIES}" \
        --stats=30s --stats-one-line
    local rc=$?
    end=$(date +%s)
    elapsed=$((end - start))

    if [ "${rc}" -eq 0 ]; then
        printf '%s\t%s\t%s\t%ss\n' "$(date -Iseconds)" "ok"   "${erda_path}" "${elapsed}" >> "${LOG_FILE}"
        echo "[ok   ] ${dest}  (${elapsed}s)"
        return 0
    else
        printf '%s\t%s\t%s\t%ss\trc=%s\n' "$(date -Iseconds)" "fail" "${erda_path}" "${elapsed}" "${rc}" >> "${LOG_FILE}"
        echo "[FAIL ] ${dest}  (${elapsed}s, rc=${rc})" >&2
        return "${rc}"
    fi
}

# Usage: rclone_delete_on_erda_list <FILE>
# FILE: one ERDA path per line; '#' / blank lines skipped.
rclone_delete_on_erda_list() {
    local list_file=${1:?Usage: rclone_delete_on_erda_list <FILE>}
    [ -r "${list_file}" ] || { echo "ERROR: cannot read ${list_file}" >&2; return 1; }

    # Load and clean the path list once.
    local paths=()
    local line
    while IFS= read -r line; do
        [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
        line=${line#"${line%%[![:space:]]*}"}   # ltrim
        line=${line%"${line##*[![:space:]]}"}   # rtrim
        paths+=("${line}")
    done < "${list_file}"
    local n=${#paths[@]}

    # head/tail preview for the confirm prompt.
    local previews=()
    local h=$(( n < PREVIEW_N ? n : PREVIEW_N ))
    local i
    for (( i=0; i<h; i++ )); do previews+=("${paths[$i]}"); done
    if [ "${n}" -gt "$((2 * PREVIEW_N))" ]; then
        previews+=("  ... ($((n - 2 * PREVIEW_N)) more) ...")
        for (( i=n-PREVIEW_N; i<n; i++ )); do previews+=("${paths[$i]}"); done
    elif [ "${n}" -gt "${PREVIEW_N}" ]; then
        for (( i=PREVIEW_N; i<n; i++ )); do previews+=("${paths[$i]}"); done
    fi

    mkdir -p "$(dirname "${LOG_FILE}")"
    _confirm_destructive "paths" "${n}" "${previews[@]}" || return 1

    local ok=0 fail=0 skip=0 idx=0 t0
    t0=$(date +%s)
    for p in "${paths[@]}"; do
        idx=$((idx + 1))
        echo "------------------------------------------------------------------"
        printf "[%d/%d] " "${idx}" "${n}"
        if rclone_delete_on_erda "${p}"; then
            # Distinguish ok vs skip by tailing the log we just wrote.
            local last
            last=$(tail -n 1 "${LOG_FILE}" | awk '{print $2}')
            if [ "${last}" = "skip" ]; then
                skip=$((skip + 1))
            else
                ok=$((ok + 1))
            fi
        else
            fail=$((fail + 1))
        fi

        # Cheap ETA every 50 items.
        if [ $((idx % 50)) -eq 0 ]; then
            local now elapsed rate eta
            now=$(date +%s)
            elapsed=$((now - t0))
            rate=$(awk "BEGIN{printf \"%.2f\", ${idx}/${elapsed}}")
            eta=$(awk "BEGIN{printf \"%.0f\", (${n}-${idx})/(${idx}/${elapsed})/60}")
            echo "  progress: ${idx}/${n} | ${rate} purges/s | ETA ${eta} min | ok=${ok} skip=${skip} fail=${fail}"
        fi
    done

    echo "=================================================================="
    echo "Done. ok=${ok} skip=${skip} fail=${fail} total=${n}"
    echo "Log : ${LOG_FILE}"
    [ "${fail}" -eq 0 ]
}

# Single-shot recursive purge tuned for "this whole tree goes away" jobs run
# under sbatch. Differences vs rclone_delete_on_erda:
#   - No pre-probe: you should know what you're nuking when you call this.
#   - Lets rclone own the audit log via `--log-file` so we capture every
#     file event, not just per-path summaries.
#   - Less chatty stats (every 5 min) — sbatch logs grow forever otherwise.
#   - Higher default LOW_LEVEL_RETRIES (15) because we're running for days
#     and want to survive a few more ERDA hiccups without giving up.
#
# Usage: rclone_purge_dir_on_erda <ERDA_PATH>
rclone_purge_dir_on_erda() {
    local erda_path=${1:?Usage: rclone_purge_dir_on_erda <ERDA_PATH>}
    _require_rclone || return 1

    local dest="${REMOTE}:${erda_path}"
    # Sanitize path for use in a filename (slashes → underscores).
    local slug=${erda_path//\//_}
    slug=${slug##_}
    local rclone_log="./logs/erda_purge_${slug}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "${rclone_log}")"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN: rclone purge ${dest} (log would go to ${rclone_log})"
        echo "DRY_RUN: peek of what would be purged:"
        rclone purge "${dest}" --dry-run 2>&1 | head -20
        return 0
    fi

    echo "Purging  : ${dest}"
    echo "Log file : ${rclone_log}"
    echo "Started  : $(date -Iseconds)"

    local start end elapsed
    start=$(date +%s)
    rclone purge "${dest}" \
        --transfers="${TRANSFERS}" \
        --retries="${RETRIES}" --retries-sleep="${RETRIES_SLEEP}" \
        --low-level-retries=15 \
        --stats=300s --stats-one-line \
        --log-file="${rclone_log}" --log-level INFO
    local rc=$?
    end=$(date +%s)
    elapsed=$((end - start))
    local h=$((elapsed / 3600)) m=$(((elapsed % 3600) / 60))

    echo "Finished : $(date -Iseconds)"
    echo "Duration : ${h}h ${m}m  (rc=${rc})"
    echo "Log file : ${rclone_log}"
    return "${rc}"
}

# ----------------------------- CLI dispatcher -------------------------------
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    cmd=${1:-}
    case "${cmd}" in
        copy)
            shift; rclone_copy_to_erda "$@" ;;
        copy-list)
            shift; rclone_copy_to_erda_list "$@" ;;
        delete)
            shift
            erda_path=${1:?Usage: $0 delete <ERDA_PATH>}
            mkdir -p "$(dirname "${LOG_FILE}")"
            _confirm_destructive "path" 1 "${erda_path}" || exit 1
            rclone_delete_on_erda "${erda_path}"
            ;;
        delete-list)
            shift
            rclone_delete_on_erda_list "$@"
            ;;
        purge)
            shift
            erda_path=${1:?Usage: $0 purge <ERDA_PATH>}
            _confirm_destructive "tree under" 1 "${erda_path}/  (recursive)" || exit 1
            rclone_purge_dir_on_erda "${erda_path}"
            ;;
        ""|-h|--help)
            sed -n '2,90p' "${BASH_SOURCE[0]}"
            ;;
        *)
            echo "ERROR: unknown subcommand '${cmd}'." >&2
            echo "Try: copy | copy-list | delete | delete-list | purge" >&2
            exit 1
            ;;
    esac
fi
