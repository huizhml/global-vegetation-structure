#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=11:00:00
#SBATCH --job-name=sc_upload
#SBATCH --output=./logs/%x-%A.out
#SBATCH --error=./logs/%x-%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
# =============================================================================
#  Publish the 2020 VSM COGs to Source Cooperative, one RH at a time.
#
#  SUBMIT LIKE THIS -- the order matters:
#      conda activate <env>
#      source-coop login --duration 12h     # browser flow, login node only
#      sbatch scripts/hendrix/run_source_coop.sh
#
#  The credential is what drives the shape of this job:
#    * `source-coop login` is an OAuth2 browser flow with a localhost callback,
#      so it CANNOT run on a compute node. The token it caches under
#      ~/.cache/source-coop is what the job picks up ($HOME is shared).
#    * That token lasts 12 h AT MOST, counted FROM LOGIN, not from job start.
#      A job that queues for 4 h has 8 h left when it finally runs. Hence
#      --time=11:00:00 and the pre-flight check below: log in immediately
#      before submitting, or the job wastes a slot and stops early.
#    * `source-coop creds` only prints the cached token, it never mints a new
#      one, so nothing renews mid-job.
#
#  Everything is resumable: each RH keeps its own <state_prefix>.done and
#  re-running skips what already landed. If the token runs out, sc_upload stops
#  submitting new objects, drains what is in flight and exits 0 -- log in again
#  and resubmit the same script; it picks up where it stopped.
#
#  Args:
#    $1  quantile        default 1        (the Q1-first priority pass)
#    $2  workers         default 128      (benchmarked knee; 64 is the safe fallback)
#    $3  RH list         default: the 15 key RHs, in publication order
#
#  Examples:
#      sbatch scripts/hendrix/run_source_coop.sh                 # 15 key RHs, Q1
#      sbatch scripts/hendrix/run_source_coop.sh 1 64            # gentler concurrency
#      sbatch scripts/hendrix/run_source_coop.sh 1 128 "98 100"  # just two RHs
# =============================================================================
set -u -o pipefail

QUANTILE=${1:-1}
WORKERS=${2:-128}
RHS=${3:-"0 10 20 25 30 40 50 60 70 75 80 90 95 98 100"}

CATALOG=${SC_CATALOG:-${HOME}/data/gvs/products/gvsm_stac_catalog/vsm_local_masked}
DST_PREFIX=${SC_DST_PREFIX:-gvsm/2020/}
STATE_DIR=${SC_STATE_DIR:-${HOME}/data/gvs/state/source_coop}
# Written by `python -m tools.run run=stac_cog_audit`. Only tiles whose COGs are
# complete may be published: an unconverted tile still has its item href on
# geotiff/, and uploading that ships the uncompressed original without erroring.
TILES=${SC_TILES:-${HOME}/data/gvs/assets/worklists/masked_2020_key_rhs_q1_have_cog.txt}
# Stop starting new RHs when the token has less than this left (minutes). One RH
# is ~35 min at 128 workers, so starting one with less than an hour left just
# means it gets interrupted partway.
MIN_MINUTES=${SC_MIN_MINUTES:-60}

echo "***************************** JOB INFO *****************************"
echo "Host: $HOSTNAME"
echo "Job Name: ${SLURM_JOB_NAME:-interactive}"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Quantile: Q${QUANTILE}   Workers: ${WORKERS}"
echo "RHs: ${RHS}"
echo "Catalog: ${CATALOG}"
echo "Destination: s3://geoai-ucph/${DST_PREFIX}"
echo "********************************************************************"

# SLURM inherits the submitting shell's env (--export=ALL), so submitting
# without an activated env leaves $CONDA_PREFIX empty and python resolves to the
# system interpreter.
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "CONDA_PREFIX is empty: activate the env before submitting" >&2
    exit 1
fi

if [ ! -f "${TILES}" ]; then
    echo "ERROR: tile worklist not found: ${TILES}
       run: python -m tools.run run=stac_cog_audit" >&2
    exit 1
fi
[ -d "${CATALOG}" ] || { echo "ERROR: STAC collection not found: ${CATALOG}" >&2; exit 1; }
echo ">>> $(wc -l < "${TILES}") tiles in the worklist"
mkdir -p "${STATE_DIR}"

# ---- credential pre-flight ------------------------------------------------

creds_minutes_left() {
    # Minutes until the cached source.coop token expires; empty if unavailable.
    # Pinned to the env's interpreter: hendrix's system python3 predates
    # datetime.fromisoformat, and a silent parse failure here reads as "no
    # credential" and aborts the whole job.
    source-coop creds 2>/dev/null | "${CONDA_PREFIX}/bin/python" -c '
import json, sys
from datetime import datetime, timezone
try:
    exp = json.load(sys.stdin)["Expiration"]
except Exception:
    sys.exit(1)
left = datetime.fromisoformat(exp) - datetime.now(timezone.utc)
print(int(left.total_seconds() // 60))
' 2>/dev/null
}

left=$(creds_minutes_left)
if [ -z "${left}" ]; then
    echo "ERROR: cannot read source.coop credentials.
       Run 'source-coop login --duration 12h' on the login node, then resubmit." >&2
    exit 1
fi
echo ">>> credential has ${left} minutes left"
if [ "${left}" -lt "${MIN_MINUTES}" ]; then
    echo "ERROR: only ${left} min left (need ${MIN_MINUTES}).
       Run 'source-coop login --duration 12h' and resubmit." >&2
    exit 1
fi

# ---- upload, one RH at a time ---------------------------------------------
# Sequential on purpose. What source.coop sees is TOTAL concurrent connections,
# not how many jobs they are spread over, so a SLURM array of RHs would just
# multiply the load on their gateway without going faster (measured: 2 procs x
# 64 workers performs the same as 1 x 128).

n_ok=0; n_skip=0; n_stopped=0; rc_all=0
expected=$(wc -l < "${TILES}")
for rh in ${RHS}; do
    # Skip an RH that is already fully published, without paying for a listing.
    # A single-RH glob yields exactly one object per tile, so a .done as long as
    # the worklist means there is nothing left. Re-listing instead would cost
    # ~13 min of STAC item reads per finished RH -- across an 86-RH rollout
    # spread over several credential windows that adds up to more time than the
    # transfers themselves.
    done_file="${STATE_DIR}/2020_RH${rh}_Q${QUANTILE}.done"
    if [ -f "${done_file}" ] && [ "$(wc -l < "${done_file}")" -ge "${expected}" ]; then
        echo ">>> RH${rh} already complete ($(wc -l < "${done_file}") objects), skipping"
        n_skip=$((n_skip + 1))
        continue
    fi

    left=$(creds_minutes_left)
    if [ -z "${left}" ] || [ "${left}" -lt "${MIN_MINUTES}" ]; then
        echo ""
        echo ">>> STOP: ${left:-0} min of credential left, not starting RH${rh}."
        echo "    Completed ${n_ok} RH(s) this run. To continue:"
        echo "      source-coop login --duration 12h"
        echo "      sbatch $0 ${QUANTILE} ${WORKERS} \"${RHS}\""
        n_stopped=1
        break
    fi

    echo ""
    echo "=================================================================="
    echo ">>> RH${rh} Q${QUANTILE}  (${left} min of credential left)"
    echo "=================================================================="
    python -m deploy.run run=sc_upload \
        run.src.kind=stac \
        run.src.catalog="${CATALOG}" \
        run.src.tiles="${TILES}" \
        run.src.glob="RH${rh}_Q${QUANTILE}.tif" \
        run.dst.prefix="${DST_PREFIX}" \
        run.state_prefix="${STATE_DIR}/2020_RH${rh}_Q${QUANTILE}" \
        run.workers="${WORKERS}"
    rc=$?

    case ${rc} in
        0) n_ok=$((n_ok + 1)); echo ">>> RH${rh} done" ;;
        *) echo ">>> RH${rh} FAILED (exit ${rc})" >&2
           # Keep going: a transient failure on one RH should not block the
           # rest, and the failed keys are recorded for the only_failed sweep.
           rc_all=1 ;;
    esac
done

echo ""
echo "=================================================================="
echo "finished ${n_ok} RH(s), skipped ${n_skip} already complete; $( [ ${n_stopped} = 1 ] && echo 'stopped early (credential)' || echo 'all requested RHs attempted')"
echo "state files: ${STATE_DIR}"
if [ ${rc_all} != 0 ]; then
    echo "some RHs had failures -- retry just those keys with:"
    echo "  python -m deploy.run run=sc_upload ... run.only_failed=true"
fi
echo "=================================================================="
exit ${rc_all}
