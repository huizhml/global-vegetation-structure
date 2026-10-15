#!/bin/bash
#SBATCH --partition=ml4good
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=gedi_extract_cols
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

# Derived GEDI copies -> base + slim sidecars (tools/gedi_extract_cols.py).
#
#   bash scripts/hendrix/gedi_extract_cols.sh inspect                # schemas only, seconds
#   bash scripts/hendrix/gedi_extract_cols.sh inspect base=all_valid/2020
#   sbatch scripts/hendrix/gedi_extract_cols.sh extract split=cal variant=original_with_sota_chms name=sota_chms
#   sbatch scripts/hendrix/gedi_extract_cols.sh extract split=cal variant=original_with_sota_chms_biome \
#       name=biome known=cols/sota_chms
#   sbatch scripts/hendrix/gedi_extract_cols.sh extract split=all_valid name=slope \
#       base=all_valid/2020 derived=all_valid/with_slope_col/2020 out=all_valid/cols/slope/2020

source scripts/core/utils.sh
export PYTHONUNBUFFERED=1

gedi_dir=${GEDI_DIR:-/projects/dereeco/data/gvs/modeling/gedi/veg_sensitivity_gt0p95}

case $1 in
inspect)
# ---------------------------------------
#   New / shared columns and row counts of every derived copy vs its base.
#   Reads parquet footers only; run it on a login node.
# ---------------------------------------
# Args: base (relative to gedi_dir; default: subset_<split>/original/2020)
parse_kv_args "${@:2}" || exit $?
for split_dir in $gedi_dir/subset_*; do
    split=$(basename $split_dir); split=${split#subset_}
    for d in $split_dir/*/; do
        variant=$(basename $d)
        case $variant in original|cols) continue ;; esac
        if [ -d $d/2020 ]; then derived=${d%/}/2020; else derived=$(ls $d/*.parquet 2>/dev/null | head -1); fi
        [ -n "$derived" ] || continue
        echo "=== subset_$split / $variant"
        python -m tools.run run=gedi_extract_cols run.gedi_dir=$gedi_dir run.split=$split \
            "run.derived='${derived%/}'" ${base:+"run.base_dir='$gedi_dir/$base'"} run.dry_run=true 2>&1 | grep -v '^\[' || true
    done
done
echo "=== all_valid / with_slope_col"
python -m tools.run run=gedi_extract_cols run.base_dir=$gedi_dir/all_valid/2020 \
    run.derived=$gedi_dir/all_valid/with_slope_col/2020 run.dry_run=true

;;
extract)
# ---------------------------------------
#   One sidecar: <gedi_dir>/subset_<split>/cols/<name>/<year>/
# ---------------------------------------
# Args: name (required), split, variant, year (default: 2020), workers (default: 8),
#       known (comma list, relative to subset_<split>/, '/<year>' appended),
#       base / derived / out (relative to gedi_dir, override the defaults),
#       overwrite (true|false)
parse_kv_args "${@:2}" || exit $?
: "${name:?name=<sidecar name> is required}"
year=${year:-2020}
sub=$gedi_dir/subset_${split:-cal}
known_list=""
for k in ${known//,/ }; do known_list+="${known_list:+,}'$sub/$k/$year'"; done

python -m tools.run run=gedi_extract_cols run.gedi_dir=$gedi_dir run.split=${split:-cal} \
    run.year=$year run.name=$name run.workers=${workers:-8} \
    ${variant:+run.variant=$variant} \
    ${base:+"run.base_dir='$gedi_dir/$base'"} \
    ${derived:+"run.derived='$gedi_dir/$derived'"} \
    ${out:+"run.out_dir='$gedi_dir/$out'"} \
    ${known:+"run.known_dirs=[$known_list]"} \
    ${overwrite:+run.overwrite=$overwrite}

;;
*)
echo "Unknown task: $1" >&2
exit 1
;;
esac
