#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=extract_pred
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

# Dispatch named tasks via just. Pass the task name as $1; per-task args use
# key=value (parsed by parse_kv_args in scripts/core/utils.sh). pa-auc is the
# exception — extra args pass through to just/Hydra unchanged.
#
#   sbatch --array=1-N run_analysis.sh diversity tile_list=tiles.csv year=2020
#   sbatch run_analysis.sh naturalness-sample-patches split=test
#   sbatch run_analysis.sh pa-auc run.max_workers=32 run.rh_level_step=5
#
# See `just --list` (group evaluation/postprocessing) for the underlying recipes.

source scripts/core/utils.sh

case $1 in
diversity)
# ---------------------------------------
#   Per-tile diversity indices
# ---------------------------------------
# Args: tile_list (default: dk.txt; .csv with "Tile name" column or .txt one per line)
#       year      (default: 2020)
tile_list=${HOME}/data/gvs/assets/worklists/dk.txt
year=2020
product_version=original
product_format=geotiff
bin_width=5
parse_kv_args "${@:2}" || exit $?
tile_id=$(pick_tile "$tile_list")
echo "Processing tile ID: $tile_id (year=$year)"
just generate-tile-diversity-maps run.tile_id=${tile_id} run.year=$year run.product_version=$product_version run.product_format=$product_format run.bin_width=$bin_width || exit $?
;;

# =======================================
#   Biome analysis
# =======================================
extract-pred-biome)
# ---------------------------------------
#   Extract predictions by tile, for biome analysis
# ---------------------------------------
loc_dir=/projects/dereeco/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome_by_tile/
save_dir=/projects/dereeco/data/gvs/analysis/biome_anlaysis/random_sample_100000_points_per_biome_preds
just post-extract-pred run.loc_dir=${loc_dir} run.save_dir=${save_dir}
source scripts/core/run_python.sh
run_sanity_check check_two_datasets ${loc_dir} ${save_dir} || exit $?
;;

# =======================================
#   Naturalness analysis
# =======================================
naturalness-prepare-loc)
# ---------------------------------------
#   Partition naturalness locations by tile
# ---------------------------------------
# NOTE: this used to call `python -m evaluation.run run=prepare_naturalness_loc_parquets`,
# but `evaluation/run.py` no longer exists. The function lives at
# `evaluation.on_naturalness.prepare_loc_parqs` but has no Hydra entry yet.
# Wire it up to an entrypoint before using.
echo "TODO: prepare_loc_parqs is not wired to a Hydra run config — see comment above." >&2
exit 2
;;

naturalness-sample-patches)
# ---------------------------------------
#   Sample VSM patches
# ---------------------------------------
# Args: split (default: val)
split=val
parse_kv_args "${@:2}" || exit $?
loc_dir=${HOME}/data/gvs/downstream_tasks/naturalness/loc_by_tile_${split}
save_dir=${HOME}/data/gvs/downstream_tasks/naturalness/vsm_patches_ps11_${split}
just post-sample-vsm-patches run.loc_dir=${loc_dir} run.save_dir=${save_dir} || exit $?
;;

naturalness-vsm-patch-stats)
# ---------------------------------------
#   Calculate VSM patch statistics
# ---------------------------------------
# Args: split (default: val)
split=val
parse_kv_args "${@:2}" || exit $?
vsm_patches_dir=${HOME}/data/gvs/downstream_tasks/naturalness/vsm_patches_ps11_${split}
save_dir=${HOME}/data/gvs/downstream_tasks/naturalness/vsm_patch_stats_ps11_${split}
just eval-cal-vsm-patch-stats run.vsm_patches_dir=${vsm_patches_dir} run.save_dir=${save_dir} || exit $?
;;

naturalness-sample-points)
# ---------------------------------------
#   Sample VSM points
# ---------------------------------------
# Args: split (default: test)
split=test
parse_kv_args "${@:2}" || exit $?
just post-sample-vsm-points run.split=${split}
;;

# =======================================
#   Protected area analysis
# =======================================
pa-auc)
# ---------------------------------------
#   Protected area structural analysis (normalized RH profile AUC)
# ---------------------------------------
# Samples primary / protected / outside-PA forest points, extracts their 10 m
# RH profiles per S2 tile, and compares AUC vs RH98 per biome. Per-point
# profiles are cached to {save_dir}/samples_profiles.parquet — delete that file
# to re-sample. I/O-bound (workers wait on network reads), so run.max_workers
# can exceed --cpus-per-task; bump the SBATCH header if you push it much higher.
#
# Hydra overrides passthrough, e.g.:
#   sbatch run_analysis.sh pa-auc run.max_workers=32 run.rh_level_step=5
just eval-pa-auc "${@:2}" || exit $?
;;

*)
echo "Usage: $0 <task> [args...]"
echo "Tasks: diversity | extract-pred-biome | naturalness-prepare-loc |"
echo "       naturalness-sample-patches | naturalness-vsm-patch-stats |"
echo "       naturalness-sample-points | pa-auc"
exit 1
;;
esac
