#!/bin/bash
#SBATCH --partition=ml4good
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --job-name=tile_diversity
#SBATCH --output=./logs/%x/%A_%a.out
#SBATCH --error=./logs/%x/%A_%a.err
#
# Per-tile diversity-indices map. One SLURM array task processes one tile.
#
# Submit (single command — sbatchs itself with the right --array range):
#     bash scripts/hendrix/tiles/compute_diversity.sh \
#         ~/data/gvs/assets/worklists/dk.txt 2020 masked
#
# Args:
#     $1  worklist          one tile_id per line
#     $2  year              default 2020
#     $3  product_version   upstream VSM version  (default masked)
#
# Knobs (env vars):
#     MAX_CONCURRENT=20    cap concurrent array tasks  (default 50)
#     DRY_RUN=1            print the sbatch command, don't submit

source scripts/hendrix/lib/per_tile_array.sh
per_tile_submit_if_login "$@"   # login node: sbatch self & exit; worker: fall through

# ---- only SLURM workers reach this point ----
set -euo pipefail

worklist=$1
year=${2:-2020}
product_version=${3:-masked}
save_dir=${HOME}/data/gvs/products/diversity_indices/${year}/${product_version}/tiles
mkdir -p "${save_dir}"

per_tile_init "${worklist}"
per_tile_skip_if_exists "${save_dir}/cog/${tile_id}.cog.tif"

python -m evaluation.diversity_maps run=create_tile_diversity_maps \
    run.save_dir="${save_dir}" \
    run.tile_id="${tile_id}" \
    run.year="${year}" \
    run.product_version="${product_version}" \
    run.max_workers="${MAX_WORKERS}"

per_tile_done
