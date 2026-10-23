#!/bin/bash
#SBATCH --account=project_465003168
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=gedi_merge_benchmark
#SBATCH --output=/users/zhanghui/scratch/logs/%x-%A_%a.out
#SBATCH --error=/users/zhanghui/scratch/logs/%x-%A_%a.err

# GEDI download from GEE: one request per orbit vs orbits merged per request,
# and GEE's size limit per request. One tile per array task (the tiles with
# the most shots in 2020); run from the repo root (~/gvs):
#   sbatch --array=0-3 scripts/lumi/gedi_merge_benchmark.sh
# then the summary (login node is fine, it only reads the tsv):
#   python -m tools.run run=gedi_merge_benchmark_summary run.results_dir=${bench_dir}/results_2020
# Extra args are Hydra overrides, e.g. run.methods=[adaptive,per_orbit]

source scripts/lumi/env_init.sh

# inputs copied from Hendrix; ~/data/gvs on LUMI is read-only
bench_dir=/scratch/project_465003168/hui/gedi_merge_benchmark
python -m tools.run run=gedi_merge_benchmark \
    run.tile_rank=${SLURM_ARRAY_TASK_ID:-0} \
    run.s2_grid_file=${bench_dir}/s2_tiles_with_growing_months.parquet \
    run.gedi_table_index_file=${bench_dir}/l2a_orbit_table_2020.parquet \
    run.state_dir=${bench_dir}/2020_state \
    run.prior_state_dir=${bench_dir}/2022_state \
    run.prior_index_file=${bench_dir}/l2a_orbit_table_2022.parquet \
    run.results_dir=${bench_dir}/results_2020 \
    "$@"
