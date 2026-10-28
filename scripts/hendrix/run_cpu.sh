#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Per-cpu, not --mem: case 3 runs several tasks per node, so the footprint has
# to scale with the allocation. The two options are mutually exclusive, so
# passing --mem on the sbatch line will be rejected -- override this one.
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24-00:00:00
#SBATCH --job-name=translate
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huzh@di.ku.dk
##SBATCH --exclude hendrixgpu26fl
##SBATCH --exclude hendrixgpu04fl,hendrixgpu03fl,hendrixgpu08fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu14fl,hendrixgpu15fl,hendrixgpu18fl #for using /scratch
echo "***************************** JOB INFO *****************************"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Number of tasks: $SLURM_NTASKS"
echo "Time requested: $SLURM_TIMELIMIT"
scontrol show job $SLURM_JOB_ID | grep "TRES="
echo "********************************************************************"

source scripts/hendrix/utils.sh

# Translate one tile-year of a worklist to COG per array task.
# Same worklist as `run_deploy.sh predict_worklist` (CSV with header tile_id,year);
# run it after the predictions are done. Progress: <worklist without .csv>_state/translate
#   conda activate inference
#   sbatch --array=1-<n_rows> scripts/hendrix/run_cpu.sh translate_worklist \
#       ~/data/gvs/01_derived_ancillary/dk_tile_years_2018_2025.csv
translate_worklist() {
    local worklist=${1:?usage: run_cpu.sh translate_worklist <worklist.csv> [variant]}
    local variant=${2:-original}
    local line_num=$(( ${SLURM_ARRAY_TASK_ID:-1} + 1 )) # line 1 is the header
    local tile_id year
    IFS=, read -r tile_id year <<< "$(sed -n "${line_num}p" $worklist | tr -d '\r')"
    [ -z "$tile_id" ] && { echo "No row $line_num in $worklist" >&2; return 1; }

    local state_dir=${worklist%.*}_state key=${year}_${tile_id}
    if [ "$(state_status $state_dir/translate $key | cut -f1)" == done ]; then
        echo "$key already translated, skipping"
        return 0
    fi
    if [ "$(state_status $state_dir/predict $key | cut -f1)" != done ]; then
        echo "$key is not predicted yet ($state_dir/predict), skipping" >&2
        return 1
    fi
    local t0=$SECONDS rc
    run_translate $tile_id $year $variant
    rc=$?
    if [ $rc -eq 0 ]; then
        state_mark $state_dir/translate $key done "$(( SECONDS - t0 ))s"
    else
        state_mark $state_dir/translate $key failed "run_translate rc=$rc"
    fi
    return $rc
}

# File manifest of a release year -> ~/data/gvs/products/vsm/<year>/released_v0.tsv
# (tools/release_manifest.py; sources per year in config/tools/config.yaml).
# Rerun for 2020 after every `stac_retarget run.dry_run=false`.
#   sbatch --job-name=release_manifest scripts/hendrix/run_cpu.sh release_manifest 2020
release_manifest() {
    local year=${1:?usage: run_cpu.sh release_manifest <year>}
    python -m tools.run run=release_manifest run.year=$year
}

# What each version dir of a year holds -> <year>/<version>/snapshot.tsv
# (+ snapshot_mosaic.tsv); tools/vsm_snapshot.py. Rerun after a prediction,
# translation or retarget changes a version.
#   sbatch --job-name=vsm_snapshot scripts/hendrix/run_cpu.sh vsm_snapshot 2024
vsm_snapshot() {
    local year=${1:?usage: run_cpu.sh vsm_snapshot <year>}
    python -m tools.run run=vsm_snapshot run.year=$year
}

# Which RH*_Q{0,1,2}.tif of each release tile are on Source Coop gvsm/<year>/
# -> <year>/source_coop_state/audit.tsv (+ audit_missing.txt: keys of partly
# present tile x Q). Anonymous listing, no login needed; ~5 min for 18k tiles.
#   sbatch --job-name=sc_audit scripts/hendrix/run_cpu.sh sc_audit 2020
sc_audit() {
    local year=${1:?usage: run_cpu.sh sc_audit <year>}
    python -m deploy.run run=sc_audit run.year=$year
}

# Upload the Q1 files sc_audit found missing (exact key list, nothing else)
# from the STAC collection the Q1 release came from. Needs
# `source-coop login --duration 12h` first; resubmit continues from
# source_coop_state/q1_fill.tsv. Rerun sc_audit afterwards to confirm.
#   sbatch --job-name=sc_fill_q1 scripts/hendrix/run_cpu.sh sc_fill_q1 2020
sc_fill_q1() {
    local year=${1:?usage: run_cpu.sh sc_fill_q1 <year> [workers]}
    local state=$HOME/data/gvs/products/vsm/$year/source_coop_state
    grep '_Q1\.tif$' $state/audit_missing.txt | sed "s#^gvsm/$year/##" > $state/q1_missing_keys.txt
    [ -s $state/q1_missing_keys.txt ] || { echo "no missing Q1 in $state/audit_missing.txt"; return 0; }
    cut -d/ -f1 $state/q1_missing_keys.txt | sort -u > $state/q1_missing_tiles.txt
    local glob
    glob=$(cut -d/ -f2 $state/q1_missing_keys.txt | sort -u | paste -sd, -)
    echo "$(wc -l < $state/q1_missing_keys.txt) keys in $(wc -l < $state/q1_missing_tiles.txt) tiles"
    python -m deploy.run run=sc_upload \
        run.src.kind=stac run.src.catalog=$HOME/data/gvs/products/gvsm_stac_catalog/vsm_local_masked \
        run.src.tiles=$state/q1_missing_tiles.txt run.src.keys=$state/q1_missing_keys.txt \
        "run.src.glob=[$glob]" \
        run.dst.prefix=gvsm/$year/ run.state_prefix=$state/q1_fill run.state_format=tsv \
        run.workers=${2:-64}
}

# <TILE>_cog/RH*_Q*.cog.tif -> <TILE>/RH*_Q*.tif in one tiles dir (default
# 2024/masked/tiles/cog; tools/rename_tile_dirs.py). Dry run unless "false".
#   sbatch --job-name=rename_tile_dirs scripts/hendrix/run_cpu.sh rename_tile_dirs false
rename_tile_dirs() {
    local dry_run=${1:-true}
    python -m tools.run run=rename_tile_dirs run.dry_run=$dry_run
}

case $1 in
1) 
# **************************************************************
#              Translate one tile in one job
# **************************************************************
tile_id_file=${2:-${HOME}/data/gvs/assets/worklists/tiles_duplicated.txt}
line_num=${SLURM_ARRAY_TASK_ID:-2}
tile_id=$(sed -n "${line_num}p" $tile_id_file)
echo "Processing tile ID: $tile_id, line $SLURM_ARRAY_TASK_ID from $tile_id_file"
year=${3:-2020}
# original | masked — masked/ is the 5674-tile backlog listed by
# `python -m tools.run run=stac_cog_audit` in masked_2020_need_cog.txt.
variant=${4:-original}
run_translate $tile_id $year $variant || exit $?
;;
2)
# **************************************************************
#       Translate a CHUNK of tiles in one job (10-job account cap)
# **************************************************************
# Each array task walks its own contiguous slice of the worklist, so 5674
# tiles fit in 10 concurrent tasks instead of 5674 one-tile tasks. Also dodges
# MaxArraySize=4001.
#   sbatch --array=1-10 --cpus-per-task=16 scripts/hendrix/run_cpu.sh 2 \
#       ~/data/gvs/assets/worklists/masked_2020_need_cog.txt 2020 masked 10
tile_id_file=${2:?usage: run_cpu.sh 2 <worklist> [year] [variant] [n_chunks]}
year=${3:-2020}
variant=${4:-masked}
n_chunks=${5:-10}
idx=${SLURM_ARRAY_TASK_ID:-1}

if [ -z "${CONDA_PREFIX}" ]; then
    echo "CONDA_PREFIX is empty: activate the env before submitting (conda activate inference)" >&2
    exit 1
fi
# Without PROJ_DATA the PROJ database cannot be opened in a non-interactive
# shell, EPSG lookups fail and the CRS written into every COG is degraded.
export PROJ_DATA=${PROJ_DATA:-${CONDA_PREFIX}/share/proj}

base=${HOME}/data/gvs/products/vsm/${year}/${variant}/tiles
python -m postprocessing.run run=translate_tiles \
    run.worklist=$tile_id_file \
    run.src_root=${base}/geotiff run.dst_root=${base}/cog \
    run.chunk=$idx run.n_chunks=$n_chunks || exit $?
;;
3)
# **************************************************************
#     One job, MANY parallel translate tasks (srun --ntasks)
# **************************************************************
# QoS `normal` caps MaxJobsPU=10 — on JOBS, not on tasks or cpus (checked with
# `sacctmgr show qos`). So the way to more parallelism is --ntasks inside one
# job, not more jobs. Each task owns one chunk of the worklist:
#     chunks = (array tasks) x SLURM_NTASKS
#     chunk  = (array_id - 1) * SLURM_NTASKS + SLURM_PROCID + 1
# so an array of A jobs x N tasks covers the worklist exactly once.
#
#   sbatch --ntasks=8 --cpus-per-task=4 --mem-per-cpu=2G \
#       scripts/hendrix/run_cpu.sh 3 \
#       ~/data/gvs/assets/worklists/masked_2020_need_cog.txt 2020 masked
#
# ml4good has 416 cpus total, so ntasks x cpus-per-task x (array size) is what
# decides how much of the partition this takes.
tile_id_file=${2:?usage: run_cpu.sh 3 <worklist> [year] [variant] [pattern]}
year=${3:-2020}
variant=${4:-masked}
# Quantile-at-a-time rollout: 'RH*_Q1.tif' converts only the median.
pattern=${5:-RH*_Q*.tif}

if [ -z "${CONDA_PREFIX}" ]; then
    echo "CONDA_PREFIX is empty: activate the env before submitting (conda activate inference)" >&2
    exit 1
fi

base=${HOME}/data/gvs/products/vsm/${year}/${variant}/tiles
# Without PROJ_DATA the PROJ database cannot be opened in a non-interactive
# shell, EPSG lookups fail and the CRS written into every COG is degraded.
export PROJ_DATA=${PROJ_DATA:-${CONDA_PREFIX}/share/proj}

# srun runs python DIRECTLY -- not `bash "$0" ...`. Inside a batch script $0 is
# /var/spool/slurmd/jobNNN/slurm_script, a node-LOCAL copy that only exists on
# the node running the batch step, so tasks landing on any other node die with
# exit 127. translate_tiles derives its chunk from SLURM_PROCID itself.
echo "job: ${SLURM_NTASKS:-1} tasks x ${SLURM_CPUS_PER_TASK:-?} cpus, array task ${SLURM_ARRAY_TASK_ID:-1}"
srun --ntasks=${SLURM_NTASKS:-1} --cpus-per-task=${SLURM_CPUS_PER_TASK:-1} \
    python -m postprocessing.run run=translate_tiles \
        run.worklist=$tile_id_file \
        run.src_root=${base}/geotiff run.dst_root=${base}/cog \
        run.pattern="$pattern" || exit $?
;;
translate_worklist|release_manifest|vsm_snapshot|rename_tile_dirs|sc_audit|sc_fill_q1)
"$@"
;;
*)
echo "Invalid option"
exit 1
;;
esac



# # **************************************************************
# #              Predict one tile in one job
# # **************************************************************
# if [ $line_num -eq 0 ]; then
#     tile_id=$1
# else
#     tile_id_file=${HOME}/data/gvs/deploy/unfinished_tiles_${year}.txt
#     line=$(sed -n "${line_num}p" $tile_id_file)
#     IFS=',' read -r tile_id idx <<< "$line"
# fi
# echo "Processing tile ID: $tile_id, line $line_num from $tile_id_file"


# translate_flag_old="${HOME}/data/gvs/deploy/translate_flags_${year}/${tile_id}_done"
# translate_flag_new="${HOME}/data/gvs/deploy/translate_flags_${year}/${tile_id}_best_images_done"

# if [ -f "$translate_flag_old" ] || [ -f "$translate_flag_new" ]; then
#     file_count=$(ls ${output_dir}/${tile_id}_cog/*.cog.tif | wc -l)
#     if [ $file_count -lt 303 ]; then
#         echo "Tile $tile_id has $file_count files, incomplete"
#         rm -rf ${output_dir}/${tile_id}_cog
#         rm ${translate_flag_new}
#     else
#         echo "Tile $tile_id already translated, skip"
#         exit 0
#     fi
# fi

# inference_flag_new="${HOME}/data/gvs/deploy/inference_flags_${year}/${tile_id}_best_images_done"
# if [ -f "$inference_flag_new" ] && [ ! -f "$translate_flag_new" ]; then
#     echo "***************************** START INFERENCE *****************************"
#     echo Translate predictions for tile $tile_id in year $year;
#     python -m postprocess.translate \
#         src_dir=$input_dir/${tile_id}_GTiff dst_dir=${HOME}/data/gvs/deploy/predictions_${year}/${tile_id}_cog \
#         hydra/job_logging=disabled hydra/hydra_logging=disabled \
#         hydra.run.dir=.  hydra.output_subdir=null  hydra.job.chdir=false
#     exit_status=$?
#     if [ $exit_status -ne 0 ]; then
#         echo "Prediction command failed with exit status $exit_status"
#     else
#         echo "Prediction command completed successfully"
#         touch ${translate_flag_new}
#     fi
#     echo "***************************** END INFERENCE *****************************"
# fi
