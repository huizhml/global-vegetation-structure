#!/bin/bash
##SBATCH --account=project_465001846
#SBATCH --partition=ml4good
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=download_gedi
#SBATCH --output=./logs/%x-%A_%a.out
#SBATCH --error=./logs/%x-%A_%a.err

# Dispatch named tasks. Pass the task name as $1; per-task args use key=value
# (parsed by parse_kv_args in scripts/core/utils.sh).
#
#   sbatch scripts/hendrix/download_gedi.sh download year=2020

source scripts/core/utils.sh

case $1 in
download)
# ---------------------------------------
#   Download all valid GEDI L2A orbits for each S2 tile's growing season.
#   Writes to ~/data/gvs/gedi/veg_sensitivity_all/all_valid/<year>/
#   (sensitivity_threshold defaults to null — no sensitivity filter).
# ---------------------------------------
# Args: year (default: 2020)
parse_kv_args "${@:2}" || exit $?
# year=${year:-2020}

python -m download.run run=download_all_valid_gedi run.year=$year run.key_file=$key_file

;;
*)
echo "Unknown task: $1" >&2
exit 1
;;
esac
