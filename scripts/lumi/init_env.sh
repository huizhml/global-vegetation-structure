#!/bin/bash

NEW_PROJECT="465002698"
ENV_NAME="ffcv"
ENV_PREFIX=/projappl/project_${NEW_PROJECT}/${ENV_NAME}
mkdir -p $ENV_PREFIX

# =========================== Clean old paths ===========================
CLEANED_PATH=$(echo $PATH | tr ':' '\n' | grep -v "project_" | paste -sd ":" -)
export PATH="$CLEANED_PATH"


# =========================== update ~/.bashrc ===========================
FILE="$HOME/.bashrc"
NEW_LINE="export PATH=\"/projappl/project_${NEW_PROJECT}/${ENV_NAME}/bin:\$PATH\""
# 1. Create a backup
cp "$FILE" "${FILE}.bak"

# 2. Remove any line that exports a PATH containing /scratch/project_
# This clears out both the old ID and the new ID if it was already there
sed -i '/export PATH=.../d' "$FILE"

# 3. Append the fresh, correct line to the end of the file
echo "" >> "$FILE"
echo "# Updated LUMI Python path" >> "$FILE"
echo "$NEW_LINE" >> "$FILE"

bash ~/.bashrc

# =========================== Load modules and create container env ===========================

module load LUMI
module load lumi-container-wrapper
conda-containerize new --prefix $ENV_PREFIX scripts/lumi/env_cnr.yml
conda-containerize update $ENV_PREFIX --post-install scripts/lumi/update_env.sh