#!/bin/bash
# Thin shim: the real logic lives in copy_from_lumio.sh as a function.
# Usage unchanged:  bash benchmark_rclone_copy.sh <TILE_ID> [YEAR]
source "$(dirname "${BASH_SOURCE[0]}")/copy_from_lumio.sh"
benchmark_rclone_copy "$@"
