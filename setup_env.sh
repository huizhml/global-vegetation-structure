module load LUMI partition/L
module load GDAL/3.9.0-cpeGNU-24.03-cray-python-3.11.7
export PATH="/scratch/project_465001846/py3/bin:$PATH"
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
# Environment setup for GDAL with libjxl
# export PREFIX=$HOME/local
# export PATH="$PREFIX/bin:$PATH"
# export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
# export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
# export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"

# echo "Environment configured for GDAL with libjxl support"
# echo "GDAL version: $(gdalinfo --version 2>/dev/null || echo 'Not found')" 