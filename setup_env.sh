#!/bin/bash
# Environment setup for GDAL with libjxl
export PREFIX=$HOME/local
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"

echo "Environment configured for GDAL with libjxl support"
echo "GDAL version: $(gdalinfo --version 2>/dev/null || echo 'Not found')" 