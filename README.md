# 5206-group-project

## Structure Flattening in K-Means Optimization

The optimization transforms the data layout from Array of Structures to Structure of Arrays, specifically flattening the feature vectors into a single contiguous array.

Speedup: 1.25

Compiler optioins: g++ -std=c++11 -O0 -fno-inline Kmeans_struct_optimize.cpp -o kmeans_optimize

## SIMD Acceleration in K-Means

The SIMD path computes squared distances with AVX2 intrinsics, using compile-time configurable vector widths (4/8/16 doubles) to process multiple centroid features per iteration and reduce horizontal reductions.

Speedup: varies by dataset; compare `kmeans` vs `kmeans_simd_w*` outputs from `run.sh`.

Compiler options: `g++ -std=c++11 -O0 -mavx2 -DVEC_WIDTH={4,8,16} Kmeans_simd.cpp -o kmeans_simd_w{4,8,16}`

Run the precompiled executables in `out/` (e.g., `./out/kmeans_simd_w4`) to see sample clustering metrics and timings without rebuilding.



