# 5206-group-project

Structure Flattening in K-Means Optimization

The optimization transforms the data layout from Array of Structures to Structure of Arrays, specifically flattening the feature vectors into a single contiguous array.

Speedup: 1.25

Compiler optioins: g++ -std=c++11 -O0 -fno-inline Kmeans_struct_optimize.cpp -o kmeans_optimize
