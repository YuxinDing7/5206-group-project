#!/bin/bash

echo "compile the base of final project"

echo "Kmeans struct optimize here: "
g++ -std=c++11 -O3 -fno-inline Kmeans_Base_extended.cpp -o kmeans_base -I/opt/homebrew/include
time -p ./kmeans_base
g++ -std=c++11 -O3 -fno-inline Kmeans_struct_optimize.cpp -o kmeans_optimize -I/opt/homebrew/include
time -p ./kmeans_optimize


echo ""
echo "compile the SIMD-optimized versions (width 4/8/16)"
# g++ -std=c++17 -O3 -DNDEBUG -mavx2 -mfma -march=native -ffast-math -DVEC_WIDTH=4  Kmeans_simd.cpp -o kmeans_simd_w4
# g++ -std=c++17 -O3 -DNDEBUG -mavx2 -mfma -march=native -ffast-math -DVEC_WIDTH=8  Kmeans_simd.cpp -o kmeans_simd_w8
# g++ -std=c++17 -O3 -DNDEBUG -mavx2 -mfma -march=native -ffast-math -DVEC_WIDTH=16 Kmeans_simd.cpp -o kmeans_simd_w16
g++ -std=c++11 -O0 -fno-inline -mavx2 -DVEC_WIDTH=4  Kmeans_simd.cpp -o kmeans_simd_w4
g++ -std=c++11 -O0 -fno-inline -mavx2 -DVEC_WIDTH=8  Kmeans_simd.cpp -o kmeans_simd_w8
g++ -std=c++11 -O0 -fno-inline -mavx2 -DVEC_WIDTH=16 Kmeans_simd.cpp -o kmeans_simd_w16
echo " ======================== run SIMD w4 =========================="
time -p ./kmeans_simd_w4
echo " ======================== run SIMD w8 =========================="
time -p ./kmeans_simd_w8
echo " ======================== run SIMD w16 =========================="
time -p ./kmeans_simd_w16


echo "Kmeans MultiThread optimize here: "
g++ -std=c++11 -O3  -fno-inline -fopenmp \
  -I/opt/homebrew/include -L/opt/homebrew/lib  \
  Kmeans_MultiThread.cpp -o kmeans_omp
time -p ./kmeans_omp
