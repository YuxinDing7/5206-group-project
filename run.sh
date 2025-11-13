#!/bin/bash

# echo "compile the base of final project"
# g++ -std=c++11 -O0 -fno-inline Kmeans_Base_extended.cpp -o kmeans 
# time -p ./kmeans


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
