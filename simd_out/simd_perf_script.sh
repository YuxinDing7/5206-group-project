#!/bin/bash

/usr/lib/linux-tools/5.15.0-161-generic/perf stat -e cache-references,cache-misses,cycles,instructions,branch-instructions \
   ./kmeans > baseline_perf.txt
/usr/lib/linux-tools/5.15.0-161-generic/perf stat -e cache-references,cache-misses,cycles,instructions,branch-instructions \
   ./kmeans_simd_w4 > simd_w4_perf.txt
/usr/lib/linux-tools/5.15.0-161-generic/perf stat -e cache-references,cache-misses,cycles,instructions,branch-instructions \
   ./kmeans_simd_w8 > simd_w8_perf.txt
/usr/lib/linux-tools/5.15.0-161-generic/perf stat -e cache-references,cache-misses,cycles,instructions,branch-instructions \
   ./kmeans_simd_w16 > simd_w16_perf.txt