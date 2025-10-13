#!/bin/bash

echo "compile the base of final project"
g++ -std=c++11 -O0 -fno-inline Kmeans_baseline.cpp -o kmeans 
time -p ./kmeans