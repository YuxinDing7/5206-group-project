#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <random>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <immintrin.h>

using json = nlohmann::json;

#ifndef VEC_WIDTH
#define VEC_WIDTH 4  // supported: 4, 8, 16 (doubles per iteration)
#endif

struct Point {
    int id;
    std::vector<double> features;
    int cluster;        // assigned cluster by algorithm
    int ground_truth;   // optional label from dataset (-1 if missing)
    Point(): id(-1), cluster(-1), ground_truth(-1) {}
};

class KMeans {
private:
    int K = 0, D = 0, N = 0;
    std::vector<Point> points;
    std::vector<double> centroids; // flattened K x D

    // squared Euclidean distance (avoid sqrt) - SIMD accelerated with AVX2 when available
    inline double calculateDistanceSquaredToCentroid(const std::vector<double>& a, const double* centroidPtr) const {
#if defined(__AVX2__)
        const int vecWidth = VEC_WIDTH; // 4, 8, or 16
        int i = 0;
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        __m256d acc2 = _mm256_setzero_pd();
        __m256d acc3 = _mm256_setzero_pd();

        const int limit = (D / vecWidth) * vecWidth;
        for (; i < limit; i += vecWidth) {
            // lane 0..3
            __m256d va0 = _mm256_loadu_pd(a.data() + i);
            __m256d vb0 = _mm256_loadu_pd(centroidPtr + i);
            __m256d diff0 = _mm256_sub_pd(va0, vb0);
        #if defined(__FMA__)
            acc0 = _mm256_fmadd_pd(diff0, diff0, acc0);
        #else
            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(diff0, diff0));
        #endif

        #if VEC_WIDTH >= 8
            __m256d va1 = _mm256_loadu_pd(a.data() + i + 4);
            __m256d vb1 = _mm256_loadu_pd(centroidPtr + i + 4);
            __m256d diff1 = _mm256_sub_pd(va1, vb1);
        #if defined(__FMA__)
            acc1 = _mm256_fmadd_pd(diff1, diff1, acc1);
        #else
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(diff1, diff1));
        #endif
        #endif

        #if VEC_WIDTH >= 16
            __m256d va2 = _mm256_loadu_pd(a.data() + i + 8);
            __m256d vb2 = _mm256_loadu_pd(centroidPtr + i + 8);
            __m256d diff2 = _mm256_sub_pd(va2, vb2);
        #if defined(__FMA__)
            acc2 = _mm256_fmadd_pd(diff2, diff2, acc2);
        #else
            acc2 = _mm256_add_pd(acc2, _mm256_mul_pd(diff2, diff2));
        #endif

            __m256d va3 = _mm256_loadu_pd(a.data() + i + 12);
            __m256d vb3 = _mm256_loadu_pd(centroidPtr + i + 12);
            __m256d diff3 = _mm256_sub_pd(va3, vb3);
        #if defined(__FMA__)
            acc3 = _mm256_fmadd_pd(diff3, diff3, acc3);
        #else
            acc3 = _mm256_add_pd(acc3, _mm256_mul_pd(diff3, diff3));
        #endif
        #endif
        }

        __m256d acc = acc0;
        #if VEC_WIDTH >= 8
        acc = _mm256_add_pd(acc, acc1);
        #endif
        #if VEC_WIDTH >= 16
        acc = _mm256_add_pd(acc, acc2);
        acc = _mm256_add_pd(acc, acc3);
        #endif

        // in-register horizontal sum
        __m256d hi = _mm256_permute2f128_pd(acc, acc, 0x1);
        __m256d sum2 = _mm256_add_pd(acc, hi);
        __m128d lo128 = _mm256_castpd256_pd128(sum2);
        __m128d hi128 = _mm_unpackhi_pd(lo128, lo128);
        __m128d sum128 = _mm_add_sd(lo128, hi128);
        double sum = _mm_cvtsd_f64(sum128);
        // tail
        for (; i < D; ++i) {
            double d = a[i] - centroidPtr[i];
            sum += d * d;
        }
        return sum;
#else
        double sum = 0.0;
        for (int i = 0; i < D; i++) {
            double d = a[i] - centroidPtr[i];
            sum += d * d;
        }
        return sum;
#endif
    }

    void initializeCentroids() {
        std::random_device rd;
        std::mt19937 gen(rd());              
        std::uniform_int_distribution<> dis(0, N - 1);

        centroids.assign(static_cast<size_t>(K) * static_cast<size_t>(D), 0.0);
        for (int i = 0; i < K; i++) {
            int randomIndex = dis(gen);
            // copy features into flat centroids
            const std::vector<double>& src = points[randomIndex].features;
            std::copy(src.begin(), src.end(), centroids.begin() + static_cast<size_t>(i) * static_cast<size_t>(D));
        }
    }

public:
    KMeans(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        json j;
        try {
            file >> j;
        } catch (const json::parse_error& e) {
            throw std::runtime_error(std::string("JSON parse error in file ") + filename + ": " + e.what());
        }

        if (!j.is_array()) {
            throw std::runtime_error("Expected top-level JSON array in file: " + filename);
        }

        N = static_cast<int>(j.size());
        if (N == 0) {
            throw std::runtime_error("Empty dataset in file: " + filename);
        }

        points.resize(N);

        for (int i = 0; i < N; i++) {
            const json& entry = j[i];
            if (!entry.contains("features") || !entry["features"].is_array()) {
                throw std::runtime_error("Entry " + std::to_string(i) + " missing 'features' array in file: " + filename);
            }

            std::vector<double> feats = entry["features"].get<std::vector<double>>();
            if (i == 0) {
                D = static_cast<int>(feats.size());
                if (D == 0) throw std::runtime_error("Feature vector has zero length in file: " + filename);
            } else if (static_cast<int>(feats.size()) != D) {
                throw std::runtime_error("Inconsistent feature dimension at entry " + std::to_string(i) + " in file: " + filename);
            }

            points[i].id = entry.contains("id") ? entry["id"].get<int>() : i;
            points[i].features = std::move(feats);
            points[i].cluster = -1;

            // optional ground-truth label: accept "cluster" or "label"
            if (entry.contains("cluster") && entry["cluster"].is_number_integer()) {
                points[i].ground_truth = entry["cluster"].get<int>();
            } else if (entry.contains("label") && entry["label"].is_number_integer()) {
                points[i].ground_truth = entry["label"].get<int>();
            } else {
                points[i].ground_truth = -1;
            }
        }
    }

    void cluster(int num_clusters, int max_iterations = 100, int min_iterations = 0) {
        K = num_clusters;

        initializeCentroids();

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            bool changed = false;

            for (int i = 0; i < N; i++) {
                double minDist = std::numeric_limits<double>::max();
                int nearestCluster = -1;
                for (int j = 0; j < K; j++) {
                    const double* centroidPtr = centroids.data() + static_cast<size_t>(j) * static_cast<size_t>(D);
                    double dist = calculateDistanceSquaredToCentroid(points[i].features, centroidPtr);
                    if (dist < minDist) {
                        minDist = dist;
                        nearestCluster = j;
                    }
                }
                if (points[i].cluster != nearestCluster) {
                    points[i].cluster = nearestCluster;
                    changed = true;
                }
            }

            std::vector<int> clusterCounts(K, 0);
            std::vector<double> newCentroids(static_cast<size_t>(K) * static_cast<size_t>(D), 0.0);
            for (int i = 0; i < N; i++) {
                int c = points[i].cluster;
                clusterCounts[c]++;
                for (int d = 0; d < D; d++) {
                    newCentroids[static_cast<size_t>(c) * static_cast<size_t>(D) + static_cast<size_t>(d)] += points[i].features[d];
                }
            }
            for (int c = 0; c < K; c++) {
                if (clusterCounts[c] > 0) {
                    for (int d = 0; d < D; d++) {
                        centroids[static_cast<size_t>(c) * static_cast<size_t>(D) + static_cast<size_t>(d)] =
                            newCentroids[static_cast<size_t>(c) * static_cast<size_t>(D) + static_cast<size_t>(d)] / static_cast<double>(clusterCounts[c]);
                    }
                }
            }

            if (!changed && (iteration + 1) >= min_iterations) break;
        }
    }

    void printResults() {
        std::cout << "Clustering Results:\n";
        for (int i = 0; i < N; i++) {
            std::cout << "Point " << points[i].id << " -> Cluster " << points[i].cluster << "\n";
        }
    }

    // Sum of squared errors (inertia)
    double computeSSE() const {
        double sse = 0.0;
        for (const auto& p : points) {
            if (p.cluster < 0 || p.cluster >= K) continue;
            const double* centroidPtr = centroids.data() + static_cast<size_t>(p.cluster) * static_cast<size_t>(D);
            // reuse the SIMD path through a small wrapper lambda to avoid duplicating code
#if defined(__AVX2__)
            // const_cast is safe here since we don't modify
            sse += const_cast<KMeans*>(this)->calculateDistanceSquaredToCentroid(p.features, centroidPtr);
#else
            sse += const_cast<KMeans*>(this)->calculateDistanceSquaredToCentroid(p.features, centroidPtr);
#endif
        }
        return sse;
    }

    // Purity: requires ground-truth labels (returns -1.0 if absent)
    double computePurity() const {
        bool has_gt = false;
        for (const auto& p : points) if (p.ground_truth != -1) { has_gt = true; break; }
        if (!has_gt) return -1.0;

        std::vector<std::unordered_map<int, int>> counts(K);
        for (const auto& p : points) {
            if (p.cluster < 0 || p.cluster >= K) continue;
            if (p.ground_truth == -1) continue;
            counts[p.cluster][p.ground_truth]++;
        }

        int sum_max = 0;
        for (int c = 0; c < K; ++c) {
            int mx = 0;
            for (const auto& kv : counts[c]) mx = std::max(mx, kv.second);
            sum_max += mx;
        }
        return static_cast<double>(sum_max) / static_cast<double>(N);
    }
};

int main() {
    // Example datasets
    std::vector<std::string> datasets = {
        "data_N200_D4_K8.json",
        "data_N200_D16_K8.json",
        "data_N800_D32_K16.json",
        "data_N800_D64_K16.json"
    };

    for (const auto& dataset : datasets) {
        std::cout << "\nProcessing " << dataset << std::endl;
        try {
            auto t_start_load = std::chrono::high_resolution_clock::now();
            KMeans kmeans(dataset);
            auto t_end_load = std::chrono::high_resolution_clock::now();

            int K = (dataset.find("N200") != std::string::npos) ? 8 : 16;

            // Allow early convergence; keep a generous upper bound
            int minIters = 0;
            int maxIters = 40000;

            auto t_start_cluster = std::chrono::high_resolution_clock::now();
            kmeans.cluster(K, /*max_iterations=*/maxIters, /*min_iterations=*/minIters);
            auto t_end_cluster = std::chrono::high_resolution_clock::now();

            // kmeans.printResults();

            double sse = kmeans.computeSSE();
            double purity = kmeans.computePurity();

            auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_load - t_start_load).count();
            auto cluster_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_cluster - t_start_cluster).count();
            auto cluster_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end_cluster - t_start_cluster).count();

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "SSE (inertia): " << sse << "\n";
            if (purity < 0.0) {
                std::cout << "Purity: (no ground-truth labels present in dataset)\n";
            } else {
                std::cout << "Purity: " << purity << "\n";
            }
            std::cout << "Load time: " << load_ms << " ms\n";
            std::cout << "Clustering time: " << cluster_ms << " ms (" << cluster_us << " us)\n";

        } catch (const std::exception& e) {
            std::cerr << "Error processing dataset '" << dataset << "': " << e.what() << "\n";
        }
    }
    return 0;
}