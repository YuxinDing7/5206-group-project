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
#include <omp.h>

using json = nlohmann::json;

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
    std::vector<std::vector<double>> centroids;

    // squared Euclidean distance (avoid sqrt)
    double calculateDistanceSquared(const std::vector<double>& a, const std::vector<double>& b) const {
        double sum = 0.0;
        if (D < 32) {
            for (int i = 0; i < D; i++) {
                double d = a[i] - b[i];
                sum += d * d;
            }
        } else {
            #pragma omp simd reduction(+:sum)
            for (int i = 0; i < D; i++) {
                double d = a[i] - b[i];
                sum += d * d;
            }
        }
        return sum;
    }

    void initializeCentroids() {
        std::random_device rd;
        std::mt19937 gen(rd());              
        std::uniform_int_distribution<> dis(0, N - 1);

        centroids.resize(K, std::vector<double>(D));
        for (int i = 0; i < K; i++) {
            int randomIndex = dis(gen);
            centroids[i] = points[randomIndex].features;
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

            #pragma omp parallel
            {
                bool local_changed = false;
                
                #pragma omp for schedule(static) nowait
                for (int i = 0; i < N; i++) {
                    double minDist = std::numeric_limits<double>::max();
                    int nearestCluster = -1;
                    
                    for (int j = 0; j < K; j++) {
                        double dist = calculateDistanceSquared(points[i].features, centroids[j]);
                        if (dist < minDist) {
                            minDist = dist;
                            nearestCluster = j;
                        }
                    }
                    
                    if (points[i].cluster != nearestCluster) {
                        points[i].cluster = nearestCluster;
                        local_changed = true;
                    }
                }
                
                if (local_changed) {
                    #pragma omp atomic write
                    changed = true;
                }
            }

            std::vector<std::vector<double>> newCentroids(K, std::vector<double>(D, 0.0));
            std::vector<int> clusterCounts(K, 0);

            #pragma omp parallel
            {
                std::vector<std::vector<double>> local_centroids(K, std::vector<double>(D, 0.0));
                std::vector<int> local_counts(K, 0);

                #pragma omp for schedule(static) nowait
                for (int i = 0; i < N; i++) {
                    int c = points[i].cluster;
                    local_counts[c]++;
                    for (int d = 0; d < D; d++) {
                        local_centroids[c][d] += points[i].features[d];
                    }
                }

                #pragma omp critical
                {
                    for (int c = 0; c < K; c++) {
                        clusterCounts[c] += local_counts[c];
                        for (int d = 0; d < D; d++) {
                            newCentroids[c][d] += local_centroids[c][d];
                        }
                    }
                }
            }

            #pragma omp parallel for schedule(static)
            for (int c = 0; c < K; c++) {
                if (clusterCounts[c] > 0) {
                    for (int d = 0; d < D; d++) {
                        centroids[c][d] = newCentroids[c][d] / clusterCounts[c];
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

    double computeSSE() const {
        double sse = 0.0;
        
        #pragma omp parallel for reduction(+:sse) schedule(static)
        for (int i = 0; i < N; i++) {
            if (points[i].cluster >= 0 && points[i].cluster < K) {
                sse += calculateDistanceSquared(points[i].features, centroids[points[i].cluster]);
            }
        }
        
        return sse;
    }

    double computePurity() const {
        bool has_gt = false;
        for (const auto& p : points) if (p.ground_truth != -1) { has_gt = true; break; }
        if (!has_gt) return -1.0;

        std::vector<std::unordered_map<int, int>> counts(K);
        
        #pragma omp parallel
        {
            std::vector<std::unordered_map<int, int>> local_counts(K);
            
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < N; i++) {
                const auto& p = points[i];
                if (p.cluster >= 0 && p.cluster < K && p.ground_truth != -1) {
                    local_counts[p.cluster][p.ground_truth]++;
                }
            }
            
            #pragma omp critical
            {
                for (int c = 0; c < K; ++c) {
                    for (const auto& kv : local_counts[c]) {
                        counts[c][kv.first] += kv.second;
                    }
                }
            }
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
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "Using " << omp_get_num_threads() << " OpenMP threads\n";
        }
    }

    // Example datasets
    std::vector<std::string> datasets = {
        // "data_N200_D4_K8.json",
        // "data_N200_D16_K8.json",
        // "data_N800_D32_K16.json",
        // "data_N800_D64_K16.json",
        // "data_N200000_D4_K8.json",
        "data_N500000_D4_K8.json",
    };

    for (const auto& dataset : datasets) {
        std::cout << "\nProcessing " << dataset << std::endl;
        try {
            auto t_start_load = std::chrono::high_resolution_clock::now();
            KMeans kmeans(dataset);
            auto t_end_load = std::chrono::high_resolution_clock::now();

            int K = (dataset.find("N200") != std::string::npos) ? 8 : 16;

            int minIters = (dataset.find("N200") != std::string::npos) ? 160 : 120;
            int maxIters = std::max(4 * minIters, 400);

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