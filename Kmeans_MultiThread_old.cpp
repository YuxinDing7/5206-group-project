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
#include <thread>
#include <mutex>
#include <omp.h>

using json = nlohmann::json;

struct Point {
    int id;
    std::vector<double> features;
    int cluster;        // assigned cluster by algorithm
    int ground_truth;   // optional label from dataset (-1 if missing)
    Point(): id(-1), cluster(-1), ground_truth(-1) {}
};

class KMeansMultiThread {
private:
    int K = 0, D = 0, N = 0;
    std::vector<Point> points;
    std::vector<std::vector<double>> centroids;
    int num_threads = 0;

    // squared Euclidean distance (avoid sqrt)
    double calculateDistanceSquared(const std::vector<double>& a, const std::vector<double>& b) const {
        double sum = 0.0;
        for (int i = 0; i < D; i++) {
            double d = a[i] - b[i];
            sum += d * d;
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
    KMeansMultiThread(const std::string& filename, int num_threads = 0) {
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

        // Set number of threads
        if (num_threads <= 0) {
            this->num_threads = std::thread::hardware_concurrency();
        } else {
            this->num_threads = num_threads;
        }
    }

    void clusterSingleThread(int num_clusters, int max_iterations = 100, int min_iterations = 0) {
        K = num_clusters;
        initializeCentroids();

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            bool changed = false;

            // Assignment step
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
                    changed = true;
                }
            }

            // Update step
            std::vector<int> clusterCounts(K, 0);
            std::vector<std::vector<double>> newCentroids(K, std::vector<double>(D, 0.0));
            for (int i = 0; i < N; i++) {
                int c = points[i].cluster;
                clusterCounts[c]++;
                for (int d = 0; d < D; d++) newCentroids[c][d] += points[i].features[d];
            }
            for (int c = 0; c < K; c++) {
                if (clusterCounts[c] > 0) {
                    for (int d = 0; d < D; d++) centroids[c][d] = newCentroids[c][d] / clusterCounts[c];
                }
            }

            if (!changed && (iteration + 1) >= min_iterations) break;
        }
    }

    void clusterMultiThreadOpenMP(int num_clusters, int max_iterations = 100, int min_iterations = 0) {
        K = num_clusters;
        initializeCentroids();
        
        omp_set_num_threads(num_threads);

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            bool changed = false;

            // Assignment step - parallel
            #pragma omp parallel for shared(changed)
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
                    #pragma omp critical
                    changed = true;
                }
            }

            // Update step - parallel
            std::vector<int> clusterCounts(K, 0);
            std::vector<std::vector<double>> newCentroids(K, std::vector<double>(D, 0.0));

            #pragma omp parallel
            {
                // Local copies for each thread
                std::vector<int> local_clusterCounts(K, 0);
                std::vector<std::vector<double>> local_newCentroids(K, std::vector<double>(D, 0.0));

                #pragma omp for nowait
                for (int i = 0; i < N; i++) {
                    int c = points[i].cluster;
                    local_clusterCounts[c]++;
                    for (int d = 0; d < D; d++) {
                        local_newCentroids[c][d] += points[i].features[d];
                    }
                }

                // Merge results from threads
                #pragma omp critical
                {
                    for (int c = 0; c < K; c++) {
                        clusterCounts[c] += local_clusterCounts[c];
                        for (int d = 0; d < D; d++) {
                            newCentroids[c][d] += local_newCentroids[c][d];
                        }
                    }
                }
            }

            // Update centroids
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

    // Sum of squared errors (inertia)
    double computeSSE() const {
        double sse = 0.0;
        for (const auto& p : points) {
            if (p.cluster < 0 || p.cluster >= K) continue;
            sse += calculateDistanceSquared(p.features, centroids[p.cluster]);
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

    int getNumThreads() const { return num_threads; }
};

int main() {
    std::vector<std::string> datasets = {
        "data_N200_D4_K8.json",
        "data_N200_D16_K8.json",
        "data_N800_D32_K16.json",
        "data_N800_D64_K16.json"
        "data_N200000_D4_K8.json"
        "data_N500000_D4_K8.json"
    };

    for (const auto& dataset : datasets) {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "Processing " << dataset << std::endl;
        std::cout << std::string(80, '=') << "\n";
        
        try {
            auto t_start_load = std::chrono::high_resolution_clock::now();
            KMeansMultiThread kmeans(dataset);
            auto t_end_load = std::chrono::high_resolution_clock::now();

            int K = (dataset.find("N200") != std::string::npos) ? 8 : 16;
            int minIters = (dataset.find("N200") != std::string::npos) ? 160 : 120;
            int maxIters = std::max(4 * minIters, 400);

            std::cout << "Number of threads available: " << kmeans.getNumThreads() << "\n\n";

            // Single-threaded baseline
            auto t_start_single = std::chrono::high_resolution_clock::now();
            KMeansMultiThread kmeans_single(dataset, 1);
            kmeans_single.clusterSingleThread(K, maxIters, minIters);
            auto t_end_single = std::chrono::high_resolution_clock::now();
            auto single_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end_single - t_start_single).count();
            auto single_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_single - t_start_single).count();

            double sse_single = kmeans_single.computeSSE();
            double purity_single = kmeans_single.computePurity();

            std::cout << "--- Single-Threaded Results ---\n";
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Clustering time: " << single_ms << " ms (" << single_us << " us)\n";
            std::cout << "SSE: " << sse_single << "\n";
            if (purity_single < 0.0) {
                std::cout << "Purity: (no ground-truth labels)\n";
            } else {
                std::cout << "Purity: " << purity_single << "\n";
            }

            // Multi-threaded OpenMP
            auto t_start_omp = std::chrono::high_resolution_clock::now();
            KMeansMultiThread kmeans_omp(dataset);
            kmeans_omp.clusterMultiThreadOpenMP(K, maxIters, minIters);
            auto t_end_omp = std::chrono::high_resolution_clock::now();
            auto omp_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end_omp - t_start_omp).count();
            auto omp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_omp - t_start_omp).count();

            double sse_omp = kmeans_omp.computeSSE();
            double purity_omp = kmeans_omp.computePurity();
            double speedup_omp = static_cast<double>(single_us) / omp_us;

            std::cout << "\n--- Multi-Threaded (OpenMP) Results ---\n";
            std::cout << "Number of threads: " << kmeans_omp.getNumThreads() << "\n";
            std::cout << "Clustering time: " << omp_ms << " ms (" << omp_us << " us)\n";
            std::cout << "SSE: " << sse_omp << "\n";
            if (purity_omp < 0.0) {
                std::cout << "Purity: (no ground-truth labels)\n";
            } else {
                std::cout << "Purity: " << purity_omp << "\n";
            }


            // Speedup summary
            std::cout << "\n--- Performance Summary ---\n";
            std::cout << "Single-threaded time: " << single_ms << " ms\n";
            std::cout << "OpenMP time: " << omp_ms << " ms (Speedup: " << speedup_omp << "x)\n";

        } catch (const std::exception& e) {
            std::cerr << "Error processing dataset '" << dataset << "': " << e.what() << "\n";
        }
    }
    return 0;
}
