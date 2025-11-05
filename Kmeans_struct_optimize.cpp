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

using json = nlohmann::json;

struct Points {
    std::vector<int> ids;
    std::vector<double> features; 
    std::vector<int> clusters;
    std::vector<int> ground_truths;
    int N;  
    int D;  
    
    Points() : N(0), D(0) {}
    
    inline double getFeature(int i, int d) const {
        return features[i * D + d];
    }
    
    inline void setFeature(int i, int d, double value) {
        features[i * D + d] = value;
    }
    
    inline const double* getFeaturePtr(int i) const {
        return &features[i * D];
    }
};

class KMeans {
private:
    int K = 0, D = 0, N = 0;
    Points points;  
    std::vector<std::vector<double>> centroids;

    // squared Euclidean distance (avoid sqrt)
    double calculateDistanceSquared(const double* point_features, const std::vector<double>& centroid) const {
        double sum = 0.0;
        for (int i = 0; i < D; i++) {
            double d = point_features[i] - centroid[i];
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
            for (int d = 0; d < D; d++) {
                centroids[i][d] = points.getFeature(randomIndex, d);
            }
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

        points.N = N;
        points.ids.resize(N);
        points.clusters.resize(N);
        points.ground_truths.resize(N);

        if (!j[0].contains("features") || !j[0]["features"].is_array()) {
            throw std::runtime_error("Entry 0 missing 'features' array in file: " + filename);
        }
        D = static_cast<int>(j[0]["features"].size());
        if (D == 0) throw std::runtime_error("Feature vector has zero length in file: " + filename);
        
        points.D = D;
        points.features.resize(N * D);  

        for (int i = 0; i < N; i++) {
            const json& entry = j[i];
            if (!entry.contains("features") || !entry["features"].is_array()) {
                throw std::runtime_error("Entry " + std::to_string(i) + " missing 'features' array in file: " + filename);
            }

            std::vector<double> feats = entry["features"].get<std::vector<double>>();
            if (static_cast<int>(feats.size()) != D) {
                throw std::runtime_error("Inconsistent feature dimension at entry " + std::to_string(i) + " in file: " + filename);
            }

            points.ids[i] = entry.contains("id") ? entry["id"].get<int>() : i;
            
            for (int d = 0; d < D; d++) {
                points.setFeature(i, d, feats[d]);
            }
            
            points.clusters[i] = -1;

            // optional ground-truth label: accept "cluster" or "label"
            if (entry.contains("cluster") && entry["cluster"].is_number_integer()) {
                points.ground_truths[i] = entry["cluster"].get<int>();
            } else if (entry.contains("label") && entry["label"].is_number_integer()) {
                points.ground_truths[i] = entry["label"].get<int>();
            } else {
                points.ground_truths[i] = -1;
            }
        }
    }

    void cluster(int num_clusters, int max_iterations = 100, int min_iterations = 0) {
        K = num_clusters;

        initializeCentroids();

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            bool changed = false;

            for (int i = 0; i < N; i++) {
                const double* point_features = points.getFeaturePtr(i);
                double minDist = std::numeric_limits<double>::max();
                int nearestCluster = -1;
                
                for (int j = 0; j < K; j++) {
                    double dist = calculateDistanceSquared(point_features, centroids[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        nearestCluster = j;
                    }
                }
                
                if (points.clusters[i] != nearestCluster) {
                    points.clusters[i] = nearestCluster;
                    changed = true;
                }
            }

            std::vector<int> clusterCounts(K, 0);
            std::vector<std::vector<double>> newCentroids(K, std::vector<double>(D, 0.0));
            
            for (int i = 0; i < N; i++) {
                int c = points.clusters[i];
                clusterCounts[c]++;
                for (int d = 0; d < D; d++) {
                    newCentroids[c][d] += points.getFeature(i, d);
                }
            }
            
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
            std::cout << "Point " << points.ids[i] << " -> Cluster " << points.clusters[i] << "\n";
        }
    }

    // Sum of squared errors (inertia)
    double computeSSE() const {
        double sse = 0.0;
        for (int i = 0; i < N; i++) {
            int c = points.clusters[i];
            if (c < 0 || c >= K) continue;
            const double* point_features = points.getFeaturePtr(i);
            sse += calculateDistanceSquared(point_features, centroids[c]);
        }
        return sse;
    }

    // Purity: requires ground-truth labels (returns -1.0 if absent)
    double computePurity() const {
        bool has_gt = false;
        for (int i = 0; i < N; i++) {
            if (points.ground_truths[i] != -1) {
                has_gt = true;
                break;
            }
        }
        if (!has_gt) return -1.0;

        std::vector<std::unordered_map<int, int>> counts(K);
        for (int i = 0; i < N; i++) {
            int c = points.clusters[i];
            if (c < 0 || c >= K) continue;
            if (points.ground_truths[i] == -1) continue;
            counts[c][points.ground_truths[i]]++;
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

            int minIters = (dataset.find("N200") != std::string::npos) ? 16000 : 12000;
            int maxIters = std::max(4 * minIters, 40000);

            auto t_start_cluster = std::chrono::high_resolution_clock::now();
            kmeans.cluster(K, /*max_iterations=*/maxIters, /*min_iterations=*/minIters);
            auto t_end_cluster = std::chrono::high_resolution_clock::now();

            kmeans.printResults();

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