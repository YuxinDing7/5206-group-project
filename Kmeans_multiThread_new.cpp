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
    int cluster;
    int ground_truth;
    Point(): id(-1), cluster(-1), ground_truth(-1) {}
};

class KMeansComprehensive {
private:
    int K = 0, D = 0, N = 0;
    std::vector<Point> points;
    std::vector<std::vector<double>> centroids;
    int num_threads = 0;
    int iteration;

    double calculateDistanceSquared(const std::vector<double>& a, const std::vector<double>& b) const {
        double sum = 0.0;
        for (int i = 0; i < D; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    void initializeCentroids() {
        std::mt19937 gen(42); // Fixed seed for reproducibility          
        std::uniform_int_distribution<> dis(0, N - 1);

        centroids.resize(K, std::vector<double>(D));
        for (int i = 0; i < K; i++) {
            int randomIndex = dis(gen);
            centroids[i] = points[randomIndex].features;
        }
    }

public:
    KMeansComprehensive(const std::string& filename, int num_threads = 0) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        json j;
        try {
            file >> j;
        } catch (const json::parse_error& e) {
            throw std::runtime_error(std::string("JSON parse error: ") + e.what());
        }

        if (!j.is_array()) {
            throw std::runtime_error("Expected top-level JSON array");
        }

        N = static_cast<int>(j.size());
        if (N == 0) {
            throw std::runtime_error("Empty dataset");
        }

        points.resize(N);

        for (int i = 0; i < N; i++) {
            const json& entry = j[i];
            if (!entry.contains("features") || !entry["features"].is_array()) {
                throw std::runtime_error("Entry missing 'features' array");
            }

            std::vector<double> feats = entry["features"].get<std::vector<double>>();
            if (i == 0) {
                D = static_cast<int>(feats.size());
                if (D == 0) throw std::runtime_error("Feature vector has zero length");
            } else if (static_cast<int>(feats.size()) != D) {
                throw std::runtime_error("Inconsistent feature dimension");
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

        if (num_threads <= 0) {
            this->num_threads = std::thread::hardware_concurrency();
        } else {
            this->num_threads = num_threads;
        }
    }

    // Single Thread
    long long clusterSingleThread(int num_clusters, int max_iterations = 100) {
        K = num_clusters;
        initializeCentroids();

        auto t_start = std::chrono::high_resolution_clock::now();
        for (iteration = 0; iteration < max_iterations; ++iteration) {
            bool changed = false;

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

            if (!changed) {
                break;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    }

    // OpenMP Multi-thread
    long long clusterOpenMP(int num_clusters, int num_threads_use, int max_iterations = 100) {
        K = num_clusters;
        initializeCentroids();
        
        omp_set_num_threads(num_threads_use);

        auto t_start = std::chrono::high_resolution_clock::now();

        for (iteration = 0; iteration < max_iterations; ++iteration) {
            bool changed = false;

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

            std::vector<int> clusterCounts(K, 0);
            std::vector<std::vector<double>> newCentroids(K, std::vector<double>(D, 0.0));

            #pragma omp parallel
            {
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

            for (int c = 0; c < K; c++) {
                if (clusterCounts[c] > 0) {
                    for (int d = 0; d < D; d++) {
                        centroids[c][d] = newCentroids[c][d] / clusterCounts[c];
                    }
                }
            }

            if (!changed) {
                break;
            }
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    }

    double computeSSE() const {
        double sse = 0.0;
        for (const auto& p : points) {
            if (p.cluster < 0 || p.cluster >= K) continue;
            sse += calculateDistanceSquared(p.features, centroids[p.cluster]);
        }
        return sse;
    }

    int getN() const { return N; }
    int getD() const { return D; }
    int getIteration() const { return iteration; }

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
    std::vector<std::string> datasets = {
        "D:\\nus-s1\\CEG5206 Algorithm\\group project\\5206-group-project\\data_N200_D4_K8.json",
        //"D:\\nus-s1\\CEG5206 Algorithm\\group project\\5206-group-project\\data_N200_D16_K8.json",
        "D:\\nus-s1\\CEG5206 Algorithm\\group project\\5206-group-project\\data_N800_D32_K16.json",
        //"data_N800_D64_K16.json"
        "D:\\nus-s1\\CEG5206 Algorithm\\group project\\5206-group-project\\data_N200000_D4_K8.json",
        //"D:\\nus-s1\\CEG5206 Algorithm\\group project\\5206-group-project\\data_N500000_D4_K8.json",
    };

    std::vector<int> thread_counts = {2, 4, 8};

    // CSV
    std::ofstream csv_file("benchmark_results.csv");
    csv_file << "Dataset,N,D,K,Single_Thread_us,Single_Thread_SSE,Single_Thread_Purity,";
    for (int t : thread_counts) {
        csv_file << "OpenMP_" << t << "T_us,";
        csv_file << "Speedup_" << t << "T,";
        csv_file << "SSE_" << t << "T,";
        csv_file << "Purity_" << t << "T,";
        csv_file << "Iterations_" << t << "T,";
    }
    csv_file << "\n";

    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════════\n";
    std::cout << "K-means Algorithm Testing (Multi-thread)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════════\n";

    for (const auto& dataset : datasets) {
        std::cout << "\nProcessing " << dataset << std::endl;
        std::cout << "───────────────────────────────────────────────────────────────────────────\n";
        
        try {
            KMeansComprehensive kmeans(dataset);
            
            int K = 0;
            if(dataset.find("K8") != std::string::npos)
                K = 8;
            else if(dataset.find("K16") != std::string::npos)
                K = 16;
            else
                throw std::runtime_error("Cannot determine K from filename.");
            int max_iters = 300;

            // Single-thread benchmark
            long long single_time = kmeans.clusterSingleThread(K, max_iters);
            double sse = kmeans.computeSSE();
            double purity = kmeans.computePurity();
            int iterations = kmeans.getIteration();

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Single-threaded:  " << single_time << " us\n";
            std::cout << "SSE:              " << sse << "\n";
            std::cout << "Purity:           " << purity << "\n";
            std::cout << "Iterations:       " << iterations << "\n\n";

            csv_file << dataset.substr(0, dataset.find(".json")) << ","
                     << kmeans.getN() << "," << kmeans.getD() << "," << K << ","
                     << single_time << "," << sse << "," << purity << "," << iterations << ",";
            std::vector<long long> omp_times;
            std::vector<double> speedups;
            std::vector<double> sse_values;
            std::vector<double> purity_values;
            std::vector<int> iterations_values;

            // Multi-thread OpenMP benchmark
            std::cout << "OpenMP Results:\n";
            for (int t : thread_counts) {
                KMeansComprehensive kmeans_omp(dataset);
                long long omp_time = kmeans_omp.clusterOpenMP(K, t, max_iters);
                omp_times.push_back(omp_time);
                
                double speedup = static_cast<double>(single_time) / omp_time;
                speedups.push_back(speedup);

                double omp_sse = kmeans_omp.computeSSE();
                sse_values.push_back(omp_sse);

                double omp_purity = kmeans_omp.computePurity();
                purity_values.push_back(omp_purity);

                std::cout << "  " << std::setw(2) << t << " threads: " 
                          << std::setw(10) << omp_time << " us  (Speedup: " 
                          << std::setw(6) << speedup << "x)" << std::setw(6)
                          << ", SSE: " << omp_sse << std::setw(6)
                          << ", Purity: " << omp_purity << std::setw(6)
                          << ", Iterations: " << kmeans_omp.getIteration() << std::setw(6)
                          << "\n";

                csv_file << omp_time << ",";
            }

            // Add speedups to CSV
            for (double s : speedups) {
                csv_file << s << ",";
            }
            for (double sse : sse_values) {
                csv_file << sse << ",";
            }
            for (double purity : purity_values) {
                csv_file << purity << ",";
            }
            for (int iter : iterations_values) {
                csv_file << iter << ",";
            }
            csv_file << "\n";

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
    }

    csv_file.close();

    std::cout << "\n═══════════════════════════════════════════════════════════════════════════\n";
    std::cout << "✓ Results saved to benchmark_results.csv\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════════\n\n";

    return 0;
}
