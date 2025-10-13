
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <random>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Point {
    int id;
    std::vector<double> features;
    int cluster;
};

class KMeans {
private:
    int K, D, N;
    std::vector<Point> points;
    std::vector<std::vector<double>> centroids;
    
    double calculateDistance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for(int i = 0; i < D; i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }

    void initializeCentroids() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, N-1);
        
        centroids.resize(K, std::vector<double>(D));
        for(int i = 0; i < K; i++) {
            int randomIndex = dis(gen);
            centroids[i] = points[randomIndex].features;
        }
    }

public:
    KMeans(const std::string& filename) {
        // Read JSON file
        std::ifstream file(filename);
        json j;
        file >> j;

        // Parse points
        N = j.size();
        D = j[0]["features"].size();
        points.resize(N);

        for(int i = 0; i < N; i++) {
            points[i].id = j[i]["id"];
            points[i].features = j[i]["features"].get<std::vector<double>>();
            points[i].cluster = -1;
        }
    }

    void cluster(int num_clusters, int max_iterations = 100) {
        K = num_clusters;
        
        // Initialize centroids
        initializeCentroids();
        
        bool changed = true;
        int iteration = 0;
        
        while(changed && iteration < max_iterations) {
            changed = false;
            
            // Assign points to nearest centroid
            for(int i = 0; i < N; i++) {
                double minDist = std::numeric_limits<double>::max();
                int nearestCluster = -1;
                
                for(int j = 0; j < K; j++) {
                    double dist = calculateDistance(points[i].features, centroids[j]);
                    if(dist < minDist) {
                        minDist = dist;
                        nearestCluster = j;
                    }
                }
                
                if(points[i].cluster != nearestCluster) {
                    points[i].cluster = nearestCluster;
                    changed = true;
                }
            }
            
            // Update centroids
            std::vector<int> clusterCounts(K, 0);
            std::vector<std::vector<double>> newCentroids(K, std::vector<double>(D, 0.0));
            
            for(int i = 0; i < N; i++) {
                int cluster = points[i].cluster;
                clusterCounts[cluster]++;
                for(int j = 0; j < D; j++) {
                    newCentroids[cluster][j] += points[i].features[j];
                }
            }
            
            for(int i = 0; i < K; i++) {
                if(clusterCounts[i] > 0) {
                    for(int j = 0; j < D; j++) {
                        centroids[i][j] = newCentroids[i][j] / clusterCounts[i];
                    }
                }
            }
            
            iteration++;
        }
    }

    void printResults() {
        std::cout << "Clustering Results:\n";
        for(int i = 0; i < N; i++) {
            std::cout << "Point " << points[i].id << " -> Cluster " << points[i].cluster << "\n";
        }
    }
};

int main() {
    // Example usage for different datasets
    std::vector<std::string> datasets = {
        "data_N200_D4_K8.json",
        "data_N200_D16_K8.json",
        "data_N800_D32_K16.json",
        "data_N800_D64_K16.json"
    };

    for(const auto& dataset : datasets) {
        std::cout << "\nProcessing " << dataset << std::endl;
        KMeans kmeans(dataset);
        
        int K = (dataset.find("N200") != std::string::npos) ? 8 : 16;
        kmeans.cluster(K);
        kmeans.printResults();
    }

    return 0;
}
