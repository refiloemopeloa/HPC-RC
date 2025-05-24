#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include "../helpers/mnist_loader.h"

class CentralizedKMeans {
public:
    CentralizedKMeans(int k, int max_iters=100, double tolerance=1e-5) 
        : k(k), max_iters(max_iters), tolerance(tolerance) {}

    void fit(const std::vector<float>& data, int dim, int samples) {
        centroids.resize(k * dim);
        
        // Initialize centroids using k-means++
        initializeCentroidsKMeansPlusPlus(data, dim, samples);

        // K-means iteration
        double prev_inertia = std::numeric_limits<double>::max();
        
        for (int iter = 0; iter < max_iters; iter++) {
            // Assign clusters
            std::vector<int> assignments(samples);
            assignClusters(data, dim, samples, assignments);

            // Update centroids
            updateCentroids(data, dim, samples, assignments);
            
            // Check convergence
            double current_inertia = calculateInertia(data, dim, samples);
            if (std::abs(prev_inertia - current_inertia) < tolerance) {
                std::cout << "Centralized K-means converged at iteration " << iter << std::endl;
                break;
            }
            prev_inertia = current_inertia;
            
            if (iter % 10 == 0) {
                std::cout << "Centralized K-means iteration " << iter << ", inertia: " << current_inertia << std::endl;
            }
        }
    }

    double calculateInertia(const std::vector<float>& data, int dim, int samples) {
        double inertia = 0.0;
        
        for (int i = 0; i < samples; i++) {
            double min_dist = std::numeric_limits<double>::max();
            
            for (int c = 0; c < k; c++) {
                double dist = 0.0;
                for (int d = 0; d < dim; d++) {
                    double diff = data[i*dim + d] - centroids[c*dim + d];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            inertia += min_dist;
        }
        
        return inertia;
    }

    const std::vector<float>& getCentroids() const { return centroids; }

private:
    int k;
    int max_iters;
    double tolerance;
    std::vector<float> centroids;

    void initializeCentroidsKMeansPlusPlus(const std::vector<float>& data, int dim, int samples) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, samples-1);
        
        // Choose first centroid randomly
        int first_idx = dis(gen);
        std::copy(data.begin() + first_idx*dim, 
                 data.begin() + (first_idx+1)*dim,
                 centroids.begin());
        
        // Choose remaining centroids using k-means++ algorithm
        for (int c = 1; c < k; c++) {
            std::vector<double> distances(samples);
            double total_distance = 0.0;
            
            // Calculate distance to nearest existing centroid for each point
            for (int i = 0; i < samples; i++) {
                double min_dist = std::numeric_limits<double>::max();
                
                for (int existing_c = 0; existing_c < c; existing_c++) {
                    double dist = 0.0;
                    for (int d = 0; d < dim; d++) {
                        double diff = data[i*dim + d] - centroids[existing_c*dim + d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                    }
                }
                
                distances[i] = min_dist;
                total_distance += min_dist;
            }
            
            // Choose next centroid with probability proportional to squared distance
            std::uniform_real_distribution<double> uniform(0.0, total_distance);
            double target = uniform(gen);
            double cumulative = 0.0;
            
            for (int i = 0; i < samples; i++) {
                cumulative += distances[i];
                if (cumulative >= target) {
                    std::copy(data.begin() + i*dim, 
                             data.begin() + (i+1)*dim,
                             centroids.begin() + c*dim);
                    break;
                }
            }
        }
    }

    void assignClusters(const std::vector<float>& data, int dim, int samples, std::vector<int>& assignments) {
        for (int i = 0; i < samples; i++) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            
            for (int c = 0; c < k; c++) {
                double dist = 0.0;
                for (int d = 0; d < dim; d++) {
                    double diff = data[i*dim + d] - centroids[c*dim + d];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }
    }

    void updateCentroids(const std::vector<float>& data, int dim, int samples, const std::vector<int>& assignments) {
        std::vector<float> new_centroids(k * dim, 0.0f);
        std::vector<int> counts(k, 0);
        
        // Accumulate points for each centroid
        for (int i = 0; i < samples; i++) {
            int c = assignments[i];
            for (int d = 0; d < dim; d++) {
                new_centroids[c*dim + d] += data[i*dim + d];
            }
            counts[c]++;
        }

        // Normalize by count
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < dim; d++) {
                    centroids[c*dim + d] = new_centroids[c*dim + d] / counts[c];
                }
            }
        }
    }
};

int main() {
    try {
        // Load MNIST data
        auto images = MNISTLoader::loadImages("../data/train-images.idx3-ubyte");
        const int dim = 28*28;
        const int samples = images.size() / dim;
        
        // Convert to float and normalize
        std::vector<float> data(images.size());
        for (size_t i = 0; i < images.size(); i++) {
            data[i] = images[i] / 255.0f;
        }

        std::cout << "Loaded " << samples << " training samples" << std::endl;

        // Train centralized K-means
        const int k = 10;  // MNIST has 10 classes
        CentralizedKMeans kmeans(k, 100, 1e-5);
        
        std::cout << "Starting centralized K-means training..." << std::endl;
        kmeans.fit(data, dim, samples);

        // Calculate final inertia
        double final_inertia = kmeans.calculateInertia(data, dim, samples);
        std::cout << "=== Centralized Baseline Results ===" << std::endl;
        std::cout << "Final inertia: " << final_inertia << std::endl;
        std::cout << "Number of clusters: " << k << std::endl;
        std::cout << "Training samples: " << samples << std::endl;
        std::cout << "=================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}