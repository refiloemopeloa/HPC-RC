#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <limits>
#include "../helpers/mnist_loader.h"  

class KMeans {
public:
    KMeans(int k, int max_iters=100) : k(k), max_iters(max_iters) {}

    void fit(const std::vector<float>& data, int dim, int samples) {
        // Initialize centroids randomly
        centroids.resize(k * dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, samples-1);
        
        for (int i = 0; i < k; i++) {
            int idx = dis(gen);
            std::copy(data.begin() + idx*dim, 
                     data.begin() + (idx+1)*dim,
                     centroids.begin() + i*dim);
        }

        // K-means iteration
        for (int iter = 0; iter < max_iters; iter++) {
            // Assign clusters
            std::vector<int> assignments(samples);
            for (int i = 0; i < samples; i++) {
                float min_dist = std::numeric_limits<float>::max();
                for (int c = 0; c < k; c++) {
                    float dist = 0;
                    for (int d = 0; d < dim; d++) {
                        float diff = data[i*dim + d] - centroids[c*dim + d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        assignments[i] = c;
                    }
                }
            }

            // Update centroids
            std::vector<float> new_centroids(k * dim, 0);
            std::vector<int> counts(k, 0);
            for (int i = 0; i < samples; i++) {
                int c = assignments[i];
                for (int d = 0; d < dim; d++) {
                    new_centroids[c*dim + d] += data[i*dim + d];
                }
                counts[c]++;
            }

            // Normalize
            for (int c = 0; c < k; c++) {
                if (counts[c] > 0) {
                    for (int d = 0; d < dim; d++) {
                        centroids[c*dim + d] = new_centroids[c*dim + d] / counts[c];
                    }
                }
            }
        }
    }

    const std::vector<float>& getCentroids() const { return centroids; }

private:
    int k;
    int max_iters;
    std::vector<float> centroids;
};

int main() {
    // Load and normalize MNIST
    auto images = MNISTLoader::loadImages("../data/train-images.idx3-ubyte");
    const int dim = 28*28;
    const int samples = images.size() / dim;
    std::vector<float> data(images.size());
    for (size_t i = 0; i < images.size(); i++) {
        data[i] = images[i] / 255.0f;
    }

    // Train K-means
    const int k = 10;  // MNIST has 10 classes
    KMeans kmeans(k);
    kmeans.fit(data, dim, samples);

    // Evaluate (simple inertia)
    float inertia = 0;
    for (int i = 0; i < samples; i++) {
        float min_dist = std::numeric_limits<float>::max();
        for (int c = 0; c < k; c++) {
            float dist = 0;
            for (int d = 0; d < dim; d++) {
                float diff = data[i*dim + d] - kmeans.getCentroids()[c*dim + d];
                dist += diff * diff;
            }
            if (dist < min_dist) min_dist = dist;
        }
        inertia += min_dist;
    }
    std::cout << "Baseline K-means inertia: " << inertia << std::endl;

    return 0;
}