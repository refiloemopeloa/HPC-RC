#include "common.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

// ---- Communication Functions ----
void sendCentroids(const std::vector<Centroid>& centroids, int dest) {
    // Flatten centroids into 1D float array
    std::vector<float> flat_data;
    for (const auto& c : centroids) {
        flat_data.insert(flat_data.end(), c.coordinates.begin(), c.coordinates.end());
    }

    // Send metadata (count) first
    int count = flat_data.size();
    MPI_Send(&count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

    // Send actual data
    MPI_Send(flat_data.data(), count, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);
}

std::vector<Centroid> receiveCentroids(int src) {
    // Receive count first
    int count;
    MPI_Recv(&count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Prepare buffer
    std::vector<float> flat_data(count);
    MPI_Recv(flat_data.data(), count, MPI_FLOAT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Reconstruct centroids (assuming 784D MNIST)
    const int dim = 784;
    std::vector<Centroid> centroids;
    for (int i = 0; i < count; i += dim) {
        Centroid c;
        c.coordinates.assign(flat_data.begin() + i, flat_data.begin() + i + dim);
        centroids.push_back(c);
    }
    return centroids;
}

// ---- Server Functions ----
void initializeRandomCentroids(std::vector<Centroid>& centroids, int dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    
    for (auto& c : centroids) {
        c.coordinates.resize(dim);
        for (float& val : c.coordinates) {
            val = dist(gen);
        }
    }
}

std::vector<Centroid> averageCentroids(const std::vector<std::vector<Centroid>>& all_centroids) {
    if (all_centroids.empty()) return {};
    
    std::vector<Centroid> avg(all_centroids[0].size());
    int num_workers = all_centroids.size();
    int dim = all_centroids[0][0].coordinates.size();

    for (int c = 0; c < avg.size(); c++) {
        avg[c].coordinates.resize(dim, 0.0f);
        for (int w = 0; w < num_workers; w++) {
            for (int d = 0; d < dim; d++) {
                avg[c].coordinates[d] += all_centroids[w][c].coordinates[d];
            }
        }
        for (int d = 0; d < dim; d++) {
            avg[c].coordinates[d] /= num_workers;
        }
    }
    return avg;
}

// ---- Worker Functions ----
std::vector<float> loadWorkerData(int rank) {
    // Path to preprocessed data (rank 0 = server, ranks 1+ = workers)
    std::string path = "../preprocess/worker_" + std::to_string(rank-1) + "_images.bin";
    
    // Debug output
    std::cout << "Worker " << rank << " loading: " << path << std::endl;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Worker " + std::to_string(rank) + 
                               " failed to open: " + path);
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Verify file size matches MNIST dimensions (28x28 = 784)
    const size_t expected_dim = 784;
    if (size % expected_dim != 0) {
        throw std::runtime_error("Invalid file size for worker " + 
                               std::to_string(rank));
    }

    std::vector<uint8_t> bytes(size);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("Worker " + std::to_string(rank) + 
                               " failed to read: " + path);
    }
    
    // Convert to normalized float
    std::vector<float> data(bytes.size());
    for (size_t i = 0; i < bytes.size(); i++) {
        data[i] = bytes[i] / 255.0f;
    }

    std::cout << "Worker " << rank << " loaded " 
              << data.size()/expected_dim << " samples" << std::endl;
    return data;
}

std::vector<Centroid> localKMeans(const std::vector<float>& data, 
                                const std::vector<Centroid>& centroids,
                                int dim) {
    std::vector<Centroid> updated = centroids;
    std::vector<int> counts(centroids.size(), 0);
    
    for (size_t i = 0; i < data.size(); i += dim) {
        float min_dist = std::numeric_limits<float>::max();
        int best_c = 0;
        
        for (int c = 0; c < centroids.size(); c++) {
            float dist = 0;
            for (int d = 0; d < dim; d++) {
                float diff = data[i+d] - centroids[c].coordinates[d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_c = c;
            }
        }
        
        for (int d = 0; d < dim; d++) {
            updated[best_c].coordinates[d] += data[i+d];
        }
        counts[best_c]++;
    }
    
    for (int c = 0; c < updated.size(); c++) {
        if (counts[c] > 0) {
            for (int d = 0; d < dim; d++) {
                updated[c].coordinates[d] /= counts[c];
            }
        }
    }
    
    return updated;
}