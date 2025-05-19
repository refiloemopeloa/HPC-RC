#pragma once
#include <vector>
#include <mpi.h>

struct Centroid {
    std::vector<float> coordinates;
};

// Communication
void sendCentroids(const std::vector<Centroid>& centroids, int dest);
std::vector<Centroid> receiveCentroids(int src);

// Server functions
void initializeRandomCentroids(std::vector<Centroid>& centroids, int dim = 784);  // Default here
std::vector<Centroid> averageCentroids(const std::vector<std::vector<Centroid>>& all_centroids);

// Worker functions
std::vector<float> loadWorkerData(int rank);
std::vector<Centroid> localKMeans(const std::vector<float>& data, 
                                 const std::vector<Centroid>& centroids,
                                 int dim = 784);  // Default here