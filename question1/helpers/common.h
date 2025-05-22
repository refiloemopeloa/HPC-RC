#pragma once
#include <vector>
#include <mpi.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdint>

using namespace std;

struct Centroid {
    std::vector<float> values;

    // Optional constructor to initialize dimensions
    Centroid(int dim = 0) : values(dim) {}
};

// Forward declarations for main functions
void runServer(int world_size);
void runWorker(int rank);

// MNIST loading functions
int32_t read_int(std::ifstream& file);
std::vector<float> loadMNISTImages(const std::string& path);
std::vector<uint8_t> loadMNISTLabels(const std::string& path);

// Communication
void sendCentroids(const std::vector<Centroid>& centroids, int dest);
std::vector<Centroid> receiveCentroids(int src);

// Server functions
void initializeRandomCentroids(std::vector<Centroid>& centroids, int dim = 784);
std::vector<Centroid> averageCentroids(const std::vector<std::vector<Centroid>>& all_centroids);

// Evaluation Functions
std::vector<float> loadTestData();
double calculateLoss(const std::vector<Centroid>& global_centroids, 
                   const std::vector<std::vector<Centroid>>& worker_updates);
bool hasConverged(double prev_loss, double current_loss, double tolerance);
void evaluateModel(const std::vector<Centroid>& centroids, 
                  const std::vector<float>& test_data);

// Termination Function
void sendTerminationSignal(int worker_rank);

inline void checkMPIError(int rc, const std::string& context) {
    if (rc != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(rc, error_string, &length);
        std::cerr << "MPI Error in " << context << ": " << error_string << std::endl;
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
}

// Worker functions
std::vector<float> loadWorkerData(int rank);
std::vector<Centroid> localKMeans(const std::vector<float>& data, 
                                 const std::vector<Centroid>& centroids,
                                 int dim = 784);