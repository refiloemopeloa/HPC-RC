#pragma once
#include <vector>
#include <mpi.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdint>

using namespace std;

/*struct Centroid {
    std::vector<float> values;

    // Optional constructor to initialize dimensions
    Centroid(int dim = 0) : values(dim) {}
};*/

struct Logistic {
    std::vector<float> weights;
    float bias;
};



void runServer(int world_size, int num_rounds);
void runWorker(int rank, int num_rounds);

// MNIST loading functions
int32_t read_int(std::ifstream& file);
std::vector<float> loadMNISTImages(const std::string& path);
std::vector<uint8_t> loadMNISTLabels(const std::string& path);

// Communication
/*void sendCentroids(const std::vector<Centroid>& centroids, int dest);
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
                  const std::vector<float>& test_data);*/

// --- Communication functions for logistic models ---
void sendModel(const Logistic& model, int dest);
Logistic receiveModel(int src);

//server stuff:
std::pair<std::vector<std::vector<float>>, std::vector<uint8_t>> loadTestData();

// --- Server functions ---
/*Logistic initializeRandomModel(int dim = 784);
Logistic averageModels(const std::vector<Logistic>& all_models);

// --- Evaluation functions ---
double calculateLoss(const Logistic& model, 
                     const std::vector<float>& data, 
                     const std::vector<uint8_t>& labels, 
                     int dim);
bool hasConverged(double prev_loss, double current_loss, double tolerance);
void evaluateModel(const Logistic& model, 
                   const std::vector<float>& test_data, 
                   const std::vector<uint8_t>& test_labels, 
                   int dim);*/



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
// In common.h or a suitable header
std::pair<std::vector<std::vector<float>>, std::vector<uint8_t>> loadWorkerData(int rank);
void deleteWorkerData(int num_workers, int rank);
std::vector<uint8_t> loadWorkerLabels(int rank);
/*Logistic localLogisticRegression(const std::vector<float>& data,
                                 const std::vector<uint8_t>& labels,
                                 const Logistic& initial_model,
                                 int dim = 784,
                                 int epochs = 10,
                                 float learning_rate = 0.01);*/