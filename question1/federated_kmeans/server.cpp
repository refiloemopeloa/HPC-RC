// server.cpp
#include "../helpers/common.h"
#include <limits>
#include <vector>
#include <iostream>
#include "mpi.h"

using namespace std;

void runServer(int world_size) {
    const int num_workers = world_size - 1;

    const int max_iterations = 20;
    const double tolerance = 1e-5;

    vector<Centroid> global_centroids(10);
    initializeRandomCentroids(global_centroids, 784);

    std::vector<float> test_data;
    try {
        test_data = loadTestData();
        std::cout << "Server loaded test data: " << test_data.size() << " values" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not load test data: " << e.what() << std::endl;
    }

    double prev_loss = numeric_limits<double>::max();
    bool converged = false;

    std::cout << "Starting federated K-means with " << num_workers << " workers..." << std::endl;

    for (int iter = 0; iter < max_iterations && !converged; iter++) {
        // Send current centroids to all workers
        for (int worker = 1; worker <= num_workers; worker++) {
            sendCentroids(global_centroids, worker);
        }

        // Collect updated centroids from all workers
        vector<vector<Centroid>> worker_updates;
        worker_updates.reserve(num_workers);
        
        for (int worker = 1; worker <= num_workers; worker++) {
            try {
                auto update = receiveCentroids(worker);
                if (!update.empty()) {
                    worker_updates.push_back(update);
                } else {
                    std::cerr << "Warning: Received empty update from worker " << worker << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error receiving from worker " << worker << ": " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        if (worker_updates.empty()) {
            std::cerr << "ERROR: No valid updates received from workers!" << std::endl;
            break;
        }

        // Average the centroids
        global_centroids = averageCentroids(worker_updates);

        // Calculate and check convergence
        double current_loss = calculateLoss(global_centroids, worker_updates);
        if (iter > 0 && hasConverged(prev_loss, current_loss, tolerance)) {
            converged = true;
            std::cout << "Converged at iteration " << iter << std::endl;
        }
        prev_loss = current_loss;

        if (iter % 5 == 0 || converged) {
            cout << "Iteration " << iter << ", Loss: " << current_loss << endl;
        }
    }

    // Send termination signal to all workers
    std::cout << "Sending termination signals to workers..." << std::endl;
    for (int worker = 1; worker <= num_workers; worker++) {
        sendTerminationSignal(worker);
    }

    // Evaluate model if test data is available
    if (!test_data.empty()) {
        evaluateModel(global_centroids, test_data);
    } else {
        std::cout << "Skipping evaluation - no test data available" << std::endl;
    }
    
    std::cout << "Server finished successfully!" << std::endl;
}