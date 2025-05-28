// server.cpp
#include "../helpers/common.h"
#include <limits>
#include <vector>
#include <iostream>
#include "mpi.h"
#include "../logisticRegression/logistic.cpp"

using namespace std;

/*void runServer(int world_size) {
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
}*/

void runServer(int num_workers, int num_rounds) {
    FederatedLogisticRegression global_model;
    
    std::cout << "=== Starting Federated Logistic Regression ===" << std::endl;
    global_model.printModelInfo();
    
    // Optionally, load test data once outside the loop
    std::vector<std::vector<float>> test_images;
    std::vector<uint8_t> test_labels;
    std::tie(test_images, test_labels) = loadTestData();

    try {
        // Implement loadTestData to return images and labels
        std::tie(test_images, test_labels) = loadTestData();
        std::cout << "Server loaded test data: " << test_images.size() << " samples" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not load test data: " << e.what() << std::endl;
    }
    
    for (int round = 0; round < num_rounds; round++) {
        std::cout << "\n--- Round " << (round + 1) << "/" << num_rounds << " ---" << std::endl;
        
        // Serialize and broadcast global model weights
        std::vector<float> global_weights = global_model.serializeWeights();
        int size = (int)global_weights.size();
        
        for (int worker = 1; worker <= num_workers; worker++) {
            MPI_Send(&size, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            MPI_Send(global_weights.data(), size, MPI_FLOAT, worker, 1, MPI_COMM_WORLD);
        }
        
        // Collect worker updates (weights + their data sizes)
        std::vector<std::vector<float>> worker_models;
        std::vector<int> worker_data_sizes;
        
        for (int worker = 1; worker <= num_workers; worker++) {
            int weights_size, data_size;
            MPI_Recv(&weights_size, 1, MPI_INT, worker, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&data_size, 1, MPI_INT, worker, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            std::vector<float> worker_weights(weights_size);
            MPI_Recv(worker_weights.data(), weights_size, MPI_FLOAT, worker, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            worker_models.push_back(worker_weights);
            worker_data_sizes.push_back(data_size);
            
            std::cout << "Received update from worker " << worker 
                      << " (data size: " << data_size << ")" << std::endl;
        }
        
        // Aggregate worker models into global model
        global_model.averageWeights(worker_models, worker_data_sizes);
        
        // Periodically evaluate model on test data if available
        if (!test_images.empty() && !test_labels.empty() && (round % 3 == 0 || round == num_rounds - 1)) {
            float accuracy = global_model.evaluateAccuracy(test_images, test_labels);
            std::cout << "Global model accuracy after round " << (round + 1) << ": " << (accuracy * 100) << "%" << std::endl;
            std::cout << "[RESULT] Final accuracy: " << (accuracy * 100) << std::endl;
        }
    }
    
    std::cout << "\nFederated training completed!" << std::endl;
}
