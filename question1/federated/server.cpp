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

void runServer(int num_workers, int max_rounds = 30) {
    FederatedLogisticRegression global_model;
    
    std::cout << "=== Starting Federated Logistic Regression ===" << std::endl;
    global_model.printModelInfo();
    
    
    std::vector<std::vector<float>> test_images;
    std::vector<uint8_t> test_labels;
    try {
        std::tie(test_images, test_labels) = loadTestData();
        std::cout << "Server loaded test data: " << test_images.size() << " samples" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not load test data: " << e.what() << std::endl;
    }
    
    
    std::vector<float> accuracy_history;
    float best_accuracy = 0.0f;
    int patience = 5; 
    int no_improvement_count = 0;
    float min_improvement = 0.005f;
    int min_rounds = 10;  // Minimum rounds before considering convergence
    
    std::cout << "Convergence criteria:" << std::endl;
    std::cout << "- Patience: " << patience << " rounds" << std::endl;
    std::cout << "- Min improvement: " << (min_improvement * 100) << "%" << std::endl;
    
    bool converged = false;
    int round = 0;
    
    for (round = 0; round < max_rounds; round++) {
        std::cout << "\n--- Round " << (round + 1) << "/" << max_rounds << " ---" << std::endl;
        
        // Serialize and broadcast global model weights
        std::vector<float> global_weights = global_model.serializeWeights();
        int size = (int)global_weights.size();
        
        for (int worker = 1; worker <= num_workers; worker++) {
            MPI_Send(&size, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            MPI_Send(global_weights.data(), size, MPI_FLOAT, worker, 1, MPI_COMM_WORLD);
        }
        
        //Collect worker updates
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
        
        //Aggregate worker models into global model
        global_model.averageWeights(worker_models, worker_data_sizes);
        
        // Evaluate model accuracy
        float current_accuracy = 0.0f;
        if (!test_images.empty() && !test_labels.empty()) {
            current_accuracy = global_model.evaluateAccuracy(test_images, test_labels);
            accuracy_history.push_back(current_accuracy);
            
            std::cout << "Global model accuracy after round " << (round + 1) 
                      << ": " << (current_accuracy * 100) << "%" << std::endl;
            
            // Check for improvement
            bool improved = current_accuracy > best_accuracy + min_improvement;
            if (improved) {
                best_accuracy = current_accuracy;
                no_improvement_count = 0;
                std::cout << "*** New best accuracy: " << (best_accuracy * 100) << "%" << std::endl;
            } else {
                no_improvement_count++;
                std::cout << "No significant improvement (" << no_improvement_count 
                          << "/" << patience << ")" << std::endl;
            }
            
            // Check convergence criteria BEFORE starting next round (only after minimum rounds)
            if (round >= min_rounds) {
                // Early stopping based on patience
                if (no_improvement_count >= patience) {
                    std::cout << "\n*** CONVERGED: No improvement for " << patience 
                              << " consecutive rounds ***" << std::endl;
                    converged = true;
                }
                
                //Check accuracy plateau (last 3 rounds within small range)
                if (accuracy_history.size() >= 3 && !converged) {
                    auto recent = accuracy_history.end() - 3;
                    float min_recent = *std::min_element(recent, accuracy_history.end());
                    float max_recent = *std::max_element(recent, accuracy_history.end());
                    
                    if (max_recent - min_recent < 0.002f) {  
                        
                        converged = true;
                    }
                }
                
                if (converged) {
                    break;
                }
            }
        } else {
            std::cout << "Skipping evaluation - no test data available" << std::endl;
        }
        
        // Print convergence status for next round
        if (!converged && round >= min_rounds) {
            std::cout << "Convergence status: " << no_improvement_count << "/" << patience 
                      << " rounds without improvement" << std::endl;
        }
    }
    

    for (int worker = 1; worker <= num_workers; worker++) {
        // Send a special termination signal (size = -1)
        int termination_signal = -1;
        MPI_Send(&termination_signal, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
    }
    
    std::cout << "\n=== FEDERATED TRAINING COMPLETED ===" << std::endl;
    std::cout << "Total rounds: " << (round + 1) << "/" << max_rounds << std::endl;
    if (converged) {
        std::cout << "Status: CONVERGED" << std::endl;
    } else {
        std::cout << "Status: MAX ROUNDS REACHED" << std::endl;
    }
    std::cout << "Best accuracy: " << (best_accuracy * 100) << "%" << std::endl;
    std::cout << "[RESULT] Final accuracy: " << (best_accuracy * 100) << std::endl;
    
    // Print accuracy progression
    if (!accuracy_history.empty()) {
        std::cout << "\nAccuracy progression:" << std::endl;
        for (size_t i = 0; i < accuracy_history.size(); i++) {
            std::cout << "Round " << (i + 1) << ": " << (accuracy_history[i] * 100) << "%" << std::endl;
        }
    }
}