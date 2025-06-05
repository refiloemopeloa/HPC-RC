// worker.cpp
#include "../helpers/common.h"
#include "../logisticRegression/logistic.cpp"
#include <tuple>


/*void runWorker(int rank) {
    auto data = loadWorkerData(rank);
    const int dim = 784; // MNIST dimension

    while (true) {
        // First, try to receive centroids normally
        auto centroids = receiveCentroids(0);
        
        // Check if this is actually a termination signal
        // (we'll modify receiveCentroids to handle this)
        if (centroids.empty()) {
            break; // Termination signal received
        }
        
        auto updated = localKMeans(data, centroids, dim);
        sendCentroids(updated, 0);
    }

}*/

void runWorker(int worker_id, int max_rounds) {
    
    std::vector<std::vector<float>> images;
    std::vector<uint8_t> labels;
    std::tie(images, labels) = loadWorkerData(worker_id);

    int num_samples = images.size(); 
    
    // Shuffle data once at the beginning
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    std::vector<std::vector<float>> shuffled_images(num_samples);
    std::vector<uint8_t> shuffled_labels(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        shuffled_images[i] = images[indices[i]];
        shuffled_labels[i] = labels[indices[i]];
    }

    int split_idx = static_cast<int>(0.8 * num_samples);
    std::vector<std::vector<float>> train_images(shuffled_images.begin(), shuffled_images.begin() + split_idx);
    std::vector<uint8_t> train_labels(shuffled_labels.begin(), shuffled_labels.begin() + split_idx);

    std::vector<std::vector<float>> val_images(shuffled_images.begin() + split_idx, shuffled_images.end());
    std::vector<uint8_t> val_labels(shuffled_labels.begin() + split_idx, shuffled_labels.end());

    FederatedLogisticRegression local_model;
    
    std::cout << "Worker " << worker_id << " loaded " << num_samples << " samples" << std::endl;
    std::cout << "Training on " << train_images.size() << " samples, validating on " << val_images.size() << std::endl;
    
    for (int round = 0; round < max_rounds; round++) {
        // Receive size of global weights from server
        int size;
        MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Check for termination signal
        if (size == -1) {
            std::cout << "Worker " << worker_id << " received termination signal at round " << round << std::endl;
            break;
        }
        
        // Receive global weights
        std::vector<float> global_weights(size);
        MPI_Recv(global_weights.data(), size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Deserialize global weights into local model
        local_model.deserializeWeights(global_weights);
        
        // Adaptive learning: more epochs in early rounds
        int epochs_per_round = (round < 5) ? 5 : 3;
        
        // Train locally with adaptive epochs
        for (int epoch = 0; epoch < epochs_per_round; epoch++) {
            local_model.trainEpoch(train_images, train_labels, 64); // Larger batch size
        }
        
        // Evaluate local accuracy on worker's validation data
        float local_accuracy = local_model.evaluateAccuracy(val_images, val_labels);
        float train_accuracy = local_model.evaluateAccuracy(train_images, train_labels);
        
        std::cout << "Worker " << worker_id << " round " << round 
                  << " - Train acc: " << (train_accuracy * 100) << "%, Val acc: " << (local_accuracy * 100) << "%" << std::endl;
        
        // Serialize updated weights
        std::vector<float> updated_weights = local_model.serializeWeights();
        int weights_size = (int)updated_weights.size();
        
        // Send training data size (not total samples) for proper weighting
        int training_samples = train_images.size();
        
        // Send updated weights size and training sample count to server
        MPI_Send(&weights_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&training_samples, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        
        // Send updated weights data
        MPI_Send(updated_weights.data(), weights_size, MPI_FLOAT, 0, 4, MPI_COMM_WORLD);
    }
    
    std::cout << "Worker " << worker_id << " finished training" << std::endl;
}