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

void runWorker(int worker_id, int num_rounds) {
    // Load images and labels from worker's dataset
    std::vector<std::vector<float>> images;
    std::vector<uint8_t> labels;
    std::tie(images, labels) = loadWorkerData(worker_id);

    int num_samples = images.size(); 

    FederatedLogisticRegression local_model;
    
    std::cout << "Worker " << worker_id << " loaded " << num_samples << " samples" << std::endl;
    
    for (int round = 0; round < num_rounds; round++) {
        // Receive size of global weights from server
        int size;
        MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Receive global weights
        std::vector<float> global_weights(size);
        MPI_Recv(global_weights.data(), size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Deserialize global weights into local model
        local_model.deserializeWeights(global_weights);
        
        // Train locally for 3 epochs
        for (int epoch = 0; epoch < 3; epoch++) {
            local_model.trainEpoch(images, labels, 32);
        }
        
        // Evaluate local accuracy on worker's data
        float local_accuracy = local_model.evaluateAccuracy(images, labels);
        std::cout << "Worker " << worker_id << " round " << round 
                  << " accuracy: " << (local_accuracy * 100) << "%" << std::endl;
        
        // Serialize updated weights
        std::vector<float> updated_weights = local_model.serializeWeights();
        int weights_size = (int)updated_weights.size();
        
        // Send updated weights size and sample count to server
        MPI_Send(&weights_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&num_samples, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        
        // Send updated weights data
        MPI_Send(updated_weights.data(), weights_size, MPI_FLOAT, 0, 4, MPI_COMM_WORLD);
    }
}
