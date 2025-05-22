// worker.cpp
#include "../helpers/common.h"

// worker.cpp
#include "../helpers/common.h"

void runWorker(int rank) {
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
}