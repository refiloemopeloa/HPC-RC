#include "../helpers/common.h"
#include "../helpers/mnist_loader.h"  

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const int num_workers = world_size - 1;

    // 1. Initialize random centroids
    std::vector<Centroid> global_centroids(10); // 10 clusters for MNIST
    initializeRandomCentroids(global_centroids);

    for (int iter = 0; iter < 20; iter++) {
        // 2. Broadcast centroids to all workers
        for (int worker = 1; worker <= num_workers; worker++) {
            sendCentroids(global_centroids, worker);
        }

        // 3. Receive updated centroids from workers
        std::vector<std::vector<Centroid>> worker_updates;
        for (int worker = 1; worker <= num_workers; worker++) {
            worker_updates.push_back(receiveCentroids(worker));
        }

        // 4. Federated averaging
        global_centroids = averageCentroids(worker_updates);
    }

    MPI_Finalize();
}