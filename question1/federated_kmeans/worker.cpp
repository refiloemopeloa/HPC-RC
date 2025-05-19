#include "../helpers/common.h"

// worker.cpp
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto data = loadWorkerData(rank); // Now implemented

    while (true) {
        auto centroids = receiveCentroids(0);
        auto updated = localKMeans(data, centroids); // Now implemented
        sendCentroids(updated, 0);
    }

    MPI_Finalize();
}