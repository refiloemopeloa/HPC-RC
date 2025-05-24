// main.cpp
#include "mpi.h"
#include <iostream>
#include "../helpers/common.h"  // Make sure this includes the function declarations
#include "../preprocess/preprocess.cpp"


// Forward declarations
void runServer(int world_size);
void runWorker(int rank);

bool dataFilesExist(int num_workers) {
    for (int i = 1; i <= num_workers; i++) {
        std::string filename = "worker_" + std::to_string(i) + "_images.bin";
        std::ifstream file(filename);
        if (!file.good()) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "ERROR: Need at least 2 processes (1 server + 1 worker)!" << std::endl;
            std::cerr << "Usage: mpirun -np <N> ./federated_kmeans (where N >= 2)" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    const int num_workers = size - 1;

    // Only rank 0 handles preprocessing
    if (rank == 0) {
        std::cout << "Checking if preprocessed data exists..." << std::endl;
        if (!dataFilesExist(num_workers)) {
            std::cout << "Preprocessing MNIST data for " << num_workers << " workers..." << std::endl;
            if (preprocessData(num_workers) != 0) {
                std::cerr << "ERROR: Failed to preprocess data!" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::cout << "Preprocessing completed successfully." << std::endl;
        } else {
            std::cout << "Using existing preprocessed data files." << std::endl;
        }
    }

    
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "Process " << rank << " of " << size << " started" << std::endl;

    if (rank == 0) {
        std::cout << "Starting server with " << num_workers << " workers" << std::endl;
        runServer(size);
    } else {
        std::cout << "Starting worker " << rank << std::endl;
        runWorker(rank);
    }

    std::cout << "Process " << rank << " finished" << std::endl;
    MPI_Finalize();
    return 0;
}