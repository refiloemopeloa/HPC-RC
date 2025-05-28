// main.cpp
#include "mpi.h"
#include <iostream>
#include "../helpers/common.h"  
#include "../preprocess/preprocess.cpp"
#include "../baseline/centralized.h"
#include "../baseline/centralized.cpp"


// Forward declarations
void runServer(int world_size, int num_rounds);
void runWorker(int rank, int num_rounds);

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

// Updated main.cpp section
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 1) {
        if (rank == 0) {
            std::cerr << "Need at least 2 processes (1 server + 1 worker) for Federated Learning!" << std::endl;
            std::cerr << "Usage: mpirun -np <N> ./federated (where N >= 2)" << std::endl;

            std::cout << "\nNow running centralized training with original MNIST data..." << std::endl;
            runCentralizedTraining(
                "../data/train-images.idx3-ubyte",
                "../data/train-labels.idx1-ubyte",
                "../data/t10k-images.idx3-ubyte",
                "../data/t10k-labels.idx1-ubyte",
                0.4f 
            );
        }
        MPI_Finalize();
        return 0;
    }

    const int num_workers = size - 1;
    
    // Configurable parameters
    const int max_rounds = 20;      
    const int min_rounds = 10;      
    
    if (rank == 0) {
        std::cout << "=== FEDERATED LEARNING CONFIGURATION ===" << std::endl;
        std::cout << "Workers: " << num_workers << std::endl;
        
    
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

    if (rank == 0 && size > 1) {
        std::cout << "Starting server with " << num_workers << " workers" << std::endl;
        runServer(num_workers, max_rounds);  
    } else {
        std::cout << "Starting worker " << rank << std::endl;
        runWorker(rank, max_rounds); 
    }

    std::cout << "Process " << rank << " finished" << std::endl;
    deleteWorkerData(num_workers, rank);
    MPI_Finalize();
    return 0;
}