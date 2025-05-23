// main.cpp
#include "mpi.h"
#include <iostream>
#include "../helpers/common.h"  // Make sure this includes the function declarations
#include "../preprocess/preprocess.cpp"

// Forward declarations
void runServer(int world_size);
void runWorker(int rank);

//declare the preprocess function here to create the files = world rank.
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Process " << rank << " of " << size << " started" << std::endl;
    kmeans(size+1);
    if (size < 2) {
        if (rank == 0) {
            std::cerr << "ERROR: Need at least 2 processes (1 server + 1 worker)!" << std::endl;
            std::cerr << "Usage: mpirun -np <N> ./federated (where N >= 2)" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "Starting server with " << (size-1) << " workers" << std::endl;
        runServer(size);
    } else {
        std::cout << "Starting worker " << rank << std::endl;
        runWorker(rank);
    }

    std::cout << "Process " << rank << " finished" << std::endl;
    MPI_Finalize();
    return 0;
}