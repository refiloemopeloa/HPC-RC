#!/bin/bash

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <num_processes>"
    exit 1
fi

NUM_PROCS=$1
EXECUTABLE=./federated

make -C federated   # Build in federated folder

chmod +x federated/"$EXECUTABLE"

# Run mpirun from federated directory so relative paths line up
(cd federated && mpirun --oversubscribe -np "$NUM_PROCS" "./$EXECUTABLE")
