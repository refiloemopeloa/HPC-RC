#!/bin/bash

#this run.sh produces csv output for performance analysis

OUTPUT=federated
LOGFILE="performance_log.csv"

# Compile
echo "Compiling..."
mpic++ -g main.cpp server.cpp worker.cpp ../helpers/common.cpp -o $OUTPUT
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# CSV header
echo "Workers,Accuracy" > $LOGFILE

MIN_PROCS=2
MAX_PROCS=20

for ((np=$MIN_PROCS; np<=$MAX_PROCS; np++)); do
    echo -e "\nRunning with $np processes..."
    
    OUTPUT_FILE="output_${np}.txt"
    mpirun --oversubscribe -np $np ./$OUTPUT > $OUTPUT_FILE
    
    # Extract accuracy from output
    ACCURACY=$(grep "\[RESULT\] Final accuracy:" $OUTPUT_FILE | awk '{print $4}')
    WORKERS=$((np - 1))

    if [ -z "$ACCURACY" ]; then
        echo "Warning: No accuracy found in output for $np processes"
        ACCURACY="N/A"
    fi

    echo "$WORKERS,$ACCURACY" >> $LOGFILE
    echo "Workers: $WORKERS | Accuracy: $ACCURACY"
done

echo -e "\nDone! Logged results to $LOGFILE"
