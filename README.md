# HPC-RC
Federated Clustering using MPI and CUDA-accelerated Ray Tracing

**HOW TO RUN QUESTION 1**
1. cd question1
2. ./run.sh n processes (assumes there is no need to --oversubscribe)
3. In the case of Makefile: 
    3.1 cd federated (you should now be in question1/federated)
    3.2 **BUILD**: make
    3.3 **RUN**: make run NP=N
4. To manually compile and run, cd federated, there is a compile.txt with instructions.
