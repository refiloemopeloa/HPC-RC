COMPILED = main
LINKS = include/vec3_gpu.cu include/ray_gpu.cu include/hitable_gpu.cu include/hitable_list_gpu.cu include/sphere_gpu.cu include/camera_gpu.cu
GENCODE_FLAGS = -gencode arch=compute_86,code=sm_86

# select one of these for Debug vs. Release
# NVCC_DBG       = -g -G
NVCC_DBG       =


CPU = main_cpu
GPU = main_gpu
MAINS = $(CPU).cu $(GPU).cu

gpu: $(GPU).cu
	nvcc $(NVCC_DBG) -m64 -Xcompiler -fopenmp $(GENCODE_FLAGS) -o $(GPU) $(GPU).cu

cpu: $(CPU).cu
	nvcc $(NVCC_DBG) -m64 -Xcompiler -fopenmp $(GENCODE_FLAGS) -o $(CPU) $(CPU).cu

main: cpu gpu
	make cpu 
	make gpu

all:
	make main

clean:
	rm -f $(COMPILED)