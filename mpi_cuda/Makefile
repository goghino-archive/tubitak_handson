NVCC := nvcc
cc := gcc
FLAGS := -g
INCLUDE := -I/opt/cuda/include -I/opt/openmpi-r27467/include
LDFLAGS := -L/opt/cuda/lib64 -L/opt/openmpi-r27467/lib
CFLAGS := -std=c99 -Wall
 

# Target rules
all: build

build: cudaMPI

cudaMPI: cudaAwareMPI.c
	$(cc) $(CFLAGS) $(INCLUDE) $(FLAGS) $(LDFLAGS) -o $@ $< -lcudart -lmpi

run: build
	export MPICH_RDMA_ENABLED_CUDA=1 && mpirun -np 4 ./cudaMPI

clean:
	rm -f cudaMPI

