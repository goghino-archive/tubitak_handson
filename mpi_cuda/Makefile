NVCC := nvcc
cc := mpicc
FLAGS := -g
INCLUDE := -I/opt/cuda/include
LDFLAGS := -L/opt/cuda/lib64
CFLAGS := -std=c99 -Wall
 

# Target rules
all: build

build: cudaMPI

cudaMPI: cudaAwareMPI.c
	$(cc) $(CFLAGS) $(INCLUDE) $(FLAGS) $(LDFLAGS) -o $@ $< -lcudart

run: build
	export MPICH_RDMA_ENABLED_CUDA=1 && mpirun -np 4 ./cudaMPI

clean:
	rm -f cudaMPI

