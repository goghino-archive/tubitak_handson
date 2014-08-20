CPUCC=g++
CPUCFLAGS=
CPUEXECUTABLE=cpu
SRC=code

GPUCC=nvcc
GPUCFLAGS=-lcurand
GPUEXECUTABLE=gpu

all: mpi


mpi: $(SRC)/mpi_version.cpp $(SRC)/mpi_version.cu $(SRC)/mpi_version.h
	nvcc -c $(SRC)/mpi_version.cu -g -o mpi_gpu.o
	mpic++ -c $(SRC)/mpi_version.cpp -g -o mpi_cpu.o
	mpic++ -L/opt/cuda/lib64 -stdlib=libstdc++ mpi_gpu.o mpi_cpu.o -o $@ -lcurand -lcudart

run: mpi
	CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/home/jkardos/hands-on/mps/nvida-mps CUDA_MPS_LOG_DIRECTORY=/home/jkardos/hands-on/mps/nvidia-log mpirun -np 3 ./mpi input5.txt

clean:
	rm -rf mpi
	rm -rf *.o
