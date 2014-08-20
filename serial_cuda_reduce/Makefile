NVCC := nvcc
FLAGS := -g -G
LDFLAGS := -I/opt/cuda/include -L/opt/cuda/lib64
 

# Target rules
all: build

build: multiGPU multiGPU_sol

multiGPU: multiGPU.cu
	$(NVCC) $(FLAGS) $(LDFLAGS) -o $@ $< -lnvToolsExt

multiGPU_sol: multiGPU_sol.cu
	$(NVCC) $(FLAGS) $(LDFLAGS) -o $@ $< -lnvToolsExt

run: build
	./multiGPU

clean:
	rm -f multiGPU
