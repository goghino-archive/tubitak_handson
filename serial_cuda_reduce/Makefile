NVCC := nvcc
FLAGS := -g -G
 

# Target rules
all: build

build: multiGPU

multiGPU: multiGPU.cu
	$(NVCC) $(FLAGS) -o $@ $<

run: build
	./multiGPU

clean:
	rm -f multiGPU
