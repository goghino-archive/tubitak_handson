NVCC := nvcc
FLAGS := -g -G 
LDFLAGS := -I/opt/cuda/include -L/opt/cuda/lib64

# Target rules
all: build

build: async

async: async.cu
	$(NVCC) $(FLAGS) $(LDFLAGS) -o $@ $< -lnvToolsExt

run: 
	./async

clean:
	rm -f async

