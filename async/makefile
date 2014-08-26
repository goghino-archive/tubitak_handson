NVCC := nvcc
FLAGS := -g -G 
LDFLAGS := -I/opt/cuda/include -L/opt/cuda/lib64

# Target rules
all: build

build: async async_multi async_multi_sol

async: async.cu
	$(NVCC) $(FLAGS) $(LDFLAGS) -o $@ $< -lnvToolsExt

async_multi_sol: async_multi_sol.cu
	$(NVCC) $(FLAGS) $(LDFLAGS) -o $@ $< -lnvToolsExt

async_multi: async_multi.cu
	$(NVCC) $(FLAGS) $(LDFLAGS) -o $@ $< -lnvToolsExt

run: 
	./async

clean:
	rm -f async

