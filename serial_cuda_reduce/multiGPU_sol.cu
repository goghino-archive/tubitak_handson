/*
 *
 * Copyright (c) 2014 Juraj Kardos
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising 
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it freely,
 * without any restrictons.
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include "nvToolsExt.h"

// NOTE: chose of grid/block config is
// also limited by device properties:
// - Maximum number of threads per block (512)
// - Maximum sizes of each dimension of a block (512 x 512 x 64)
// - Maximum sizes of each dimension of a grid (65535 x 65535 x 1)
#define PER_DEVICE_DATA 262144
#define BLOCK_N (PER_DEVICE_DATA/THREAD_N)
#define THREAD_N 512

typedef struct
{
    //pointer to data, device and pinned host
    float *data_d;
    float *data_h;

    float *sum_d;
    float *sum_h;

    //Stream for asynchronous command execution
    cudaStream_t stream;

} TGPUdata;

// Gets last error and prints message when error is present
void check_cuda_error(const char *message);

// Computes SUM reduction using shared memory
// Outputs partial sums for each block, need to further
// reduce them on CPU.
__global__ void
reduce(float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}



/*
*    Example of serial application using multiple GPUs
*   
*    Compute sum of (PER_DEVICE_DATA*GPU_cnt) elements
*    using 4 GPUs @tesla-cmc server
*         
*/
int main(int argc, char **argv)
{
    //Get all available CUDA-capable GPUs
    int GPU_cnt;
    cudaGetDeviceCount(&GPU_cnt);
    check_cuda_error("Get device count error!");
    printf("CUDA-capable device count: %d\n", GPU_cnt);

    TGPUdata *gpuData = (TGPUdata *)malloc(sizeof(TGPUdata)*GPU_cnt);

    //initialize random data
    float *data = (float *)malloc(sizeof(float) * GPU_cnt*PER_DEVICE_DATA);
    printf("Total size of data: %e\n\n", (double)GPU_cnt*PER_DEVICE_DATA);

    for(int i=0; i<GPU_cnt*PER_DEVICE_DATA; i++)
    {
        //data[i] = (float)rand() / (float)RAND_MAX; // result = ???
        data[i] = 0.5f; //result = (PER_DEVICE_DATA*GPU_cnt)*0.5
    }

    //Set up computation
    for (int i = 0; i < GPU_cnt; i++)
    {
        //select current device
        cudaSetDevice(i);
        check_cuda_error("Setting up device!");

        //create cuda stream for current device
        cudaStreamCreate(&gpuData[i].stream);
        check_cuda_error("Create stream");

        //Allocate memory
        cudaMalloc((void **)&gpuData[i].data_d, PER_DEVICE_DATA * sizeof(float));
        check_cuda_error("Device memory allocation");
        cudaMalloc((void **)&gpuData[i].sum_d, BLOCK_N*sizeof(float));
        check_cuda_error("Device memory allocation");

        cudaMallocHost((void **)&gpuData[i].data_h, PER_DEVICE_DATA* sizeof(float));
        check_cuda_error("Host page-locked memory allocation");
        cudaMallocHost((void **)&gpuData[i].sum_h, BLOCK_N*sizeof(float));
        check_cuda_error("Host page-locked memory allocation");

        //copy our data to pinned host memory
        for (int j=0; j<PER_DEVICE_DATA; j++)
        {
            gpuData[i].data_h[j] = data[i*PER_DEVICE_DATA+j];
        }

    }

    //Copy data to GPU, launch the kernel and copy data back. All asynchronously
    for (int i = 0; i < GPU_cnt; i++)
    {
        //Set device
        cudaSetDevice(i);
        check_cuda_error("Setting up device!");

        //Copy input data from pinned host memory
        cudaMemcpyAsync(gpuData[i].data_d, gpuData[i].data_h, PER_DEVICE_DATA*sizeof(float),
                        cudaMemcpyHostToDevice, gpuData[i].stream);
        check_cuda_error("Copying data H2D");

        //Perform GPU computations
        int smemSize =  THREAD_N * sizeof(float);
        reduce<<<BLOCK_N, THREAD_N, smemSize, gpuData[i].stream>>>(gpuData[i].data_d, gpuData[i].sum_d, PER_DEVICE_DATA);
        check_cuda_error("Kernel execution failed.\n");

        //Copy back GPU results
        cudaMemcpyAsync(gpuData[i].sum_h, gpuData[i].sum_d, THREAD_N  *sizeof(float),
                        cudaMemcpyDeviceToHost, gpuData[i].stream);
        check_cuda_error("Copying data D2H");
    }

    float globalSUM = 0;

    //Process GPU results
    for (int i = 0; i < GPU_cnt; i++)
    {
        float sum;

        //Set device
        cudaSetDevice(i);
        check_cuda_error("Setting up device!");

        //Wait for all operations in device's stream to finish
        cudaStreamSynchronize(gpuData[i].stream);
        check_cuda_error("Synchronization error");

        //Finalize GPU reduction for current subvector
        sum = 0;

        for (int j = 0; j < BLOCK_N; j++)
        {
            sum += gpuData[i].sum_h[j];
        }

        printf("GPU%d result is %.2f\n", i, sum);
        globalSUM += sum;
    }

    printf("Global result is %f\n", globalSUM);

    // Cleanup and shutdown
    for (int i = 0; i < GPU_cnt; i++)
    {
        //Set device
        cudaSetDevice(i);
        check_cuda_error("Setting up device!");

        //Shut down this GPU
        cudaFreeHost(gpuData[i].data_h);
        check_cuda_error("Free host memory");
        cudaFreeHost(gpuData[i].sum_h);
        check_cuda_error("Free host memory");
        cudaFree(gpuData[i].data_d);
        check_cuda_error("Free device memory");
        cudaFree(gpuData[i].sum_d);
        check_cuda_error("Free device memory");
        cudaStreamDestroy(gpuData[i].stream);
        check_cuda_error("Stream destroy");

        cudaDeviceReset();
    }
    

    //no longer needed
    free(data);
    free(gpuData);
}

void check_cuda_error(const char *message)
{
        cudaError_t err = cudaGetLastError();
            if (err!=cudaSuccess){
             printf("\033[31mERROR: %s: %s\n\033[0m", message, cudaGetErrorString(err));
             exit(1);
            }
}
