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

#include <stdio.h>

#include <cuda_runtime.h>

#include "nvToolsExt.h"

#define n (16*1014*1024)

__global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx > n/4)
      return;

    //to be more computational intensive, repeat same task multiple times
    for(int i=0; i<30; i++)
        g_data[idx] = g_data[idx] + inc_value;
}

#define checkCudaErrors(cuda_call)  \
{  \
        cudaError_t err = (cuda_call);  \
            if (err!=cudaSuccess){  \
             printf("\033[31mERROR: %s\n\033[0m", cudaGetErrorString(err));  \
             exit(1);  \
            }  \
}


int main(int argc, char *argv[])
{
    int nbytes = n * sizeof(int);
    int value = 26;

    // allocate host memory
    int *a = 0;
    checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
    memset(a, 0, nbytes);

    // allocate device memory
    int *d_a=0;
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    dim3 threads = dim3(512, 1);
    dim3 blocks  = dim3(n / threads.x, 1);

    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaDeviceSynchronize());

    //multi GPU related set-up
    int ndevices = 0;
    cudaGetDeviceCount(&ndevices);

    int *d_as[ndevices];
    cudaEvent_t stop_ev[ndevices];
    // create cuda streams for each device
    cudaStream_t stream_multi[4];

    for(int i=0; i<ndevices; i++)
    {
        //select current device
        cudaSetDevice(i);

        //create cuda stream for current device
        cudaStreamCreate(&stream_multi[i]);

        // allocate device memory
        checkCudaErrors(cudaMalloc((void **)&d_as[i], nbytes));
        checkCudaErrors(cudaMemset(d_as[i], 255, nbytes/4));

	    //create events
	    cudaEventCreate(&stop_ev[i]);
    }

    cudaSetDevice(0);
//------------------------------------------------------------------------------

    cudaEventRecord(start, 0);
    cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice);
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaEventRecord(stop, 0);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter=0;

    nvtxRangePushA("CPU Compute");
    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }
    nvtxRangePop();
    cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();

    float gpu_time_block = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&gpu_time_block, start, stop));
    printf("One big kernel compute time (blocking): %fms\n", gpu_time_block);

//------------------------------------------------------------------------------
    // asynchronously copy data, run kernel and copy back
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);

    // have CPU do some work while waiting for stage 1 to finish
    counter=0;

    nvtxRangePushA("CPU Compute");
    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }
    nvtxRangePop();

    float gpu_time = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("One big kernel compute time (async): %fms\n", gpu_time);

//------------------------------------------------------------------------------
// run kernel on partial data multiple times, overlap computation and communication

    // set kernel launch configuration
    threads = dim3(512, 1, 1);
    blocks  = dim3(n / 4 / threads.x, 1, 1);

    // create cuda streams
    cudaStream_t stream[4];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaStreamCreate(&stream[3]);
    cudaStreamCreate(&stream[2]);
    checkCudaErrors(cudaDeviceSynchronize());

    int offset = n/4;

    // asynchronously copy data, run kernel and copy back
    cudaEventRecord(start, stream[0]);
    cudaMemcpyAsync(d_a, a, nbytes/4, cudaMemcpyHostToDevice, stream[0]);
    increment_kernel<<<blocks, threads, 0, stream[0]>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes/4, cudaMemcpyDeviceToHost, stream[0]);

    cudaMemcpyAsync(d_a+offset, a+offset, nbytes/4, cudaMemcpyHostToDevice, stream[1]);
    increment_kernel<<<blocks, threads, 0, stream[1]>>>(d_a+offset, value);
    cudaMemcpyAsync(a+offset, d_a+offset, nbytes/4, cudaMemcpyDeviceToHost, stream[1]);

    cudaMemcpyAsync(d_a+2*offset, a+2*offset, nbytes/4, cudaMemcpyHostToDevice, stream[2]);
    increment_kernel<<<blocks, threads, 0, stream[2]>>>(d_a+2*offset, value);
    cudaMemcpyAsync(a+2*offset, d_a+2*offset, nbytes/4, cudaMemcpyDeviceToHost, stream[2]);

    cudaMemcpyAsync(d_a+3*offset, a+3*offset, nbytes/4, cudaMemcpyHostToDevice, stream[3]);
    increment_kernel<<<blocks, threads, 0, stream[3]>>>(d_a+3*offset, value);
    cudaMemcpyAsync(a+3*offset, d_a+3*offset, nbytes/4, cudaMemcpyDeviceToHost, stream[3]);
    cudaEventRecord(stop, stream[2]);

    // have CPU do some work while waiting for stage 1 to finish
    counter=0;

    nvtxRangePushA("CPU Compute");
    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }
    nvtxRangePop();

    float gpu_time1 = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&gpu_time1, start, stop));
    printf("Many small kernels compute time: %fms\n", gpu_time1);
    printf("Speedup is %f\n", gpu_time/gpu_time1);

//------------------------------------------------------------------------------

    printf("Found %d CUDA capable devices\n", ndevices);    

    //submit work to GPU devices
    for(int i=0; i<ndevices; i++)
    {    
        cudaSetDevice(i);

        cudaMemcpyAsync(d_as[i], a+i*offset, nbytes/4, cudaMemcpyHostToDevice, stream_multi[i]);
	    increment_kernel<<<blocks, threads, 0, stream_multi[i]>>>(d_as[i], value);
        /*
	    cudaError_t errSync  = cudaGetLastError();
	    cudaError_t errAsync = cudaDeviceSynchronize();
	    if (errSync != cudaSuccess) 
	        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	    if (errAsync != cudaSuccess)
	        printf("Async kernel %d error: %s\n", i, cudaGetErrorString(errAsync));*/
        cudaMemcpyAsync(a+i*offset, d_as[i], nbytes/4, cudaMemcpyDeviceToHost, stream_multi[i]);
        cudaEventRecord(stop_ev[i], stream_multi[i]);
    }

    // have CPU do some work while waiting for stage 1 to finish
    counter=0;

    nvtxRangePushA("CPU Compute");
    while (cudaEventQuery(stop_ev[0]) == cudaErrorNotReady ||
           cudaEventQuery(stop_ev[1]) == cudaErrorNotReady ||
           cudaEventQuery(stop_ev[2]) == cudaErrorNotReady ||
           cudaEventQuery(stop_ev[3]) == cudaErrorNotReady )  
    {
        counter++;
    }

    /*
    for(int i=0; i<ndevices; i++)
    {
      cudaEventSynchronize(stop_ev[i]);  
    }
    */

    //cudaDeviceSynchronize();

    nvtxRangePop();

    //checkCudaErrors(cudaEventElapsedTime(&gpu_time1, start, stop));
    //printf("Many small kernels compute time: %fms\n", gpu_time1);
    //printf("Speedup is %f\n", gpu_time/gpu_time1);

//------------------------------------------------------------------------------
    // release resources
    cudaSetDevice(0);
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(stop_ev[0]));
    checkCudaErrors(cudaEventDestroy(stop_ev[1]));
    checkCudaErrors(cudaEventDestroy(stop_ev[2]));
    checkCudaErrors(cudaEventDestroy(stop_ev[3]));
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    cudaStreamDestroy(stream[3]);

    //free memory
    for(int i=0; i<ndevices; i++)
    {
        cudaSetDevice(i);
        cudaFree(d_as[i]);
        cudaStreamDestroy(stream_multi[i]);
    }

    // flush all profile data
    cudaDeviceReset();

}
