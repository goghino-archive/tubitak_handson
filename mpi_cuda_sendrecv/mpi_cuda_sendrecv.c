/*
 * MSU CUDA Course Examples and Exercises.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising 
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it freely,
 * without any restrictons.
 */

#include "pattern2d.h"

#include <cuda_runtime.h>
#include <malloc.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[])
{
	// Initialize MPI. From this point the specified
	// number of processes will be executed in parallel.
	int mpi_status = MPI_Init(&argc, &argv);
	int mpi_error_msg_length;
	char mpi_error_msg[MPI_MAX_ERROR_STRING];
	if (mpi_status != MPI_SUCCESS)
	{
		MPI_Error_string(mpi_status, mpi_error_msg, &mpi_error_msg_length);
		fprintf(stderr, "Cannot initialize MPI, status = %s\n",
			mpi_error_msg);
		return 1;
	}
	
	// Get the size of the MPI global communicator,
	// that is get the total number of MPI processes.
	int nprocesses;
	mpi_status = MPI_Comm_size(MPI_COMM_WORLD, &nprocesses);
	if (mpi_status != MPI_SUCCESS)
	{
		MPI_Error_string(mpi_status, mpi_error_msg, &mpi_error_msg_length);
		fprintf(stderr, "Cannot retrieve the number of MPI processes, status = %s\n",
			mpi_error_msg);
		return 1;
	}
	
	// Get the rank (index) of the current MPI process
	// in the global communicator.
	int iprocess;
	mpi_status = MPI_Comm_rank(MPI_COMM_WORLD, &iprocess);
	if (mpi_status != MPI_SUCCESS)
	{
		MPI_Error_string(mpi_status, mpi_error_msg, &mpi_error_msg_length);
		fprintf(stderr, "Cannot retrieve the rank of current MPI process, status = %s\n",
			mpi_error_msg);
		return 1;
	}

	int ndevices = 0;
	cudaError_t cuda_status = cudaGetDeviceCount(&ndevices);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot get the cuda device count by process %d, status = %s\n",
			iprocess, cudaGetErrorString(cuda_status));
		return 1;
	}

	// Return if no cuda devices present.
	if (iprocess == 0)
		printf("%d CUDA device(s) found\n", ndevices);
	if (!ndevices) return 0;

	// Get problem size from the command line.
	if (argc != 3)
	{
		printf("Usage: %s <n> <npasses>\n", argv[0]);
		return 0;
	}
	
	int n = atoi(argv[1]);
	int npasses = atoi(argv[2]);
	size_t size = n * n * sizeof(float);
	
	if ((n <= 0) || (npasses <= 0)) return 0;

        // Assign unique device to each MPI process.
	cuda_status = cudaSetDevice(iprocess);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot set CUDA device by process d, status= %s\n",
			iprocess, cudaGetErrorString(cuda_status));
		return 1;
	}

	// Create two device input buffers.
	float *din1, *din2;
	cuda_status = cudaMalloc((void**)&din1, size);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate input device buffer by process %d, status = %s\n",
			iprocess, cudaGetErrorString(cuda_status));
		return 1;
	}
	cuda_status = cudaMalloc((void**)&din2, size);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate input device buffer by process %d, status = %s\n",
			iprocess, cudaGetErrorString(cuda_status));
		return 1;
	}
	
	// Create device output buffer.
	float* dout;
	cuda_status = cudaMalloc((void**)&dout, size);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate output device buffer by process %d, status = %s\n",
			iprocess, cudaGetErrorString(cuda_status));
		return 1;
	}

	float* hin = (float*)malloc(size);
	float* hout = (float*)malloc(size);
	
	// Generate random input data.
	double dinvrmax = 1.0 / RAND_MAX;
	for (int i = 0; i < n * n; i++)
	{
		for (int j = 0; j < iprocess + 1; j++)
			hin[i] += rand() * dinvrmax;
		hin[i] /= iprocess + 1;
	}
	
	// Copy input data generated on host to device buffer.
	cuda_status = cudaMemcpy(din1, hin, size, cudaMemcpyHostToDevice); 
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy input data from host to device by process %d, status = %s\n",
			iprocess, cudaGetErrorString(cuda_status));
		return 1;
	}
	
	// Perform the specified number of processing passes.
	for (int ipass = 0; ipass < npasses; ipass++)
	{
		// Fill output device buffer will zeros.
		cuda_status = cudaMemset(dout, 0, size);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot fill output device buffer with zeros by process %d, status = %s\n",
				iprocess, cudaGetErrorString(cuda_status));
			return 1;
		}

		// Process data on GPU.
		pattern2d_gpu(1, n, 1, 1, n, 1, din1, dout);

		// Wait for GPU kernels to finish processing.
		cuda_status = cudaThreadSynchronize();
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot synchronize GPU kernel by process %d, status = %s\n",
				iprocess, cudaGetErrorString(cuda_status));
			return 1;
		}
	
		// Copy output data back from device to host.
		cuda_status = cudaMemcpy(hout, dout, size, cudaMemcpyDeviceToHost); 
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot copy output data from device to host by process %d, status = %s\n",
				iprocess, cudaGetErrorString(cuda_status));
			return 1;
		}
	
		// Output average value of the resulting field.
		float avg = 0.0;
		for (int i = 0; i < n * n; i++)
			avg += hout[i];
		avg /= n * n;
		printf("Sending process %d resulting field with average = %f to process %d\n",
			iprocess, avg, (iprocess + 1) % nprocesses);

		MPI_Request request;
                int inext = (iprocess + 1) % nprocesses;
                int iprev = (iprocess - 1) % nprocesses; iprev += (iprev < 0) ? nprocesses : 0;
	
		// Pass entire process input device buffer directly to input device buffer
		// of next process.
		mpi_status = MPI_Isend(din1, n * n, MPI_FLOAT, inext, 0, MPI_COMM_WORLD, &request);
		mpi_status = MPI_Recv(din2, n * n, MPI_FLOAT, iprev, 0,	MPI_COMM_WORLD, NULL);
		mpi_status = MPI_Wait(&request, MPI_STATUS_IGNORE);
		
		// Swap buffers.
		float* swap = din1; din1 = din2; din2 = swap;
	}
	
	cuda_status = cudaFree(din1);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot free input device buffer by process %d, status = %d\n",
			iprocess, cudaGetErrorString(cuda_status));
		return 1;
	}
	
	cuda_status = cudaFree(dout);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot free output device buffer by process d, status = %d\n",
			iprocess, cudaGetErrorString(cuda_status));
		return 1;
	}

	free(hin);
	free(hout);

	mpi_status = MPI_Finalize();
	if (mpi_status != MPI_SUCCESS)
	{
		MPI_Error_string(mpi_status, mpi_error_msg, &mpi_error_msg_length);
		fprintf(stderr, "Cannot finalize MPI, status = %s\n",
			mpi_error_msg);
		return 1;
	}
	
	return 0;
}

