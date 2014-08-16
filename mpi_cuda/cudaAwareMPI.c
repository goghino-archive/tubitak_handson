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
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mpi.h>
 
int main( int argc, char** argv )
{
    MPI_Init (&argc, &argv);
 
    // Get MPI rank and size
    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
 

    // Allocate device buffer and copy process' rank value to GPU memory
    int *rank_d = NULL;
    cudaError_t status = cudaMalloc((void **)&rank_d, sizeof(int));
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate GPU memory, status = %s\n",
			cudaGetErrorString(status));
		return status;
	}

    status = cudaMemcpy(rank_d, &rank, sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy to GPU memory, status = %s\n",
			cudaGetErrorString(status));
		return status;
	}
 

    // Do MPI communication,
    // send data directly from GPU buffers into another GPU buffer
    // withou staging at host memory
    int *rank_recv_d;
    if(rank ==0){
        // Master proces receives ranks of other processes
        status = cudaMalloc((void **)&rank_recv_d, (size)*sizeof(int));
	    if (status != cudaSuccess)
	    {
		    fprintf(stderr, "Cannot allocate GPU memory, status = %s\n",
			    cudaGetErrorString(status));
		    return status;
	    }
        for(int i=1; i<size; i++){        
            int err_status = MPI_Recv(&rank_recv_d[i], 1, MPI_INT, i, i,
                                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(err_status != MPI_SUCCESS) {
                fprintf(stderr, "MPI Receive error\n");
                return err_status;    
            }

        }
        
        // copies it from device into host memory (so that we can print it)
        int *rank_recv_h = malloc((size)*sizeof(int));
        status = cudaMemcpy(rank_recv_h, rank_recv_d, size*sizeof(int), cudaMemcpyDeviceToHost);
	    if (status != cudaSuccess)
	    {
		    fprintf(stderr, "Cannot copy from GPU memory, status = %s\n",
			    cudaGetErrorString(status));
		    return status;
	    } 
        cudaFree(rank_recv_d);        

        // and actually prints it
        printf("[master] Ranks recieived from slave processes:");
        for(int i=1; i<size; i++)
            printf(" %d ",rank_recv_h[i]);
        printf("\n");
        free(rank_recv_h);
    }else{
        //print info
        printf("[slave] Sending my rank: %d\n", rank);
        
        // Processes send their rank to master process (rank==0)
        int err_status = MPI_Send(rank_d, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
        if(err_status != MPI_SUCCESS){
            fprintf(stderr,"MPI Send error\n");
            return err_status;
        }
    }
 
    // Clean up
    MPI_Finalize();
 
    return 0;
}
