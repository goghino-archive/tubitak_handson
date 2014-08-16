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
 

    // Allocate host and device buffers and copy rank value to GPU
    int *rank_d = NULL;
    cudaMalloc((void **)&rank_d, sizeof(int));
    cudaMemcpy(rank_d, &rank, sizeof(int), cudaMemcpyHostToDevice);
 

    int *rank_recv_d;
    if(rank ==0){
        // Master proces receives ranks of other processes and prints it
        cudaMalloc((void **)&rank_recv_d, (size)*sizeof(int));
        for(int i=1; i<size; i++)        
            MPI_Recv(&rank_recv_d[i], 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        int *rank_recv_h = malloc((size)*sizeof(int));
        cudaMemcpy(rank_recv_h, rank_recv_d, size*sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(rank_recv_d);        

        printf("Ranks recieived from other processes:\n");
        for(int i=1; i<size; i++)
            printf("\t%d\n",rank_recv_h[i]);
        free(rank_recv_h);
    }else{
        // Processes send their rank to master process (rank==0)
        MPI_Send(rank_d, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
    }


    if(rank==0)
        printf("Success!\n");
 
    // Clean up
    MPI_Finalize();
 
    return 0;
}
