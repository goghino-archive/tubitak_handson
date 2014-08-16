HOWTO: Starting MPS client application
https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf


Set the following variables in the client process’s environment. Note that
CUDA_VISIBLE_DEVICES should not be set in the client’s environment.

# Set to the same location as the MPS control daemon (see startMPS.sh)
export CUDA_MPS_PIPE_DIRECTORY=...

# Set to the same location as the MPS control daemon (see startMPS.sh)
export CUDA_MPS_LOG_DIRECTORY=...

Run application as usual

------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
Difference you should see when running without/with MPS:


jkardos@tesla-cmc:~/hands-on/mps$ mpirun -np 3 ./mpi input5.txt
Reading file - success!
Reading file - success!
Reading file - success!
Process 0 compute time: 5.58
Process 2 compute time: 5.61
Process 1 compute time: 5.59
------------------------------------------------------------
Finished! Found Solution at process 1: 
	c0 = -1.17863
	c1 = 3.31422
	c2 = -1.21134
	c3 = 3.33414
	c4 = -1.10314
	c5 = -1.72653
Best fitness: 340.474
Generations: 1500
Time for GPU calculation equals 5.15 seconds


jkardos@tesla-cmc:~/hands-on/mps$ ./startMPS.sh 


jkardos@tesla-cmc:~/hands-on/mps$ make run
CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/home/jkardos/hands-on/mps/nvida-mps CUDA_MPS_LOG_DIRECTORY=/home/jkardos/hands-on/mps/nvidia-log mpirun -np 3 ./mpi input5.txt
Reading file - success!
Reading file - success!
Reading file - success!
Process 0 compute time: 2.23
Process 1 compute time: 2.29
Process 2 compute time: 2.35
------------------------------------------------------------
Finished! Found Solution at process 2: 
	c0 = 1.41085
	c1 = -0.280552
	c2 = -0.3653
	c3 = -0.215853
	c4 = 1.29387
	c5 = -2.11262
Best fitness: 335.573
Generations: 1500
Time for GPU calculation equals 2.29 seconds

----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
HOWTO: Generating performance profile to see how kernels are executed
http://docs.nvidia.com/cuda/profiler-users-guide/#mps-profiling

1)nvprof --profile-all-processes -o output_%p
2)open new terminal
3)run multiprocess mpi application (i.e. mpirun)
4)return to nvprof terminal
5)ctrl^c
6)import generated profile filen into nvvp multiprocess view
