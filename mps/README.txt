HOWTO: Starting MPS client application
https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
(5.1.2. On a Single-User System, page 23)

Set the following variables in the client process’s environment. Note that
CUDA_VISIBLE_DEVICES should not be set in the client’s environment.

# Set to the same location as the MPS control daemon (see startMPS.sh)
export CUDA_MPS_PIPE_DIRECTORY=...

# Set to the same location as the MPS control daemon (see startMPS.sh)
export CUDA_MPS_LOG_DIRECTORY=...

Run application as usual

--------------------------------------------------------------------------------
Difference you should see when running without/with MPS:
--------------------------------------------------------------------------------

*********************
WITHOUT MPS enabled *
*********************
Check if deamon is running, usually enabled by default
jkardos@tesla-cmc:~/hands-on/mps$ ps ax | grep mps
16617 ?        Sl     0:00 /usr/sbin/mpssd
18567 pts/6    S+     0:00 grep --color=auto mps
31298 ?        Ssl    0:05 nvidia-cuda-mps-control -d
31342 ?        Sl     1:30 nvidia-cuda-mps-server

Stop MPS daemon if it is running(process id=31298 in above example):
$echo quit | nvidia-cuda-mps-control
or
$kill -9 31298

jkardos@tesla-cmc:~/hands-on/mps$ ps ax | grep mps
16617 ?        Sl     0:00 /usr/sbin/mpssd
18809 pts/6    S+     0:00 grep --color=auto mps


jkardos@tesla-cmc:~/hands-on/mps$ make run
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


******************
WITH MPS enabled *
******************
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
HOWTO: Generating performance profile to see how different processes' kernels
are executed in parallel
http://docs.nvidia.com/cuda/profiler-users-guide/#mps-profiling

1)nvprof --profile-all-processes -o output_%p
2)open new terminal
3)run multiprocess mpi application (i.e. mpirun)
4)return to nvprof terminal
5)ctrl^c
6)import generated profile filen into nvvp multiprocess view
