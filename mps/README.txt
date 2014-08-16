Starting MPS client application
https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
-------------------------------

Set the following variables in the client process’s environment. Note that

CUDA_VISIBLE_DEVICES should not be set in the client’s environment.

# Set to the same location as the MPS control daemon
export CUDA_MPS_PIPE_DIRECTORY=...

# Set to the same location as the MPS control daemon
export CUDA_MPS_LOG_DIRECTORY=...

Run application as usual
