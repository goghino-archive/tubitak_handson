#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

# Select a location that’saccessible to the given user
export CUDA_MPS_PIPE_DIRECTORY=/home/jkardos/hands-on/mps/nvida-mps
export CUDA_MPS_LOG_DIRECTORY=/home/jkardos/hands-on/mps/nvidia-log

nvidia-cuda-mps-control -d # Start the daemon.



