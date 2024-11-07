#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "cuda_helpers.h"

size_t getFreeGlobalMemory(int gpu_num)
{
  size_t free_mem = 0;
  size_t total_mem = 0;
  cuchk(cudaSetDevice(gpu_num));
  cuchk(cudaMemGetInfo(&free_mem, &total_mem));
  return free_mem;
}