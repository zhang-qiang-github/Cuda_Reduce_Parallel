#ifndef reduce0_header_h
#define reduce0_header_h

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void reduce0(int * input, int *output, int N);
int gpu_reduce0(int * input, int *output, int N, dim3 grid, dim3 block);

#endif // !reduce0_header_h


