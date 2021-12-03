#include "kernel.h"
#include <stdio.h>

__global__ void reduce0(int * input, int *output, int N)
{
	extern __shared__ int sdata[1024];

	////printf("%d, %d\n", tid, N);

	//int *idata = input + blockIdx.x*blockDim.x;
	
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x * 2 + threadIdx.x;
	//boundary check
	if (tid >= N) return;

	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	
	//if (idx + blockDim.x < N)
	//{
	//	input[idx] += input[idx + blockDim.x];
	//}

	//convert global data pointer to the
	int *idata = input + blockIdx.x*blockDim.x * 2;
	if (idx + blockDim.x<N)
	{
		input[idx] += input[idx + blockDim.x];
		//printf("idx is: %d, blockDim is: %d, N is: %d\n", idx, blockDim.x, N);
	}
	__syncthreads();

	//sdata[tid] = input[i];
	__syncthreads();

	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}

	if (tid == 0)
	{
		output[blockIdx.x] = idata[0];
	}
}

int gpu_reduce0(int * input, int *output, int N, dim3 grid, dim3 block)
{
	int * cpu_sum = new int[1];
	int * gpu_sum;
	cudaMalloc(&gpu_sum, sizeof(int));
	reduce0<<<grid.x/2, block >>>(input, output, N);
	cudaDeviceSynchronize();
	int * cpu_output = new int[grid.x];
	cudaMemcpy(cpu_output, output, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

	int s = 0;
	for (int i = 0; i < grid.x; i++)
	{
		s += cpu_output[i];
	}

	/*reduce0 << <1, grid.x >> > (output, gpu_sum, grid.x);
	cudaDeviceSynchronize();
	cudaMemcpy(cpu_sum, gpu_sum, sizeof(int), cudaMemcpyDeviceToHost);
	return cpu_sum[0];*/
	return s;
}