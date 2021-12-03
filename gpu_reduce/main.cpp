#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"

int cpu_array_sum(int * array, int N)
{
	int sum = 0;
	for (int i=0; i<N; i++)
	{
		sum += array[i];
	}
	return sum;
}

void main()
{
	int N = 1 << 24; // size of array
	//int N = 1024; // size of array
	int * array = (int *)malloc(N*sizeof(int));
	// array initialization
	srand(100);
	for (int i=0; i<N; i++)
	{
		array[i] = int(rand()&0xff);
	}
	int cpu_result, gpu_result;
	double cpu_start_time, cpu_end_time, cpu_cal_time;
	double gpu_start_time, gpu_end_time, gpu_cal_time;
	cpu_start_time = clock();
	cpu_result = cpu_array_sum(array, N);
	cpu_end_time = clock();
	cpu_cal_time = double(cpu_end_time - cpu_start_time) / CLOCKS_PER_SEC;
	printf("cpu: result is:%d, calculation time is: %f\n", cpu_result, cpu_cal_time);

	int blocksize = 1024;
	dim3 block(blocksize);
	dim3 grid(N / blocksize + 1);
	cudaError_t status;
	int * gpu_array = nullptr;
	int * block_array = nullptr;
	status = cudaMalloc(&gpu_array, N * sizeof(int));
	if (status != cudaSuccess)
	{
		printf("%s", cudaGetErrorString(status));
	}
	status = cudaMalloc(&block_array, grid.x * sizeof(int));
	if (status != cudaSuccess)
	{
		printf("%s", cudaGetErrorString(status));
	}
	status = cudaMemcpy(gpu_array, array, N * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		printf("%s", cudaGetErrorString(status));
	}
	//int * test = new int[N];
	//cudaMemcpy(test, gpu_array, N * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_start_time = clock();
	gpu_result = gpu_reduce0(gpu_array, block_array, N, grid, block);
	gpu_end_time = clock();
	gpu_cal_time = double(gpu_end_time - gpu_start_time) / CLOCKS_PER_SEC;
	printf("reduce0: result is:%d, calculation time is: %f\n", gpu_result, gpu_cal_time);

}