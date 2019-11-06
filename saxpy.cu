#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_VALUE 10

__global__ void
saxpy(float *X, float *Y, float *Z, int A, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N){
        Z[i] = A * X[i] + Y[i];
    }
}

int main()
{

    //Define error variable
    cudaError_t err = cudaSuccess;

    //Getting Cuda Device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop,0);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector X (error code %s)!\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print GPU device name, the maximum number of thread blocks, and the maximum number of threads per block
    printf("  Device name: %s\n", prop.name);
    printf("  The maximum number of thread blocks dim[0]: %d\n", prop.maxThreadsDim[0]);
    printf("  The maximum number of thread blocks dim[1]: %d\n", prop.maxThreadsDim[1]);
    printf("  The maximum number of thread blocks dim[2]: %d\n", prop.maxThreadsDim[2]);
    printf("  The maximum number of thread per block: %d\n", prop.maxThreadsPerBlock);
  
    srand((unsigned int)time(NULL));

    int N,A; // N is number of elements of  array, A is scalar number 
   
    printf("Write the size of array N: ");  
    
    scanf("%d", &N);  
    
    
    printf("Write the scalar value A: ");
    scanf("%d", &A);

    // Define size 
    size_t size = N * sizeof(float);
    


    // Allocate the host vector X
    float *h_X = (float *)malloc(size);

    // Allocate the host vector Y
    float *h_Y = (float *)malloc(size);

    // Allocate the host vector Z
    float *h_Z = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_X == NULL || h_Y == NULL || h_Z == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    //Assign random values the host vector X and Y
    for(int i = 0; i < N; i++){
	h_X[i] = ((float)rand()/(float)(RAND_MAX) * MAX_VALUE);
	printf("h_X[%d] : %f \n",i,h_X[i]);
        h_Y[i] = ((float)rand()/(float)(RAND_MAX) * MAX_VALUE);
	printf("h_Y[%d] : %f \n",i,h_Y[i]);
    }

    // Allocate the device input vector X
    float *d_X = NULL;
    err = cudaMalloc((void **)&d_X, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector X (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector Y
    float *d_Y = NULL;
    err = cudaMalloc((void **)&d_Y, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector Y (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector Z
    float *d_Z = NULL;
    err = cudaMalloc((void **)&d_Z, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector Y (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copy vectors from host to device
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector X from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_Y, h_Y, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector Y from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_Z, h_Z, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector Y from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Launch the SAXPY Cuda Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, d_Z, A, N);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  
    //Copy result vector from device to host
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_Z, d_Z, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Show results

    for (int i = 0; i < N; ++i)
    {
       printf("h_Z[%d] : %f \n",i,h_Z[i]);
    }
 
    //Clean device and host memory
    err = cudaFree(d_X);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_Y);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_Z);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(h_X);
    free(h_Y);
    free(h_Z);

    printf("Done");
    
    return 0;
}
