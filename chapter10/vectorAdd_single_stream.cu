/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <omp.h>
#include <math.h>

#include "../common/book.h"

#define START_GPU                                 \
    {                                             \
        cudaEvent_t start, stop;                  \
        float elapsedTime;                        \
        checkCudaErrors(cudaEventCreate(&start)); \
        checkCudaErrors(cudaEventCreate(&stop));  \
        checkCudaErrors(cudaEventRecord(start, 0));

#define END_GPU                                                       \
    checkCudaErrors(cudaEventRecord(stop, 0));                        \
    checkCudaErrors(cudaEventSynchronize(stop));                      \
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
    printf("GPU Time used:  %3.1f ms\n", elapsedTime);                \
    checkCudaErrors(cudaEventDestroy(start));                         \
    checkCudaErrors(cudaEventDestroy(stop));                          \
    }

#define START_CPU \
    {             \
        double start = omp_get_wtime();

#define END_CPU                                           \
    double end = omp_get_wtime();                         \
    double duration = end - start;                        \
    printf("CPU Time used: %3.1f ms\n", duration * 1000); \
    }

#define BLOCK_SIZE 256

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const double *A, const double *B, double *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = cos(A[i]) / sin(B[i]);
    }
}

/**
 * Host main routine
 */
int main(void)
{
    // Print the vector length to be used, and compute its size
    size_t numElements = 100 * 1024 * 1024;
    size_t num_chunks = 100;                      // number of chunks to slice the original vector
    size_t chunk_size = numElements / num_chunks; // size of each chunk

    printf("[Vector addition of %ld elements]\n", numElements);

    // Initialize the stream
    cudaStream_t stream;
    HANDLE_ERROR(cudaStreamCreate(&stream));

    // Allocate the device input vector A
    double *d_A = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_A, chunk_size * sizeof(double)));

    // Allocate the device input vector B
    double *d_B = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_B, chunk_size * sizeof(double)));

    // Allocate the device output vector C
    double *d_C = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_C, chunk_size * sizeof(double)));

    printf("Alloc Device A,B,C. size = %ld. ok!\n", chunk_size);

    double *h_A = NULL, *h_B = NULL, *h_C = NULL;
    HANDLE_ERROR(cudaHostAlloc((void **)&h_A, numElements * sizeof(double),
                               cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void **)&h_B, numElements * sizeof(double),
                               cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void **)&h_C, numElements * sizeof(double),
                               cudaHostAllocDefault));

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (double)RAND_MAX;
        h_B[i] = rand() / (double)RAND_MAX;
    }

    printf("Alloc Host A,B,C ok!\n");

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    // process the whole vector in multiple overlapping chunks
    START_GPU
    for (size_t i = 0; i < numElements; i += chunk_size)
    {
        size_t num = std::min(numElements - i, chunk_size);
        // copy the locked memory to the device, async
        HANDLE_ERROR(cudaMemcpyAsync(d_A, h_A + i,
                                     num * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream));
        HANDLE_ERROR(cudaMemcpyAsync(d_B, h_B + i,
                                     num * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream));

        // launch kernel
        vectorAdd<<<(num + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    BLOCK_SIZE, 0, stream>>>(d_A, d_B, d_C, num);

        // copy the data from device to the locked memory
        HANDLE_ERROR(cudaMemcpyAsync(h_C + i, d_C,
                                     num * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     stream));
    }
    END_GPU

    HANDLE_ERROR(cudaStreamSynchronize(stream));

    START_CPU
    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(cos(h_A[i]) / sin(h_B[i]) - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    END_CPU

    printf("Test PASSED\n");

    // clean up
    HANDLE_ERROR(cudaFreeHost(h_A));
    HANDLE_ERROR(cudaFreeHost(h_B));
    HANDLE_ERROR(cudaFreeHost(h_C));
    HANDLE_ERROR(cudaFree(d_A));
    HANDLE_ERROR(cudaFree(d_B));
    HANDLE_ERROR(cudaFree(d_C));
    HANDLE_ERROR(cudaStreamDestroy(stream));

    printf("Done\n");
    return 0;
}
