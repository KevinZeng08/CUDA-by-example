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
#include <assert.h>

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
    cudaStream_t stream0, stream1, stream2;
    HANDLE_ERROR(cudaStreamCreate(&stream0));
    HANDLE_ERROR(cudaStreamCreate(&stream1));
    HANDLE_ERROR(cudaStreamCreate(&stream2));

    // Allocate the device input vector
    double *d_A0 = NULL, *d_B0 = NULL, *d_C0 = NULL;
    double *d_A1 = NULL, *d_B1 = NULL, *d_C1 = NULL;
    double *d_A2 = NULL, *d_B2 = NULL, *d_C2 = NULL;

    HANDLE_ERROR(cudaMalloc((void **)&d_A0, chunk_size * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&d_B0, chunk_size * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&d_C0, chunk_size * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&d_A1, chunk_size * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&d_B1, chunk_size * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&d_C1, chunk_size * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&d_A2, chunk_size * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&d_B2, chunk_size * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **)&d_C2, chunk_size * sizeof(double)));

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
    for (size_t i = 0; i < numElements; i += 3 * chunk_size)
    {
        size_t num = std::min(numElements - i, 3 * chunk_size);
        size_t num1 = num / 3;
        size_t num2 = (num - num1) / 2;
        size_t num3 = num - num1 - num2;
        assert(num == num1 + num2 + num3);

        // copy the locked memory to the device, async
        HANDLE_ERROR(cudaMemcpyAsync(d_A0, h_A + i,
                                     num1 * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream0));
        HANDLE_ERROR(cudaMemcpyAsync(d_A1, h_A + i + num1,
                                     num2 * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream1));
        HANDLE_ERROR(cudaMemcpyAsync(d_A2, h_A + i + num1 + num2,
                                     num3 * sizeof(double), cudaMemcpyHostToDevice, stream2));
        HANDLE_ERROR(cudaMemcpyAsync(d_B0, h_B + i,
                                     num1 * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream0));
        HANDLE_ERROR(cudaMemcpyAsync(d_B1, h_B + i + num1,
                                     num2 * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream1));
        HANDLE_ERROR(cudaMemcpyAsync(d_B2, h_B + i + num1 + num2,
                                     num3 * sizeof(double), cudaMemcpyHostToDevice, stream2));

        // launch kernel
        vectorAdd<<<(num1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    BLOCK_SIZE, 0, stream0>>>(d_A0, d_B0, d_C0, num1);
        vectorAdd<<<(num2 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    BLOCK_SIZE, 0, stream1>>>(d_A1, d_B1, d_C1, num2);
        vectorAdd<<<(num3 + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    BLOCK_SIZE, 0, stream2>>>(d_A2, d_B2, d_C2, num3);

        // copy the data from device to the locked memory
        HANDLE_ERROR(cudaMemcpyAsync(h_C + i, d_C0,
                                     num1 * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     stream0));
        HANDLE_ERROR(cudaMemcpyAsync(h_C + i + num1, d_C1,
                                     num2 * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     stream1));
        HANDLE_ERROR(cudaMemcpyAsync(h_C + i + num1 + num2, d_C2,
                                     num3 * sizeof(double), cudaMemcpyDeviceToHost, stream2));
    }
    END_GPU

    HANDLE_ERROR(cudaStreamSynchronize(stream0));
    HANDLE_ERROR(cudaStreamSynchronize(stream1));
    HANDLE_ERROR(cudaStreamSynchronize(stream2));

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
    HANDLE_ERROR(cudaFree(d_A0));
    HANDLE_ERROR(cudaFree(d_B0));
    HANDLE_ERROR(cudaFree(d_C0));
    HANDLE_ERROR(cudaFree(d_A1));
    HANDLE_ERROR(cudaFree(d_B1));
    HANDLE_ERROR(cudaFree(d_C1));
    HANDLE_ERROR(cudaFree(d_A2));
    HANDLE_ERROR(cudaFree(d_B2));
    HANDLE_ERROR(cudaFree(d_C2));
    HANDLE_ERROR(cudaStreamDestroy(stream0));
    HANDLE_ERROR(cudaStreamDestroy(stream1));
    HANDLE_ERROR(cudaStreamDestroy(stream2));

    printf("Done\n");
    return 0;
}
