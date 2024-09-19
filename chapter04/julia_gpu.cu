/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000
#define BLOCK_SIZE 8

struct cuComplex {
    float   r;
    float   i;
    // cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ cuComplex( float a, float b ) : r(a), i(b) {} // Fix error for calling host function from device
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y, int maxIter=200 ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<maxIter; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return i;
    }

    return maxIter;
}

__global__ void kernel( unsigned char *ptr ) {
    // map from blockIdx to pixel position
    // int x = blockIdx.x;
    // int y = blockIdx.y;
    // int offset = x + y * gridDim.x;

    // map from threadIdx and blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // 行优先访问: [y][x] = [y * width + x], 避免bank conflict
    // 因为x永远是增长最快的那一维度 (0,0)->(1,0)->(2,0)->(3,0)...
    int offset = x + y * gridDim.x * blockDim.x;
    int maxIter = 200;
    // now calculate the value at that position
    int juliaValue = julia( x, y, maxIter );
    ptr[offset*4 + 0] = (int) ((float)juliaValue / maxIter * 255); // r
    ptr[offset*4 + 1] = juliaValue * 2 % 256; // g
    ptr[offset*4 + 2] = 0; // b
    ptr[offset*4 + 3] = 255; // alpha 透明度
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;

    dim3    block(BLOCK_SIZE, BLOCK_SIZE);
    dim3    grid( (DIM + BLOCK_SIZE - 1)/BLOCK_SIZE, (DIM + BLOCK_SIZE - 1)/(BLOCK_SIZE) );
    kernel<<<grid,block>>>( dev_bitmap );

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
                              
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
                              
    // bitmap.display_and_exit();
    bitmap.save_image("julia_gpu.png");
}

