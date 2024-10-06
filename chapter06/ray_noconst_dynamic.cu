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


#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include "../common/utils.h"

#define DIM 1024
#define NUM_FRAME 100

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

struct Sphere {
    float   vx,vy,vz; // velocity
    float   r,b,g;
    float   radius;
    float   x,y,z;
    __device__ float hit( float ox, float oy, float *n ) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};
#define SPHERES 20

__global__ void updatePosition(Sphere *s, const double delta_time, const int n) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < n) {
        s[x].x += s[x].vx * delta_time;
        s[x].y += s[x].vy * delta_time;
        s[x].z += s[x].vz * delta_time;
        if (s[x].x > 500 || s[x].x < -500) {
            s[x].vx = -s[x].vx;
        }
        if (s[x].y > 500 || s[x].y < -500) {
            s[x].vy = -s[x].vy;
        }
        if (s[x].z > 500 || s[x].z < -500) {
            s[x].vz = -s[x].vz;
        }
    }
}

__global__ void kernel( Sphere *s, unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - DIM/2);
    float   oy = (y - DIM/2);

    float   r=0, g=0, b=0;
    float   maxz = -INF;
    for(int i=0; i<SPHERES; i++) {
        float   n;
        float   t = s[i].hit( ox, oy, &n );
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    } 

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
    Sphere          *s;
};

int main( void ) {
    DataBlock   data;

    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char   *dev_bitmap;
    Sphere          *s;


    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap,
                              bitmap.image_size() * NUM_FRAME ) );
    // allocate memory for the Sphere dataset
    HANDLE_ERROR( cudaMalloc( (void**)&s,
                              sizeof(Sphere) * SPHERES ) );

    // allocate temp memory, initialize it, copy to
    // memory on the GPU, then free our temp memory
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
    for (int i=0; i<SPHERES; i++) {
        temp_s[i].vx = rnd(100.0f) - 50.0f;
        temp_s[i].vy = rnd(100.0f) - 50.0f;
        temp_s[i].vz = rnd(100.0f) - 50.0f;
    }
    for (int i=0; i<SPHERES; i++) {
        temp_s[i].r = rnd( 1.0f );
        temp_s[i].g = rnd( 1.0f );
        temp_s[i].b = rnd( 1.0f );
        temp_s[i].x = rnd( 1000.0f ) - 500;
        temp_s[i].y = rnd( 1000.0f ) - 500;
        temp_s[i].z = rnd( 1000.0f ) - 500;
        temp_s[i].radius = rnd( 100.0f ) + 20;
    }
    HANDLE_ERROR( cudaMemcpy( s, temp_s,
                                sizeof(Sphere) * SPHERES,
                                cudaMemcpyHostToDevice ) );
    free( temp_s );

    // generate a bitmap from our sphere data
    const double delta_time = 1.0;
    dim3    grids(DIM/16,DIM/16);
    dim3    threads(16,16);
    START_GPU
    for (int i = 0; i < NUM_FRAME; i ++) {
        int frame_offset = i * bitmap.image_size();       
        kernel<<<grids,threads>>>( s, dev_bitmap + frame_offset );
        updatePosition<<<1, SPHERES>>>(s, delta_time, SPHERES);
    }
    END_GPU("Ray tracing")

    // output result
    clock_t start_time = clock();
    for (int frame = 0; frame < NUM_FRAME; ++frame) {
        int frame_offset = frame * bitmap.image_size();
        // copy our bitmap back from the GPU for display
        HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap + frame_offset,
                                bitmap.image_size(),
                                cudaMemcpyDeviceToHost ) );
        char filename[256];
        sprintf(filename, "results/ray_%04d.png", frame);
        bitmap.save_image(filename);
    }
    clock_t end_time = clock();
    printf("Time to save images: %f ms\n", (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000);

    HANDLE_ERROR( cudaFree( dev_bitmap ) );
    HANDLE_ERROR( cudaFree( s ) );

    // display
    // bitmap.display_and_exit();
    // bitmap.save_image("ray_noconst.png");
}

