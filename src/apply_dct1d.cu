#include <cuda_runtime.h>
#include "apply_dct1d.hpp"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

/* 
Each thread computes exactly one output value.
The DCT coefficient at index [w][z][y][k]
in which k is the X-dimension coef index (DCT output position)
and w,z,y specify the location of the 1D slice (sub-vector) I'm transforming
*/

// BUFFER ORDER: WZYX


/*
    TODO:
    Write down the dct basis coefficients by using local memory
*/

__constant__ float BASIS16[16 * 16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.4074,1.35332,1.24723,1.0932,0.897168,0.666656,0.410524,0.138617,-0.138617,-0.410525,-0.666655,-0.897168,-1.0932,-1.24722,-1.35332,-1.4074,1.38704,1.17588,0.785695,0.275899,-0.275899,-0.785695,-1.17588,-1.38704,-1.38704,-1.17588,-0.785695,-0.275899,0.275899,0.785695,1.17588,1.38704,1.35332,0.897168,0.138617,-0.666655,-1.24722,-1.4074,-1.0932,-0.410524,0.410524,1.0932,1.4074,1.24722,
0.666656,-0.138617,-0.897168,-1.35332,1.30656,0.541196,-0.541196,-1.30656,-1.30656,-0.541196,0.541196,1.30656,1.30656,0.541196,-0.541196,-1.30656,-1.30656,-0.541197,0.541196,1.30656,1.24723,0.138617,-1.0932,-1.35332,-0.410524,0.897168,1.4074,0.666656,-0.666655,-1.4074,-0.897168,0.410524,1.35332,1.0932,-0.138618,-1.24723,1.17588,-0.275899,-1.38704,-0.785695,0.785695,1.38704,0.2759,-1.17588,-1.17588,0.2759,1.38704,0.785695,-0.785695,-1.38704,-0.275898,1.17588,1.0932,-0.666655,-1.35332,0.138617,1.4074,0.410525,-1.24722,-0.897168,
0.897167,1.24723,-0.410524,-1.4074,-0.138618,1.35332,0.666657,-1.0932,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-0.999999,-1,1,0.897168,-1.24722,-0.410524,1.4074,-0.138617,-1.35332,0.666656,1.0932,-1.0932,-0.666656,1.35332,0.138617,-1.4074,0.410526,1.24722,-0.897167,0.785695,-1.38704,0.275899,1.17588,-1.17588,-0.275899,1.38704,-0.785695,-0.785695,1.38704,-0.275898,-1.17588,1.17588,0.275898,-1.38704,0.785695,0.666656,-1.4074,0.897168,0.410525,
-1.35332,1.0932,0.138618,-1.24722,1.24723,-0.138616,-1.0932,1.35332,-0.410525,-0.897167,1.4074,-0.666655,0.541196,-1.30656,1.30656,-0.541196,-0.541197,1.30656,-1.30656,0.541197,0.541197,-1.30656,1.30656,-0.541197,-0.541197,1.30656,-1.30656,0.541197,0.410524,-1.0932,1.4074,-1.24722,0.666656,0.138618,-0.897168,1.35332,-1.35332,0.897168,-0.138617,-0.666656,1.24722,-1.4074,1.0932,-0.410527,0.275899,-0.785695,1.17588,-1.38704,1.38704,-1.17588,0.785694,-0.275898,-0.2759,0.785695,-1.17588,1.38704,-1.38704,1.17587,-0.785693,0.275898,
0.138617,-0.410524,0.666656,-0.897168,1.0932,-1.24722,1.35332,-1.4074,1.4074,-1.35332,1.24722,-1.0932,0.897169,-0.666654,0.410523,-0.138617,};

__constant__ float BASIS13[13 * 13] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1.4039,1.32231,1.16387,0.937797,0.657218,0.338444,-6.18172e-08,-0.338443,-0.657218,-0.937797,-1.16387,-1.32231,-1.4039,1.37312,1.05855,0.501487,-0.170465,-0.803365,-1.25222,-1.41421,-1.25222,-0.803365,-0.170465,0.501487,1.05855,1.37312,1.32231,0.657218,-0.338443,-1.16387,-1.4039,-0.937797,1.68643e-08,0.937797,1.4039,1.16387,0.338444,-0.657217,-1.32231,1.25222,0.170465,-1.05855,-1.37312,-0.501487,0.803365,1.41421,0.803365,
-0.501486,-1.37312,-1.05855,0.170464,1.25222,1.16387,-0.338443,-1.4039,-0.657218,0.937797,1.32231,1.96676e-07,-1.32231,-0.937797,0.657218,1.4039,0.338444,-1.16388,1.05855,-0.803365,-1.25222,0.501487,1.37312,-0.170464,-1.41421,-0.170465,1.37312,0.501487,-1.25222,-0.803366,1.05855,0.937797,-1.16387,-0.657218,1.32231,0.338444,-1.4039,-4.10216e-07,1.4039,-0.338444,-1.32231,0.657217,1.16387,-0.937796,0.803365,-1.37312,0.170465,1.25222,-1.05855,-0.501488,1.41421,-0.501487,-1.05855,1.25222,0.170464,-1.37312,0.803365,0.657218,-1.4039,0.937797,
0.338444,-1.32231,1.16387,-5.0593e-08,-1.16387,1.32231,-0.338445,-0.937797,1.4039,-0.657217,0.501487,-1.25222,1.37312,-0.803364,-0.170465,1.05855,-1.41421,1.05855,-0.170466,-0.803365,1.37312,-1.25222,0.501488,0.338444,-0.937797,1.32231,-1.4039,1.16387,-0.657218,-8.37297e-07,0.657217,-1.16388,1.4039,-1.32231,0.937796,-0.338441,0.170465,-0.501487,0.803365,-1.05855,1.25222,-1.37312,1.41421,-1.37312,1.25222,-1.05855,0.803364,-0.501485,0.170464,};

__global__ void dct4d_x_kernel(const float* d_input, float* d_output,
                             int W, int Z, int Y, int X)
{
    // calculates one output value
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int X_stride = 1;         // stride between elements along X
    int Y_stride = X;         // stride between elements along Y
    int Z_stride = Y * X;     // stride between elements along Z
    int W_stride = Z * Y * X; // stride between elements along W

    int k = global_idx % X;
    int y = (global_idx / X) % Y;
    int z = (global_idx / (X * Y)) % Z;
    int w = global_idx / (X * Y * Z);

    
    // ensure this thread is within the bounds of the output data.
    if (w >= W || z >= Z || y >= Y || k >= X) {
        return;
    }

    // this thread calculates output[w][z][y][k]
    float sum = 0.0f;

    for (int n = 0; n < X; ++n) {
        // calculate the linear index for the input element input[w][z][y][n]
        int input_linear_idx = 
              w * W_stride 
            + z * Z_stride 
            + y * Y_stride  // if dim was Y, n would walk here
            + n * X_stride; // n walks here

        // accumulate the sum
        sum += d_input[input_linear_idx] * BASIS13[X * k + n];
    }

    // calculate the linear index for the output element output[w][z][y][k]
    int output_linear_idx = 
          w * W_stride 
        + z * Z_stride 
        + y * Y_stride 
        + k * X_stride;

    d_output[output_linear_idx] = sum;
}

__global__ void dct4d_y_kernel(float *d_input, float *d_output,
                               int W, int Z, int Y, int X) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int X_stride = 1;
    int Y_stride = X;
    int Z_stride = Y * X;
    int W_stride = Y * X * Z;

    int x = global_idx % X;
    int k = (global_idx / X) % Y;
    int z = (global_idx / (X * Y)) % Z;
    int w = (global_idx / (X * Y * Z));

    // ensure this thread is within the bounds of the output data.
    if (w >= W || z >= Z || k >= Y || x >= X) {
        return;
    }

    // calculate dct sum
    // this thread calculates output[w][z][y][k]
    float sum = 0.0f;

    /*
    ----------------- REMINDER ---------------
    for a given k, it takes its correct sum from the kth wave,
    so it doesnt walk through the whole slice, just one coef
    */
    for (int n = 0; n < Y; ++n) {
        // Calculate the linear index for the input element input[w][z][y][n]
        int input_linear_idx = 
              w * W_stride 
            + z * Z_stride 
            + n * Y_stride  // if dim was Y, n would walk here
            + x * X_stride; // n walks here

        // accumulate the sum
        sum += d_input[input_linear_idx] * BASIS13[Y * k + n];
    }


    // calculate the linear index for the output element output[w][z][y][k]
    // this is the same as the global_idx if the grid/block structure matches perfectly.
    // it just flattens again
    int output_linear_idx = 
          w * W_stride 
        + z * Z_stride 
        + k * Y_stride 
        + x * X_stride;

    // the output element is the position of the DCT basis wave coef at thread's WZY
    d_output[output_linear_idx] = sum;
}

__global__ void dct4d_z_kernel(float *d_input, float *d_output,
    int W, int Z, int Y, int X) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int X_stride = 1;
    int Y_stride = X;
    int Z_stride = Y * X;
    int W_stride = Y * X * Z;

    int x = global_idx % X;
    int y = (global_idx / X) % Y;
    int k = (global_idx / (X * Y)) % Z;
    int w = (global_idx / (X * Y * Z));

    // ensure this thread is within the bounds of the output data.
    if (w >= W || k >= Z || y >= Y || x >= X) {
        return;
    }

    // calculate dct sum
    // this thread calculates output[w][z][y][k]
    float sum = 0.0f;

    /*
    for a given k, it takes its correct sum from the kth wave,
    so it doesnt walk through the whole slice, just one coef
    */
    for (int n = 0; n < Z; ++n) {
        // Calculate the linear index for the input element input[w][z][y][n]
        int input_linear_idx = 
        w * W_stride 
        + n * Z_stride 
        + y * Y_stride  // if dim was Y, n would walk here
        + x * X_stride;

        // accumulate the sum
        sum += d_input[input_linear_idx] * BASIS16[Z * k + n];
    }

    // calculate the linear index for the output element output[w][z][y][k]
    // this is the same as the global_idx if the grid/block structure matches perfectly.
    // it just flattens again
    int output_linear_idx = 
    w * W_stride 
    + k * Z_stride 
    + y * Y_stride 
    + x * X_stride;

    float normalizer = 1.3919439868f;

    // the output element is the position of the DCT basis wave coef at thread's WZY
    d_output[output_linear_idx] = sum * normalizer;
}

__global__ void dct4d_w_kernel(float *d_input, float *d_output,
    int W, int Z, int Y, int X) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int X_stride = 1;
    int Y_stride = X;
    int Z_stride = Y * X;
    int W_stride = Y * X * Z;

    int x = global_idx % X;
    int y = (global_idx / X) % Y;
    int z = (global_idx / (X * Y)) % Z;
    int k = (global_idx / (X * Y * Z));

    // ensure this thread is within the bounds of the output data.
    if (k >= W || z >= Z || y >= Y || x >= X) {
        return;
    }

    // calculate dct sum
    // this thread calculates output[w][z][y][k]
    float sum = 0.0f;

    /*
    for a given k, it takes its correct sum from the kth wave,
    so it doesnt walk through the whole slice, just one coef
    */
    for (int n = 0; n < W; ++n) {
        // Calculate the linear index for the input element input[w][z][y][n]
        int input_linear_idx = 
        n * W_stride 
        + z * Z_stride 
        + y * Y_stride  // if dim was Y, n would walk here
        + x * X_stride; // n walks here

        // accumulate the sum

        // K determines which wave to take from the 2D basis waves matrix
        // each of these waves correspond to one index of the vector result of coefs for a fixated ZYX.
        // the d_input would be broadcasted if the kernel computed whole coef vectors
        sum += d_input[input_linear_idx] * BASIS16[W * k + n];
    }

    // calculate the linear index for the output element output[w][z][y][k]
    // this is the same as the global_idx if the grid/block structure matches perfectly.
    // it just flattens again
    int output_linear_idx = 
    k * W_stride 
    + z * Z_stride 
    + y * Y_stride 
    + x * X_stride;

    float normalizer = 1.3919439868f;

    // the output element is the position of the DCT basis wave coef at thread's WZY
    d_output[output_linear_idx] = sum * normalizer;
}

void apply_dct1d_gpu(float* data,
                     int U, int V, int S, int T, int selectedDim) {
    float *d_output, *d_data;

    size_t size_out = (size_t)U*V*S*T * sizeof(float);
    size_t size_data = (size_t)U*V*S*T * sizeof(float);

    cudaMalloc(&d_output, size_out);
    cudaMalloc(&d_data, size_data);

    cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice);

    int total_threads = U * V * S * T;
    int threads_per_block = 256;

    dim3 num_blocks((total_threads + threads_per_block - 1) / threads_per_block);
    dim3 block_dims(threads_per_block);

    switch (selectedDim) {
        case 3: {
            dct4d_x_kernel<<<num_blocks, block_dims>>>(d_data, d_output, U, V, S, T);
            break;
        }

        case 2: {
            dct4d_y_kernel<<<num_blocks, block_dims>>>(d_data, d_output, U, V, S, T);
            break;
        }

        case 1: {
            dct4d_z_kernel<<<num_blocks, block_dims>>>(d_data, d_output, U, V, S, T);
            break;
        }

        case 0: {
            dct4d_w_kernel<<<num_blocks, block_dims>>>(d_data, d_output, U, V, S, T);
            break;
        }
    }


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(data, d_output, size_out, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after memcpy: %s\n", cudaGetErrorString(err));
    }


    cudaFree(d_output);
    cudaFree(d_data);
}