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

__global__ void dct4d_x_kernel(const float* d_input, float* d_output, float* basisWaves,
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
        sum += d_input[input_linear_idx] * basisWaves[X * k + n];
    }

    // calculate the linear index for the output element output[w][z][y][k]
    int output_linear_idx = 
          w * W_stride 
        + z * Z_stride 
        + y * Y_stride 
        + k * X_stride;

    d_output[output_linear_idx] = sum;
}

__global__ void dct4d_y_kernel(float *d_input, float *d_output, float *basisWaves,
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
        sum += d_input[input_linear_idx] * basisWaves[Y * k + n];
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

__global__ void dct4d_z_kernel(float *d_input, float *d_output, float *basisWaves,
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
        sum += d_input[input_linear_idx] * basisWaves[Z * k + n];
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

__global__ void dct4d_w_kernel(float *d_input, float *d_output, float *basisWaves,
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
        sum += d_input[input_linear_idx] * basisWaves[W * k + n];
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

void apply_dct1d_gpu(float* dctMatrix, float* data,
                     int U, int V, int S, int T, int selectedDim) {
    float *d_output, *d_data, *d_basis;

    size_t size_out = (size_t)U*V*S*T * sizeof(float);
    size_t size_data = (size_t)U*V*S*T * sizeof(float);
    size_t size_basis;

    switch (selectedDim) {
        //case 0: size_basis = (size_t)T * T * sizeof(float); break;
        //case 1: size_basis = (size_t)S * S * sizeof(float); break;
        //case 2: size_basis = (size_t)V * V * sizeof(float); break;
        //case 3: size_basis = (size_t)U * U * sizeof(float); break;
        case 0: size_basis = (size_t)U * U * sizeof(float); break;       
        case 1: size_basis = (size_t)V * V * sizeof(float); break;
        case 2: size_basis = (size_t)S * S * sizeof(float); break;
        case 3: size_basis = (size_t)T * T * sizeof(float); break;
        default:
            printf("Error: Invalid selectedDim in apply_dct1d_gpu\n");
            return;
    }

    cudaMalloc(&d_output, size_out);
    cudaMalloc(&d_data, size_data);
    cudaMalloc((void**)&d_basis, size_basis);

    cudaMemcpy(d_basis, dctMatrix, size_basis, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice);

    int total_threads = U * V * S * T;
    int threads_per_block = 256;

    dim3 num_blocks((total_threads + threads_per_block - 1) / threads_per_block);
    dim3 block_dims(threads_per_block);

    switch (selectedDim) {
        case 3: {
            dct4d_x_kernel<<<num_blocks, block_dims, 0>>>(d_data, d_output, d_basis, U, V, S, T);
            break;
        }

        case 2: {
            dct4d_y_kernel<<<num_blocks, block_dims, 0>>>(d_data, d_output, d_basis, U, V, S, T);
            break;
        }

        case 1: {
            dct4d_z_kernel<<<num_blocks, block_dims, 0>>>(d_data, d_output, d_basis, U, V, S, T);
            break;
        }

        case 0: {
            dct4d_w_kernel<<<num_blocks, block_dims, 0>>>(d_data, d_output, d_basis, U, V, S, T);
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
    cudaFree(d_basis);
}