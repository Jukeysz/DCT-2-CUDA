#include <cuda_runtime.h>
#include "apply_dct1d_7x7x8x8.hpp"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#define LF_SIZE 45841250

void checkCUDAError(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#define gpuErrchk(ans) { checkCUDAError((ans), __FILE__, __LINE__); }

__constant__ double BASIS7[7 * 7] = 
{
1,1,1,1,1,1,1,
1.3787562757436209,1.1056766859965506,0.61360426835320059,8.6595605623549341e-17,-0.61360426835320048,-1.1056766859965501,-1.3787562757436209,
1.2741623922635348,0.31469212271294766,-0.88174773378993476,-1.4142135623730951,-0.88174773378993498,0.31469212271294611,1.274162392263535,
1.1056766859965506,-0.61360426835320048,-1.3787562757436209,-2.5978681687064801e-16,1.3787562757436207,0.61360426835320092,-1.1056766859965501,
0.88174773378993487,-1.2741623922635346,-0.31469212271294789,1.4142135623730951,-0.31469212271294716,-1.2741623922635361,0.88174773378993632,
0.61360426835320059,-1.3787562757436209,1.1056766859965503,4.3297802811774672e-16,-1.1056766859965508,1.3787562757436207,-0.61360426835319981,
0.31469212271294766,-0.88174773378993498,1.2741623922635354,-1.4142135623730951,1.2741623922635343,-0.8817477337899341,0.31469212271294666,
};

__constant__ double BASIS8[8 * 8] = 
{
1,1,1,1,1,1,1,1,
1.3870398453221475,1.1758756024193588,0.78569495838710235,0.27589937928294311,-0.27589937928294295,-0.7856949583871019,-1.1758756024193588,-1.3870398453221475,
1.3065629648763766,0.54119610014619712,-0.54119610014619701,-1.3065629648763766,-1.3065629648763768,-0.54119610014619779,0.54119610014619735,1.3065629648763764,
1.1758756024193588,-0.27589937928294295,-1.3870398453221475,-0.78569495838710213,0.78569495838710168,1.3870398453221475,0.27589937928294372,-1.1758756024193586,
1.0000000000000002,-1,-1.0000000000000002,0.99999999999999978,1.0000000000000002,-0.99999999999999889,-0.99999999999999956,0.99999999999999878,
0.78569495838710235,-1.3870398453221475,0.27589937928294306,1.1758756024193591,-1.1758756024193586,-0.27589937928294267,1.3870398453221477,-0.78569495838710124,
0.54119610014619712,-1.3065629648763768,1.3065629648763764,-0.54119610014619723,-0.54119610014619812,1.3065629648763766,-1.3065629648763761,0.54119610014619668,
0.27589937928294311,-0.78569495838710213,1.1758756024193591,-1.3870398453221477,1.3870398453221475,-1.1758756024193584,0.78569495838710124,-0.27589937928294345,
};

// Paralelizar sobre as espaciais e iterar sobre as angulares

__global__ void dct4d_x_kernel(const double* d_input, double* d_output,
                             int W, int Z, int Y, int X)
{
    double normalizer = 1.9685019685029528;

    int ANGULAR_DIM = 7;
    int SPATIAL_DIM = 8;

    int X_stride = 1;         
    int Y_stride = X;         
    int Z_stride = Y * X;     
    int W_stride = Z * Y * X;

    int macroblock_x = blockIdx.x * 31;
    int macroblock_y = blockIdx.y * 31;

    int positions[3] = {7, 15, 23};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int subblock_x = macroblock_x + positions[i];
            int subblock_y = macroblock_y + positions[j];

            if (subblock_x >= X || subblock_y >= Y) return;

            int global_x = subblock_x + threadIdx.x;
            int global_y = subblock_y + threadIdx.y;

            if (global_x >= X || global_y >= Y) return;

            int ANGULAR_OFFSET_Z = 6;
            int ANGULAR_OFFSET_W = 6;

            for (int z = 0; z < ANGULAR_DIM; ++z) {
                int global_z = ANGULAR_OFFSET_Z + z;
                for (int w = 0; w < ANGULAR_DIM; ++w) {
                    int global_w = ANGULAR_OFFSET_W + w;

                    double sum = 0.0;
                    for (int x_in_local = 0; x_in_local < SPATIAL_DIM; ++x_in_local) {
                        // int x_in_global = block_offset_x + x_in_local;
                        int x_in_global = subblock_x + x_in_local;

                        int input_idx = global_w*W_stride + global_z*Z_stride + global_y*Y_stride + x_in_global*X_stride;

                        int basis_idx = threadIdx.x * SPATIAL_DIM + x_in_local;

                        sum += d_input[input_idx] * BASIS8[basis_idx];
                    }

                    // int x_out_global = block_offset_x + tx;
                    int x_out_global = subblock_x + threadIdx.x;

                    int output_idx = global_w*W_stride + global_z*Z_stride + global_y*Y_stride + x_out_global*X_stride;

                    d_output[output_idx] = sum * normalizer;
                }
            }
        }
    }
}

__global__ void dct4d_y_kernel(const double* d_input, double* d_output,
                               int W, int Z, int Y, int X) 
{
    double normalizer = 1.9685019685029528;
    int ANGULAR_DIM = 7;
    int SPATIAL_DIM = 8;

    int X_stride = 1;
    int Y_stride = X;
    int Z_stride = Y * X;
    int W_stride = Y * X * Z;

    int macroblock_x = blockIdx.x * 31;
    int macroblock_y = blockIdx.y * 31;

    int positions[3] = {7, 15, 23};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int subblock_x = macroblock_x + positions[i];
            int subblock_y = macroblock_y + positions[j];

            if (subblock_x >= X || subblock_y >= Y) return;

            int global_x = subblock_x + threadIdx.x;
            int global_y = subblock_y + threadIdx.y;

            if (global_x >= X || global_y >= Y) return;

            int ANGULAR_OFFSET_Z = 6;
            int ANGULAR_OFFSET_W = 6;

            for (int z = 0; z < ANGULAR_DIM; ++z) {
                int global_z = ANGULAR_OFFSET_Z + z;
                for (int w = 0; w < ANGULAR_DIM; ++w) {
                    int global_w = ANGULAR_OFFSET_W + w;

                    double sum = 0.0;
                    for (int y_in_local = 0; y_in_local < SPATIAL_DIM; ++y_in_local) {
                        // int y_in_global = block_offset_y + y_in_local;
                        int y_in_global = subblock_y + y_in_local;

                        int input_idx = global_w*W_stride + global_z*Z_stride + y_in_global*Y_stride + global_x*X_stride;
                        int basis_idx = threadIdx.y * SPATIAL_DIM + y_in_local;

                        sum += d_input[input_idx] * BASIS8[basis_idx];
                    }

                    // int y_out_global = block_offset_y + ty;
                    int y_out_global = subblock_y + threadIdx.y;
                    int output_idx = global_w*W_stride + global_z*Z_stride + y_out_global*Y_stride + global_x*X_stride;

                    d_output[output_idx] = sum * normalizer;
                }
            }
        }
    }
}

__global__ void dct4d_z_kernel(const double* d_input, double* d_output,
    int W, int Z, int Y, int X) 
{   
    int SPATIAL_DIM = 8;
    int ANGULAR_DIM = 7;
    double normalizer = 1.3627702877384937;
    
    int X_stride = 1;
    int Y_stride = X;
    int Z_stride = Y * X;
    int W_stride = Y * X * Z;

    int macroblock_x = blockIdx.x * 31;
    int macroblock_y = blockIdx.y * 31;

    int positions[3] = {7, 15, 23};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int subblock_x = macroblock_x + positions[i];
            int subblock_y = macroblock_y + positions[j];

            if (subblock_x >= X || subblock_y >= Y) return;

            int global_x = subblock_x + threadIdx.x;
            int global_y = subblock_y + threadIdx.y;

            if (global_x >= X || global_y >= Y) return;

            int ANGULAR_OFFSET_Z = 6;
            int ANGULAR_OFFSET_W = 6;

            for (int w = 0; w < ANGULAR_DIM; ++w) {
                int global_w = ANGULAR_OFFSET_W + w;
                for (int z_out = 0; z_out < ANGULAR_DIM; ++z_out) {
                    int global_z_out = ANGULAR_OFFSET_Z + z_out;
                    double sum = 0.0;

                    for (int z_in = 0; z_in < ANGULAR_DIM; ++z_in) {
                        int global_z_in = ANGULAR_OFFSET_Z + z_in;
                        int input_idx = global_w*W_stride + global_z_in*Z_stride + global_y*Y_stride + global_x*X_stride;

                        int basis_idx = z_out * ANGULAR_DIM + z_in;
                        
                        sum += d_input[input_idx] * BASIS7[basis_idx];
                    }

                    int output_idx = global_w*W_stride + global_z_out*Z_stride + global_y*Y_stride + global_x*X_stride;

                    d_output[output_idx] = sum * normalizer;
                }
            }
        }
    }
}

__global__ void dct4d_w_kernel(const double* d_input, double* d_output,
    int W, int Z, int Y, int X) 
{
    const int SPATIAL_DIM = 8;
    const int ANGULAR_DIM = 7;
    const int ANGULAR_OFFSET_Z = 6;
    const int ANGULAR_OFFSET_W = 6;
    const double normalizer = 1.3627702877384937;

    const int X_stride = 1;
    const int Y_stride = X;
    const int Z_stride = Y * X;
    const int W_stride = Y * X * Z;    
    /*
        Qual é a posição do macrobloco que o threadblock atual processa?
        Utilizamos o espaçamento entre o início de cada macrobloco pra x e pra y
    */

    int macroblock_x = blockIdx.x * 31;
    int macroblock_y = blockIdx.y * 31;

    int positions[3] = {7, 15, 23};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            /*
            Qual é a posição do subbloco que o threadblock atual processa?
            Aplicar o deslocamento necessário para encontrar a posição inicial do subbloco
            */

            int subblock_x = macroblock_x + positions[i];
            int subblock_y = macroblock_y + positions[j];

            /*
                Qual é a posição da amostra que a thread processa
            */    
            int global_x = subblock_x + threadIdx.x;
            int global_y = subblock_y + threadIdx.y;

            if (global_x >= X || global_y >= Y) return;

            /* CARREGAR COLABORATIVEMENTE NA SMEM AQUI! */

            /*
            ==========================================================================================
            A idéia é iterar sobre todas as vistas (óbviamente).
            A onda a ser utilizada na dimensão W dependerá da linha de vistas.
            Os elementos de d_input a serem pegos serão [:][z][global_y][global_x].
            O output resultante dependerá de [w_out][z][global_x][global_y], ou seja, a posição global
            do respectivo thread dependendo da iteração de vista.
            ==========================================================================================
            */

            for (int z = 0; z < ANGULAR_DIM; ++z) {
                int global_z = ANGULAR_OFFSET_Z + z;
                for (int w_out = 0; w_out < ANGULAR_DIM; ++w_out) {
                    int global_w_out = ANGULAR_OFFSET_W + w_out;

                    double sum = 0.0;
                    for (int w_in = 0; w_in < ANGULAR_DIM; ++w_in) {
                        int global_w_in = ANGULAR_OFFSET_W + w_in;
                        int input_idx = global_w_in*W_stride + global_z*Z_stride + global_y*Y_stride + global_x*X_stride;
                        int basis_idx = w_out * ANGULAR_DIM + w_in;

                        sum += d_input[input_idx] * BASIS7[basis_idx];
                    }
                    int output_idx = global_w_out*W_stride + global_z*Z_stride + global_y*Y_stride + global_x*X_stride;

                    d_output[output_idx] = sum * normalizer;
                }
            }
        }
    }
}

void apply_dct1d_7x7x8x8_gpu(double* data,
                     int U, int V, int S, int T) {
    double *d_output, *d_data;

    size_t size_out = (size_t)U*V*S*T * sizeof(double);
    size_t size_data = (size_t)U*V*S*T * sizeof(double);

    gpuErrchk(cudaMalloc(&d_output, size_out));
    gpuErrchk(cudaMalloc(&d_data, size_data));
    // gpuErrchk(cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice));
    

    const int TILE_DIM = 8;
    const int MACROBLOCK = 31; 
    dim3 block_dims(TILE_DIM, TILE_DIM);
    int num_blocks_x = (T + MACROBLOCK - 1) / MACROBLOCK;
    int num_blocks_y = (S + MACROBLOCK - 1) / MACROBLOCK;
    dim3 grid_dims(num_blocks_x, num_blocks_y);

    // Warmup run
    // dct4d_x_kernel<<<grid_dims, block_dims>>>(d_data, d_output, U, V, S, T);
    // dct4d_y_kernel<<<grid_dims, block_dims>>>(d_output, d_data, U, V, S, T);
    // dct4d_z_kernel<<<grid_dims, block_dims>>>(d_data, d_output, U, V, S, T);
    // dct4d_w_kernel<<<grid_dims, block_dims>>>(d_output, d_data, U, V, S, T);
    // // Ensure the warmup completion
    // gpuErrchk(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start_event, stop_event;
    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));

    gpuErrchk(cudaEventRecord(start_event));
    
    gpuErrchk(cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice));
    dct4d_x_kernel<<<grid_dims, block_dims>>>(d_data, d_output, U, V, S, T);
    gpuErrchk(cudaGetLastError());
    dct4d_y_kernel<<<grid_dims, block_dims>>>(d_output, d_data, U, V, S, T);
    gpuErrchk(cudaGetLastError());
    dct4d_z_kernel<<<grid_dims, block_dims>>>(d_data, d_output, U, V, S, T);
    gpuErrchk(cudaGetLastError());
    dct4d_w_kernel<<<grid_dims, block_dims>>>(d_output, d_data, U, V, S, T);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpy(data, d_data, size_out, cudaMemcpyDeviceToHost)); 

    gpuErrchk(cudaEventRecord(stop_event));
    gpuErrchk(cudaEventSynchronize(stop_event));

    float milliseconds = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    std::cout << "Kernel execution time with memory transfer: " << milliseconds << "\n";

    gpuErrchk(cudaEventDestroy(start_event));
    gpuErrchk(cudaEventDestroy(stop_event));
    cudaFree(d_output);
    cudaFree(d_data);
}
