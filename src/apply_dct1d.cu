#include <cuda_runtime.h>
#include "apply_dct1d.hpp"
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

__constant__ double BASIS13[13 * 13] = {
1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,
1.40390235323759338,1.32231265144484689,1.16387494476104925,0.93779705680103165,0.65721781265334278,0.33844345812379117,-0.00000000000000023,-0.33844345812379040,-0.65721781265334267,-0.93779705680103120,-1.16387494476104925,-1.32231265144484689,-1.40390235323759338,
1.37311908647910430,1.05855405164560357,0.50148704053933324,-0.17046460798050689,-0.80336486913323779,-1.25222392036374819,-1.41421356237309515,-1.25222392036374908,-0.80336486913323812,-0.17046460798050786,0.50148704053933324,1.05855405164560312,1.37311908647910430,
1.32231265144484689,0.65721781265334278,-0.33844345812379040,-1.16387494476104925,-1.40390235323759338,-0.93779705680103143,-0.00000000000000026,0.93779705680103143,1.40390235323759338,1.16387494476104969,0.33844345812379090,-0.65721781265334178,-1.32231265144484667,
1.25222392036374863,0.17046460798050705,-1.05855405164560379,-1.37311908647910452,-0.50148704053933368,0.80336486913323646,1.41421356237309515,0.80336486913323912,-0.50148704053933302,-1.37311908647910408,-1.05855405164560379,0.17046460798050575,1.25222392036374863,
1.16387494476104925,-0.33844345812379040,-1.40390235323759338,-0.65721781265334345,0.93779705680103143,1.32231265144484689,0.00000000000000043,-1.32231265144484667,-0.93779705680103220,0.65721781265334167,1.40390235323759316,0.33844345812379001,-1.16387494476104836,
1.05855405164560357,-0.80336486913323779,-1.25222392036374908,0.50148704053933324,1.37311908647910452,-0.17046460798050719,-1.41421356237309515,-0.17046460798050697,1.37311908647910408,0.50148704053933413,-1.25222392036374863,-0.80336486913323968,1.05855405164560268,
0.93779705680103165,-1.16387494476104925,-0.65721781265334345,1.32231265144484689,0.33844345812379090,-1.40390235323759338,-0.00000000000000061,1.40390235323759316,-0.33844345812378851,-1.32231265144484711,0.65721781265334134,1.16387494476105080,-0.93779705680102898,
0.80336486913323812,-1.37311908647910430,0.17046460798050733,1.25222392036374908,-1.05855405164560290,-0.50148704053933635,1.41421356237309515,-0.50148704053933035,-1.05855405164560379,1.25222392036374730,0.17046460798050747,-1.37311908647910497,0.80336486913323779,
0.65721781265334278,-1.40390235323759338,0.93779705680103143,0.33844345812379090,-1.32231265144484711,1.16387494476104991,0.00000000000000078,-1.16387494476105080,1.32231265144484644,-0.33844345812379062,-0.93779705680103087,1.40390235323759294,-0.65721781265334078,
0.50148704053933324,-1.25222392036374908,1.37311908647910430,-0.80336486913323635,-0.17046460798050697,1.05855405164560379,-1.41421356237309515,1.05855405164560268,-0.17046460798050525,-0.80336486913324001,1.37311908647910386,-1.25222392036374930,0.50148704053932958,
0.33844345812379117,-0.93779705680103143,1.32231265144484689,-1.40390235323759338,1.16387494476104991,-0.65721781265334367,0.00000000000000156,0.65721781265334756,-1.16387494476104814,1.40390235323759338,-1.32231265144484644,0.93779705680103242,-0.33844345812379484,
0.17046460798050705,-0.50148704053933368,0.80336486913323912,-1.05855405164560379,1.25222392036374930,-1.37311908647910430,1.41421356237309515,-1.37311908647910452,1.25222392036374708,-1.05855405164560223,0.80336486913323746,-0.50148704053932947,0.17046460798050439,
};

__constant__ double BASIS16[16 * 16] = {
1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,1.00000000000000000,
1.40740373752638259,1.35331800117435264,1.24722501298667132,1.09320186700175759,0.89716758634263638,0.66665565847774677,0.41052452752235735,0.13861716919909170,-0.13861716919909153,-0.41052452752235713,-0.66665565847774666,-0.89716758634263616,-1.09320186700175759,-1.24722501298667110,-1.35331800117435264,-1.40740373752638237,
1.38703984532214752,1.17587560241935885,0.78569495838710235,0.27589937928294311,-0.27589937928294295,-0.78569495838710190,-1.17587560241935885,-1.38703984532214752,-1.38703984532214752,-1.17587560241935907,-0.78569495838710213,-0.27589937928294361,0.27589937928294306,0.78569495838710168,1.17587560241935885,1.38703984532214730,
1.35331800117435264,0.89716758634263638,0.13861716919909170,-0.66665565847774666,-1.24722501298667110,-1.40740373752638259,-1.09320186700175781,-0.41052452752235752,0.41052452752235696,1.09320186700175714,1.40740373752638259,1.24722501298667132,0.66665565847774699,-0.13861716919909056,-0.89716758634263538,-1.35331800117435219,
1.30656296487637658,0.54119610014619712,-0.54119610014619701,-1.30656296487637658,-1.30656296487637680,-0.54119610014619779,0.54119610014619735,1.30656296487637635,1.30656296487637658,0.54119610014619801,-0.54119610014619723,-1.30656296487637613,-1.30656296487637658,-0.54119610014619812,0.54119610014619701,1.30656296487637613,
1.24722501298667132,0.13861716919909170,-1.09320186700175759,-1.35331800117435286,-0.41052452752235752,0.89716758634263649,1.40740373752638259,0.66665565847774699,-0.66665565847774633,-1.40740373752638259,-0.89716758634263616,0.41052452752235785,1.35331800117435219,1.09320186700175848,-0.13861716919909023,-1.24722501298667088,
1.17587560241935885,-0.27589937928294295,-1.38703984532214752,-0.78569495838710213,0.78569495838710168,1.38703984532214752,0.27589937928294372,-1.17587560241935862,-1.17587560241935929,0.27589937928294150,1.38703984532214752,0.78569495838710268,-0.78569495838710124,-1.38703984532214775,-0.27589937928294550,1.17587560241935707,
1.09320186700175759,-0.66665565847774666,-1.35331800117435286,0.13861716919909076,1.40740373752638259,0.41052452752235769,-1.24722501298667110,-0.89716758634263616,0.89716758634263527,1.24722501298667177,-0.41052452752235769,-1.40740373752638259,-0.13861716919909445,1.35331800117435286,0.66665565847774766,-1.09320186700175581,
1.00000000000000022,-1.00000000000000000,-1.00000000000000022,0.99999999999999978,1.00000000000000022,-0.99999999999999889,-0.99999999999999956,0.99999999999999878,0.99999999999999967,-0.99999999999999856,-0.99999999999999978,0.99999999999999845,1.00000000000000000,-0.99999999999999845,-1.00000000000000022,0.99999999999999822,
0.89716758634263638,-1.24722501298667110,-0.41052452752235752,1.40740373752638259,-0.13861716919909056,-1.35331800117435264,0.66665565847774610,1.09320186700175848,-1.09320186700175759,-0.66665565847774744,1.35331800117435286,0.13861716919909461,-1.40740373752638281,0.41052452752235719,1.24722501298667310,-0.89716758634263460,
0.78569495838710235,-1.38703984532214752,0.27589937928294306,1.17587560241935907,-1.17587560241935862,-0.27589937928294267,1.38703984532214775,-0.78569495838710124,-0.78569495838710279,1.38703984532214752,-0.27589937928294345,-1.17587560241935840,1.17587560241935685,0.27589937928294600,-1.38703984532214797,0.78569495838710057,
0.66665565847774677,-1.40740373752638259,0.89716758634263649,0.41052452752235769,-1.35331800117435264,1.09320186700175759,0.13861716919909176,-1.24722501298667288,1.24722501298667199,-0.13861716919909237,-1.09320186700175737,1.35331800117435264,-0.41052452752235702,-0.89716758634263694,1.40740373752638237,-0.66665565847774066,
0.54119610014619712,-1.30656296487637680,1.30656296487637635,-0.54119610014619723,-0.54119610014619812,1.30656296487637658,-1.30656296487637613,0.54119610014619668,0.54119610014619857,-1.30656296487637791,1.30656296487637680,-0.54119610014619624,-0.54119610014619912,1.30656296487637791,-1.30656296487637480,0.54119610014619102,
0.41052452752235735,-1.09320186700175781,1.40740373752638259,-1.24722501298667110,0.66665565847774610,0.13861716919909176,-0.89716758634263649,1.35331800117435352,-1.35331800117435286,0.89716758634263472,-0.13861716919909203,-0.66665565847774810,1.24722501298667332,-1.40740373752638237,1.09320186700175515,-0.41052452752235152,
0.27589937928294311,-0.78569495838710213,1.17587560241935907,-1.38703984532214775,1.38703984532214752,-1.17587560241935840,0.78569495838710124,-0.27589937928294345,-0.27589937928294583,0.78569495838710313,-1.17587560241935840,1.38703984532214819,-1.38703984532214641,1.17587560241935929,-0.78569495838710013,0.27589937928293734,
0.13861716919909170,-0.41052452752235752,0.66665565847774699,-0.89716758634263616,1.09320186700175848,-1.24722501298667288,1.35331800117435352,-1.40740373752638281,1.40740373752638237,-1.35331800117435264,1.24722501298667177,-1.09320186700175848,0.89716758634263416,-0.66665565847774033,0.41052452752235619,-0.13861716919908601,
};

// Paralelizar sobre as espaciais e iterar sobre as angulares

__global__ void dct4d_x_kernel(const double* d_input, double* d_output,
                             int W, int Z, int Y, int X)
{
    double normalizer = 1.3919410907075054;
    int TILE_DIM = 16;

    int X_stride = 1;         
    int Y_stride = X;         
    int Z_stride = Y * X;     
    int W_stride = Z * Y * X;

    int macroblock_x = blockIdx.x * 31;
    int macroblock_y = blockIdx.y * 31;

    int subblock_x = macroblock_x + 15;
    int subblock_y = macroblock_y + 15;

    if (subblock_x >= X || subblock_y >= Y) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_x = subblock_x + tx;
    int global_y = subblock_y + ty;

    if (global_x >= X || global_y >= Y) return;

    // Collaboratively have each thread load one coefficient for the threadblock
    // into smem.

    for (int z = 0; z < Z; ++z) {
        for (int w = 0; w < W; ++w) {
            double sum = 0.0;

            for (int x_in_local = 0; x_in_local < TILE_DIM; ++x_in_local) {
                // int x_in_global = block_offset_x + x_in_local;
                int x_in_global = subblock_x + x_in_local;

                int input_idx = w*W_stride + z*Z_stride + global_y*Y_stride + x_in_global*X_stride;

                int basis_idx = tx * TILE_DIM + x_in_local;

                sum += d_input[input_idx] * BASIS16[basis_idx];
            }

            // int x_out_global = block_offset_x + tx;
            int x_out_global = subblock_x + tx;

            int output_idx = w*W_stride + z*Z_stride + global_y*Y_stride + x_out_global*X_stride;

            d_output[output_idx] = sum;
        }
    }
}

__global__ void dct4d_y_kernel(const double* d_input, double* d_output,
                               int W, int Z, int Y, int X) 
{
    double normalizer = 1.3919410907075054;
    int TILE_DIM = 16;

    int X_stride = 1;
    int Y_stride = X;
    int Z_stride = Y * X;
    int W_stride = Y * X * Z;

    int macroblock_x = blockIdx.x * 31;
    int macroblock_y = blockIdx.y * 31;

    int subblock_x = macroblock_x + 15;
    int subblock_y = macroblock_y + 15;

    if (subblock_x >= X || subblock_y >= Y) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_x = subblock_x + tx;
    int global_y = subblock_y + ty;

    if (global_x >= X || global_y >= Y) return;


    for (int z = 0; z < Z; ++z) {
        for (int w = 0; w < W; ++w) {
            double sum = 0.0;
            for (int y_in_local = 0; y_in_local < TILE_DIM; ++y_in_local) {
                // int y_in_global = block_offset_y + y_in_local;
                int y_in_global = subblock_y + y_in_local;

                int input_idx = w*W_stride + z*Z_stride + y_in_global*Y_stride + global_x*X_stride;
                int basis_idx = ty * TILE_DIM + y_in_local;

                sum += d_input[input_idx] * BASIS16[basis_idx];
            }

            // int y_out_global = block_offset_y + ty;
            int y_out_global = subblock_y + ty;
            int output_idx = w*W_stride + z*Z_stride + y_out_global*Y_stride + global_x*X_stride;

            d_output[output_idx] = sum;
        }
    }
}

__global__ void dct4d_z_kernel(const double* d_input, double* d_output,
    int W, int Z, int Y, int X) 
{   
    int TILE_DIM = 16;
    int ANGULAR_DIM = 13;
    double normalizer = 1.3919410907075054;
    
    int X_stride = 1;
    int Y_stride = X;
    int Z_stride = Y * X;
    int W_stride = Y * X * Z;

    int macroblock_x = blockIdx.x * 31;
    int macroblock_y = blockIdx.y * 31;

    int subblock_x = macroblock_x + 15;
    int subblock_y = macroblock_y + 15;

    if (subblock_x >= X || subblock_y >= Y) return;
               
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_x = subblock_x + tx;
    int global_y = subblock_y + ty;

    if (global_x >= X || global_y >= Y) return;

    for (int w = 0; w < W; ++w) {
        for (int z_out = 0; z_out < ANGULAR_DIM; ++z_out) {
            double sum = 0.0;

            for (int z_in = 0; z_in < ANGULAR_DIM; ++z_in) {
                int input_idx = w*W_stride + z_in*Z_stride + global_y*Y_stride + global_x*X_stride;

                int basis_idx = z_out * ANGULAR_DIM + z_in;
                
                sum += d_input[input_idx] * BASIS13[basis_idx];
            }

            int output_idx = w*W_stride + z_out*Z_stride + global_y*Y_stride + global_x*X_stride;

            d_output[output_idx] = sum * normalizer;
        }
    }
}

__global__ void dct4d_w_kernel(const double* d_input, double* d_output,
    int W, int Z, int Y, int X) 
{
    const int TILE_DIM = 16;
    const int ANGULAR_DIM = 13;
    double normalizer = 1.3919410907075054;

    int X_stride = 1;
    int Y_stride = X;
    int Z_stride = Y * X;
    int W_stride = Y * X * Z;    
    /*
        Qual é a posição do macrobloco que o threadblock atual processa?
        Utilizamos o espaçamento entre o início de cada macrobloco pra x e pra y
    */

    int macroblock_x = blockIdx.x * 31;
    int macroblock_y = blockIdx.y * 31;

    /*
        Qual é a posição do subbloco que o threadblock atual processa?
        Aplicar o deslocamento necessário para encontrar a posição inicial do subbloco
    */

    int subblock_x = macroblock_x + 15;
    int subblock_y = macroblock_y + 15;

    /*
        Qual é a posição da amostra que a thread processa
    */    
    int global_x = subblock_x + threadIdx.x;
    int global_y = subblock_y + threadIdx.y;

    if (global_x >= X || global_y >= Y) return;

    /*
    ==========================================================================================
    A idéia é iterar sobre todas as vistas (óbviamente).
    A onda a ser utilizada na dimensão W dependerá da linha de vistas.
    Os elementos de d_input a serem pegos serão [:][z][global_y][global_x].
    O output resultante dependerá de [w_out][z][global_x][global_y], ou seja, a posição global
    do respectivo thread dependendo da iteração de vista.
    ==========================================================================================
    */

    for (int z = 0; z < Z; ++z) {
        for (int w_out = 0; w_out < ANGULAR_DIM; ++w_out) {
            double sum = 0.0;

            for (int w_in = 0; w_in < W; ++w_in) {
                int input_idx = w_in*W_stride + z*Z_stride + global_y*Y_stride + global_x*X_stride;
                int basis_idx = w_out * ANGULAR_DIM + w_in;

                sum += d_input[input_idx] * BASIS13[basis_idx];
            }
            int output_idx = w_out*W_stride + z*Z_stride + global_y*Y_stride + global_x*X_stride;

            d_output[output_idx] = sum * normalizer;
        }
    }
}

void apply_dct1d_gpu(double* data,
                     int U, int V, int S, int T) {
    double *d_output, *d_data;

    size_t size_out = (size_t)U*V*S*T * sizeof(double);
    size_t size_data = (size_t)U*V*S*T * sizeof(double);

    gpuErrchk(cudaMalloc(&d_output, size_out));
    gpuErrchk(cudaMalloc(&d_data, size_data));
    // gpuErrchk(cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice));
    

    const int TILE_DIM = 16;
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

    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    std::cout << "Kernel execution time with memory transfer: " << milliseconds << "\n";

    gpuErrchk(cudaEventDestroy(start_event));
    gpuErrchk(cudaEventDestroy(stop_event));
    cudaFree(d_output);
    cudaFree(d_data);
}
