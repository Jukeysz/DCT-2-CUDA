#include <cstdio>
#include <cuda_runtime.h>

void checkCUDAError(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#define gpuErrchk(ans) { checkCUDAError((ans), __FILE__, __LINE__); }

int main() {
    int device = 0;
    gpuErrchk(cudaSetDevice(device));

    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    int smem_per_block = 0;
    int smem_per_block_optin = 0;
    int smem_per_sm = 0;

    gpuErrchk(cudaDeviceGetAttribute(
        &smem_per_block,
        cudaDevAttrMaxSharedMemoryPerBlock,
        device));

    gpuErrchk(cudaDeviceGetAttribute(
        &smem_per_block_optin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device));

    gpuErrchk(cudaDeviceGetAttribute(
        &smem_per_sm,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor,
        device));

    printf("\nShared Memory Info:\n");
    printf("  Max shared memory per block        : %d bytes (%.1f KB)\n",
           smem_per_block, smem_per_block / 1024.0);

    printf("  Max shared memory per block (opt-in): %d bytes (%.1f KB)\n",
           smem_per_block_optin, smem_per_block_optin / 1024.0);

    printf("  Max shared memory per SM           : %d bytes (%.1f KB)\n",
           smem_per_sm, smem_per_sm / 1024.0);

    printf("\nOther relevant limits:\n");
    printf("  Max threads per block : %d\n", prop.maxThreadsPerBlock);
    printf("  Max blocks per SM     : %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  Warp size             : %d\n", prop.warpSize);

    return 0;
}
