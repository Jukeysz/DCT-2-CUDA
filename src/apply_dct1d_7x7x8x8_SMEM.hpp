#ifndef APPLY_DCT1D_7X7X8X8_SMEM_HPP
#define APPLY_DCT1D_7X7X8X8_SMEM_HPP
#include "LightFieldLoader.hpp"

#ifdef __cplusplus
extern "C" {
#endif


void apply_dct1d_7x7x8x8_SMEM_gpu(double *data,
                    int U, int V, int S, int T);


#ifdef __cplusplus
}
#endif

#endif