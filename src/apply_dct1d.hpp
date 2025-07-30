#ifndef APPLY_DCT1D_HPP
#define APPLY_DCT1D_HPP
#include "LightFieldLoader.hpp"

#ifdef __cplusplus
extern "C" {
#endif


void apply_dct1d_gpu(float *data,
                    int U, int V, int S, int T, int selectedDim);


#ifdef __cplusplus
}
#endif

#endif