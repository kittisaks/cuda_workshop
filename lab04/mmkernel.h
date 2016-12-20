#ifndef _MMKERNEL_H
#define _MMKERNEL_H

#include "cuda_runtime_api.h"

cudaError_t MatMul(float * matA, float * matB, float * matC, size_t dim);

#endif //_MMKERNEL_H

