#include "mmkernel.h"
#include <stdio.h>

__global__ void MatMulA(float * matA, float * matB, float * matC, unsigned int dim) {

}


#define WIDTH  25
#define HEIGHT 25

__global__ void MatMulB(float * matA, float * matB, float * matC, unsigned int dim) {

}

cudaError_t MatMul(float * matA, float * matB, float * matC, size_t dim) {

    //MatMul <<<32, 512>>> (matA, matB, matC, dim);

    dim3 grid(16, 16, 1);
    dim3 block(WIDTH, HEIGHT ,1);
    MatMulB <<<grid, block>>> (matA, matB, matC, dim);
            
    return cudaDeviceSynchronize();
}

