#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "bin_reader.h"
#include "timer.h"
#include "cuda_runtime_api.h"
#include "mmkernel.h"

#define CHECK(exp)                                          \
    do {                                                    \
        if (exp != 0) {                                     \
            printf("Runtime error at line %d\n", __LINE__); \
            exit(-1);                                       \
        }                                                   \
    } while(0)

#define CUCHECK(exp)                                                                         \
    do {                                                                                     \
        cudaError_t ce = exp;                                                                \
        if (ce != cudaSuccess) {                                                             \
            printf("CUDA runtime error at line %d: %s\n", __LINE__, cudaGetErrorString(ce)); \
            exit(-1);                                                                        \
        }                                                                                    \
    } while(0)

int main(int argc, char ** argv) {

    if (argc < 2) {
        printf("Error: please specify matrix dimension.\n");
        exit(-1);
    }
    size_t dim = atoi(argv[1]);
    if (dim <= 0) {
        printf("Error: invalid matrix dimension\n");
        exit(-1);
    }

    float * mata_h, * matb_h, * matc_h, * matc_hv;
    float * mata_d, * matb_d, * matc_d;
    size_t  count;
    char    mata_fn[128], matb_fn[128], matc_fn[128];

    Timer dataLoad_tm, cpu2gpu_tm, gpu2cpu_tm, comp_tm;

    //Loading matrix-A and -B from file for matrix multiplication
    //Loading matrix-C from file for verifying the result from GPU
    dataLoad_tm.Start();
    sprintf(mata_fn, "vecA_%ld.bin", dim);
    sprintf(matb_fn, "vecB_%ld.bin", dim);
    sprintf(matc_fn, "vecC_%ld.bin", dim);
    CHECK(    binReadAsArrayNP<float>(mata_fn, NULL, &mata_h, &count));
    CHECK(    binReadAsArrayNP<float>(matb_fn, NULL, &matb_h, &count));
    CHECK(    binReadAsArrayNP<float>(matc_fn, NULL, &matc_hv, &count));
    matc_h = new float [dim * dim];
    dataLoad_tm.Stop();

    //Allocate memory on GPU for matrices
    cpu2gpu_tm.Start();
    size_t memSize = dim * dim * sizeof(float);
    CUCHECK(    cudaMalloc((void **) &mata_d, memSize));
    CUCHECK(    cudaMalloc((void **) &matb_d, memSize));
    CUCHECK(    cudaMalloc((void **) &matc_d, memSize));

    //Copy matrix-A and -B from the host to the device
    CUCHECK(    cudaMemcpy(mata_d, mata_h, memSize, cudaMemcpyHostToDevice));
    CUCHECK(    cudaMemcpy(matb_d, matb_h, memSize, cudaMemcpyHostToDevice));
    CUCHECK(    cudaMemset(matc_d, 0, memSize));
    cpu2gpu_tm.Stop();

    //Perform matrix multiplication by invoking the kernel
    comp_tm.Start();
    CUCHECK(    MatMul(mata_d, matb_d, matc_d, dim));
    comp_tm.Stop();

    //Copy matrix-C which is the result from GPU back to the host
    gpu2cpu_tm.Start();
    CUCHECK(    cudaMemcpy(matc_h, matc_d, memSize, cudaMemcpyDeviceToHost));
    gpu2cpu_tm.Stop();

    //Verify the results copied back from GPU
    for (size_t idx=0;idx<dim * dim;idx++) {
        float diff = fabsf(matc_h[idx] - matc_hv[idx]) / matc_hv[idx];
        if (diff > 0.001) {
            printf("Verification: FAILED (%ld)\n", idx);
            printf("h: %.6f / hv: %.6f\n", matc_h[idx], matc_hv[idx]);
            exit(-1);
        }
    }
    printf("Verification: PASSED\n");


    //Release the resources
    CHECK(    binDiscardArrayNP(mata_h));
    CHECK(    binDiscardArrayNP(matb_h));
    CHECK(    binDiscardArrayNP(matc_hv));
    delete [] matc_h;
    CUCHECK(    cudaFree(mata_d));
    CUCHECK(    cudaFree(matb_d));
    CUCHECK(    cudaFree(matc_d));

    //Show timing results
    Timer::Duration dataLoad_dur = dataLoad_tm.GetDuration();
    Timer::Duration cpu2gpu_dur = cpu2gpu_tm.GetDuration();
    Timer::Duration comp_dur = comp_tm.GetDuration();
    Timer::Duration gpu2cpu_dur = gpu2cpu_tm.GetDuration();

    printf("\n===== Time used =====\n");
    printf("Data load:           %.2f\tus\n", dataLoad_dur.raw);
    printf("CPU-to-GPU transfer: %.2f\tus\n", cpu2gpu_dur.raw);
    printf("Computation:         %.2f\tus\n", comp_dur.raw);
    printf("GPU-to-CPU transfer: %.2f\tus\n\n\n", gpu2cpu_dur.raw);

    return 0;
}

