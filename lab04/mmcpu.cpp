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
    size_t  count;
    char    mata_fn[128], matb_fn[128], matc_fn[128];

    //Timers
    Timer dataLoad_tm, comp_tm;

    //Loading matrix-A and -B from file for matrix multiplication
    //Loading matrix-C from file for verifying the result produced by CPU
    dataLoad_tm.Start();
    sprintf(mata_fn, "vecA_%ld.bin", dim);
    sprintf(matb_fn, "vecB_%ld.bin", dim);
    sprintf(matc_fn, "vecC_%ld.bin", dim);
    CHECK(    binReadAsArrayNP<float>(mata_fn, NULL, &mata_h, &count));
    CHECK(    binReadAsArrayNP<float>(matb_fn, NULL, &matb_h, &count));
    CHECK(    binReadAsArrayNP<float>(matc_fn, NULL, &matc_hv, &count));
    matc_h = new float [dim * dim];
    dataLoad_tm.Stop();

    //Perform matrix multiplication on CPU
    comp_tm.Start();
    #pragma omp parallel for
    for (size_t iIdx=0;iIdx<dim;iIdx++) {
        size_t i = iIdx * dim;

        for (size_t jIdx=0;jIdx<dim;jIdx++) {
            size_t j = jIdx * dim;

            float temp = 0.0;
            for (size_t kIdx=0;kIdx<dim;kIdx++) {
                temp += mata_h[i + kIdx] * matb_h[j + kIdx];
                matc_h[(iIdx * dim) + jIdx] = temp;
            }
        }
    }
    comp_tm.Stop();

    //Verify the results
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

    //Show timing results
    Timer::Duration dataLoad_dur = dataLoad_tm.GetDuration();
    Timer::Duration comp_dur = comp_tm.GetDuration();
    printf("\n===== Time used =====\n");
    printf("Data load:   %.2f\tus\n", dataLoad_dur.raw);
    printf("Computation: %.2f\tus\n\n\n", comp_dur.raw);

    return 0;
}

