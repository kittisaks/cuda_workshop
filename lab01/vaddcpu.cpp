#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "bin_reader.h"
#include "timer.h"

#define CHECK(exp)                                          \
    do {                                                    \
        if (exp != 0) {                                     \
            printf("Runtime error at line %d\n", __LINE__); \
            exit(-1);                                       \
        }                                                   \
    } while(0)

int main(int argc, char ** argv) {

    float * veca_h, * vecb_h, * vecc_h, * vecc_hv;
    size_t  count;

    //Loading vector-A and -B from file for calculation on GPU
    //Loading vector-C from file for verifying the results from GPU
    CHECK(    binReadAsArrayNP<float>("vecA.bin", NULL, &veca_h, &count));
    CHECK(    binReadAsArrayNP<float>("vecB.bin", NULL, &vecb_h, &count));
    CHECK(    binReadAsArrayNP<float>("vecC.bin", NULL, &vecc_hv, &count));
    vecc_h = new float [count];


    //Perform the vector addition by calling the kernel
    //Play with scaling and configuration (WBB)
    Timer compTimer;
    compTimer.Start();

    omp_set_num_threads(4);

    #pragma omp parallel for shared(vecc_h, veca_h, vecb_h)
    for (size_t i=0;i<count;i++)
        vecc_h[i] = veca_h[i] + vecb_h[i];

    compTimer.Stop();
    Timer::Duration d = compTimer.GetDuration();
    printf("Computation time: %.2f us\n", d.raw);


    //Verification
    for (size_t idx=0;idx<count;idx++) {
        if (vecc_h[idx] != vecc_hv[idx]) {
            printf("Verification: FAILED (%ld)\n", idx);
            exit(-1);
        }
    }
    printf("Verification: PASSED\n");

    //Release resources after completed calculation (WBB)
    CHECK(    binDiscardArrayNP(veca_h));
    CHECK(    binDiscardArrayNP(vecb_h));
    CHECK(    binDiscardArrayNP(vecc_hv));

    return 0;
}

