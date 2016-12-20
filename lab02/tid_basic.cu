#include <iostream>

#define DEFAULT_BLOCK_COUNT 128
#define DEFAULT_TPB_COUNT   128

using namespace std;

int   blockCnt  = DEFAULT_BLOCK_COUNT;
int   tpbCnt    = DEFAULT_TPB_COUNT;
int   totalThreads;

int * id; 

//Declaration of pointers to CPU memory (host)
int * blockx_h;
int * idx_h;

//Declaration of pointers to GPU memory (device)
int * blockx_d;
int * idx_d;

__global__ void MyFirstKernel(int * blkx, int * idx) {

    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    blkx[threadId] = blockIdx.x;
    idx[threadId]  = threadIdx.x;
}

int ParseArguments(int argc, char ** argv) {

    if (argc == 1)
        return 0;

    if (argc != 3) {
        cout << "Error: Not enough arguments specified." << endl;
        return -1;
    }
    
    for (int i=1;i<3;i++) {
        if (atoi(argv[i]) <= 0) {
            cout << "Error: Invalid arguments" << endl;
            return -1;
        }
    }

    blockCnt = atoi(argv[1]);
    tpbCnt = atoi(argv[2]);

    if (tpbCnt > 1024) {
        cout << "Error: Too many threads per block (<= 1024)" << endl;
        return -1;
    }

    return 0;
}

void CheckCudaError(cudaError_t ce) {

    if (ce == cudaSuccess)
        return;

    cout << "Error: " << cudaGetErrorString(ce) << endl;
    exit(-1);
}

int AllocateHostMemory(int totalThreads) {

    try { 
        blockx_h = new int[totalThreads];
        idx_h = new int[totalThreads];
    }
    catch(bad_alloc e) {
        return -1;
    }

    return 0;
}

int main(int argc, char ** argv) {

    if (ParseArguments(argc, argv))
        exit(-1);

    totalThreads = blockCnt * tpbCnt;
    int totalMem = totalThreads * sizeof(int);

    if (AllocateHostMemory(totalThreads)) {
        cout << "Error: Memory allocation on host failed." << endl;
        exit(-1);
    }

    //Allocate memory on GPU to store block identifiers
    CheckCudaError(    cudaMalloc(&blockx_d, totalMem));

    //Allocate memory on GPU to store thread identifiers
    CheckCudaError(    cudaMalloc(&idx_d, totalMem));

    //Clear allocated memory block on GPU for storing block identifiers to 0
    CheckCudaError(    cudaMemset(blockx_d, 0, totalMem));

    //Clear allocated memory block on GPU for storing thread identifiers to 0
    CheckCudaError(    cudaMemset(idx_d, 0, totalMem));

    //Invoke the kernel
    MyFirstKernel <<<blockCnt, tpbCnt>>>(blockx_d, idx_d);
    cudaDeviceSynchronize();

    //Copying data generated by the kernel from GPU back to CPU
    CheckCudaError(
        cudaMemcpy(blockx_h, blockx_d, totalMem, cudaMemcpyDeviceToHost));
    CheckCudaError(    
        cudaMemcpy(idx_h, idx_d, totalMem, cudaMemcpyDeviceToHost));

    for (int i=0;i<totalThreads;i++)
        cout << "[" << i << "]\t" << 
            blockx_h[i] << "\t" <<
            idx_h[i] << endl;

    return 0;
}
