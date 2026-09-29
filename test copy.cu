#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

//kernel definition example
__global__ void doAdd(float* A, float* B, float* C)
{
    int workIdx = threadIdx.x + blockDim.x * blockIdx.x;
    C[workIdx] = B[workIdx] + A[workIdx];  
    //no loops needed here, becuase all calulations happen simultaneously, 
    //workIndex tells each thread which memory to handle
}

int main()
{
    //dim3 grid(4,4);
    //dim3 block(64,64);
    //doAdd<<<grid, block>>>(A, B, C);
    // don't know how to work index multidimentional stuff yet

    float* A;
    float* B;
    float* C;
    doAdd<<<16, 64>>>(A,B,C);
}