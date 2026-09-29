#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

//kernel definition example
__global__ void vecAdd(float* a, float* b, float* out, int vectorLength)
{
    int workIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIdx < vectorLength)
    //^^This handles imperfect vector lengths, where one block will launch with some inactive threads
    {
        //no loops needed here, because all calculations happen simultaneously,
        //workIdx tells each thread which memory to handle
        out[workIdx] = a[workIdx] + b[workIdx];
    }

}

__global__ void vecSub(float* a, float* b, float* out, int vectorLength)
{
    int workIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIdx < vectorLength)
    //^^This handles imperfect vector lengths, where one block will launch with some inactive threads
    {
        //no loops needed here, because all calculations happen simultaneously,
        //workIdx tells each thread which memory to handle
        out[workIdx] = a[workIdx] - b[workIdx];
    }

}

__global__ void vecMult(float* a, float* b, float* out, int vectorLength)
{
    int workIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIdx < vectorLength)
    //^^This handles imperfect vector lengths, where one block will launch with some inactive threads
    {
        //no loops needed here, because all calculations happen simultaneously,
        //workIdx tells each thread which memory to handle
        out[workIdx] = a[workIdx] * b[workIdx];
    }

}

__global__ void vecDivide(float* a, float* b, float* out, int vectorLength)
{
    int workIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIdx < vectorLength)
    //^^This handles imperfect vector lengths, where one block will launch with some inactive threads
    {
        //no loops needed here, because all calculations happen simultaneously,
        //workIdx tells each thread which memory to handle
        out[workIdx] = a[workIdx] / b[workIdx]; // b == 0 gives inf
    }

}

// named vecPow so it doesn't clash with the built-in pow() math function
__global__ void vecPow(float* a, float* b, float* out, int vectorLength)
{
    int workIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIdx < vectorLength)
    //^^This handles imperfect vector lengths, where one block will launch with some inactive threads
    {
        //powf is a free function taking (base, exponent); powf = float version, pow = double
        out[workIdx] = powf(a[workIdx], b[workIdx]);
    }

}

// named vecExp so it doesn't clash with the built-in exp() math function
// exp only needs one input: out = e^a
__global__ void vecExp(float* a, float* out, int vectorLength)
{
    int workIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIdx < vectorLength)
    //^^This handles imperfect vector lengths, where one block will launch with some inactive threads
    {
        out[workIdx] = expf(a[workIdx]);
    }

}

int main()
{
    //dim3 grid(4,4);
    //dim3 block(32,32); // max 1024 threads per block, so 64x64 (4096) would fail to launch
    //add<<<grid, block>>>(A, B, C, n);
    // don't know how to work index multidimentional stuff yet

    const int n = 1000;
    float* A;
    float* B;
    float* C;

}
