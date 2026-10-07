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

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                   \
                    cudaGetErrorString(err), __FILE__, __LINE__);         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

int main()
{
    //dim3 grid(4,4);
    //dim3 block(32,32); // max 1024 threads per block, so 64x64 (4096) would fail to launch
    //add<<<grid, block>>>(A, B, C, n);
    // don't know how to work index multidimentional stuff yet
    const int n = 1<<10;              // ~1 million elements
    const size_t bytes = n * sizeof(float);
    // 1. Allocate and fill arrays in CPU memory
    float *h_a   = (float *)malloc(bytes);
    float *h_b   = (float *)malloc(bytes);
    float *h_c   = (float *)malloc(bytes);
    float *h_d   = (float *)malloc(bytes);
    float *h_e   = (float *)malloc(bytes);
    float *h_f   = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
        h_c[i] = 3.5f;
        h_d[i] = 2.0f;
        h_e[i] = 10.0f;
        h_f[i] = 0.5f;
    }


    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    CUDA_CHECK(cudaMalloc(&d_d, bytes));
    CUDA_CHECK(cudaMalloc(&d_e, bytes));
    CUDA_CHECK(cudaMalloc(&d_f, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, h_d, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e, h_e, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, h_f, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, h_out, bytes, cudaMemcpyHostToDevice));

    const int threadsPerBlock  = 256;
    const int blocks =  (n+ threadsPerBlock -1) / threadsPerBlock; //this rounds up
    vecAdd<<<blocks, threadsPerBlock>>>(d_a,d_b,d_c, n);
    vecMult<<<blocks, threadsPerBlock>>>(d_a,d_b,d_d, n);
    vecAdd<<<blocks, threadsPerBlock>>>(d_c,d_d,d_e, n);
    vecPow<<<blocks, threadsPerBlock>>>(d_e,d_f,d_out, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_e));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_out));

    // 7. Verify on the CPU
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_out[i] < 2.2f || h_out[i] > 2.4f) errors++;
    }
    printf("Launched %d blocks x %d threads. Errors: %d\n",
           blocks, threadsPerBlock, errors);
    printf("Sample value = %.6f\n", h_out[3]);
}
