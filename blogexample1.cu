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
    const int n = 1<<20;              // ~1 million elements
    const size_t bytes = n * sizeof(float);
    // 1. Allocate and fill arrays in CPU memory
    float *h_a   = (float *)malloc(bytes);
    float *h_b   = (float *)malloc(bytes);
    float *h_c   = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
        h_c[i] = 3.5f;
    }


    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice));

    const int threadsPerBlock  = 256;
    const int blocks =  (n+ threadsPerBlock -1) / threadsPerBlock; //this rounds up
    vecAdd<<<blocks, threadsPerBlock>>>(d_a,d_b,d_c, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // 7. Verify on the CPU
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if((h_a[i] + h_b[i]) != h_c[i]  ){errors++;}
    }
    printf("Launched %d blocks x %d threads. Errors: %d\n",
           blocks, threadsPerBlock, errors);
    printf("Sample value = %.6f\n", h_c[3]);
}
