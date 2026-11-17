// vector_add.cu
// Compile:  nvcc vector_add.cu -o vector_add
// Run:      ./vector_add

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Most CUDA API calls return an error code. This macro checks it and
// stops with a readable message instead of failing silently.
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                   \
                    cudaGetErrorString(err), __FILE__, __LINE__);         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// ---------------------------------------------------------------------
// The kernel: runs on the GPU. Each thread handles one element.
// ---------------------------------------------------------------------
__global__ void add(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < n) {                                    // extra threads do nothing
        out[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------
// Host code: runs on the CPU and orchestrates everything.
// Naming convention: h_ = host (CPU RAM), d_ = device (GPU VRAM).
// ---------------------------------------------------------------------
int main() {
    const int n = 1 << 20;                 // ~1 million elements
    const size_t bytes = n * sizeof(float);

    // 1. Allocate and fill arrays in CPU memory
    float *h_a   = (float *)malloc(bytes);
    float *h_b   = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 2. Allocate arrays in GPU memory
    //    These pointers hold GPU addresses. The CPU must never dereference them.
    float *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // 3. Copy inputs from CPU RAM to GPU VRAM (across the PCIe bus)
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 4. Choose a launch configuration and launch the kernel
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;  // round up
    add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_out, n);

    // 5. Check for launch errors, then wait for the GPU to finish
    CUDA_CHECK(cudaGetLastError());        // catches bad launch configurations
    CUDA_CHECK(cudaDeviceSynchronize());   // catches errors during execution

    // 6. Copy the result back from GPU VRAM to CPU RAM
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // 7. Verify on the CPU
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_out[i] != 3.0f) errors++;
    }
    printf("Launched %d blocks x %d threads. Errors: %d\n",
           blocks, threadsPerBlock, errors);

    // 8. Free GPU and CPU memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    free(h_a);
    free(h_b);
    free(h_out);

    return 0;
}
