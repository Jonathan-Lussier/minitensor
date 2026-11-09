#include "tensor.h"
#include "internal.cuh"
using namespace detail;

namespace {
    //Transpose
    __global__ void transposeKernel(const float* a, int rows, int cols, float* out){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= rows * cols) return;
        int r = i / cols, c = i % cols;
        out[c * rows + r] = a[i];
    }

    //naive matmul product
    __global__ void naiveMatMulKernel(const float* a_mat, const float * b_mat, float * out, 
                                    int rows, int innerDim, int cols, int flattendn) {
                                        //flattendn is rows*cols
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= flattendn) return;

        float temp = 0.0f;
        int curr_row = i / cols;
        int curr_col = i % cols;
        for(int j = 0; j < innerDim; j++){
            temp += a_mat[curr_row * innerDim + j] * b_mat[j * cols + curr_col];
        } 
        out[i] = temp;
    }

    //matmul a different way
    __global__ void jonsMatMul(){
    //need transpose for this
    }


    //summing kernel
    template <int BLOCK_SIZE>
    __global__ void sumKernel(const float* in, float* out, int n) {
        __shared__ float sdata[BLOCK_SIZE];          // fast memory shared by this block's threads
        int tid = threadIdx.x;

        // Step 1: each thread sums several elements into a register (grid-stride loop)
        float local = 0.0f;
        for (int i = blockIdx.x * BLOCK_SIZE + tid; i < n; i += BLOCK_SIZE * gridDim.x) {
            local += in[i];
        }
        sdata[tid] = local;
        __syncthreads();                         // wait until every thread has written

        // Step 2: tree reduction, halving the active threads each round
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();                     // finish this round before starting the next
        }

        // Step 3: thread 0 writes this block's result
        if (tid == 0) out[blockIdx.x] = sdata[0];
    }

        //kernels that sum along an axis and collapse 1 dimension, spilt into 2 functions 
    __global__ void sumAxis0Kernel(const float* in, float* out, int rows, int cols) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= cols) return;
        float acc = 0.0f;
        for (int r = 0; r < rows; r++) acc += in[r * cols + c];
        out[c] = acc;
    }

    //this is more complcated bc there's no easy way to coalesce
    template <int BLOCK_SIZE>
    __global__ void sumAxis1Kernel(const float* in, float* out, int rows, int cols) {
        __shared__ float sdata[BLOCK_SIZE];
        int tid = threadIdx.x;
        const float* row = in + blockIdx.x * cols;   // this block's row

        //threads stride across the row, so neighbors read neighbors (coalesced)
        float local = 0.0f;
        for (int c = tid; c < cols; c += BLOCK_SIZE) local += row[c];
        sdata[tid] = local;
        __syncthreads();

        //tree reduction
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) out[blockIdx.x] = sdata[0];
    }

    //maxxing kernel
    template <int BLOCK_SIZE>
    __global__ void maxKernel(const float* in, float* out, int n) {
        __shared__ float sdata[BLOCK_SIZE];          // fast memory shared by this block's threads
        int tid = threadIdx.x;

        // Step 1: each thread sums several elements into a register (grid-stride loop)
        float local = -INFINITY;
        for (int i = blockIdx.x * BLOCK_SIZE + tid; i < n; i += BLOCK_SIZE * gridDim.x) {
            if (local< in[i]) local = in[i];
        }
        sdata[tid] = local;
        __syncthreads();                         // wait until every thread has written

            // Step 2: tree reduction, halving the active threads each round
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
            __syncthreads();                     // finish this round before starting the next
        }

        // Step 3: thread 0 writes this block's result
        if (tid == 0) out[blockIdx.x] = sdata[0];
    }

    //minning kernel
    template <int BLOCK_SIZE>
    __global__ void minKernel(const float* in, float* out, int n) {
        __shared__ float sdata[BLOCK_SIZE];          // fast memory shared by this block's threads
        int tid = threadIdx.x;

        // Step 1: each thread sums several elements into a register (grid-stride loop)
        float local = INFINITY;
        for (int i = blockIdx.x * BLOCK_SIZE + tid; i < n; i += BLOCK_SIZE * gridDim.x) {
            if (local > in[i]) local = in[i];
        }
        sdata[tid] = local;
        __syncthreads();                         // wait until every thread has written

            // Step 2: tree reduction, halving the active threads each round
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = fminf(sdata[tid], sdata[tid+s]);
            __syncthreads();                     // finish this round before starting the next
        }

        // Step 3: thread 0 writes this block's result
        if (tid == 0) out[blockIdx.x] = sdata[0];
    }

    //tiled matmul,  much faster
    template <int TILE>
    __global__ void tiledMatMulKernel(const float * A, const float * B, float * out,
                                        int M, int K, int N){
        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];
        int tx = threadIdx.x; 
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE + ty;   // output row this thread works on
        int col = blockIdx.x * TILE + tx;   // output column this thread works on

        float acc = 0.0f;
        int numtiles = (K + TILE -1)/TILE;

        for (int t = 0; t < numtiles; t++){
            int aCol = t * TILE + tx;
            int bRow = t * TILE + ty;
            As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
            Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

            __syncthreads();
            #pragma unroll
            for (int k = 0; k < TILE; k++){
                acc += As[ty][k] * Bs[k][tx];
            }
            __syncthreads();
        }
        
        if (row < M && col < N) out[row * N + col] = acc;
    }
}

//public tensor functions
Tensor transpose(const Tensor& a){
    if (a.shape().size() != 2)
        throw std::runtime_error("transpose: expected a 2D tensor, got " + shapeToString(a.shape()));

    int rows =  a.shape()[0];
    int cols =  a.shape()[1];
    Tensor out({cols,rows});
    int n = out.numel(), threads = 256, blocks = (n + threads - 1) / threads;
    if (n > 0){
        transposeKernel<<<blocks, threads>>>(a.data(), rows, cols, out.data());
        cudaCheck(cudaGetLastError(), "transpose launch");
    }

    if (needsGrad(a)){
        recordNode(out, "transpose", {a}, [](const Tensor& g) {
            return std::vector<Tensor>{ transpose(g) };
        });
    }

    return out;
    }

    Tensor naiveMatMul(const Tensor& a, const Tensor& b){
    
    if (a.shape().size() > 2 || b.shape().size() > 2){
        throw std::runtime_error("matmul got 3D shapes: " +shapeToString(a.shape()) +
                                ", " + shapeToString(b.shape()));
    }
    
    if (a.shape().size() == 1 || b.shape().size() == 1){
        throw std::runtime_error("matmul cannot handle vector-matrix multiplication: " +shapeToString(a.shape()) +
                                ", " + shapeToString(b.shape()));
    }
    int M =  a.shape()[0];
    int K =  a.shape()[1];
    int N =  b.shape()[1];
    int numdot = M*N;
    if (b.shape()[0] != K){
        throw std::runtime_error("inner shapes: " + std::to_string(K) + ", "
                                  + std::to_string(b.shape()[1]) + " don't match");
    }

    Tensor out({M,N});
    if (numdot == 0) return out;                       // launching 0 blocks is an error

    int threads = 256;
    int blocks = (numdot + threads - 1)/ threads;
    naiveMatMulKernel<<<blocks, threads>>>(a.data(), b.data(), out.data(), M, K, N, numdot);
    cudaCheck(cudaGetLastError(), "matmul launch");
    return out;
}

Tensor tiledMatMul(const Tensor& a, const Tensor& b){
    
    if (a.shape().size() > 2 || b.shape().size() > 2){
        throw std::runtime_error("matmul got 3D shapes: " +shapeToString(a.shape()) +
                                ", " + shapeToString(b.shape()));
    }
    
    if (a.shape().size() == 1 || b.shape().size() == 1){
        throw std::runtime_error("matmul cannot handle vector-matrix multiplication: " +shapeToString(a.shape()) +
                                ", " + shapeToString(b.shape()));
    }
    int M =  a.shape()[0];
    int K =  a.shape()[1];
    int N =  b.shape()[1];
    int numdot = M*N;
    if (b.shape()[0] != K){
        throw std::runtime_error("inner shapes: " + std::to_string(K) + ", "
                                  + std::to_string(b.shape()[0]) + " don't match");
    }

    Tensor out({M,N});
    if (numdot == 0) return out;                       // launching 0 blocks is an error

    constexpr int tile_size = 16;
    dim3 threads(tile_size, tile_size);
    dim3 blocks((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);
    tiledMatMulKernel<tile_size><<<blocks, threads>>>(a.data(), b.data(), out.data(), M, K, N);
    cudaCheck(cudaGetLastError(), "matmul launch");
    return out;
}

Tensor jonsMatMul(const Tensor& a, const Tensor& b){
    
    if (a.shape().size() > 2 || b.shape().size() > 2){
        throw std::runtime_error("matmul got 3D shapes: " +shapeToString(a.shape()) +
                                ", " + shapeToString(b.shape()));
    }
    
    if (a.shape().size() == 1 || b.shape().size() == 1){
        throw std::runtime_error("matmul cannot handle vector-matrix multiplication: " +shapeToString(a.shape()) +
                                ", " + shapeToString(b.shape()));
    }
    int M =  a.shape()[0];
    int K =  a.shape()[1];
    int N =  b.shape()[1];
    int numdot = M*N;
    if (b.shape()[0] != K){
        throw std::runtime_error("inner shapes: " + std::to_string(K) + ", "
                                  + std::to_string(b.shape()[1]) + " don't match");
    }

    Tensor out({M,N});
    if (numdot == 0) return out;                       // launching 0 blocks is an error

    int threads = 256;
    int blocks = (numdot + threads - 1)/ threads;
    naiveMatMulKernel<<<blocks, threads>>>(a.data(), b.data(), out.data(), M, K, N, numdot);
    cudaCheck(cudaGetLastError(), "matmul launch");
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    Tensor out = tiledMatMul(a, b);
    if (needsGrad(a, b)) {
        recordNode(out, "matmul", {a, b}, [a, b](const Tensor& g) {
            return std::vector<Tensor>{ a.requiresGrad() ? matmul(g, transpose(b)) : a,
                                        b.requiresGrad() ? matmul(transpose(a), g) : b };
        });
    }
    return out;
}

//pairwise sum
Tensor sum(const Tensor& a) {
    constexpr int threads = 256;
    int n = a.numel();
    if (n == 0) return Tensor::scalar(0.0f);   // launching 0 blocks is an error

    int blocks = std::min((n + threads - 1) / threads, 1024);
    Tensor partials({blocks});
    sumKernel<threads><<<blocks, threads>>>(a.data(), partials.data(), n);
    cudaCheck(cudaGetLastError(), "sum pass 1");

    Tensor out(std::vector<int>{});             // scalar output, shape []
    sumKernel<threads><<<1, threads>>>(partials.data(), out.data(), blocks);
    cudaCheck(cudaGetLastError(), "sum pass 2");

    
    if (needsGrad(a)){
        recordNode(out, "sum", {a}, [shape = a.shape()](const Tensor& g) {
            return std::vector<Tensor>{ Tensor::full(shape, 1.0f) * g };   // broadcasts g up to the input's shape
        });
    }
    return out;
}

Tensor sumAxis(const Tensor& a, int axis) {
    if (a.shape().size() != 2)
        throw std::runtime_error("sumAxis: expected a 2D tensor, got " + shapeToString(a.shape()));
    if (axis != 0 && axis != 1)
        throw std::runtime_error("sumAxis: axis must be 0 or 1, got " + std::to_string(axis));

    int rows = a.shape()[0], cols = a.shape()[1];
    Tensor out(axis == 0 ? std::vector<int>{1, cols} : std::vector<int>{rows, 1});
    if (out.numel() == 0) return out;            // launching 0 blocks is an error

    constexpr int threads = 256;
    if (axis == 0) {
        int blocks = (cols + threads - 1) / threads;
        sumAxis0Kernel<<<blocks, threads>>>(a.data(), out.data(), rows, cols);
    } else {
        sumAxis1Kernel<threads><<<rows, threads>>>(a.data(), out.data(), rows, cols);
    }
    cudaCheck(cudaGetLastError(), "sumAxis launch");

    if (needsGrad(a)) {
        recordNode(out, "sumAxis", {a}, [shape = a.shape()](const Tensor& g) {
            return std::vector<Tensor>{ Tensor::full(shape, 1.0f) * g };
        });
    }
    return out;
}

Tensor dot(const Tensor & a, const Tensor & b){
    if (a.shape().size() != 1 || b.shape().size() != 1){throw std::runtime_error("dot product got non vector shapes:"
                                                        + shapeToString(a.shape()) + shapeToString(b.shape()));}
    if (a.shape()[0] != b.shape()[0]){throw std::runtime_error("dot product got non equal vector lengths:"
                                                    + shapeToString(a.shape()) + shapeToString(b.shape()));}
    return(sum(a*b));
}

Tensor mean(const Tensor& a) {
    if (a.numel() == 0){
        throw std::runtime_error("empty tensor, devide by 0 error" + a.numel()); 
    }
    Tensor out = unaryOp(sum(a), ScaleOp{1.0f / a.numel()});

    
    if (needsGrad(a)) {
        recordNode(out, "mean", {a}, [shape = a.shape(), n = a.numel()](const Tensor& g) {
        return std::vector<Tensor>{ Tensor::full(shape, 1.0f / n) * g };
        });
    }

    return out;
}

Tensor max(const Tensor& a) {
    constexpr int threads = 256;
    int n = a.numel();
    if (n == 0) return Tensor::scalar(0.0f);   // launching 0 blocks is an error

    int blocks = std::min((n + threads - 1) / threads, 1024);
    Tensor partials({blocks});
    maxKernel<threads><<<blocks, threads>>>(a.data(), partials.data(), n);
    cudaCheck(cudaGetLastError(), "max pass 1");

    Tensor out(std::vector<int>{});             // scalar output, shape []
    maxKernel<threads><<<1, threads>>>(partials.data(), out.data(), blocks);
    cudaCheck(cudaGetLastError(), "max pass 2");
    return out;
}

Tensor min(const Tensor& a) {
    constexpr int threads = 256;
    int n = a.numel();
    if (n == 0) return Tensor::scalar(0.0f);   // launching 0 blocks is an error

    int blocks = std::min((n + threads - 1) / threads, 1024);
    Tensor partials({blocks});
    minKernel<threads><<<blocks, threads>>>(a.data(), partials.data(), n);
    cudaCheck(cudaGetLastError(), "min pass 1");

    Tensor out(std::vector<int>{});             // scalar output, shape []
    minKernel<threads><<<1, threads>>>(partials.data(), out.data(), blocks);
    cudaCheck(cudaGetLastError(), "min pass 2");
    return out;
}