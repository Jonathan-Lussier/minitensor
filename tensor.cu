#include "tensor.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <algorithm>

//simaliar to p1, but built for python binding

namespace {

//helper functions
inline void cudaCheck(cudaError_t err, const char* what){
    if (err != cudaSuccess){
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }

}

// Pad a shape to 2D with leading 1s: {10} -> {1, 10}, {} -> {1, 1}
std::vector<int> to2D(const std::vector<int>& shape) {
    if (shape.size() > 2) throw std::runtime_error("broadcast: only up to 2D supported");
    std::vector<int> s = shape;
    while (s.size() < 2) s.insert(s.begin(), 1);
    return s;
}


//operator functions
//Binary
struct AddOp { __device__ float operator()(float x, float y) const { return x + y; } };
struct SubOp { __device__ float operator()(float x, float y) const { return x - y; } };
struct MulOp { __device__ float operator()(float x, float y) const { return x * y; } };
struct DivOp { __device__ float operator()(float x, float y) const { return x / y; } };
//Unary
struct NegOp { __device__ float operator()(float x) const { return x * (-1); } };
struct ScaleOp {
    float s;
    __device__ float operator()(float x) const { return x * s; }
};

//kernel defintions
//binary kernel defintion
template <typename Op>
__global__ void binaryKernel(const float* a, const float* b, float* out, int n, Op op) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}

// 2D broadcasting kernel
template <typename Op>
__global__ void broadcastKernel2D(const float* a, int aStrideR, int aStrideC,
                                  const float* b, int bStrideR, int bStrideC,
                                  float* out, int rows, int cols, Op op) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < rows * cols) {
        int r = i / cols, c = i % cols;
        out[i] = op(a[r * aStrideR + c * aStrideC], b[r * bStrideR + c * bStrideC]);
    }
}

//unary kernel
template <typename Op>
__global__ void unaryKernel(const float* a, float* out, int n, Op op) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = op(a[i]);
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

//host side kernel wrappers
//2D broadcasting wrapper
template <typename Op>
Tensor broadcastOp(const Tensor& a, const Tensor& b, Op op){
    std::vector<int> as = to2D(a.shape()), bs = to2D(b.shape());

    std::vector<int> out2D(2);
    for (int d = 0; d < 2; d++) {
        if (as[d] == bs[d])  out2D[d] = as[d];
        else if (as[d] == 1) out2D[d] = bs[d];
        else if (bs[d] == 1) out2D[d] = as[d];
        else throw std::runtime_error("broadcast: incompatible shapes " +
                                    shapeToString(a.shape()) + " and " +
                                    shapeToString(b.shape()));    
    }
    // Output keeps the larger of the two original ranks: (3) + (1) -> (3), not (1, 3)
    size_t maxRank = std::max(a.shape().size(), b.shape().size());
    Tensor out(std::vector<int>(out2D.end() - maxRank, out2D.end()));

    int aStrideR = (as[0] == 1) ? 0 : as[1], aStrideC = (as[1] == 1) ? 0 : 1;
    int bStrideR = (bs[0] == 1) ? 0 : bs[1], bStrideC = (bs[1] == 1) ? 0 : 1;

        int rows = out2D[0], cols = out2D[1], n = rows * cols;
    int threads = 256, blocks = (n + threads - 1) / threads;
    broadcastKernel2D<<<blocks, threads>>>(a.data(), aStrideR, aStrideC,
                                           b.data(), bStrideR, bStrideC,
                                           out.data(), rows, cols, op);
    cudaCheck(cudaGetLastError(), "broadcast kernel launch");
    return out;
}

//host side 2in-1out operator wrapper (binary operations)
template <typename Op>
Tensor binaryOp(const Tensor& a, const Tensor& b, Op op) {
    if (a.shape() != b.shape()) return broadcastOp(a, b, op);   // if shapes are different. otherwise this kernel is faster
    Tensor out(a.shape());
    int n = out.numel(), threads = 256, blocks = (n + threads - 1) / threads;
    binaryKernel<<<blocks, threads>>>(a.data(), b.data(), out.data(), n, op);
    cudaCheck(cudaGetLastError(), "kernel launch");
    return out;
}

//host side 1in-1out operator wrapper
template <typename Op>
Tensor unaryOp(const Tensor& a, Op op) {
    Tensor out(a.shape());
    int n = out.numel(), threads = 256, blocks = (n + threads - 1) / threads;
    unaryKernel<<<blocks, threads>>>(a.data(), out.data(), n, op);
    cudaCheck(cudaGetLastError(), "kernel launch");
    return out;
}



}//end of unnamed namespace


// CPU side tensor class
Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    float* ptr = nullptr;
    cudaCheck(cudaMalloc(&ptr, numel() * sizeof(float)), "cudaMalloc");
    // shared_ptr frees the GPU memory when the last Tensor using it is destroyed
    data_ = std::shared_ptr<float>(ptr, [](float* p) { cudaFree(p); });
}

int Tensor::numel() const {
    int n = 1;                       // a scalar (shape {}) has 1 element
    for (int d : shape_) n *= d;
    return n;
}

Tensor Tensor::fromHostData(const std::vector<int>& shape, const float* values) {
    Tensor t(shape);
    cudaCheck(cudaMemcpy(t.data(), values, t.numel() * sizeof(float),
                         cudaMemcpyHostToDevice), "cudaMemcpy host to device");
    return t;
}

Tensor Tensor::fromHost(const std::vector<int>& shape, const std::vector<float>& values) {
    int expected = 1;
    for (int d : shape) expected *= d;
    if ((int)values.size() != expected) {
        throw std::runtime_error("fromHost: got " + std::to_string(values.size()) +
                                 " values for shape " + shapeToString(shape));
    }
    return fromHostData(shape, values.data());
}

Tensor Tensor::scalar(float value) {
    return fromHostData({}, &value);
}

void Tensor::copyToHost(float* dest) const {
    cudaCheck(cudaMemcpy(dest, data(), numel() * sizeof(float),
                         cudaMemcpyDeviceToHost), "cudaMemcpy device to host");
}

std::vector<float> Tensor::toHost() const {
    std::vector<float> values(numel());
    copyToHost(values.data());
    return values;
}

//operators
Tensor operator+(const Tensor& a, const Tensor& b) { return binaryOp(a, b, AddOp{}); }
Tensor operator-(const Tensor& a, const Tensor& b) { return binaryOp(a, b, SubOp{}); }
Tensor operator*(const Tensor& a, const Tensor& b) { return binaryOp(a, b, MulOp{}); }
Tensor operator/(const Tensor& a, const Tensor& b) { return binaryOp(a, b, DivOp{}); }
Tensor operator-(const Tensor& a) { return unaryOp(a, NegOp{}); }

//public tensor functions
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
    return out;
}

Tensor mean(const Tensor& a) {
    if (a.numel() == 0){
        throw std::runtime_error("empty tensor, devide by 0 error" + a.numel()); 
    } 
    return unaryOp(sum(a), ScaleOp{1.0f / a.numel()});
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


//public helpers 
std::string shapeToString(const std::vector<int>& shape){
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) s += ", ";
        s += std::to_string(shape[i]);
    }
    return s + "]";
}
// prints tensors up to 2D
void printTensor(const char* name, const Tensor& t) {
    std::vector<float> v = t.toHost();
    const std::vector<int>& s = t.shape();
    printf("%s %s =\n", name, shapeToString(s).c_str());

    if (s.size() == 0) {                       // scalar
        printf("   %6.2f\n", v[0]);
    } else if (s.size() == 1) {                // vector: one row
        printf("   [");
        for (int i = 0; i < s[0]; i++) printf(" %6.2f", v[i]);
        printf(" ]\n");
    } else if (s.size() == 2) {                // matrix: one line per row
        for (int r = 0; r < s[0]; r++) {
            printf("   [");
            for (int c = 0; c < s[1]; c++) printf(" %6.2f", v[r * s[1] + c]);
            printf(" ]\n");
        }
    } else {
        throw std::runtime_error("printTensor: only up to 2D supported, got " + shapeToString(s));
    }
}











