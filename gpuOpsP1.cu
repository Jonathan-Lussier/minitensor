#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <memory>

class Tensor{
public:
    Tensor(std::vector<int> shape) : shape_(shape) {
        float* ptr;
        cudaMalloc(&ptr, numel() * sizeof(float));
        data_ = std::shared_ptr<float>(ptr, [](float* p) {cudaFree(p); });
    }
    float* data() const { return data_.get(); }
    const std::vector<int>& shape() const { return shape_; }
    int numel() const { int n = 1; for (int d : shape_) n *= d; return n; }
private:
    std::shared_ptr<float> data_;
    std::vector<int> shape_;
};

template <typename Op>
__global__ void binaryKernel(const float* a, const float* b, float* out, int n, Op op) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}
struct AddOp { __device__ float operator()(float x, float y) const { return x + y; } };
struct SubOp { __device__ float operator()(float x, float y) const { return x - y; } };
struct MulOp { __device__ float operator()(float x, float y) const { return x * y; } };

template <typename Op>
Tensor binaryOp(const Tensor& a, const Tensor& b, Op op) {
    if (a.shape() != b.shape()) {
        fprintf(stderr, "binaryOp: shape mismatch\n");
        exit(EXIT_FAILURE);
    }
    Tensor out(a.shape());
    int n = out.numel(), threads = 256, blocks = (n + threads - 1) / threads;
    binaryKernel<<<blocks, threads>>>(a.data(), b.data(), out.data(), n, op);
    return out;
}

Tensor operator+(const Tensor& a, const Tensor& b) { return binaryOp(a, b, AddOp{}); }
Tensor operator-(const Tensor& a, const Tensor& b) { return binaryOp(a, b, SubOp{}); }
Tensor operator*(const Tensor& a, const Tensor& b) { return binaryOp(a, b, MulOp{}); }


template <typename Op>
__global__ void unaryKernel(const float* a, float* out, int n, Op op) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = op(a[i]);
}
struct NegOp { __device__ float operator()(float x) const { return x * (-1); } };

template <typename Op>
Tensor unaryOp(const Tensor& a, Op op) {
    Tensor out(a.shape());
    int n = out.numel(), threads = 256, blocks = (n + threads - 1) / threads;
    unaryKernel<<<blocks, threads>>>(a.data(), out.data(), n, op);
    return out;
}
Tensor operator-(const Tensor& a) { return unaryOp(a, NegOp{}); }