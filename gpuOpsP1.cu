#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <memory>

inline void cudaCheck(cudaError_t err, const char* what){
    if (err != cudaSuccess){
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }

}

// CPU side tensor class
class Tensor{
public:
    // Allocate an uninitialized tensor on the GPU
    explicit Tensor(const std::vector<int>& shape) : shape_(shape) {
        float* ptr = nullptr;
        cudaCheck(cudaMalloc(&ptr, numel() * sizeof(float)), "cudaMalloc");
        // shared_ptr frees the GPU memory when the last Tensor using it is destroyed
        data_ = std::shared_ptr<float>(ptr, [](float* p) { cudaFree(p); });
    }

    // copy a tensor from host(CPU)
    static Tensor fromHost(const std::vector<int>& shape, const std::vector<float>& values){
        Tensor t(shape);
        if ((int)values.size() != t.numel()) {
            throw std::runtime_error("fromHost: number of values doesn't match shape");
        }
        cudaCheck(cudaMemcpy(t.data(), values.data(), t.numel() * sizeof(float),
                            cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        return t;
    }

    // copy data out of GPU back to host
    std::vector<float> toHost() const {
        std::vector<float> values(numel());
        cudaCheck(cudaMemcpy(values.data(), data(), numel() * sizeof(float),
                            cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        return values;

    }


    float* data() const { return data_.get(); }
    const std::vector<int>& shape() const { return shape_; }
    int numel() const { 
        int n = 1; 
        for (int d : shape_) n *= d; 
        return n; 
    }

private:
    std::shared_ptr<float> data_;
    std::vector<int> shape_;
};


//binary kernel defintion
template <typename Op>
__global__ void binaryKernel(const float* a, const float* b, float* out, int n, Op op) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}
struct AddOp { __device__ float operator()(float x, float y) const { return x + y; } };
struct SubOp { __device__ float operator()(float x, float y) const { return x - y; } };
struct MulOp { __device__ float operator()(float x, float y) const { return x * y; } };
struct DivOp { __device__ float operator()(float x, float y) const { return x / y; } };

//host side 2in-1out operator wrapper (binary operations)
template <typename Op>
Tensor binaryOp(const Tensor& a, const Tensor& b, Op op) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("binaryOp: shape mismatch");
    }
    Tensor out(a.shape());
    int n = out.numel(), threads = 256, blocks = (n + threads - 1) / threads;
    binaryKernel<<<blocks, threads>>>(a.data(), b.data(), out.data(), n, op);
    cudaCheck(cudaGetLastError(), "kernel launch");
    return out;
}

Tensor operator+(const Tensor& a, const Tensor& b) { return binaryOp(a, b, AddOp{}); }
Tensor operator-(const Tensor& a, const Tensor& b) { return binaryOp(a, b, SubOp{}); }
Tensor operator*(const Tensor& a, const Tensor& b) { return binaryOp(a, b, MulOp{}); }
Tensor operator/(const Tensor& a, const Tensor& b) { return binaryOp(a, b, DivOp{}); }

//1in-1out operator support (unary operations)
template <typename Op>
__global__ void unaryKernel(const float* a, float* out, int n, Op op) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = op(a[i]);
}
struct NegOp { __device__ float operator()(float x) const { return x * (-1); } };


//host side 1in-1out operator wrapper
template <typename Op>
Tensor unaryOp(const Tensor& a, Op op) {
    Tensor out(a.shape());
    int n = out.numel(), threads = 256, blocks = (n + threads - 1) / threads;
    unaryKernel<<<blocks, threads>>>(a.data(), out.data(), n, op);
    cudaCheck(cudaGetLastError(), "kernel launch");
    return out;
}
Tensor operator-(const Tensor& a) { return unaryOp(a, NegOp{}); }
