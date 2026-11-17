// tensor_example.cu
// Compile:  nvcc -std=c++17 tensor_example.cu -o tensor_example
// Run:      ./tensor_example

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------
inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------
// Tensor: a CPU-side object describing a block of GPU memory
// ---------------------------------------------------------------------
class Tensor {
public:
    // Allocate an uninitialized tensor on the GPU
    explicit Tensor(const std::vector<int>& shape) : shape_(shape) {
        float* ptr = nullptr;
        cudaCheck(cudaMalloc(&ptr, numel() * sizeof(float)), "cudaMalloc");
        // shared_ptr frees the GPU memory when the last Tensor using it is destroyed
        data_ = std::shared_ptr<float>(ptr, [](float* p) { cudaFree(p); });
    }

    // Create a tensor and copy CPU data into it
    static Tensor fromHost(const std::vector<int>& shape, const std::vector<float>& values) {
        Tensor t(shape);
        if ((int)values.size() != t.numel()) {
            throw std::runtime_error("fromHost: number of values doesn't match shape");
        }
        cudaCheck(cudaMemcpy(t.data(), values.data(), t.numel() * sizeof(float),
                             cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        return t;
    }

    // Copy the tensor's data back to the CPU
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

// ---------------------------------------------------------------------
// Generic elementwise kernel and operation functors
// ---------------------------------------------------------------------
template <typename Op>
__global__ void binaryKernel(const float* a, const float* b, float* out, int n, Op op) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}

struct AddOp { __device__ float operator()(float x, float y) const { return x + y; } };
struct SubOp { __device__ float operator()(float x, float y) const { return x - y; } };
struct MulOp { __device__ float operator()(float x, float y) const { return x * y; } };
struct DivOp { __device__ float operator()(float x, float y) const { return x / y; } };

// ---------------------------------------------------------------------
// Host-side wrapper: checks shapes, allocates output, launches kernel
// ---------------------------------------------------------------------
template <typename Op>
Tensor binaryOp(const Tensor& a, const Tensor& b, Op op) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("binaryOp: shapes don't match (no broadcasting yet)");
    }
    Tensor out(a.shape());
    int n = out.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    binaryKernel<<<blocks, threads>>>(a.data(), b.data(), out.data(), n, op);
    cudaCheck(cudaGetLastError(), "kernel launch");
    return out;
}

Tensor operator+(const Tensor& a, const Tensor& b) { return binaryOp(a, b, AddOp{}); }
Tensor operator-(const Tensor& a, const Tensor& b) { return binaryOp(a, b, SubOp{}); }
Tensor operator*(const Tensor& a, const Tensor& b) { return binaryOp(a, b, MulOp{}); }
Tensor operator/(const Tensor& a, const Tensor& b) { return binaryOp(a, b, DivOp{}); }

// ---------------------------------------------------------------------
// Helper to print a 2D tensor
// ---------------------------------------------------------------------
void print2D(const char* name, const Tensor& t) {
    std::vector<float> v = t.toHost();
    int rows = t.shape()[0], cols = t.shape()[1];
    printf("%s =\n", name);
    for (int r = 0; r < rows; r++) {
        printf("  [");
        for (int c = 0; c < cols; c++) printf(" %6.2f", v[r * cols + c]);
        printf(" ]\n");
    }
}

// ---------------------------------------------------------------------
// main
// ---------------------------------------------------------------------
int main() {
    try {
        std::vector<int> shape = {2, 3};   // a 2x3 matrix

        Tensor a = Tensor::fromHost(shape, {1.0f, 2.0f, 3.0f,
                                            4.0f, 5.0f, 6.0f});
        Tensor b = Tensor::fromHost(shape, {10.0f, 10.0f, 10.0f,
                                            10.0f, 10.0f, 10.0f});
        Tensor c = Tensor::fromHost(shape, {0.5f, 1.0f, 1.5f,
                                            2.0f, 2.5f, 3.0f});

        // b * c runs first (normal precedence), producing a temporary Tensor.
        // Then a + temporary produces out. The temporary is freed automatically.
        Tensor out = a + b * c;

        print2D("a", a);
        print2D("b", b);
        print2D("c", c);
        print2D("a + b * c", out);   // expected: 6 12 18 / 24 30 36

        // Shape mismatches are caught on the CPU before any kernel launches
        Tensor wrong = Tensor::fromHost({3}, {1.0f, 2.0f, 3.0f});
        Tensor bad = a + wrong;
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
    }
    return 0;
}
