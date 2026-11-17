#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <algorithm>


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


// helper Functions
// converts a shape to string, obv
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

// Pad a shape to 2D with leading 1s: {10} -> {1, 10}, {} -> {1, 1}
std::vector<int> to2D(const std::vector<int>& shape) {
    if (shape.size() > 2) throw std::runtime_error("broadcast: only up to 2D supported");
    std::vector<int> s = shape;
    while (s.size() < 2) s.insert(s.begin(), 1);
    return s;
}



struct AddOp { __device__ float operator()(float x, float y) const { return x + y; } };
struct SubOp { __device__ float operator()(float x, float y) const { return x - y; } };
struct MulOp { __device__ float operator()(float x, float y) const { return x * y; } };
struct DivOp { __device__ float operator()(float x, float y) const { return x / y; } };

struct NegOp { __device__ float operator()(float x) const { return x * (-1); } };
struct PowOp {
    float p;
    __device__ float operator()(float x) const { return powf(x, p); }
};


//binary kernel definition
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


Tensor operator+(const Tensor& a, const Tensor& b) { return binaryOp(a, b, AddOp{}); }
Tensor operator-(const Tensor& a, const Tensor& b) { return binaryOp(a, b, SubOp{}); }
Tensor operator*(const Tensor& a, const Tensor& b) { return binaryOp(a, b, MulOp{}); }
Tensor operator/(const Tensor& a, const Tensor& b) { return binaryOp(a, b, DivOp{}); }

Tensor operator-(const Tensor& a) { return unaryOp(a, NegOp{}); }
Tensor pow(const Tensor& a, float p) { return unaryOp(a, PowOp{p}); }



int main(){
    try{
        std::vector<int> shape = {2,3};

        Tensor a = Tensor::fromHost(shape, {1.0f, 2.0f, 3.0f,
                                            4.0f, 5.0f, 6.0f});
        Tensor b = Tensor::fromHost(shape, {10.0f, 10.0f, 10.0f,
                                            10.0f, 10.0f, 10.0f});
        Tensor c = Tensor::fromHost(shape, {0.5f, 1.0f, 1.5f,
                                            2.0f, 2.5f, 3.0f});
        
        Tensor out = a + b * c;
        printTensor("a", a);
        printTensor("b", b);
        printTensor("c", c);
        printTensor("a + b * c", out);   // expected: 6 12 18 / 24 30 36


        Tensor m = Tensor::fromHost({2,3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        Tensor arow = Tensor::fromHost({3}, {10.00f, 20.0f, 30.0f});    
        Tensor acol = Tensor::fromHost({2, 1}, {100.0f, 200.0f});


        printTensor("m", m);      
        printTensor("arow", arow);      
        printTensor("acol", acol);      
        printTensor("m + arow", m + arow);   // [ 11 22 33 ] [ 14 25 36 ]
        printTensor("m + acol", m + acol);   // [ 101 102 103 ] [ 204 205 206 ]

    } catch (const std::exception& e){
        printf("Error: %s\n", e.what());
    }
    return 0;
}