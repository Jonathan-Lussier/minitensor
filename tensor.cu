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
#include <unordered_set>

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

//transpose

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

//grad stuff
bool needsGrad(const Tensor& a, const Tensor& b) {
    return isGradEnabled() && (a.requiresGrad() || b.requiresGrad());
}
bool needsGrad(const Tensor& a) {
    return isGradEnabled() && a.requiresGrad();
}

void recordNode(Tensor& out, const char* name, std::vector<Tensor> inputs,
                std::function<std::vector<Tensor>(const Tensor&)> backward) {
    auto node = std::make_shared<Node>();
    node->inputs = std::move(inputs);
    node->op_name = name;
    node->backward = std::move(backward);
    out.impl()->grad_fn = node;
    out.setRequiresGrad(true);   // so ops that use `out` will record nodes too
}

void accumulateGrad(const Tensor& t, const Tensor& g) {
    TensorImpl* impl = t.impl();
    if (!impl->grad) impl->grad = g.implPtr();                       // first gradient: store it
    else             impl->grad = (Tensor(impl->grad) + g).implPtr();  // later ones: new sum
}

}
//end of unnamed namespace

//grad stuff
static bool g_gradEnabled = true;
bool isGradEnabled() { return g_gradEnabled; }
NoGradGuard::NoGradGuard() : prev_(g_gradEnabled) { g_gradEnabled = false; }
NoGradGuard::~NoGradGuard() { g_gradEnabled = prev_; }

void Tensor::backward() {
    if (numel() != 1)
        throw std::runtime_error("backward: only supported for single-element tensors, got shape " +
                                 shapeToString(shape()));
    if (!requiresGrad())
        throw std::runtime_error("backward: tensor does not require grad");

    //Topological order like micrograd's build_topo
    std::vector<Tensor> topo;
    std::unordered_set<TensorImpl*> visited;
    std::function<void(const Tensor&)> build = [&](const Tensor& t) {
        if (!visited.insert(t.impl()).second) return;// if already visited return
        if (t.impl()->grad_fn)
            for (const Tensor& input : t.impl()->grad_fn->inputs) build(input);
        topo.push_back(t);
    };
    build(*this);

    NoGradGuard guard;   // gradient doesn't make new graph nodes

    //d(loss)/d(loss) = 1  like micrograd's self.grad = 1
    accumulateGrad(*this, Tensor::full(shape(), 1.0f));

    // 3. Walk backward, applying the chain rule at each node
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const Tensor& t = *it;
        Node* node = t.impl()->grad_fn.get();
        if (!node || !t.hasGrad()) continue;                    // leaf, or nothing flowed here

        std::vector<Tensor> grads = node->backward(t.grad());
        for (size_t i = 0; i < node->inputs.size(); i++) {
            const Tensor& input = node->inputs[i];
            if (!input.requiresGrad()) continue;
            if (grads[i].shape() != input.shape())
                throw std::runtime_error(std::string(node->op_name) + " backward: gradient shape " +
                                         shapeToString(grads[i].shape()) + " doesn't match input shape " +
                                         shapeToString(input.shape()));
            accumulateGrad(input, grads[i]);
        }
    }
}


// CPU side tensor class
Tensor::Tensor(const std::vector<int>& shape) : impl_(std::make_shared<TensorImpl>()) {
    impl_->shape = shape;                         
    float* ptr = nullptr;
    cudaCheck(cudaMalloc(&ptr, numel() * sizeof(float)), "cudaMalloc");
    // shared_ptr frees the GPU memory when the last Tensor using it is destroyed
    impl_->data = std::shared_ptr<float>(ptr, [](float* p) { cudaFree(p); });
}

int Tensor::numel() const {
    int n = 1;                       // a scalar (shape {}) has 1 element
    for (int d : impl_->shape) n *= d;
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

Tensor Tensor::full(const std::vector<int>& shape, float value) {
    int n = 1;
    for (int d : shape) n *= d;
    return fromHost(shape, std::vector<float>(n, value));
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
Tensor operator+(const Tensor& a, const Tensor& b){
    Tensor out = binaryOp(a, b, AddOp{});
    if (needsGrad(a,b)){
        if (a.shape() != b.shape() )
            throw std::runtime_error("add: gradients for broadcast shapes not supported... yet");
        recordNode(out, "add", {a,b}, [](const Tensor& g){
            return std::vector<Tensor>{g, g};
        });
    }
    return out;
}
Tensor operator-(const Tensor& a, const Tensor& b){
    Tensor out = binaryOp(a, b, SubOp{});
    if (needsGrad(a,b)){
        if (a.shape() != b.shape() )
            throw std::runtime_error("sub: gradients for broadcast shapes not supported... yet");
        recordNode(out, "sub", {a,b}, [](const Tensor& g){
            return std::vector<Tensor>{g, -g};
        });
    }
    return out;
}
Tensor operator*(const Tensor& a, const Tensor& b){
    Tensor out = binaryOp(a, b, MulOp{});
    if (needsGrad(a,b)){
        if (a.shape() != b.shape() )
            throw std::runtime_error("mul: gradients for broadcast shapes not supported... yet");
        recordNode(out, "mul", {a,b}, [a, b](const Tensor& g){
            return std::vector<Tensor>{g*b, g*a};  //here's dat chain rule
        });
    }
    return out;
}
Tensor operator-(const Tensor& a) { 
    Tensor out = unaryOp(a, NegOp{});
    if (needsGrad(a)){
        recordNode(out, "neg", {a}, [](const Tensor& g){
            return std::vector<Tensor>{-g};
        });
    }
    return out; 
}
Tensor operator/(const Tensor& a, const Tensor& b){
    Tensor out = binaryOp(a, b, DivOp{});
    if (needsGrad(a,b)){
        if (a.shape() != b.shape() )
            throw std::runtime_error("div: gradients for broadcast shapes not supported... yet");
        recordNode(out, "div", {a,b}, [a, b](const Tensor& g){
            return std::vector<Tensor>{g/b, -g*a/(b*b)};  //da chain 
        });
    }
    return out;
}



//public tensor functions
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

//public tensor functions
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

void synchronize() { cudaCheck(cudaDeviceSynchronize(), "synchronize"); }












