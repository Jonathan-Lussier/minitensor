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


//Stuct operations
struct ReluOp { __device__ float operator()(float x) const { return fmaxf(x, 0.0f); } };
struct ReluBackOp { __device__ float operator()(float g, float x) const { return x > 0.0f ? g : 0.0f; } };
struct ScaleOp {
    float s;
    __device__ float operator()(float x) const { return x * s; }
};
struct MaxOp { __device__ float operator()(float a, float b) const { return fmaxf(a, b); } };
struct ExpOp { __device__ float operator()(float x) const { return expf(x); } };
struct LogOp { __device__ float operator()(float x) const { return logf(x); } };

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

//Transpose
__global__ void transposeKernel(const float* a, int rows, int cols, float* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i % cols;
    out[c * rows + r] = a[i];
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

// Reduce one value per thread down to one value for the whole block, combining with `op`.
// Every thread gets the result back.
template <int BLOCK_SIZE, typename Op>
__device__ float blockReduce(float v, float* sdata, Op op) {
    int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = op(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float result = sdata[0];
    __syncthreads();         // every thread reads the result before sdata can be reused
    return result;
}

template <int BLOCK_SIZE>
__global__ void softmaxKernel(const float* in, float* out, int rows, int cols) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    const float* x = in  + blockIdx.x * cols;   
    float*       y = out + blockIdx.x * cols;

    float localMax = -INFINITY;
    for (int c = tid; c < cols; c += BLOCK_SIZE) localMax = fmaxf(localMax, x[c]);
    float rowMax = blockReduce<BLOCK_SIZE>(localMax, sdata, MaxOp{});

    float localSum = 0.0f;
    for (int c = tid; c < cols; c += BLOCK_SIZE) localSum += expf(x[c] - rowMax);
    float rowSum = blockReduce<BLOCK_SIZE>(localSum, sdata, AddOp{});

    for (int c = tid; c < cols; c += BLOCK_SIZE) y[c] = expf(x[c] - rowMax) / rowSum;
}

template <int BLOCK_SIZE>
__global__ void softmaxCrossEntropyKernel(const float* logits, const float* labels,
                                          float* probs, float* rowLoss, int rows, int cols) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x, r = blockIdx.x;
    const float* x = logits + r * cols;
    float*       p = probs  + r * cols;

    float localMax = -INFINITY;
    for (int c = tid; c < cols; c += BLOCK_SIZE) localMax = fmaxf(localMax, x[c]);
    float rowMax = blockReduce<BLOCK_SIZE>(localMax, sdata, MaxOp{});

    float localSum = 0.0f;
    for (int c = tid; c < cols; c += BLOCK_SIZE) localSum += expf(x[c] - rowMax);
    float rowSum = blockReduce<BLOCK_SIZE>(localSum, sdata, AddOp{});

    for (int c = tid; c < cols; c += BLOCK_SIZE) p[c] = expf(x[c] - rowMax) / rowSum;

    if (tid == 0) {
        int label = (int)labels[r];
        rowLoss[r] = (label >= 0 && label < cols)
                   ? rowMax + logf(rowSum) - x[label]   // -log softmax(x)[label]
                   : NAN;                               // bad label: make it visible
    }
}

__global__ void crossEntropyBackwardKernel(const float* probs, const float* labels, const float* g,
                                           float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i % cols;
    float target = (c == (int)labels[r]) ? 1.0f : 0.0f;
    out[i] = (probs[i] - target) * g[0] / rows;
}

Tensor unbroadcast(const Tensor& g, const std::vector<int>& target) {
    if (g.shape() == target) return g;                 // nothing was broadcast
    std::vector<int> gs = to2D(g.shape()), ts = to2D(target);
    Tensor r = reshape(g, gs);                         // view as 2D
    if (ts[0] == 1 && gs[0] != 1) r = sumAxis(r, 0);   // input was stretched down the rows
    if (ts[1] == 1 && gs[1] != 1) r = sumAxis(r, 1);   // input was stretched across the columns
    return reshape(r, target);                         // back to the input's real shape
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

//this just changes shape labels, keeps same data, creates new TensorImpl
Tensor reshape(const Tensor& a, const std::vector<int>& shape) {
    int n = 1;
    for (int d : shape) n *= d;
    if (n != a.numel())
        throw std::runtime_error("reshape: can't reshape " + shapeToString(a.shape()) +
                                 " to " + shapeToString(shape));
    auto impl = std::make_shared<TensorImpl>();
    impl->data = a.impl()->data;     
    impl->shape = shape;
    Tensor out(impl);
    if (needsGrad(a)) {
        recordNode(out, "reshape", {a}, [orig = a.shape()](const Tensor& g) {
            return std::vector<Tensor>{ reshape(g, orig) };
        });
    }
    return out;
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
        recordNode(out, "add", {a,b}, [sa = a.shape(), sb = b.shape()](const Tensor& g) {
            return std::vector<Tensor>{ unbroadcast(g, sa), unbroadcast(g, sb) };
        });
    }
    return out;
}
Tensor operator-(const Tensor& a, const Tensor& b){
    Tensor out = binaryOp(a, b, SubOp{});
    if (needsGrad(a, b)) {
        recordNode(out, "sub", {a, b}, [sa = a.shape(), sb = b.shape()](const Tensor& g) {
            return std::vector<Tensor>{ unbroadcast(g, sa), unbroadcast(-g, sb) };
        });
    }
    return out;
}
Tensor operator*(const Tensor& a, const Tensor& b){
    Tensor out = binaryOp(a, b, MulOp{});
    if (needsGrad(a, b)) {
        recordNode(out, "mul", {a, b}, [a, b](const Tensor& g) {
            return std::vector<Tensor>{ unbroadcast(g * b, a.shape()),
                                        unbroadcast(g * a, b.shape()) };
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
        recordNode(out, "div", {a, b}, [a, b](const Tensor& g) {
            return std::vector<Tensor>{ unbroadcast(g / b, a.shape()),
                                        unbroadcast(-g * a / (b * b), b.shape()) };
        });
    }
    return out;
}

Tensor exp(const Tensor& a) {
    Tensor out = unaryOp(a, ExpOp{});
    if (needsGrad(a)) {
        // d/dx exp(x) = exp(x): reuse the output, detached to avoid a reference cycle
        recordNode(out, "exp", {a}, [y = detach(out)](const Tensor& g) {
            return std::vector<Tensor>{ g * y };
        });
    }
    return out;
}

Tensor log(const Tensor& a) {
    Tensor out = unaryOp(a, LogOp{});
    if (needsGrad(a)) {
        // d/dx log(x) = 1 / x
        recordNode(out, "log", {a}, [a](const Tensor& g) {
            return std::vector<Tensor>{ g / a };
        });
    }
    return out;
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

Tensor matmul(const Tensor& a, const Tensor& b) {
    Tensor out = tiledMatMul(a, b);
    if (needsGrad(a, b)) {
        recordNode(out, "matmul", {a, b}, [a, b](const Tensor& g) {
            return std::vector<Tensor>{ matmul(g, transpose(b)), matmul(transpose(a), g) };
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

Tensor relu(const Tensor& a) {
    Tensor out = unaryOp(a, ReluOp{});
    if (needsGrad(a)) {
        recordNode(out, "relu", {a}, [a](const Tensor& g) {
            return std::vector<Tensor>{ binaryOp(g, a, ReluBackOp{}) };
        });
    }
    return out;
}

Tensor softmax(const Tensor& a) {
    if (a.shape().size() != 2)
        throw std::runtime_error("softmax: expected a 2D tensor, got " + shapeToString(a.shape()));
    int rows = a.shape()[0], cols = a.shape()[1];
    Tensor out({rows, cols});
    if (out.numel() > 0) {
        constexpr int threads = 256;
        softmaxKernel<threads><<<rows, threads>>>(a.data(), out.data(), rows, cols);
        cudaCheck(cudaGetLastError(), "softmax launch");
    }

    if (needsGrad(a)) {
        recordNode(out, "softmax", {a}, [y = detach(out)](const Tensor& g) {
            return std::vector<Tensor>{ y * (g - sumAxis(g * y, 1)) };
        });
    }
    return out;
}

Tensor softmaxCrossEntropy(const Tensor& logits, const Tensor& labels) {
    if (logits.shape().size() != 2)
        throw std::runtime_error("cross_entropy: logits must be 2D (batch, classes), got " +
                                 shapeToString(logits.shape()));
    int rows = logits.shape()[0], cols = logits.shape()[1];
    if (labels.shape() != std::vector<int>{rows})
        throw std::runtime_error("cross_entropy: labels must have shape [" + std::to_string(rows) +
                                 "], got " + shapeToString(labels.shape()));
    if (rows == 0 || cols == 0)
        throw std::runtime_error("cross_entropy: empty input");

    Tensor probs({rows, cols});
    Tensor rowLoss({rows});
    constexpr int threads = 256;
    softmaxCrossEntropyKernel<threads><<<rows, threads>>>(logits.data(), labels.data(),
                                                          probs.data(), rowLoss.data(), rows, cols);
    cudaCheck(cudaGetLastError(), "cross_entropy launch");

    // Mean over the batch. rowLoss doesn't require grad, so these steps record nothing.
    Tensor out = unaryOp(sum(rowLoss), ScaleOp{1.0f / rows});

    if (needsGrad(logits)) {
        recordNode(out, "cross_entropy", {logits}, [probs, labels, rows, cols](const Tensor& g) {
            Tensor dx({rows, cols});
            int n = rows * cols, threads = 256, blocks = (n + threads - 1) / threads;
            crossEntropyBackwardKernel<<<blocks, threads>>>(probs.data(), labels.data(), g.data(),
                                                            dx.data(), rows, cols);
            cudaCheck(cudaGetLastError(), "cross_entropy backward launch");
            return std::vector<Tensor>{ dx };
        });
    }
    return out;
}

Tensor detach(const Tensor& t) {
    auto impl = std::make_shared<TensorImpl>();
    impl->data = t.impl()->data;      // same values, no copy
    impl->shape = t.shape();          // no grad_fn, requires_grad = false
    return Tensor(impl);
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












