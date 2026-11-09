#include "tensor.h"
#include "internal.cuh"
using namespace detail;

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
#include <unordered_map>
#include <mutex>

//simaliar to p1, but built for python binding

namespace {
    bool g_gradEnabled = true;

    //GPU buffers ready to reuse without calling the driver
    constexpr size_t kMaxCachedPerSize = 64;
    struct BufferPool {
        std::mutex mutex;
        std::unordered_map<size_t, std::vector<float*>> buffers;
    };
    BufferPool& bufferPool() { static BufferPool* pool = new BufferPool; return *pool; }

    void dropCachedBuffers() {
        BufferPool& pool = bufferPool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        for (auto& [bytes, list] : pool.buffers) {
            for (float* p : list) cudaFree(p);
            list.clear();
        }
    }

    float* acquireBuffer(size_t bytes) {
        BufferPool& pool = bufferPool();
        {
            std::lock_guard<std::mutex> lock(pool.mutex);
            auto it = pool.buffers.find(bytes);
            if (it != pool.buffers.end() && !it->second.empty()) {
                float* p = it->second.back();
                it->second.pop_back();
                return p;
            }
        }
        float* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err == cudaErrorMemoryAllocation) {        // out of memory: give back the cache and retry
            dropCachedBuffers();
            err = cudaMalloc(&ptr, bytes);
        }
        cudaCheck(err, "cudaMalloc");
        return ptr;
    }

    void releaseBuffer(float* p, size_t bytes) {
        BufferPool& pool = bufferPool();
        {
            std::lock_guard<std::mutex> lock(pool.mutex);
            auto& list = pool.buffers[bytes];
            if (list.size() < kMaxCachedPerSize) { list.push_back(p); return; }
        }
        cudaFree(p);
    }

    void accumulateGrad(const Tensor& t, const Tensor& g) {
        TensorImpl* impl = t.impl();
        if (!impl->grad) impl->grad = g.implPtr();                       // first gradient: store it
        else             impl->grad = (Tensor(impl->grad) + g).implPtr();  // later ones: new sum
    }
}
//end of unnamed namespace

namespace detail {
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

    Tensor unbroadcast(const Tensor& g, const std::vector<int>& target) {
        if (g.shape() == target) return g;                 // nothing was broadcast
        std::vector<int> gs = to2D(g.shape()), ts = to2D(target);
        Tensor r = reshape(g, gs);                         // view as 2D
        if (ts[0] == 1 && gs[0] != 1) r = sumAxis(r, 0);   // input was stretched down the rows
        if (ts[1] == 1 && gs[1] != 1) r = sumAxis(r, 1);   // input was stretched across the columns
        return reshape(r, target);                         // back to the input's real shape
    }
}
//end of detail namespace

//public stuff
//grad stuff 
bool isGradEnabled() { return g_gradEnabled; }
void setGradEnabled(bool enabled) { g_gradEnabled = enabled; }
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
    size_t bytes = numel() * sizeof(float);
    float* ptr = acquireBuffer(bytes); //uses a pointer so the buffer is freed when not used by any tensors
    impl_->data = std::shared_ptr<float>(ptr, [bytes](float* p) { releaseBuffer(p, bytes); });
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












