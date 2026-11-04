#include "tensor.h"
#include "internal.cuh"
using namespace detail;

namespace {

    
    __global__ void crossEntropyBackwardKernel(const float* probs, const float* labels, const float* g,
                                            float* out, int rows, int cols) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= rows * cols) return;
        int r = i / cols, c = i % cols;
        float target = (c == (int)labels[r]) ? 1.0f : 0.0f;
        out[i] = (probs[i] - target) * g[0] / rows;
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
}


//public stuffs
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