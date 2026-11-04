// internal.cuh
#pragma once
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace detail{
    //--------------------------------------------------------------------------------
    //helper functions
    inline void cudaCheck(cudaError_t err, const char* what){
        if (err != cudaSuccess){
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
        }

    }

    // Pad a shape to 2D with leading 1s: {10} -> {1, 10}, {} -> {1, 1}
    inline std::vector<int> to2D(const std::vector<int>& shape) {
        if (shape.size() > 2) throw std::runtime_error("broadcast: only up to 2D supported");
        std::vector<int> s = shape;
        while (s.size() < 2) s.insert(s.begin(), 1);
        return s;
    }




    //--------------------------------------------------------------------------------
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




    //--------------------------------------------------------------------------------
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



    //--------------------------------------------------------------------------------
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


    //--------------------------------------------------------------------------------
    //header declarations
    bool needsGrad(const Tensor& a);
    bool needsGrad(const Tensor& a, const Tensor& b);
    void recordNode(Tensor& out, const char* name, std::vector<Tensor> inputs,
                std::function<std::vector<Tensor>(const Tensor&)> backward);
    Tensor unbroadcast(const Tensor& g, const std::vector<int>& target);

}