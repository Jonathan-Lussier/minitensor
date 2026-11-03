#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <memory>
#include "tensor.h"

namespace nb = nanobind;

using CpuArray = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

NB_MODULE(minitensor, m){
    m.doc() = "Minimal float32 tensor library backed by CUDA.";

    nb::class_<Tensor>(m, "Tensor", "A float32 tensor stored on the GPU.")
        .def_static("from_list", &Tensor::fromHost, nb::arg("shape"), nb::arg("values"),
            "Create a GPU tensor from a shape and a flat list of row-major values.")
        .def_static("from_numpy", [](CpuArray arr) {
            std::vector<int> shape;
            for (size_t i = 0; i < arr.ndim(); i++) shape.push_back((int)arr.shape(i));
            return Tensor::fromHostData(shape, arr.data());
    }, nb::arg("array"), "Create a GPU tensor by copying a C-contiguous float32 NumPy array.")
    .def("to_list", &Tensor::toHost, "Copy the tensor to the CPU as a flat list of row-major values.")
    .def("to_numpy", [](const Tensor& t) {
        auto host = std::make_unique<float[]>(t.numel());
        t.copyToHost(host.get());
        std::vector<size_t> shape(t.shape().begin(), t.shape().end());
        float* raw = host.release();
        nb::capsule owner(raw, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        return nb::ndarray<nb::numpy, float>(raw, shape.size(), shape.data(), owner);
    }, "Copy the tensor to the CPU as a float32 NumPy array with the same shape.")
    .def_prop_ro("shape", &Tensor::shape, "The tensor's dimensions as a list of ints.")
    .def(nb::self + nb::self, nb::arg("other"), "Elementwise add (2D broadcasting supported).")
    .def(nb::self - nb::self, nb::arg("other"), "Elementwise subtract (2D broadcasting supported).")
    .def(nb::self * nb::self, nb::arg("other"), "Elementwise multiply (2D broadcasting supported).")
    .def(nb::self / nb::self, nb::arg("other"), "Elementwise divide (2D broadcasting supported).")
    .def(-nb::self, "Elementwise negation.")
    .def("__mul__",     [](const Tensor& a, float s) { return a * Tensor::scalar(s); },
        nb::arg("scalar"), "Multiply every element by a scalar.")
    .def("__rmul__",    [](const Tensor& a, float s) { return Tensor::scalar(s) * a; },
        nb::arg("scalar"), "Multiply every element by a scalar.")
    .def("__add__",     [](const Tensor& a, float s) { return a + Tensor::scalar(s); },
        nb::arg("scalar"), "Add a scalar to every element.")
    .def("__radd__",    [](const Tensor& a, float s) { return Tensor::scalar(s) + a; },
        nb::arg("scalar"), "Add a scalar to every element.")
    .def("__rsub__",    [](const Tensor& a, float s) { return Tensor::scalar(s) - a; },
        nb::arg("scalar"), "Subtract every element from a scalar.")
    .def("__rtruediv__",[](const Tensor& a, float s) { return Tensor::scalar(s) / a; },
        nb::arg("scalar"), "Divide a scalar by every element.")
    .def("sum", [](const Tensor& t) { return sum(t); },         
        "Sum of all elements, returned as a tensor.")

    .def("sumAxis", &sumAxis, nb::arg("axis"),
            "Sum a 2D tensor along one axis, keeping it as size 1. "
            "axis=0: sum down each column, (rows, cols) -> (1, cols). "
            "axis=1: sum across each row, (rows, cols) -> (rows, 1).")
    .def("reshape", &reshape, nb::arg("shape"),
            "Return a tensor with a new shape and the same number of elements. "
            "The result shares memory with the original (no copy).")

    .def("dot", [](const Tensor& t, const Tensor& s) { return dot(t, s); },         
        "dot product of two vectors.")

    .def("matmul", [](const Tensor& t, const Tensor& s) { return matmul(t, s); },         
        "matmul of two matrices.")

        .def("naiveMatMul", [](const Tensor& t, const Tensor& s) { return naiveMatMul(t, s); },         
        "Naive matmul, inefficient for long and narrow matrices.")
    .def("tiledMatMul", [](const Tensor& t, const Tensor& s) { return tiledMatMul(t, s); },         
        "tiledMatMul matmul, more efficient.")
    .def("max", [](const Tensor& t) { return max(t); },         
        "Maximum of all elements, returned as a tensor.")
    .def("min", [](const Tensor& t) { return min(t); },         
        "Minimum of all elements, returned as a tensor.")
    .def("mean", [](const Tensor& t) { return mean(t); },         
        "Sum of all elements, deivde by number of elements, returned as a tensor.")
    .def("transpose", [](const Tensor& t) { return transpose(t); },
        "Transpose tensor t")
    .def("exp", &exp, "Elementwise e^x.")
    .def("log", &log, "Elementwise natural log.")
    .def("__repr__", [](const Tensor& t) {
        return "Tensor(shape=" + shapeToString(t.shape()) + ")";})
    .def("__str__", [](nb::handle self) {
        const Tensor& t = nb::cast<const Tensor&>(self);
        std::string header = "Tensor(shape=" + shapeToString(t.shape());
        if (t.requiresGrad()) header += ", requires_grad=True";
        header += ")\n";
        nb::object arr = self.attr("to_numpy")();
        return header + std::string(nb::str(arr).c_str()); })
    .def("__array__", [](nb::handle self, nb::args, nb::kwargs) {
        return self.attr("to_numpy")();})
    .def("item", [](const Tensor& t) {
        if (t.numel() != 1)
            throw std::runtime_error("item: only works on single-element tensors, got shape " +
                                        shapeToString(t.shape()));
        float v;
        t.copyToHost(&v);          // copies just 4 bytes from the GPU
        return v;                  // a C++ float becomes a Python float
        })

    // grad stuff
    .def_prop_rw("requires_grad", &Tensor::requiresGrad, &Tensor::setRequiresGrad)
            .def_prop_ro("grad", [](const Tensor& t) -> std::optional<Tensor> {
                if (!t.hasGrad()) return std::nullopt;   // shows up as None in Python
                return t.grad();
        })
    .def("backward", &Tensor::backward)
    .def("relu", &relu, "Elementwise max(x, 0).")
    .def("softmax", &softmax, "Row-wise softmax of a 2D tensor shaped (batch, classes)")
    .def("softmaxCrossEntropy", &softmaxCrossEntropy, "Mean softmax cross-entropy loss over a batch.")
    
    ;




    m.def("sum", &sum, nb::arg("a"), "Sum of all elements of a tensor.");
    m.def("sumAxis", &sumAxis, nb::arg("a"), nb::arg("axis"),
        "Sum a 2D tensor along one axis, keeping it as size 1 (see Tensor.sum_axis).");
    m.def("reshape", &reshape, nb::arg("a"), nb::arg("shape"),
          "Reshape a tensor without copying its data (see Tensor.reshape).");
    m.def("max", &max, nb::arg("a"), "Maximum of all elements of a tensor.");
    m.def("min", &min, nb::arg("a"), "Minimum of all elements of a tensor.");
    m.def("mean", &mean, nb::arg("a"), "mean of all elements of a tensor.");
    m.def("dot", &dot, nb::arg("a"), nb::arg("b"), "dot product of two vectors.");
    m.def("matmul", &matmul, nb::arg("a"), nb::arg("b"), "default matrix multiplication of 2D tensors a and b.");
    m.def("naiveMatMul", &naiveMatMul, nb::arg("a"), nb::arg("b"), "naive matrix multiplication of 2D tensors a and b.");
    m.def("tiledMatMul", &tiledMatMul, nb::arg("a"), nb::arg("b"), "tiled matrix multiplication of 2D tensors a and b.");
    m.def("transpose", &transpose, nb::arg("a"), "simple transpose of tensor a.");
    m.def("exp", &exp, nb::arg("a"), "Elementwise e^x.");
    m.def("log", &log, nb::arg("a"), "Elementwise natural log.");
    m.def("synchronize", &synchronize);
    m.def("relu", &relu, nb::arg("a"), "Elementwise max(x, 0).");
    m.def("softmax", &softmax, nb::arg("x"),
        "Row-wise softmax of a 2D tensor shaped (batch, classes)");
    m.def("softmax_cross_entropy", &softmaxCrossEntropy, nb::arg("logits"), nb::arg("labels"),
          "Mean softmax cross-entropy loss over a batch.\n"
          "A seperate softmax call is NOT needed.");
}