#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
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
    .def("max", [](const Tensor& t) { return max(t); },         
        "Maximum of all elements, returned as a tensor.")
    .def("min", [](const Tensor& t) { return min(t); },         
        "Minimum of all elements, returned as a tensor.")
    .def("mean", [](const Tensor& t) { return mean(t); },         
        "Sum of all elements, deivde by number of elements, returned as a tensor.")
    .def("__repr__", [](const Tensor& t) {
        return "Tensor(shape=" + shapeToString(t.shape()) + ")";
    });

    m.def("sum", &sum, nb::arg("a"), "Sum of all elements of a tensor.");
    m.def("max", &max, nb::arg("a"), "Maximum of all elements of a tensor.");
    m.def("min", &min, nb::arg("a"), "Minimum of all elements of a tensor.");
    m.def("mean", &mean, nb::arg("a"), "mean of all elements of a tensor.");
}