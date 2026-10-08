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
    nb::class_<Tensor>(m, "Tensor")
        .def_static("from_list", &Tensor::fromHost)
        .def_static("from_numpy", [](CpuArray arr) {
            std::vector<int> shape;
            for (size_t i = 0; i < arr.ndim(); i++) shape.push_back((int)arr.shape(i));
            return Tensor::fromHostData(shape, arr.data());
    })
    .def("to_list", &Tensor::toHost)
    .def("to_numpy", [](const Tensor& t) {
        auto host = std::make_unique<float[]>(t.numel());
        t.copyToHost(host.get());
        std::vector<size_t> shape(t.shape().begin(), t.shape().end());
        float* raw = host.release();
        nb::capsule owner(raw, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        return nb::ndarray<nb::numpy, float>(raw, shape.size(), shape.data(), owner);
    })
    .def_prop_ro("shape", &Tensor::shape)
    .def(nb::self + nb::self)
    .def(nb::self - nb::self)
    .def(nb::self * nb::self)
    .def(nb::self / nb::self)
    .def(-nb::self)
    .def("__mul__",     [](const Tensor& a, float s) { return a * Tensor::scalar(s); })
    .def("__rmul__",    [](const Tensor& a, float s) { return Tensor::scalar(s) * a; })
    .def("__add__",     [](const Tensor& a, float s) { return a + Tensor::scalar(s); })
    .def("__radd__",    [](const Tensor& a, float s) { return Tensor::scalar(s) + a; })
    .def("__rsub__",    [](const Tensor& a, float s) { return Tensor::scalar(s) - a; })
    .def("__rtruediv__",[](const Tensor& a, float s) { return Tensor::scalar(s) / a; })
    .def("__repr__", [](const Tensor& t) {
        return "Tensor(shape=" + shapeToString(t.shape()) + ")";
    });


}