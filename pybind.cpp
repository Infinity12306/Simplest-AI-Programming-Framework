#include "tensor.cu"
#include "activation_functions/relu.cu"
#include "activation_functions/sigmoid.cu"
#include "layers/fc.cu"
#include "layers/conv.cu"
#include "layers/cross_entropy.cu"
#include "layers/pooling.cu"
#include "layers/softmax.cu"
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>

namespace py = pybind11;

template <typename T>
tensor from_numpy(py::array_t<T> data) {
    // get the input shape
    std::vector<int> shape(data.ndim());
    for (int i = 0; i < shape.size(); ++i) {
        shape[i] = data.shape(i);
    }
    // create a tensor
    tensor tensor_(shape, "cpu");
    for (int i = 0; i < tensor_.size(); ++i) {
        tensor->data[i] = data.data()[i];
    }
    return tensor;
}

PYBIND11_MODULE(mytensor, m) {
    py::class_<tensor>(m,"tensor")
    .def(py::init<const std::vector<int> , const char*>())
    // .def("size", &tensor::size)
    // .def("set_data", &tensor::set_data)
    .def("cpu", &tensor::cpu)
    .def("gpu", &tensor::gpu)
    // .def("print_data", &tensor::print_data);
    .def("deepcopy", &tensor::deepcopy)
    .def("from_numpy", &from_numpy)
};

PYBIND11_MODULE(myactivation, m) {
    py::class_<relu>(m,"relu")
    .def(py::init<>())
    .def("forward", &relu::forward)
    .def("backward", &relu::backward);

    py::class_<sigmoid>(m,"sigmoid")
    .def(py::init<>())
    .def("forward", &sigmoid::forward)
    .def("backward", &sigmoid::backward)
};

PYBIND11_MODULE(mylayer, m) {
    py::class_<fc>(m,"fc")
    .def(py::init<int, int>())
    .def("forward", &fc::forward)
    .def("backward", &fc::backward);

    py::class_<conv>(m,"conv")
    .def(py::init<>())
    .def("forward", &conv::forward)
    .def("backward", &conv::backward);

    py::class_<csEntropy>(m,"csEntropy")
    .def(py::init<>())
    .def("forward", &csEntropy::forward)
    .def("backward", &csEntropy::backward);

    py::class_<softmax>(m,"softmax")
    .def(py::init<>())
    .def("forward", &softmax::forward)
    .def("backward", &softmax::backward);

    py::class_<pooling>(m,"pooling")
    .def(py::init<>())
    .def("forward", &pooling::forward)
    .def("backward", &pooling::backward)
};