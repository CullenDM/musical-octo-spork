/*
 * bindings.cpp — PyTorch C++ extension bindings for mospork_kernels.
 *
 * Exposes the CUDA kernel entry points to Python via pybind11/torch.
 */

#include "kernels.cuh"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "mospork_kernels: CUDA-accelerated linear forward/backward pass";

    m.def(
        "linear_forward",
        &linear_forward_cuda,
        "Linear forward pass (CUDA). "
        "Dispatches to WMMA FP16/BF16 tensor-core kernel on Volta+/Ampere+ "
        "or falls back to a scalar tiled GEMM on Pascal and older architectures.",
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("bias")
    );

    m.def(
        "linear_backward",
        &linear_backward_cuda,
        "Linear backward pass (CUDA). "
        "Returns {grad_x, grad_weight, grad_bias}. "
        "grad_bias is an empty tensor when has_bias is false.",
        pybind11::arg("grad_output"),
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("has_bias")
    );
}
