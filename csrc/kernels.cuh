/*
 * kernels.cuh — declarations and shared helpers for mospork_kernels.
 *
 * Architecture strategy
 * ─────────────────────
 *   sm_60 / sm_61  (Pascal)   — standard CUDA SGEMM / HGEMM
 *   sm_70          (Volta)    — WMMA FP16 tensor cores
 *   sm_75          (Turing)   — WMMA FP16 / INT8 tensor cores
 *   sm_80 / sm_86  (Ampere)   — WMMA FP16 + BF16 tensor cores
 *   sm_90          (Hopper)   — WMMA FP16 + BF16 tensor cores
 *
 * The compile-time macros __CUDA_ARCH__ and __CUDA_ARCH__ >= 700 are used
 * inside *.cu to select the appropriate kernel variant.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#  include <cuda_bf16.h>
#endif

#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Compile-time architecture helpers (evaluated per PTX target)
// ---------------------------------------------------------------------------

// WMMA tensor-core API is available from Volta (sm_70) onwards.
#define ARCH_HAS_WMMA      (__CUDA_ARCH__ >= 700)
// BF16 tensor cores were introduced on Ampere (sm_80).
#define ARCH_HAS_BF16_TC   (__CUDA_ARCH__ >= 800)

// Tile dimensions for WMMA — must be multiples of 16.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Shared-memory tile size used by the scalar (non-tensor-core) kernel.
constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// Kernel declarations
// ---------------------------------------------------------------------------

// Linear forward  :  out = x @ W^T + b   (b may be nullptr)
torch::Tensor linear_forward_cuda(
    const torch::Tensor& x,        // (*, in_features)  — any leading dims
    const torch::Tensor& weight,   // (out_features, in_features)
    const c10::optional<torch::Tensor>& bias  // (out_features,) or nullopt
);

// Linear backward : returns {grad_x, grad_w, grad_b}
// grad_b is an empty tensor when has_bias == false.
std::vector<torch::Tensor> linear_backward_cuda(
    const torch::Tensor& grad_output,  // same shape as forward output
    const torch::Tensor& x,
    const torch::Tensor& weight,
    bool has_bias
);
