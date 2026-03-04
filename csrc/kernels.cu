/*
 * kernels.cu — CUDA kernel implementations for mospork_kernels.
 *
 * Provides optimised linear-layer forward and backward passes that
 * automatically select the best available path at runtime:
 *
 *   FP32 input on any arch → tiled SGEMM (scalar, no tensor cores)
 *   FP16 input on Volta+   → WMMA FP16 tensor-core GEMM
 *   BF16 input on Ampere+  → WMMA BF16 tensor-core GEMM
 *
 * All kernels fall back gracefully (via AT_DISPATCH_FLOATING_TYPES_AND2)
 * to a scalar path for unexpected dtypes.
 */

#include "kernels.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// WMMA API (Volta+)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#  include <mma.h>
using namespace nvcuda;
#endif

// ============================================================================
// Scalar tiled GEMM kernel  (any arch, any floating-point type)
// Computes  C = A @ B^T  where A:(M,K), B:(N,K), C:(M,N)
// ============================================================================
template <typename scalar_t>
__global__ void gemm_tiled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t*       __restrict__ C,
    int M, int N, int K)
{
    __shared__ scalar_t sA[TILE][TILE];
    __shared__ scalar_t sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    scalar_t acc = 0;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bCol = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : scalar_t(0);
        sB[threadIdx.y][threadIdx.x] = (col < N && bCol < K) ? B[col * K + bCol] : scalar_t(0);
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[threadIdx.x][k];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ============================================================================
// WMMA FP16 kernel  (Volta sm_70, Turing sm_75, Ampere sm_80/86, Hopper sm_90)
// Computes  C_fp16 = A_fp16 @ B_fp16^T  (accumulates in FP32)
// ============================================================================
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
__global__ void gemm_wmma_fp16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    int M, int N, int K)
{
    // Each warp handles one (WMMA_M × WMMA_N) output tile.
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN =  blockIdx.y * blockDim.y + threadIdx.y;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    // Accumulate in FP32 for numerical stability, then convert to FP16 on store.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>  acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> out_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < (K + WMMA_K - 1) / WMMA_K; ++k) {
        int aRow = warpM * WMMA_M, aCol = k * WMMA_K;
        int bRow = warpN * WMMA_N, bCol = k * WMMA_K;

        if (aRow < M && aCol < K && bRow < N && bCol < K) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            // B is (N, K); col_major fragment treats it as B^T for the GEMM.
            wmma::load_matrix_sync(b_frag, B + bRow * K + bCol, K);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Convert each FP32 accumulator element to FP16 before writing.
    for (int i = 0; i < out_frag.num_elements; ++i)
        out_frag.x[i] = __float2half(acc_frag.x[i]);

    if (warpM * WMMA_M < M && warpN * WMMA_N < N)
        wmma::store_matrix_sync(
            C + warpM * WMMA_M * N + warpN * WMMA_N,
            out_frag,
            N,
            wmma::mem_row_major
        );
}
#endif  // __CUDA_ARCH__ >= 700

// ============================================================================
// WMMA BF16 kernel  (Ampere sm_80+)
// ============================================================================
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__global__ void gemm_wmma_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16*       __restrict__ C,
    int M, int N, int K)
{
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN =  blockIdx.y * blockDim.y + threadIdx.y;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    // Accumulate in FP32, then convert to BF16 on store.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>         acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16> out_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < (K + WMMA_K - 1) / WMMA_K; ++k) {
        int aRow = warpM * WMMA_M, aCol = k * WMMA_K;
        int bRow = warpN * WMMA_N, bCol = k * WMMA_K;

        if (aRow < M && aCol < K && bRow < N && bCol < K) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * K + bCol, K);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Convert each FP32 accumulator element to BF16 before writing.
    for (int i = 0; i < out_frag.num_elements; ++i)
        out_frag.x[i] = __float2bfloat16(acc_frag.x[i]);

    if (warpM * WMMA_M < M && warpN * WMMA_N < N)
        wmma::store_matrix_sync(
            C + warpM * WMMA_M * N + warpN * WMMA_N,
            out_frag,
            N,
            wmma::mem_row_major
        );
}
#endif  // __CUDA_ARCH__ >= 800

// ============================================================================
// Bias addition kernel
// ============================================================================
template <typename scalar_t>
__global__ void add_bias_kernel(
    scalar_t*       __restrict__ C,
    const scalar_t* __restrict__ bias,
    int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        C[row * N + col] += bias[col];
}

// ============================================================================
// Host-side dispatch helpers
// ============================================================================

// Returns the CUDA device compute capability as an integer (e.g. 80 for sm_80).
static int device_sm(int device_index) {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_index);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_index);
    return major * 10 + minor;
}

// ---------------------------------------------------------------------------
// linear_forward_cuda
// ---------------------------------------------------------------------------
torch::Tensor linear_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias)
{
    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Flatten leading dims → (M, K)
    const int K  = x.size(-1);
    const int M  = x.numel() / K;
    const int N  = weight.size(0);

    auto x_2d   = x.reshape({M, K}).contiguous();
    auto w_cont = weight.contiguous();
    auto out    = torch::empty({M, N}, x.options());

    const int sm = device_sm(x.device().index());
    const auto dtype = x.scalar_type();

    // ------------------------------------------------------------------
    // Dispatch to the best available kernel
    // ------------------------------------------------------------------
    if (dtype == torch::kFloat16 && sm >= 70) {
        // WMMA FP16 tensor-core path
        dim3 block(128, 4);
        dim3 grid(
            (M + WMMA_M - 1) / WMMA_M * (128 / 32),  // warps along M
            (N + WMMA_N - 1) / WMMA_N
        );
        gemm_wmma_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x_2d.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(w_cont.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M, N, K
        );
    }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    else if (dtype == torch::kBFloat16 && sm >= 80) {
        // WMMA BF16 tensor-core path (Ampere+)
        dim3 block(128, 4);
        dim3 grid(
            (M + WMMA_M - 1) / WMMA_M * (128 / 32),
            (N + WMMA_N - 1) / WMMA_N
        );
        gemm_wmma_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_2d.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(w_cont.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            M, N, K
        );
    }
#endif
    else {
        // Scalar tiled GEMM fallback (Pascal / FP32 / unknown dtype)
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, dtype, "linear_forward_scalar", [&] {
                gemm_tiled_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    x_2d.data_ptr<scalar_t>(),
                    w_cont.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    M, N, K
                );
            }
        );
    }

    // Bias
    if (bias.has_value()) {
        dim3 block2(TILE, TILE);
        dim3 grid2((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, dtype, "add_bias", [&] {
                add_bias_kernel<scalar_t><<<grid2, block2, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    bias.value().contiguous().data_ptr<scalar_t>(),
                    M, N
                );
            }
        );
    }

    // Restore original leading dimensions
    auto out_shape = x.sizes().vec();
    out_shape.back() = N;
    return out.view(out_shape);
}

// ---------------------------------------------------------------------------
// linear_backward_cuda
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> linear_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    bool has_bias)
{
    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int K = x.size(-1);
    const int M = x.numel() / K;
    const int N = weight.size(0);

    auto go_2d  = grad_output.reshape({M, N}).contiguous();
    auto x_2d   = x.reshape({M, K}).contiguous();
    auto w_cont = weight.contiguous();

    // torch::mm dispatches to cuBLAS on GPU, which uses tensor cores for
    // FP16/BF16 and TF32 for FP32 on Ampere+.  This is the optimal path for
    // the non-square backward GEMMs; the custom WMMA kernels are oriented
    // toward the square/large forward projection tiles.
    auto grad_x = torch::mm(go_2d, w_cont);    // (M, K)
    auto grad_w = torch::mm(go_2d.t(), x_2d);  // (N, K)

    torch::Tensor grad_b;
    if (has_bias)
        grad_b = go_2d.sum(0);  // (N,)
    else
        grad_b = torch::Tensor();

    auto gx_shape = x.sizes().vec();
    return {grad_x.view(gx_shape), grad_w, grad_b};
}
