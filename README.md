# musical-octo-spork

CUDA-oriented kernels for the unique parts of a Recursive PEER-GR-KAN style MoE block.

## What was added

- `peer_grkan_kernels.py`:
  - `FlashRationalGateFn`: fused forward/backward for grouped rational gate with a single `scatter_add_` coefficient reduction.
  - `moe_ffn_chunked_fast_gather_einsum`: chunked gather/einsum MoE path preserving model behavior and keeping rational math in fp32.
  - adaptive expert gather strategy:
    - direct gather for tiny `(tokens × topk)` workloads,
    - unique-compressed gather when workloads are large enough for compression wins.
  - strict shape/index validation to catch bad dispatch state early.
  - `recommend_tuning`: architecture-aware defaults for T4 (SM75) and newer GPUs like RTX 3060 (SM86), including AMP dtype + gather threshold recommendations.
  - `configure_cuda_math`: one-call setup for TF32 toggles and float32 matmul precision hint.
  - `maybe_autocast`, `maybe_compile`, and `maybe_inference_mode` helpers for consistent fast-path runtime wiring.
  - explicit null-expert support via `null_expert_id` to mask null routes before expert gather.
  - per-step rational support via `step_idx`, base per-step `coeffs` (`[S,G,5]`) and additive deltas (`rat_delta`, `step_rat_delta`).

## Suggested training profile

- **T4 (SM75)**
  - Use fp16 autocast (`amp_dtype=torch.float16`).
  - Keep `moe_chunk_size` around `768`.
  - Disable TF32.
  - Lower unique-compression threshold (default `1536`) so tiny micro-batches don't pay unique/sort overhead.
- **RTX 3060 12GB (SM86)**
  - Prefer bf16 if your PyTorch/CUDA stack reports support, otherwise fp16.
  - `moe_chunk_size` around `1536`.
  - Enable TF32 for matmuls.
  - Keep the default larger unique-compression threshold (`2048`) for better dispatch balance.

## Minimal integration

```python
import torch
from peer_grkan_kernels import (
    recommend_tuning,
    configure_cuda_math,
    maybe_autocast,
    maybe_compile,
    moe_ffn_chunked_fast_gather_einsum,
)

tuning = recommend_tuning(torch.device("cuda"))
configure_cuda_math(tuning)

moe_kernel = maybe_compile(moe_ffn_chunked_fast_gather_einsum)

with maybe_autocast(tuning):
    out = moe_kernel(
        x_flat=z,
        bank_u=bank.U,
        bank_v=bank.V,
        group_id=bank.group_id,
        coeffs=rat.coeffs,
        expert_ids=expert_ids,
        route_w=route_w,
        moe_chunk_size=tuning.moe_chunk_size,
        unique_compression_min_k_tokens=tuning.unique_compression_min_k_tokens,
        null_expert_id=cfg.n_experts - 1,
        step_idx=recursion_step,
        rat_delta=block_rat_delta,           # optional [G,5]
        step_rat_delta=block_step_rat_delta, # optional [S,G,5]
        dispatch_backend="auto",            # torch|triton|triton_fused|auto
        eps=cfg.eps,
    )
```


## Kernelization guidance

- **Kernelize now**: rational gate math + MoE gather/einsum dispatch (already done).
- **Keep in PyTorch for now**: LoRA/post-MoE adapters and lightweight per-step scalar gates, unless profiling proves they are a hotspot.
- **Why**: the dispatch + rational path dominates FLOPs/memory traffic; small per-layer adapters are usually bandwidth-light and easier to iterate in eager/compile mode.


## Benchmark + autotuning

Yes — benchmark here means comparing **naive/reference PyTorch** vs the optimized kernel path and sweeping tuning knobs.

Run:

```bash
python benchmarks/autotune_peer_grkan.py --device cuda --backend auto --save-profile profiles/t4-auto.json
```

This reports:
- naive latency,
- best optimized latency over (`moe_chunk_size`, `unique_compression_min_k_tokens`) sweep,
- speedup vs naive,
- full sweep table as JSON.


## Compile/autocast observability

- `maybe_compile(..., log_fn=...)` now attaches compile telemetry (`CompileTelemetry`) to the returned callable.
- Use `get_compile_telemetry(compiled_fn)` to inspect whether compile succeeded or fell back and why.
- Use `explain_graph_breaks(fn, *example_args)` for best-effort graph-break diagnostics when TorchDynamo support is present.


## Stress/regression tests

The test suite includes stress coverage for:
- high-null routing rates,
- signed route weights (sign-head-like behavior),
- multi-seed randomized runs,
- direct-vs-unique gather parity and naive-reference parity checks.


## Experimental Triton fused gate+dispatch path

- `moe_ffn_chunked_fast_gather_einsum(..., dispatch_backend=...)` supports:
  - `"torch"` (default eager/compile path),
  - `"triton"` (Triton gate*mul path with custom autograd),
  - `"triton_fused"` (adds Triton fused project-reduce in inference-safe scenarios),
  - `"auto"` (prefer Triton-capable paths then fallback to torch).
- Dispatch telemetry is attached to the function after a call; inspect with `get_dispatch_telemetry(moe_ffn_chunked_fast_gather_einsum)`.
- If Triton is unavailable on a machine, behavior falls back to torch path while preserving numerics.
- Training gradients are preserved end-to-end for both `dispatch_backend="torch"` and `dispatch_backend="triton"`; backend parity is covered by tests.


## CUDA-only Triton parity tests

The unit suite includes CUDA-gated tests that run when CUDA is available to verify:
- Triton backend usage/fallback telemetry behavior,
- forward/backward parity between `dispatch_backend="torch"` and `dispatch_backend="triton"` on-device.


Use a saved profile to re-run only the selected config:

```bash
python benchmarks/autotune_peer_grkan.py --device cuda --backend auto --load-profile profiles/t4-auto.json
```

You can load and apply saved profiles in training code with:
- `load_autotune_profile(path)`
- `apply_autotune_profile(profile)`

## CUDA CI automation

A GitHub Actions workflow is included at `.github/workflows/cuda-kernels.yml`:
- CPU sanity job (py_compile + unittest discovery)
- CUDA job for self-hosted GPU runners, including CUDA-gated Triton tests and autotune profile artifact upload



For deterministic CI, pin torch/triton versions in your GPU runner environment and keep runner labels aligned with `.github/workflows/cuda-kernels.yml`.
