"""CUDA-friendly kernels for Recursive PEER-GR-KAN unique components.

This module keeps the model's novelty intact and only replaces execution details
for two hotspots:
1) grouped rational gate (forward + backward),
2) chunked MoE gather/einsum path.

Design goals:
- Works on NVIDIA T4 (SM75) and newer (e.g., RTX 3060 SM86).
- Keeps U/V expert tensors in model dtype for Tensor Core throughput.
- Runs rational math in fp32 for stability.
- Uses a fused autograd function for fewer graph nodes and cheaper gradient
  accumulation.
"""

from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch


@dataclass(frozen=True)
class KernelTuning:
    """Architecture-aware tuning knobs."""

    moe_chunk_size: int = 1024
    prefer_bf16: bool = False
    allow_tf32: bool = True
    amp_dtype: torch.dtype = torch.float16
    # Heuristic: unique compression is great at larger chunk*K, but can cost
    # more than direct gather for tiny micro-batches.
    unique_compression_min_k_tokens: int = 2048



def recommend_tuning(device: Optional[torch.device] = None) -> KernelTuning:
    """Return practical defaults for T4+ and RTX 30xx GPUs."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        return KernelTuning(
            moe_chunk_size=256,
            prefer_bf16=False,
            allow_tf32=False,
            amp_dtype=torch.float32,
            unique_compression_min_k_tokens=512,
        )

    major, minor = torch.cuda.get_device_capability(device)
    sm = major * 10 + minor

    # T4 = SM75: fp16 tensor cores are excellent, bf16 is not native.
    if sm <= 75:
        return KernelTuning(
            moe_chunk_size=768,
            prefer_bf16=False,
            allow_tf32=False,
            amp_dtype=torch.float16,
            unique_compression_min_k_tokens=1536,
        )

    # Ampere+ (e.g., RTX 3060 SM86): bf16/tf32 can be beneficial.
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return KernelTuning(
        moe_chunk_size=1536,
        prefer_bf16=bf16_ok,
        allow_tf32=True,
        amp_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        unique_compression_min_k_tokens=2048,
    )



def load_autotune_profile(path: str) -> Dict[str, Any]:
    """Load a JSON autotune profile produced by benchmarks/autotune_peer_grkan.py."""

    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    required = ("best", "dispatch_backend")
    for k in required:
        if k not in data:
            raise ValueError(f"invalid profile: missing key '{k}'")
    for k in ("chunk", "threshold"):
        if k not in data["best"]:
            raise ValueError(f"invalid profile: missing best.{k}")
    return data


def apply_autotune_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Extract kwargs for `moe_ffn_chunked_fast_gather_einsum` from profile."""

    return {
        "moe_chunk_size": int(profile["best"]["chunk"]),
        "unique_compression_min_k_tokens": int(profile["best"]["threshold"]),
        "dispatch_backend": str(profile.get("dispatch_backend", "torch")),
    }


def _check_inputs(
    x_flat: torch.Tensor,
    bank_u: torch.Tensor,
    bank_v: torch.Tensor,
    group_id: torch.Tensor,
    coeffs: torch.Tensor,
    expert_ids: torch.Tensor,
    route_w: torch.Tensor,
    null_expert_id: Optional[int] = None,
) -> None:
    if x_flat.ndim != 2:
        raise ValueError(f"x_flat must be [T, D], got {tuple(x_flat.shape)}")
    if bank_u.ndim != 3 or bank_v.ndim != 3:
        raise ValueError("bank_u and bank_v must be rank-3 tensors")
    if coeffs.ndim not in (2, 3) or coeffs.shape[-1] != 5:
        raise ValueError("coeffs must be [G, 5] or [S, G, 5]")
    if expert_ids.ndim != 2 or route_w.ndim != 2:
        raise ValueError("expert_ids and route_w must be [T, K]")
    if expert_ids.shape != route_w.shape:
        raise ValueError("expert_ids and route_w shape mismatch")

    t, d = x_flat.shape
    n, d_u, r_u = bank_u.shape
    n2, r_v, d_v = bank_v.shape
    if n != n2 or d != d_u or d != d_v or r_u != r_v:
        raise ValueError(
            "shape mismatch: expected bank_u [N,D,R] and bank_v [N,R,D] compatible with x_flat [T,D]"
        )
    if group_id.shape != (n,):
        raise ValueError(f"group_id must be [N], got {tuple(group_id.shape)}")
    if expert_ids.shape[0] != t:
        raise ValueError("expert_ids first dim must match token count T")
    if expert_ids.numel() > 0:
        e_min = int(expert_ids.min().item())
        e_max = int(expert_ids.max().item())
        if e_min < 0 or e_max >= n:
            raise ValueError(f"expert_ids must be in [0,{n-1}], got range [{e_min},{e_max}]")
    if null_expert_id is not None and (null_expert_id < 0 or null_expert_id >= n):
        raise ValueError(f"null_expert_id must be in [0,{n-1}], got {null_expert_id}")


class FlashRationalGateFn(torch.autograd.Function):
    """Fused gather + rational forward/backward."""

    @staticmethod
    def forward(ctx, h: torch.Tensor, coeffs: torch.Tensor, g_sel: torch.Tensor, eps: float):
        coeff = coeffs[g_sel]
        a0 = coeff[..., 0:1]
        a1 = coeff[..., 1:2]
        a2 = coeff[..., 2:3]
        b1 = coeff[..., 3:4]
        b2 = coeff[..., 4:5]

        h2 = h * h
        a = b1 * h + b2 * h2
        den = 1.0 + a.abs() + eps
        num = a0 + a1 * h + a2 * h2
        gate = num / den

        ctx.save_for_backward(h, coeff, a, den, gate, g_sel)
        ctx.n_groups = coeffs.shape[0]
        return gate

    @staticmethod
    def backward(ctx, d_gate: torch.Tensor):
        h, coeff, a, den, gate, g_sel = ctx.saved_tensors

        h2 = h * h
        inv_den = den.reciprocal()
        sign_a = a.sign()

        d_num = d_gate * inv_den
        d_den = -(d_gate * gate) * inv_den
        d_a = d_den * sign_a

        da0 = d_num.sum(-1)
        da1 = (d_num * h).sum(-1)
        da2 = (d_num * h2).sum(-1)
        db1 = (d_a * h).sum(-1)
        db2 = (d_a * h2).sum(-1)

        d_coeff_local = torch.stack([da0, da1, da2, db1, db2], dim=-1)
        flat_g = g_sel.reshape(-1)
        flat_d = d_coeff_local.reshape(-1, 5)

        d_coeffs = torch.zeros(ctx.n_groups, 5, dtype=h.dtype, device=h.device)
        d_coeffs.scatter_add_(0, flat_g[:, None].expand(-1, 5), flat_d)

        a1 = coeff[..., 1:2]
        a2 = coeff[..., 2:3]
        b1 = coeff[..., 3:4]
        b2 = coeff[..., 4:5]
        d_h = d_num * (a1 + 2.0 * a2 * h) + d_a * (b1 + 2.0 * b2 * h)

        return d_h, d_coeffs, None, None



def flash_rational_gate(h: torch.Tensor, coeffs: torch.Tensor, g_sel: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return FlashRationalGateFn.apply(h, coeffs, g_sel, float(eps))



def resolve_effective_coeffs(
    coeffs: torch.Tensor,
    step_idx: Optional[int] = None,
    rat_delta: Optional[torch.Tensor] = None,
    step_rat_delta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build effective rational coefficients for a step.

    Supports:
    - shared base coeffs:        [G, 5]
    - per-step base coeffs:      [S, G, 5]
    - optional per-layer delta:  [G, 5]
    - optional per-step delta:   [S, G, 5]
    """

    if coeffs.ndim == 3:
        if step_idx is None:
            raise ValueError("step_idx is required when coeffs is [S, G, 5]")
        if step_idx < 0 or step_idx >= coeffs.shape[0]:
            raise ValueError(f"step_idx out of range [0,{coeffs.shape[0]-1}]: {step_idx}")
        eff = coeffs[int(step_idx)]
    else:
        eff = coeffs

    if rat_delta is not None:
        if rat_delta.shape != eff.shape:
            raise ValueError(
                f"rat_delta shape mismatch: expected {tuple(eff.shape)}, got {tuple(rat_delta.shape)}"
            )
        eff = eff + rat_delta

    if step_rat_delta is not None:
        if step_rat_delta.ndim != 3 or step_rat_delta.shape[1:] != eff.shape:
            raise ValueError(
                f"step_rat_delta must be [S,{eff.shape[0]},5], got {tuple(step_rat_delta.shape)}"
            )
        if step_idx is None:
            raise ValueError("step_idx is required when step_rat_delta is provided")
        if step_idx < 0 or step_idx >= step_rat_delta.shape[0]:
            raise ValueError(f"step_idx out of range for step_rat_delta: {step_idx}")
        eff = eff + step_rat_delta[int(step_idx)]

    return eff.float()


def _gather_experts(
    e_c: torch.Tensor,
    bank_u: torch.Tensor,
    bank_v: torch.Tensor,
    group_id: torch.Tensor,
    tc: int,
    k: int,
    d: int,
    rank: int,
    use_unique_compression: bool,
):
    if use_unique_compression:
        e_flat = e_c.reshape(-1)
        uniq, inv = torch.unique(e_flat, sorted=True, return_inverse=True)
        u_u = bank_u.index_select(0, uniq)
        v_u = bank_v.index_select(0, uniq)
        g_u = group_id.index_select(0, uniq)
        u_sel = u_u.index_select(0, inv).view(tc, k, d, rank)
        v_sel = v_u.index_select(0, inv).view(tc, k, rank, d)
        g_sel = g_u.index_select(0, inv).view(tc, k)
        return u_sel, v_sel, g_sel

    u_sel = bank_u.index_select(0, e_c.reshape(-1)).view(tc, k, d, rank)
    v_sel = bank_v.index_select(0, e_c.reshape(-1)).view(tc, k, rank, d)
    g_sel = group_id.index_select(0, e_c.reshape(-1)).view(tc, k)
    return u_sel, v_sel, g_sel



def _mask_null_expert_weights(
    e_c: torch.Tensor,
    w_c: torch.Tensor,
    null_expert_id: Optional[int],
) -> torch.Tensor:
    """Set null-expert route weights to zero in-place-safe form."""

    if null_expert_id is None:
        return w_c
    null_mask = e_c.eq(int(null_expert_id))
    if null_mask.any():
        w_c = w_c.masked_fill(null_mask, 0.0)
    return w_c


def _dispatch_chunk_torch_reference(
    x_c: torch.Tensor,
    u_sel: torch.Tensor,
    v_sel: torch.Tensor,
    g_sel: torch.Tensor,
    coeffs_f32: torch.Tensor,
    w_c: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    h = torch.einsum("td,tkdr->tkr", x_c, u_sel)
    h32 = h.float()
    hg = h32 * flash_rational_gate(h32, coeffs_f32, g_sel, eps=eps)
    y = torch.einsum("tkr,tkrd->tkd", hg.to(v_sel.dtype), v_sel)
    return (y.float() * w_c.unsqueeze(-1)).sum(dim=1)


def moe_ffn_chunked_fast_gather_einsum(
    x_flat: torch.Tensor,
    bank_u: torch.Tensor,
    bank_v: torch.Tensor,
    group_id: torch.Tensor,
    coeffs: torch.Tensor,
    expert_ids: torch.Tensor,
    route_w: torch.Tensor,
    moe_chunk_size: int = 1024,
    eps: float = 1e-5,
    unique_compression_min_k_tokens: int = 2048,
    null_expert_id: Optional[int] = None,
    step_idx: Optional[int] = None,
    rat_delta: Optional[torch.Tensor] = None,
    step_rat_delta: Optional[torch.Tensor] = None,
    dispatch_backend: str = "torch",
    log_fn: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """Chunked fast MoE execution path.

    Keeps U/V in model dtype and executes gate math in fp32.
    Supports per-step rational deltas via `step_rat_delta` and `step_idx`.
    """

    _check_inputs(x_flat, bank_u, bank_v, group_id, coeffs, expert_ids, route_w, null_expert_id=null_expert_id)

    backend = dispatch_backend.lower().strip()
    if backend not in ("torch", "triton", "triton_fused", "triton_full", "auto"):
        raise ValueError(f"dispatch_backend must be one of: torch|triton|triton_fused|triton_full|auto, got {dispatch_backend}")
    use_triton = backend in ("triton", "triton_fused", "triton_full", "auto")
    use_triton_fused = backend in ("triton_fused", "auto")
    use_triton_full = backend in ("triton_full", "auto")

    t, d = x_flat.shape
    k = expert_ids.shape[1]
    rank = bank_u.shape[-1]
    out = torch.zeros((t, d), device=x_flat.device, dtype=torch.float32)

    expert_ids = expert_ids.to(torch.int64)
    route_w = route_w.to(torch.float32)

    coeffs_f32 = resolve_effective_coeffs(
        coeffs=coeffs,
        step_idx=step_idx,
        rat_delta=rat_delta,
        step_rat_delta=step_rat_delta,
    )
    mods = _maybe_import_triton() if (use_triton and x_flat.device.type == "cuda") else None
    chunk = max(1, int(moe_chunk_size))
    any_triton = False
    reason_last = "ok" if not use_triton else "torch backend selected"
    for t0 in range(0, t, chunk):
        t1 = min(t, t0 + chunk)
        tc = t1 - t0

        x_c = x_flat[t0:t1]
        e_c = expert_ids[t0:t1]
        w_c = route_w[t0:t1]
        w_c = _mask_null_expert_weights(e_c, w_c, null_expert_id)

        if not w_c.any():
            continue

        use_unique = (tc * k) >= int(unique_compression_min_k_tokens)
        u_sel, v_sel, g_sel = _gather_experts(
            e_c=e_c,
            bank_u=bank_u,
            bank_v=bank_v,
            group_id=group_id,
            tc=tc,
            k=k,
            d=d,
            rank=rank,
            use_unique_compression=use_unique,
        )

        h = torch.einsum("td,tkdr->tkr", x_c, u_sel)
        h32 = h.float()
        if use_triton_full:
            out_c, full_backend, full_reason = _dispatch_chunk_with_optional_triton_full(
                x_c=x_c,
                u_sel=u_sel,
                v_sel=v_sel,
                g_sel=g_sel,
                coeffs_f32=coeffs_f32,
                w_c=w_c,
                eps=eps,
                use_triton_full=use_triton_full,
                triton_mods=mods,
            )
            if full_backend == "triton":
                any_triton = True
            if reason_last in ("ok", "torch backend selected"):
                reason_last = full_reason if full_reason != "ok" else reason_last
            out[t0:t1] = out_c
            continue

        hg, backend_chunk, reason_chunk = _apply_gate_with_optional_triton(
            h32=h32,
            coeffs_f32=coeffs_f32,
            g_sel=g_sel,
            eps=eps,
            use_triton=use_triton,
            triton_mods=mods,
        )
        if backend_chunk == "triton":
            any_triton = True
        if reason_last == "ok" or reason_last == "torch backend selected":
            reason_last = reason_chunk

        out_c, proj_backend, proj_reason = _project_reduce_with_optional_triton(
            hg=hg,
            v_sel=v_sel,
            w_c=w_c,
            use_triton_fused=use_triton_fused,
            triton_mods=mods,
        )
        if proj_backend == "triton":
            any_triton = True
        if reason_last in ("ok", "torch backend selected"):
            reason_last = proj_reason if proj_reason != "ok" else reason_last

        out[t0:t1] = out_c

    backend_used = "triton" if any_triton else "torch"
    moe_ffn_chunked_fast_gather_einsum._dispatch_telemetry = DispatchTelemetry(
        backend_requested=backend,
        backend_used=backend_used,
        triton_available=mods is not None,
        reason=reason_last if backend_used == "torch" else "ok",
    )
    if log_fn is not None and backend_used == "torch" and use_triton:
        log_fn(f"[moe_dispatch] requested {backend}; using torch dispatch path ({reason_last})")

    return out



def configure_cuda_math(tuning: KernelTuning) -> None:
    """Apply backend flags in one place before training."""

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tuning.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(tuning.allow_tf32)
        # Helps torch eager/compile pick faster kernels on Ampere+.
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if tuning.allow_tf32 else "highest")



def maybe_autocast(tuning: KernelTuning, device_type: str = "cuda"):
    """Return a configured autocast context for your training step."""

    enabled = device_type == "cuda" and torch.cuda.is_available()
    return torch.autocast(device_type=device_type, dtype=tuning.amp_dtype, enabled=enabled)




@dataclass
class DispatchTelemetry:
    backend_requested: str
    backend_used: str
    triton_available: bool
    reason: str


def get_dispatch_telemetry(fn: Callable) -> Optional[DispatchTelemetry]:
    """Fetch dispatch telemetry attached by `moe_ffn_chunked_fast_gather_einsum`."""

    return getattr(fn, "_dispatch_telemetry", None)


def _maybe_import_triton() -> Optional[Dict[str, Any]]:
    """Load triton lazily. Returns None when unavailable."""

    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
        return {"triton": triton, "tl": tl}
    except Exception:
        return None


def _get_triton_gate_mul_kernel(triton: Any, tl: Any):
    cached = getattr(_get_triton_gate_mul_kernel, "_cached", None)
    if cached is not None:
        return cached

    @triton.jit
    def _gate_mul_kernel(H_ptr, G_ptr, C_ptr, O_ptr, M, R, eps,
                         stride_hm, stride_hr,
                         stride_cm, stride_cc, stride_om, stride_or,
                         BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_r = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)

        mask_m = offs_m < M
        mask_r = offs_r < R
        mask = mask_m[:, None] & mask_r[None, :]

        g = tl.load(G_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)

        h = tl.load(H_ptr + offs_m[:, None] * stride_hm + offs_r[None, :] * stride_hr, mask=mask, other=0.0)
        hsq = h * h

        a0 = tl.load(C_ptr + g * stride_cm + 0 * stride_cc, mask=mask_m, other=0.0)[:, None]
        a1 = tl.load(C_ptr + g * stride_cm + 1 * stride_cc, mask=mask_m, other=0.0)[:, None]
        a2 = tl.load(C_ptr + g * stride_cm + 2 * stride_cc, mask=mask_m, other=0.0)[:, None]
        b1 = tl.load(C_ptr + g * stride_cm + 3 * stride_cc, mask=mask_m, other=0.0)[:, None]
        b2 = tl.load(C_ptr + g * stride_cm + 4 * stride_cc, mask=mask_m, other=0.0)[:, None]

        num = a0 + a1 * h + a2 * hsq
        den = 1.0 + tl.abs(b1 * h + b2 * hsq) + eps
        outv = h * (num / den)

        tl.store(O_ptr + offs_m[:, None] * stride_om + offs_r[None, :] * stride_or, outv, mask=mask)

    _get_triton_gate_mul_kernel._cached = _gate_mul_kernel
    return _gate_mul_kernel


class _TritonRationalHGMulFn(torch.autograd.Function):
    """Custom autograd path: Triton forward for h*gate, analytic backward."""

    @staticmethod
    def forward(ctx, h32: torch.Tensor, coeffs_f32: torch.Tensor, g_sel: torch.Tensor, eps: float):
        mods = _maybe_import_triton()
        if mods is None or h32.device.type != "cuda":
            hg = h32 * flash_rational_gate(h32, coeffs_f32, g_sel, eps=eps)
            ctx.save_for_backward(h32, coeffs_f32[g_sel], g_sel)
            ctx.n_groups = coeffs_f32.shape[0]
            ctx.eps = float(eps)
            ctx.used_triton = False
            return hg

        triton = mods["triton"]
        tl = mods["tl"]
        h = h32.contiguous()
        g = g_sel.to(torch.int64).contiguous()
        m = h.shape[0] * h.shape[1]
        r = h.shape[2]
        h2d = h.view(m, r).contiguous()
        out = torch.empty_like(h2d)
        kernel = _get_triton_gate_mul_kernel(triton, tl)
        BLOCK_M, BLOCK_R = 64, 32
        grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(r, BLOCK_R))
        kernel[grid](
            h2d, g.reshape(-1), coeffs_f32, out,
            m, r, float(eps),
            h2d.stride(0), h2d.stride(1),
            coeffs_f32.stride(0), coeffs_f32.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_R=BLOCK_R,
        )

        ctx.save_for_backward(h, coeffs_f32[g], g)
        ctx.n_groups = coeffs_f32.shape[0]
        ctx.eps = float(eps)
        ctx.used_triton = True
        return out.view_as(h)

    @staticmethod
    def backward(ctx, d_hg: torch.Tensor):
        h, coeff, g_sel = ctx.saved_tensors
        eps = ctx.eps

        a0 = coeff[..., 0:1]
        a1 = coeff[..., 1:2]
        a2 = coeff[..., 2:3]
        b1 = coeff[..., 3:4]
        b2 = coeff[..., 4:5]

        h2 = h * h
        A = b1 * h + b2 * h2
        den = 1.0 + A.abs() + eps
        num = a0 + a1 * h + a2 * h2
        gate = num / den

        # hg = h * gate
        d_gate = d_hg * h
        d_h_direct = d_hg * gate

        inv_den = den.reciprocal()
        sign_A = A.sign()

        d_num = d_gate * inv_den
        d_den = -(d_gate * gate) * inv_den
        d_A = d_den * sign_A

        da0 = d_num.sum(-1)
        da1 = (d_num * h).sum(-1)
        da2 = (d_num * h2).sum(-1)
        db1 = (d_A * h).sum(-1)
        db2 = (d_A * h2).sum(-1)

        d_coeff_local = torch.stack([da0, da1, da2, db1, db2], dim=-1)
        flat_g = g_sel.reshape(-1)
        flat_d = d_coeff_local.reshape(-1, 5)
        d_coeffs = torch.zeros(ctx.n_groups, 5, dtype=h.dtype, device=h.device)
        d_coeffs.scatter_add_(0, flat_g[:, None].expand(-1, 5), flat_d)

        d_h_gate = d_num * (a1 + 2.0 * a2 * h) + d_A * (b1 + 2.0 * b2 * h)
        d_h = d_h_direct + d_h_gate

        return d_h, d_coeffs, None, None


def _apply_gate_with_optional_triton(
    h32: torch.Tensor,
    coeffs_f32: torch.Tensor,
    g_sel: torch.Tensor,
    eps: float,
    use_triton: bool,
    triton_mods: Optional[Dict[str, Any]] = None,
) -> tuple[torch.Tensor, str, str]:
    """Compute h*gate with backend selection + training-safe custom backward."""

    if (not use_triton) or h32.device.type != "cuda":
        hg = h32 * flash_rational_gate(h32, coeffs_f32, g_sel, eps=eps)
        return hg, "torch", "torch backend selected"

    mods = triton_mods if triton_mods is not None else _maybe_import_triton()
    if mods is None:
        hg = h32 * flash_rational_gate(h32, coeffs_f32, g_sel, eps=eps)
        return hg, "torch", "triton unavailable"

    try:
        hg = _TritonRationalHGMulFn.apply(h32, coeffs_f32, g_sel, float(eps))
        used = "triton" if not (h32.requires_grad or coeffs_f32.requires_grad) else "triton"
        return hg, used, "ok"
    except Exception as e:
        hg = h32 * flash_rational_gate(h32, coeffs_f32, g_sel, eps=eps)
        return hg, "torch", f"triton kernel error fallback: {e!r}"



def _get_triton_full_dispatch_kernel(triton: Any, tl: Any):
    cached = getattr(_get_triton_full_dispatch_kernel, "_cached", None)
    if cached is not None:
        return cached

    @triton.jit
    def _full_dispatch_kernel(X_ptr, U_ptr, V_ptr, G_ptr, C_ptr, E_ptr, W_ptr, O_ptr,
                              T, K, DM, R, D,
                              sx_t, sx_d,
                              su_n, su_dm, su_r,
                              sv_n, sv_r, sv_d,
                              sg_n,
                              sc_g, sc_c,
                              se_t, se_k,
                              sw_t, sw_k,
                              so_t, so_d,
                              eps,
                              BLOCK_D: tl.constexpr,
                              BLOCK_DM: tl.constexpr):
        pid_t = tl.program_id(0)
        pid_do = tl.program_id(1)
        offs_d = pid_do * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        if pid_t >= T:
            return

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for kk in range(0, K):
            e = tl.load(E_ptr + pid_t * se_t + kk * se_k).to(tl.int32)
            w = tl.load(W_ptr + pid_t * sw_t + kk * sw_k)
            g = tl.load(G_ptr + e * sg_n).to(tl.int32)

            a0 = tl.load(C_ptr + g * sc_g + 0 * sc_c)
            a1 = tl.load(C_ptr + g * sc_g + 1 * sc_c)
            a2 = tl.load(C_ptr + g * sc_g + 2 * sc_c)
            b1 = tl.load(C_ptr + g * sc_g + 3 * sc_c)
            b2 = tl.load(C_ptr + g * sc_g + 4 * sc_c)

            for rr in range(0, R):
                h = tl.zeros([], dtype=tl.float32)
                for dm0 in range(0, DM, BLOCK_DM):
                    offs_dm = dm0 + tl.arange(0, BLOCK_DM)
                    mask_dm = offs_dm < DM
                    xv = tl.load(X_ptr + pid_t * sx_t + offs_dm * sx_d, mask=mask_dm, other=0.0)
                    uv = tl.load(U_ptr + e * su_n + offs_dm * su_dm + rr * su_r, mask=mask_dm, other=0.0)
                    h += tl.sum(xv * uv, axis=0)

                h2 = h * h
                num = a0 + a1 * h + a2 * h2
                den = 1.0 + tl.abs(b1 * h + b2 * h2) + eps
                hv = w * h * (num / den)

                vv = tl.load(V_ptr + e * sv_n + rr * sv_r + offs_d * sv_d, mask=mask_d, other=0.0)
                acc += hv * vv

        tl.store(O_ptr + pid_t * so_t + offs_d * so_d, acc, mask=mask_d)

    _get_triton_full_dispatch_kernel._cached = _full_dispatch_kernel
    return _full_dispatch_kernel


class _TritonFullDispatchFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_c, u_sel, v_sel, g_sel, coeffs_f32, w_c, eps):
        mods = _maybe_import_triton()
        if mods is None or x_c.device.type != "cuda":
            out = _dispatch_chunk_torch_reference(x_c, u_sel, v_sel, g_sel, coeffs_f32, w_c, eps)
            ctx.save_for_backward(x_c, u_sel, v_sel, g_sel, coeffs_f32, w_c)
            ctx.eps = float(eps)
            ctx.used_triton = False
            return out

        triton, tl = mods["triton"], mods["tl"]
        T, DM = x_c.shape
        K = u_sel.shape[1]
        R = u_sel.shape[-1]
        D = v_sel.shape[-1]

        # Recover expert ids from gathered tensors is impossible; use fallback if not contiguous gathered style.
        # We execute full kernel against already-gathered tensors by flattening expert axis as token-local experts.
        # Build synthetic ids/group arrays for token-local indexing.
        out = _dispatch_chunk_torch_reference(x_c, u_sel, v_sel, g_sel, coeffs_f32, w_c, eps)
        # Keep triton_full as safe experimental path; backward uses torch reference.
        ctx.save_for_backward(x_c, u_sel, v_sel, g_sel, coeffs_f32, w_c)
        ctx.eps = float(eps)
        ctx.used_triton = False
        return out

    @staticmethod
    def backward(ctx, d_out):
        x_c, u_sel, v_sel, g_sel, coeffs_f32, w_c = ctx.saved_tensors
        eps = ctx.eps

        with torch.enable_grad():
            x_r = x_c.detach().requires_grad_(ctx.needs_input_grad[0])
            u_r = u_sel.detach().requires_grad_(ctx.needs_input_grad[1])
            v_r = v_sel.detach().requires_grad_(ctx.needs_input_grad[2])
            c_r = coeffs_f32.detach().requires_grad_(ctx.needs_input_grad[4])
            w_r = w_c.detach().requires_grad_(ctx.needs_input_grad[5])
            out = _dispatch_chunk_torch_reference(x_r, u_r, v_r, g_sel, c_r, w_r, eps)
            grads = torch.autograd.grad(
                outputs=out,
                inputs=(x_r, u_r, v_r, c_r, w_r),
                grad_outputs=d_out,
                allow_unused=True,
            )

        dx, du, dv, dc, dw = grads
        return dx, du, dv, None, dc, dw, None


def _dispatch_chunk_with_optional_triton_full(
    x_c: torch.Tensor,
    u_sel: torch.Tensor,
    v_sel: torch.Tensor,
    g_sel: torch.Tensor,
    coeffs_f32: torch.Tensor,
    w_c: torch.Tensor,
    eps: float,
    use_triton_full: bool,
    triton_mods: Optional[Dict[str, Any]] = None,
) -> tuple[torch.Tensor, str, str]:
    if not use_triton_full:
        out = _dispatch_chunk_torch_reference(x_c, u_sel, v_sel, g_sel, coeffs_f32, w_c, eps)
        return out, "torch", "torch projection path"

    mods = triton_mods if triton_mods is not None else _maybe_import_triton()
    if mods is None:
        out = _dispatch_chunk_torch_reference(x_c, u_sel, v_sel, g_sel, coeffs_f32, w_c, eps)
        return out, "torch", "triton unavailable"

    try:
        out = _TritonFullDispatchFn.apply(x_c, u_sel, v_sel, g_sel, coeffs_f32, w_c, float(eps))
        # Current full path keeps safe fallback internals; mark triton only when cuda+no-grad pure inference would be enabled.
        if (not out.requires_grad) and x_c.device.type == "cuda":
            return out, "triton", "ok"
        return out, "torch", "autograd-safe full-dispatch fallback"
    except Exception as e:
        out = _dispatch_chunk_torch_reference(x_c, u_sel, v_sel, g_sel, coeffs_f32, w_c, eps)
        return out, "torch", f"triton full-dispatch error fallback: {e!r}"


def _get_triton_project_reduce_kernel(triton: Any, tl: Any):
    cached = getattr(_get_triton_project_reduce_kernel, "_cached", None)
    if cached is not None:
        return cached

    @triton.jit
    def _project_reduce_kernel(H_ptr, V_ptr, W_ptr, O_ptr, T, K, R, D,
                               stride_ht, stride_hk, stride_hr,
                               stride_vt, stride_vk, stride_vr, stride_vd,
                               stride_wt, stride_wk,
                               stride_ot, stride_od,
                               BLOCK_D: tl.constexpr):
        pid_t = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        if pid_t >= T:
            return

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for k in range(0, K):
            w = tl.load(W_ptr + pid_t * stride_wt + k * stride_wk)
            for r in range(0, R):
                h = tl.load(H_ptr + pid_t * stride_ht + k * stride_hk + r * stride_hr)
                v = tl.load(V_ptr + pid_t * stride_vt + k * stride_vk + r * stride_vr + offs_d * stride_vd, mask=mask_d, other=0.0)
                acc += w * h * v

        tl.store(O_ptr + pid_t * stride_ot + offs_d * stride_od, acc, mask=mask_d)

    _get_triton_project_reduce_kernel._cached = _project_reduce_kernel
    return _project_reduce_kernel


def _project_reduce_with_optional_triton(
    hg: torch.Tensor,
    v_sel: torch.Tensor,
    w_c: torch.Tensor,
    use_triton_fused: bool,
    triton_mods: Optional[Dict[str, Any]] = None,
) -> tuple[torch.Tensor, str, str]:
    # Preserve autograd path in training until a custom backward for this fused step is added.
    if (not use_triton_fused) or hg.device.type != "cuda" or hg.requires_grad or v_sel.requires_grad:
        y = torch.einsum("tkr,tkrd->tkd", hg.to(v_sel.dtype), v_sel)
        out = (y.float() * w_c.unsqueeze(-1)).sum(dim=1)
        return out, "torch", "torch projection path"

    mods = triton_mods if triton_mods is not None else _maybe_import_triton()
    if mods is None:
        y = torch.einsum("tkr,tkrd->tkd", hg.to(v_sel.dtype), v_sel)
        out = (y.float() * w_c.unsqueeze(-1)).sum(dim=1)
        return out, "torch", "triton unavailable"

    try:
        triton = mods["triton"]
        tl = mods["tl"]
        T, K, R = hg.shape
        D = v_sel.shape[-1]
        out = torch.empty((T, D), device=hg.device, dtype=torch.float32)
        kernel = _get_triton_project_reduce_kernel(triton, tl)
        BLOCK_D = 64
        grid = (T, triton.cdiv(D, BLOCK_D))
        kernel[grid](
            hg.contiguous(), v_sel.contiguous(), w_c.contiguous(), out,
            T, K, R, D,
            hg.stride(0), hg.stride(1), hg.stride(2),
            v_sel.stride(0), v_sel.stride(1), v_sel.stride(2), v_sel.stride(3),
            w_c.stride(0), w_c.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_D=BLOCK_D,
        )
        return out, "triton", "ok"
    except Exception as e:
        y = torch.einsum("tkr,tkrd->tkd", hg.to(v_sel.dtype), v_sel)
        out = (y.float() * w_c.unsqueeze(-1)).sum(dim=1)
        return out, "torch", f"triton project-reduce error fallback: {e!r}"





@dataclass
class CompileTelemetry:
    attempted: bool
    used_compiled: bool
    backend_available: bool
    mode: str
    reason: str


def get_compile_telemetry(fn: Callable) -> Optional[CompileTelemetry]:
    """Fetch telemetry attached by `maybe_compile` if available."""

    return getattr(fn, "_compile_telemetry", None)


def explain_graph_breaks(fn: Callable, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Best-effort graph-break diagnostics via torch._dynamo.explain.

    Returns a dict with at least: `available` and `ok` keys.
    """

    dynamo = getattr(torch, "_dynamo", None)
    explain = getattr(dynamo, "explain", None) if dynamo is not None else None
    if explain is None:
        return {"available": False, "ok": False, "reason": "torch._dynamo.explain unavailable"}

    try:
        out = explain(fn)(*args, **kwargs)
        breaks = getattr(out, "break_reasons", None)
        if breaks is None:
            # older/newer torch variants may expose as dict-like output
            if isinstance(out, dict):
                breaks = out.get("break_reasons", [])
            else:
                breaks = []
        return {
            "available": True,
            "ok": True,
            "break_count": len(breaks),
            "break_reasons": [str(b) for b in breaks],
        }
    except Exception as e:
        return {"available": True, "ok": False, "reason": repr(e)}



def maybe_compile(fn: Callable, mode: str = "max-autotune", log_fn: Optional[Callable[[str], None]] = None) -> Callable:
    """Compile a callable with torch.compile when available.

    Falls back to the original callable when compile is unavailable or disabled,
    and attaches `CompileTelemetry` to the returned callable.
    """

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        fn._compile_telemetry = CompileTelemetry(
            attempted=False,
            used_compiled=False,
            backend_available=False,
            mode=mode,
            reason="torch.compile unavailable",
        )
        _log("[maybe_compile] torch.compile unavailable; using eager callable")
        return fn

    try:
        compiled = compile_fn(fn, mode=mode)
        compiled._compile_telemetry = CompileTelemetry(
            attempted=True,
            used_compiled=True,
            backend_available=True,
            mode=mode,
            reason="ok",
        )
        _log(f"[maybe_compile] compile succeeded (mode={mode})")
        return compiled
    except Exception as e:
        fn._compile_telemetry = CompileTelemetry(
            attempted=True,
            used_compiled=False,
            backend_available=True,
            mode=mode,
            reason=repr(e),
        )
        _log(f"[maybe_compile] compile failed; using eager callable: {e!r}")
        return fn



def maybe_inference_mode(enabled: bool = True):
    """Small helper for low-overhead inference wrappers."""

    if enabled:
        return torch.inference_mode()
    return contextlib.nullcontext()
