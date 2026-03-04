"""Benchmark + autotune utility for peer_grkan_kernels.

Compares:
- naive Python-loop reference,
- stronger vectorized torch baseline,
- optimized chunked kernel path with backend telemetry,

and sweeps shape/chunk/threshold configurations.

Run:
  python benchmarks/autotune_peer_grkan.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from peer_grkan_kernels import (
    flash_rational_gate,
    get_dispatch_telemetry,
    moe_ffn_chunked_fast_gather_einsum,
    recommend_tuning,
)


def naive_moe(x, U, V, group_id, coeffs, expert_ids, route_w, eps=1e-5):
    t, d = x.shape
    k = expert_ids.shape[1]
    out = torch.zeros((t, d), device=x.device, dtype=torch.float32)
    for ti in range(t):
        token = x[ti]
        acc = torch.zeros((d,), device=x.device, dtype=torch.float32)
        for ki in range(k):
            eid = int(expert_ids[ti, ki].item())
            w = route_w[ti, ki]
            u = U[eid]
            v = V[eid]
            gid = int(group_id[eid].item())
            c = coeffs[gid]
            h = torch.einsum("d,dr->r", token, u).float()
            h2 = h * h
            num = c[0] + c[1] * h + c[2] * h2
            den = 1.0 + (c[3] * h + c[4] * h2).abs() + eps
            gate = num / den
            y = torch.einsum("r,rd->d", (h * gate).to(v.dtype), v).float()
            acc = acc + y * w.float()
        out[ti] = acc
    return out


def vectorized_torch_moe(x, U, V, group_id, coeffs, expert_ids, route_w, eps=1e-5):
    """Stronger eager torch baseline: fully vectorized gather+einsum path."""

    t, d = x.shape
    k = expert_ids.shape[1]
    r = U.shape[-1]

    e_flat = expert_ids.reshape(-1)
    u_sel = U.index_select(0, e_flat).view(t, k, d, r)
    v_sel = V.index_select(0, e_flat).view(t, k, r, d)
    g_sel = group_id.index_select(0, e_flat).view(t, k)

    h = torch.einsum("td,tkdr->tkr", x, u_sel)
    h32 = h.float()
    hg = h32 * flash_rational_gate(h32, coeffs.float(), g_sel, eps=eps)
    y = torch.einsum("tkr,tkrd->tkd", hg.to(v_sel.dtype), v_sel)
    return (y.float() * route_w.float().unsqueeze(-1)).sum(dim=1)


def shape_grid_from_args(args: argparse.Namespace) -> List[Dict[str, int]]:
    if args.shape_sweep == "single":
        return [{"T": args.t, "D": args.d, "N": args.n, "R": args.r, "K": args.k, "G": args.g}]
    if args.shape_sweep == "t4_train":
        return [
            {"T": 128, "D": 768, "N": 1024, "R": 4, "K": 4, "G": 64},
            {"T": 256, "D": 768, "N": 1024, "R": 4, "K": 4, "G": 64},
            {"T": 512, "D": 768, "N": 1024, "R": 4, "K": 4, "G": 64},
            {"T": 768, "D": 768, "N": 1024, "R": 4, "K": 4, "G": 64},
            {"T": 1024, "D": 768, "N": 1024, "R": 4, "K": 4, "G": 64},
        ]
    raise ValueError(f"unsupported --shape-sweep={args.shape_sweep}")


def summarize_telemetry(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    backend_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    for s in samples:
        backend = str(s.get("backend_used", "unknown"))
        reason = str(s.get("reason", "unknown"))
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "samples": list(samples),
        "backend_used_counts": backend_counts,
        "reason_counts": reason_counts,
    }


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--t", type=int, default=256)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--g", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--backend", choices=["torch", "triton", "triton_fused", "auto"], default="auto")
    ap.add_argument("--shape-sweep", choices=["single", "t4_train"], default="single")
    ap.add_argument("--skip-naive", action="store_true", help="Skip very-slow Python-loop baseline")
    ap.add_argument("--telemetry-samples", type=int, default=3, help="How many post-bench calls to sample dispatch telemetry")
    ap.add_argument("--save-profile", default="", help="Optional path to save best profile JSON")
    ap.add_argument("--load-profile", default="", help="Optional path to load profile JSON and evaluate only that config")
    args = ap.parse_args()

    device = torch.device(args.device)
    tuning = recommend_tuning(device if device.type == "cuda" else None)

    loaded = json.loads(Path(args.load_profile).read_text()) if args.load_profile else None
    chunk_grid = sorted(set([256, 512, 768, 1024, 1536, tuning.moe_chunk_size]))
    thr_grid = sorted(set([0, 512, 1024, 1536, 2048, tuning.unique_compression_min_k_tokens]))
    if loaded is not None:
        chunk_grid = [int(loaded["best"]["chunk"])]
        thr_grid = [int(loaded["best"]["threshold"])]

    per_shape_results: List[Dict[str, object]] = []
    global_best: Dict[str, object] = {"latency_s": 1e9}

    for shape in shape_grid_from_args(args):
        t, d, n, r, k, g = shape["T"], shape["D"], shape["N"], shape["R"], shape["K"], shape["G"]
        x = torch.randn(t, d, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        U = torch.randn(n, d, r, device=device, dtype=x.dtype)
        V = torch.randn(n, r, d, device=device, dtype=x.dtype)
        group_id = torch.randint(0, g, (n,), device=device, dtype=torch.int64)
        coeffs = torch.randn(g, 5, device=device, dtype=torch.float32)
        expert_ids = torch.randint(0, n, (t, k), device=device, dtype=torch.int64)
        route_w = torch.softmax(torch.randn(t, k, device=device, dtype=torch.float32), dim=-1)

        naive_t = None if args.skip_naive else bench(
            lambda: naive_moe(x, U, V, group_id, coeffs, expert_ids, route_w),
            iters=max(2, args.iters // 4),
        )
        vectorized_t = bench(
            lambda: vectorized_torch_moe(x, U, V, group_id, coeffs, expert_ids, route_w),
            iters=args.iters,
        )

        best: Dict[str, float] = {"latency_s": 1e9}
        results = []
        for chunk in chunk_grid:
            for thr in thr_grid:
                latency_opt = bench(
                    lambda: moe_ffn_chunked_fast_gather_einsum(
                        x_flat=x,
                        bank_u=U,
                        bank_v=V,
                        group_id=group_id,
                        coeffs=coeffs,
                        expert_ids=expert_ids,
                        route_w=route_w,
                        moe_chunk_size=chunk,
                        unique_compression_min_k_tokens=thr,
                        dispatch_backend=args.backend,
                    ),
                    iters=args.iters,
                )

                telemetry_samples = []
                for _ in range(max(1, args.telemetry_samples)):
                    _ = moe_ffn_chunked_fast_gather_einsum(
                        x_flat=x,
                        bank_u=U,
                        bank_v=V,
                        group_id=group_id,
                        coeffs=coeffs,
                        expert_ids=expert_ids,
                        route_w=route_w,
                        moe_chunk_size=chunk,
                        unique_compression_min_k_tokens=thr,
                        dispatch_backend=args.backend,
                    )
                    telem = get_dispatch_telemetry(moe_ffn_chunked_fast_gather_einsum)
                    telemetry_samples.append(
                        {
                            "backend_requested": None if telem is None else telem.backend_requested,
                            "backend_used": None if telem is None else telem.backend_used,
                            "triton_available": None if telem is None else telem.triton_available,
                            "reason": None if telem is None else telem.reason,
                        }
                    )

                row = {
                    "chunk": chunk,
                    "threshold": thr,
                    "latency_s": latency_opt,
                    "telemetry": summarize_telemetry(telemetry_samples),
                }
                results.append(row)
                if latency_opt < best["latency_s"]:
                    best = row

        per_shape = {
            "shape": shape,
            "dtype": str(x.dtype),
            "naive_latency_s": naive_t,
            "vectorized_torch_latency_s": vectorized_t,
            "best": best,
            "speedup_vs_vectorized_torch": (vectorized_t / best["latency_s"]) if best["latency_s"] > 0 else float("inf"),
            "speedup_vs_naive": (None if naive_t is None else (naive_t / best["latency_s"]) if best["latency_s"] > 0 else float("inf")),
            "all_results": results,
        }
        per_shape_results.append(per_shape)

        if float(best["latency_s"]) < float(global_best["latency_s"]):
            global_best = {"shape": shape, **best}

    payload: Dict[str, object] = {
        "device": str(device),
        "dispatch_backend": args.backend,
        "shape_sweep": args.shape_sweep,
        "chunk_grid": chunk_grid,
        "threshold_grid": thr_grid,
        "best": global_best,
        "per_shape": per_shape_results,
    }
    if args.save_profile:
        out_path = Path(args.save_profile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
