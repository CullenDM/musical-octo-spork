"""Benchmark + autotune utility for peer_grkan_kernels.

Compares naive reference vs optimized chunked kernel and sweeps chunk/threshold.
Run:
  python benchmarks/autotune_peer_grkan.py
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

from peer_grkan_kernels import get_dispatch_telemetry, moe_ffn_chunked_fast_gather_einsum, recommend_tuning


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
    """Vectorized torch baseline (no Python token/expert loop)."""

    u_sel = U.index_select(0, expert_ids.reshape(-1)).view(*expert_ids.shape, x.shape[1], U.shape[-1])
    v_sel = V.index_select(0, expert_ids.reshape(-1)).view(*expert_ids.shape, V.shape[-2], x.shape[1])
    g_sel = group_id.index_select(0, expert_ids.reshape(-1)).view(*expert_ids.shape)

    h = torch.einsum("td,tkdr->tkr", x, u_sel).float()
    c = coeffs[g_sel]
    h2 = h * h
    num = c[..., 0] + c[..., 1] * h + c[..., 2] * h2
    den = 1.0 + (c[..., 3] * h + c[..., 4] * h2).abs() + eps
    gate = num / den
    y = torch.einsum("tkr,tkrd->tkd", (h * gate).to(v_sel.dtype), v_sel).float()
    return (y * route_w.unsqueeze(-1)).sum(dim=1)


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


def collect_dispatch_telemetry(fn, kwargs: Dict[str, Any], samples: int) -> Dict[str, Dict[str, int]]:
    backend_used_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for _ in range(max(0, samples)):
        _ = fn(**kwargs)
        telem = get_dispatch_telemetry(fn)
        if telem is None:
            reason_counts["missing_telemetry"] += 1
            continue
        backend_used_counts[str(telem.backend_used)] += 1
        reason_counts[str(telem.reason)] += 1
    return {
        "backend_used_counts": dict(backend_used_counts),
        "reason_counts": dict(reason_counts),
    }


def shape_sweep(mode: str, t: int) -> List[int]:
    if mode == "single":
        return [t]
    if mode == "t4_train":
        return [64, 128, 256, 512, 1024, 1536]
    raise ValueError(f"unsupported shape sweep mode: {mode}")


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
    ap.add_argument("--skip-naive", action="store_true", help="Skip very slow naive Python baseline")
    ap.add_argument("--telemetry-samples", type=int, default=5, help="How many dispatch telemetry samples to collect per config")
    ap.add_argument("--save-profile", default="", help="Optional path to save best profile JSON")
    ap.add_argument("--load-profile", default="", help="Optional path to load profile JSON and evaluate only that config")
    args = ap.parse_args()

    device = torch.device(args.device)
    tuning = recommend_tuning(device if device.type == "cuda" else None)

    chunk_grid = sorted(set([256, 512, 768, 1024, 1536, tuning.moe_chunk_size]))
    thr_grid = sorted(set([0, 512, 1024, 1536, 2048, tuning.unique_compression_min_k_tokens]))

    if args.load_profile:
        loaded = json.loads(Path(args.load_profile).read_text())
        chunk_grid = [int(loaded["best"]["chunk"])]
        thr_grid = [int(loaded["best"]["threshold"])]

    per_shape: List[Dict[str, Any]] = []
    for shape_t in shape_sweep(args.shape_sweep, args.t):
        x = torch.randn(shape_t, args.d, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        U = torch.randn(args.n, args.d, args.r, device=device, dtype=x.dtype)
        V = torch.randn(args.n, args.r, args.d, device=device, dtype=x.dtype)
        group_id = torch.randint(0, args.g, (args.n,), device=device, dtype=torch.int64)
        coeffs = torch.randn(args.g, 5, device=device, dtype=torch.float32)
        expert_ids = torch.randint(0, args.n, (shape_t, args.k), device=device, dtype=torch.int64)
        route_w = torch.softmax(torch.randn(shape_t, args.k, device=device, dtype=torch.float32), dim=-1)

        naive_t = None
        if not args.skip_naive:
            naive_t = bench(lambda: naive_moe(x, U, V, group_id, coeffs, expert_ids, route_w), iters=max(2, args.iters // 4))

        vectorized_t = bench(lambda: vectorized_torch_moe(x, U, V, group_id, coeffs, expert_ids, route_w), iters=args.iters)

        best: Dict[str, Any] = {"latency_s": 1e9}
        results: List[Dict[str, Any]] = []
        for chunk in chunk_grid:
            for thr in thr_grid:
                run_kwargs = {
                    "x_flat": x,
                    "bank_u": U,
                    "bank_v": V,
                    "group_id": group_id,
                    "coeffs": coeffs,
                    "expert_ids": expert_ids,
                    "route_w": route_w,
                    "moe_chunk_size": chunk,
                    "unique_compression_min_k_tokens": thr,
                    "dispatch_backend": args.backend,
                }
                t = bench(lambda: moe_ffn_chunked_fast_gather_einsum(**run_kwargs), iters=args.iters)
                telemetry = collect_dispatch_telemetry(
                    moe_ffn_chunked_fast_gather_einsum,
                    run_kwargs,
                    samples=args.telemetry_samples,
                )
                row = {
                    "chunk": chunk,
                    "threshold": thr,
                    "latency_s": t,
                    "telemetry": telemetry,
                }
                results.append(row)
                if t < float(best["latency_s"]):
                    best = row

        speedup_vs_naive = (naive_t / best["latency_s"]) if (naive_t is not None and best["latency_s"] > 0) else None
        speedup_vs_vectorized = (vectorized_t / best["latency_s"]) if best["latency_s"] > 0 else float("inf")
        per_shape.append(
            {
                "shape": {"T": shape_t, "D": args.d, "N": args.n, "R": args.r, "K": args.k, "G": args.g},
                "naive_latency_s": naive_t,
                "vectorized_torch_latency_s": vectorized_t,
                "best": best,
                "speedup_vs_naive": speedup_vs_naive,
                "speedup_vs_vectorized_torch": speedup_vs_vectorized,
                "all_results": results,
            }
        )

    payload: Dict[str, object] = {
        "device": str(device),
        "shape_sweep": args.shape_sweep,
        "dispatch_backend": args.backend,
        "skip_naive": bool(args.skip_naive),
        "telemetry_samples": int(args.telemetry_samples),
        "dtype": str(x.dtype),
        "per_shape": per_shape,
    }

    if args.shape_sweep == "single" and per_shape:
        payload.update(per_shape[0])
    if args.save_profile:
        out_path = Path(args.save_profile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
