"""Benchmark + autotune utility for peer_grkan_kernels.

Compares naive reference vs optimized chunked kernel and sweeps chunk/threshold.
Run:
  python benchmarks/autotune_peer_grkan.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch

from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum, recommend_tuning


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
    ap.add_argument("--save-profile", default="", help="Optional path to save best profile JSON")
    ap.add_argument("--load-profile", default="", help="Optional path to load profile JSON and evaluate only that config")
    args = ap.parse_args()

    device = torch.device(args.device)
    tuning = recommend_tuning(device if device.type == "cuda" else None)

    x = torch.randn(args.t, args.d, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    U = torch.randn(args.n, args.d, args.r, device=device, dtype=x.dtype)
    V = torch.randn(args.n, args.r, args.d, device=device, dtype=x.dtype)
    group_id = torch.randint(0, args.g, (args.n,), device=device, dtype=torch.int64)
    coeffs = torch.randn(args.g, 5, device=device, dtype=torch.float32)
    expert_ids = torch.randint(0, args.n, (args.t, args.k), device=device, dtype=torch.int64)
    route_w = torch.softmax(torch.randn(args.t, args.k, device=device, dtype=torch.float32), dim=-1)

    naive_t = bench(lambda: naive_moe(x, U, V, group_id, coeffs, expert_ids, route_w), iters=max(2, args.iters // 4))

    chunk_grid = sorted(set([256, 512, 768, 1024, 1536, tuning.moe_chunk_size]))
    thr_grid = sorted(set([0, 512, 1024, 1536, 2048, tuning.unique_compression_min_k_tokens]))

    if args.load_profile:
        loaded = json.loads(Path(args.load_profile).read_text())
        chunk_grid = [int(loaded["best"]["chunk"])]
        thr_grid = [int(loaded["best"]["threshold"])]

    best: Dict[str, float] = {"latency_s": 1e9}
    results = []

    for chunk in chunk_grid:
        for thr in thr_grid:
            t = bench(
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
            row = {"chunk": chunk, "threshold": thr, "latency_s": t}
            results.append(row)
            if t < best["latency_s"]:
                best = row

    speedup = naive_t / best["latency_s"] if best["latency_s"] > 0 else float("inf")
    payload: Dict[str, object] = {
        "device": str(device),
        "shape": {"T": args.t, "D": args.d, "N": args.n, "R": args.r, "K": args.k, "G": args.g},
        "dispatch_backend": args.backend,
        "dtype": str(x.dtype),
        "naive_latency_s": naive_t,
        "best": best,
        "speedup_vs_naive": speedup,
        "all_results": results,
    }
    if args.save_profile:
        out_path = Path(args.save_profile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
