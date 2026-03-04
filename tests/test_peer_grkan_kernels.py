import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class PeerGRKANKernelParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _naive_rational_gate(self, h, coeff, eps=1e-6):
        a0 = coeff[..., 0:1]
        a1 = coeff[..., 1:2]
        a2 = coeff[..., 2:3]
        b1 = coeff[..., 3:4]
        b2 = coeff[..., 4:5]
        h2 = h * h
        num = a0 + a1 * h + a2 * h2
        den = 1.0 + (b1 * h + b2 * h2).abs() + eps
        return num / den

    def _naive_moe(self, x, U, V, group_id, coeffs_eff, expert_ids, route_w, null_expert_id=None, eps=1e-5):
        t, d = x.shape
        k = expert_ids.shape[1]
        out = torch.zeros((t, d), dtype=torch.float32)
        for ti in range(t):
            token = x[ti]
            acc = torch.zeros((d,), dtype=torch.float32)
            for ki in range(k):
                eid = int(expert_ids[ti, ki].item())
                w = route_w[ti, ki].float()
                if null_expert_id is not None and eid == int(null_expert_id):
                    w = torch.tensor(0.0, dtype=torch.float32)
                if float(w) == 0.0:
                    continue
                u = U[eid]
                v = V[eid]
                g = int(group_id[eid].item())
                h = torch.einsum("d,dr->r", token, u).float()
                gate = self._naive_rational_gate(h.unsqueeze(0), coeffs_eff[g].view(1, 5), eps=eps).squeeze(0)
                y = torch.einsum("r,rd->d", (h * gate).to(v.dtype), v).float()
                acc = acc + y * w
            out[ti] = acc
        return out

    def test_flash_rational_gate_forward_backward_parity(self):
        from peer_grkan_kernels import flash_rational_gate

        t, k, r, g = 5, 3, 4, 7
        h1 = torch.randn(t, k, r, dtype=torch.float32, requires_grad=True)
        h2 = h1.detach().clone().requires_grad_(True)
        coeff1 = torch.randn(g, 5, dtype=torch.float32, requires_grad=True)
        coeff2 = coeff1.detach().clone().requires_grad_(True)
        g_sel = torch.randint(0, g, (t, k), dtype=torch.int64)

        out_fast = flash_rational_gate(h1, coeff1, g_sel, eps=1e-6)
        out_ref = self._naive_rational_gate(h2, coeff2[g_sel], eps=1e-6)

        self.assertTrue(torch.allclose(out_fast, out_ref, atol=1e-6, rtol=1e-5))

        loss_fast = (out_fast ** 2).mean()
        loss_ref = (out_ref ** 2).mean()
        loss_fast.backward()
        loss_ref.backward()

        self.assertTrue(torch.allclose(h1.grad, h2.grad, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(coeff1.grad, coeff2.grad, atol=1e-5, rtol=1e-4))

    def test_resolve_effective_coeffs_step_and_deltas(self):
        from peer_grkan_kernels import resolve_effective_coeffs

        s, g = 3, 8
        coeffs = torch.randn(s, g, 5, dtype=torch.float32)
        rat_delta = torch.randn(g, 5, dtype=torch.float32)
        step_rat_delta = torch.randn(s, g, 5, dtype=torch.float32)
        step_idx = 2

        got = resolve_effective_coeffs(
            coeffs=coeffs,
            step_idx=step_idx,
            rat_delta=rat_delta,
            step_rat_delta=step_rat_delta,
        )
        exp = coeffs[step_idx] + rat_delta + step_rat_delta[step_idx]
        self.assertTrue(torch.allclose(got, exp.float(), atol=0.0, rtol=0.0))


    def test_resolve_effective_coeffs_requires_step_for_3d(self):
        from peer_grkan_kernels import resolve_effective_coeffs

        coeffs = torch.randn(2, 4, 5, dtype=torch.float32)
        with self.assertRaises(ValueError):
            _ = resolve_effective_coeffs(coeffs=coeffs, step_idx=None)

    def test_moe_direct_vs_unique_paths_parity(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum

        t, d, n, r, k, g = 7, 12, 25, 4, 4, 5
        x = torch.randn(t, d, dtype=torch.float32)
        U = torch.randn(n, d, r, dtype=torch.float32)
        V = torch.randn(n, r, d, dtype=torch.float32)
        group_id = torch.randint(0, g, (n,), dtype=torch.int64)
        coeffs = torch.randn(g, 5, dtype=torch.float32)
        expert_ids = torch.randint(0, n, (t, k), dtype=torch.int64)
        route_w = torch.softmax(torch.randn(t, k, dtype=torch.float32), dim=-1)

        out_direct = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x,
            bank_u=U,
            bank_v=V,
            group_id=group_id,
            coeffs=coeffs,
            expert_ids=expert_ids,
            route_w=route_w,
            unique_compression_min_k_tokens=10**9,  # force direct gather
            moe_chunk_size=3,
        )
        out_unique = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x,
            bank_u=U,
            bank_v=V,
            group_id=group_id,
            coeffs=coeffs,
            expert_ids=expert_ids,
            route_w=route_w,
            unique_compression_min_k_tokens=0,  # force unique compression
            moe_chunk_size=3,
        )
        self.assertTrue(torch.allclose(out_direct, out_unique, atol=1e-5, rtol=1e-4))

    def test_moe_input_validation_raises_for_bad_null_id(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum

        t, d, n, r, k, g = 2, 8, 9, 4, 2, 3
        x = torch.randn(t, d, dtype=torch.float32)
        U = torch.randn(n, d, r, dtype=torch.float32)
        V = torch.randn(n, r, d, dtype=torch.float32)
        group_id = torch.randint(0, g, (n,), dtype=torch.int64)
        coeffs = torch.randn(g, 5, dtype=torch.float32)
        expert_ids = torch.randint(0, n, (t, k), dtype=torch.int64)
        route_w = torch.softmax(torch.randn(t, k, dtype=torch.float32), dim=-1)

        with self.assertRaises(ValueError):
            _ = moe_ffn_chunked_fast_gather_einsum(
                x_flat=x,
                bank_u=U,
                bank_v=V,
                group_id=group_id,
                coeffs=coeffs,
                expert_ids=expert_ids,
                route_w=route_w,
                null_expert_id=n,
            )

    def test_moe_chunked_matches_naive_with_null_and_step_delta(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum, resolve_effective_coeffs

        t, d, n, r, k, g, s = 6, 10, 16, 4, 3, 6, 4
        x = torch.randn(t, d, dtype=torch.float32)
        U = torch.randn(n, d, r, dtype=torch.float32)
        V = torch.randn(n, r, d, dtype=torch.float32)
        group_id = torch.randint(0, g, (n,), dtype=torch.int64)

        coeffs = torch.randn(s, g, 5, dtype=torch.float32)
        rat_delta = torch.randn(g, 5, dtype=torch.float32)
        step_rat_delta = torch.randn(s, g, 5, dtype=torch.float32)
        step_idx = 1

        expert_ids = torch.randint(0, n, (t, k), dtype=torch.int64)
        null_expert_id = n - 1
        # force some null selections
        expert_ids[0, 0] = null_expert_id
        expert_ids[3, 2] = null_expert_id

        route_w = torch.softmax(torch.randn(t, k, dtype=torch.float32), dim=-1)

        out_fast = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x,
            bank_u=U,
            bank_v=V,
            group_id=group_id,
            coeffs=coeffs,
            expert_ids=expert_ids,
            route_w=route_w,
            moe_chunk_size=2,
            unique_compression_min_k_tokens=4,
            null_expert_id=null_expert_id,
            step_idx=step_idx,
            rat_delta=rat_delta,
            step_rat_delta=step_rat_delta,
            eps=1e-5,
        )

        coeff_eff = resolve_effective_coeffs(
            coeffs=coeffs,
            step_idx=step_idx,
            rat_delta=rat_delta,
            step_rat_delta=step_rat_delta,
        )
        out_ref = self._naive_moe(
            x=x,
            U=U,
            V=V,
            group_id=group_id,
            coeffs_eff=coeff_eff,
            expert_ids=expert_ids,
            route_w=route_w,
            null_expert_id=null_expert_id,
            eps=1e-5,
        )

        self.assertTrue(torch.allclose(out_fast, out_ref, atol=1e-5, rtol=1e-4))


    def test_maybe_compile_telemetry_attached(self):
        from peer_grkan_kernels import maybe_compile, get_compile_telemetry

        def f(x):
            return x + 1

        c = maybe_compile(f)
        telem = get_compile_telemetry(c)
        self.assertIsNotNone(telem)
        self.assertTrue(hasattr(telem, "used_compiled"))

    def test_explain_graph_breaks_returns_dict(self):
        from peer_grkan_kernels import explain_graph_breaks

        def f(x):
            return x * 2

        out = explain_graph_breaks(f, torch.randn(2, 3))
        self.assertIn("available", out)
        self.assertIn("ok", out)

    def test_stress_high_null_and_signed_routes(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum, resolve_effective_coeffs

        t, d, n, r, k, g, s = 32, 32, 64, 4, 4, 8, 3
        null_expert_id = n - 1

        for seed in (1, 7, 17):
            torch.manual_seed(seed)
            x = torch.randn(t, d, dtype=torch.float32)
            U = torch.randn(n, d, r, dtype=torch.float32)
            V = torch.randn(n, r, d, dtype=torch.float32)
            group_id = torch.randint(0, g, (n,), dtype=torch.int64)
            coeffs = torch.randn(s, g, 5, dtype=torch.float32)
            rat_delta = torch.randn(g, 5, dtype=torch.float32)
            step_rat_delta = torch.randn(s, g, 5, dtype=torch.float32)
            step_idx = seed % s

            expert_ids = torch.randint(0, n, (t, k), dtype=torch.int64)
            # high-null routing: 70% slots go to null expert
            null_mask = torch.rand(t, k) < 0.7
            expert_ids = torch.where(null_mask, torch.full_like(expert_ids, null_expert_id), expert_ids)

            # signed routes to emulate sign-head behavior
            route_w = torch.tanh(torch.randn(t, k, dtype=torch.float32))

            out = moe_ffn_chunked_fast_gather_einsum(
                x_flat=x,
                bank_u=U,
                bank_v=V,
                group_id=group_id,
                coeffs=coeffs,
                expert_ids=expert_ids,
                route_w=route_w,
                null_expert_id=null_expert_id,
                step_idx=step_idx,
                rat_delta=rat_delta,
                step_rat_delta=step_rat_delta,
                moe_chunk_size=8,
                unique_compression_min_k_tokens=0,
            )
            self.assertTrue(torch.isfinite(out).all())

            # compare against naive for one seed to keep test fast
            if seed == 7:
                coeff_eff = resolve_effective_coeffs(
                    coeffs=coeffs,
                    step_idx=step_idx,
                    rat_delta=rat_delta,
                    step_rat_delta=step_rat_delta,
                )
                ref = self._naive_moe(
                    x=x,
                    U=U,
                    V=V,
                    group_id=group_id,
                    coeffs_eff=coeff_eff,
                    expert_ids=expert_ids,
                    route_w=route_w,
                    null_expert_id=null_expert_id,
                )
                self.assertTrue(torch.allclose(out, ref, atol=1e-4, rtol=1e-3))



    def test_triton_fused_backend_matches_torch_forward(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum

        t, d, n, r, k, g = 5, 16, 25, 4, 3, 6
        x = torch.randn(t, d, dtype=torch.float32)
        U = torch.randn(n, d, r, dtype=torch.float32)
        V = torch.randn(n, r, d, dtype=torch.float32)
        group_id = torch.randint(0, g, (n,), dtype=torch.int64)
        coeffs = torch.randn(g, 5, dtype=torch.float32)
        expert_ids = torch.randint(0, n, (t, k), dtype=torch.int64)
        route_w = torch.tanh(torch.randn(t, k, dtype=torch.float32))

        y_torch = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x, bank_u=U, bank_v=V, group_id=group_id,
            coeffs=coeffs, expert_ids=expert_ids, route_w=route_w,
            dispatch_backend="torch",
        )
        y_fused = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x, bank_u=U, bank_v=V, group_id=group_id,
            coeffs=coeffs, expert_ids=expert_ids, route_w=route_w,
            dispatch_backend="triton_fused",
        )
        self.assertTrue(torch.allclose(y_torch, y_fused, atol=1e-5, rtol=1e-4))

    def test_moe_input_validation_raises_for_bad_backend(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum

        t, d, n, r, k, g = 2, 8, 9, 4, 2, 3
        x = torch.randn(t, d, dtype=torch.float32)
        U = torch.randn(n, d, r, dtype=torch.float32)
        V = torch.randn(n, r, d, dtype=torch.float32)
        group_id = torch.randint(0, g, (n,), dtype=torch.int64)
        coeffs = torch.randn(g, 5, dtype=torch.float32)
        expert_ids = torch.randint(0, n, (t, k), dtype=torch.int64)
        route_w = torch.softmax(torch.randn(t, k, dtype=torch.float32), dim=-1)

        with self.assertRaises(ValueError):
            _ = moe_ffn_chunked_fast_gather_einsum(
                x_flat=x,
                bank_u=U,
                bank_v=V,
                group_id=group_id,
                coeffs=coeffs,
                expert_ids=expert_ids,
                route_w=route_w,
                dispatch_backend="invalid",
            )

    def test_dispatch_telemetry_attached(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum, get_dispatch_telemetry

        t, d, n, r, k, g = 3, 8, 9, 4, 2, 3
        x = torch.randn(t, d, dtype=torch.float32)
        U = torch.randn(n, d, r, dtype=torch.float32)
        V = torch.randn(n, r, d, dtype=torch.float32)
        group_id = torch.randint(0, g, (n,), dtype=torch.int64)
        coeffs = torch.randn(g, 5, dtype=torch.float32)
        expert_ids = torch.randint(0, n, (t, k), dtype=torch.int64)
        route_w = torch.softmax(torch.randn(t, k, dtype=torch.float32), dim=-1)

        _ = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x,
            bank_u=U,
            bank_v=V,
            group_id=group_id,
            coeffs=coeffs,
            expert_ids=expert_ids,
            route_w=route_w,
            dispatch_backend="auto",
        )
        telem = get_dispatch_telemetry(moe_ffn_chunked_fast_gather_einsum)
        self.assertIsNotNone(telem)
        self.assertTrue(hasattr(telem, "backend_used"))


    def test_gradients_preserved_when_triton_requested(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum, get_dispatch_telemetry

        t, d, n, r, k, g = 4, 8, 16, 4, 2, 5
        x = torch.randn(t, d, dtype=torch.float32, requires_grad=True)
        U = torch.randn(n, d, r, dtype=torch.float32, requires_grad=True)
        V = torch.randn(n, r, d, dtype=torch.float32, requires_grad=True)
        group_id = torch.randint(0, g, (n,), dtype=torch.int64)
        coeffs = torch.randn(g, 5, dtype=torch.float32, requires_grad=True)
        expert_ids = torch.randint(0, n, (t, k), dtype=torch.int64)
        route_w = torch.softmax(torch.randn(t, k, dtype=torch.float32), dim=-1)

        out = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x,
            bank_u=U,
            bank_v=V,
            group_id=group_id,
            coeffs=coeffs,
            expert_ids=expert_ids,
            route_w=route_w,
            dispatch_backend="triton",
        )
        loss = out.square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(U.grad)
        self.assertIsNotNone(V.grad)
        self.assertIsNotNone(coeffs.grad)

        telem = get_dispatch_telemetry(moe_ffn_chunked_fast_gather_einsum)
        self.assertIsNotNone(telem)
        self.assertIn(telem.backend_used, ("torch", "triton"))


    def test_triton_requested_matches_torch_gradients(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum

        t, d, n, r, k, g = 4, 8, 16, 4, 2, 5
        x1 = torch.randn(t, d, dtype=torch.float32, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        U1 = torch.randn(n, d, r, dtype=torch.float32, requires_grad=True)
        U2 = U1.detach().clone().requires_grad_(True)
        V1 = torch.randn(n, r, d, dtype=torch.float32, requires_grad=True)
        V2 = V1.detach().clone().requires_grad_(True)
        group_id = torch.randint(0, g, (n,), dtype=torch.int64)
        c1 = torch.randn(g, 5, dtype=torch.float32, requires_grad=True)
        c2 = c1.detach().clone().requires_grad_(True)
        expert_ids = torch.randint(0, n, (t, k), dtype=torch.int64)
        route_w = torch.tanh(torch.randn(t, k, dtype=torch.float32))

        y_torch = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x1, bank_u=U1, bank_v=V1, group_id=group_id,
            coeffs=c1, expert_ids=expert_ids, route_w=route_w,
            dispatch_backend="torch",
        )
        y_triton_req = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x2, bank_u=U2, bank_v=V2, group_id=group_id,
            coeffs=c2, expert_ids=expert_ids, route_w=route_w,
            dispatch_backend="triton",
        )

        l1 = y_torch.square().mean(); l2 = y_triton_req.square().mean()
        l1.backward(); l2.backward()

        self.assertTrue(torch.allclose(y_torch, y_triton_req, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(U1.grad, U2.grad, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(V1.grad, V2.grad, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(c1.grad, c2.grad, atol=1e-5, rtol=1e-4))


@unittest.skipIf(torch is None or (not torch.cuda.is_available()), "cuda is not available")
class PeerGRKANCudaTritonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_triton_backend_reports_usage_or_clean_fallback(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum, get_dispatch_telemetry

        t, d, n, r, k, g = 8, 32, 64, 4, 4, 8
        dev = torch.device("cuda")
        x = torch.randn(t, d, device=dev, dtype=torch.float16)
        U = torch.randn(n, d, r, device=dev, dtype=torch.float16)
        V = torch.randn(n, r, d, device=dev, dtype=torch.float16)
        group_id = torch.randint(0, g, (n,), device=dev, dtype=torch.int64)
        coeffs = torch.randn(g, 5, device=dev, dtype=torch.float32)
        expert_ids = torch.randint(0, n, (t, k), device=dev, dtype=torch.int64)
        route_w = torch.tanh(torch.randn(t, k, device=dev, dtype=torch.float32))

        _ = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x, bank_u=U, bank_v=V, group_id=group_id, coeffs=coeffs,
            expert_ids=expert_ids, route_w=route_w, dispatch_backend="triton"
        )
        telem = get_dispatch_telemetry(moe_ffn_chunked_fast_gather_einsum)
        self.assertIsNotNone(telem)
        self.assertIn(telem.backend_used, ("torch", "triton"))
        self.assertTrue(isinstance(telem.reason, str) and len(telem.reason) > 0)


    def test_cuda_triton_fused_backend_runs_or_falls_back_cleanly(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum, get_dispatch_telemetry

        t, d, n, r, k, g = 6, 24, 36, 4, 3, 7
        dev = torch.device("cuda")
        x = torch.randn(t, d, device=dev, dtype=torch.float16)
        U = torch.randn(n, d, r, device=dev, dtype=torch.float16)
        V = torch.randn(n, r, d, device=dev, dtype=torch.float16)
        group_id = torch.randint(0, g, (n,), device=dev, dtype=torch.int64)
        coeffs = torch.randn(g, 5, device=dev, dtype=torch.float32)
        expert_ids = torch.randint(0, n, (t, k), device=dev, dtype=torch.int64)
        route_w = torch.tanh(torch.randn(t, k, device=dev, dtype=torch.float32))

        y = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x, bank_u=U, bank_v=V, group_id=group_id,
            coeffs=coeffs, expert_ids=expert_ids, route_w=route_w,
            dispatch_backend="triton_fused", moe_chunk_size=3,
        )
        self.assertTrue(torch.isfinite(y).all())
        telem = get_dispatch_telemetry(moe_ffn_chunked_fast_gather_einsum)
        self.assertIn(telem.backend_used, ("torch", "triton"))

    def test_cuda_triton_and_torch_backward_parity_small(self):
        from peer_grkan_kernels import moe_ffn_chunked_fast_gather_einsum

        t, d, n, r, k, g = 6, 24, 49, 4, 3, 7
        dev = torch.device("cuda")
        x1 = torch.randn(t, d, device=dev, dtype=torch.float16).float().requires_grad_(True)
        x2 = x1.detach().clone().requires_grad_(True)
        U1 = torch.randn(n, d, r, device=dev, dtype=torch.float16).float().requires_grad_(True)
        U2 = U1.detach().clone().requires_grad_(True)
        V1 = torch.randn(n, r, d, device=dev, dtype=torch.float16).float().requires_grad_(True)
        V2 = V1.detach().clone().requires_grad_(True)
        group_id = torch.randint(0, g, (n,), device=dev, dtype=torch.int64)
        c1 = torch.randn(g, 5, device=dev, dtype=torch.float32, requires_grad=True)
        c2 = c1.detach().clone().requires_grad_(True)
        expert_ids = torch.randint(0, n, (t, k), device=dev, dtype=torch.int64)
        route_w = torch.tanh(torch.randn(t, k, device=dev, dtype=torch.float32))

        y_t = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x1, bank_u=U1, bank_v=V1, group_id=group_id, coeffs=c1,
            expert_ids=expert_ids, route_w=route_w, dispatch_backend="torch", moe_chunk_size=3,
        )
        y_k = moe_ffn_chunked_fast_gather_einsum(
            x_flat=x2, bank_u=U2, bank_v=V2, group_id=group_id, coeffs=c2,
            expert_ids=expert_ids, route_w=route_w, dispatch_backend="triton", moe_chunk_size=3,
        )

        l1 = y_t.square().mean(); l2 = y_k.square().mean()
        l1.backward(); l2.backward()

        self.assertTrue(torch.allclose(y_t, y_k, atol=2e-4, rtol=2e-3))
        self.assertTrue(torch.allclose(x1.grad, x2.grad, atol=2e-4, rtol=2e-3))
        self.assertTrue(torch.allclose(U1.grad, U2.grad, atol=2e-4, rtol=2e-3))
        self.assertTrue(torch.allclose(V1.grad, V2.grad, atol=2e-4, rtol=2e-3))
        self.assertTrue(torch.allclose(c1.grad, c2.grad, atol=2e-4, rtol=2e-3))


@unittest.skipIf(torch is None, "torch is not installed")
class PeerGRKANProfileHelpersTests(unittest.TestCase):
    def test_profile_roundtrip_helpers(self):
        import json
        import tempfile
        from pathlib import Path
        from peer_grkan_kernels import load_autotune_profile, apply_autotune_profile

        payload = {
            "dispatch_backend": "triton",
            "best": {"chunk": 768, "threshold": 1536, "latency_s": 0.01},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "profile.json"
            p.write_text(json.dumps(payload))
            loaded = load_autotune_profile(str(p))
            cfg = apply_autotune_profile(loaded)
            self.assertEqual(cfg["moe_chunk_size"], 768)
            self.assertEqual(cfg["unique_compression_min_k_tokens"], 1536)
            self.assertEqual(cfg["dispatch_backend"], "triton")


if __name__ == "__main__":
    unittest.main()
