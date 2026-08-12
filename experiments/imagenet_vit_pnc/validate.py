"""Correctness gates that must pass before any preflight measurement is believed.

  1. cached_prefix_parity   prefix+tail logits == full model logits        (< 1e-4)
  2. cls_only_parity        CLS-only tail == full-token tail at CLS        (< 1e-5)
  3. solver_parity          ridge_solve_from_stats == ridge_solve         (bitwise)
  4. cho_parity            shared-Cholesky solve == ridge_solve           (< 1e-8 rel)
  5. mutation_restore       in-place W1 mutation then restore == pristine (bitwise)
  6. token_sampling         deterministic and CLS always present
  7. basis_parity           basis/coeff/scale == Banking77 construct.py   (bitwise)

Gate 7 needs jax (Banking77 imports it), so it is skipped in the torch-only venv and
run separately with the main .venv:
    JAX_PLATFORMS=cpu .venv/bin/python -m experiments.imagenet_vit_pnc.validate --only basis_parity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "scripts"))

from pnc_theory.linalg import ridge_solve  # noqa: E402

RESULTS = []


def gate(name, ok, detail):
    RESULTS.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


# ---------------------------------------------------------------- torch gates
def torch_gates(device="cuda", batch=4, seed=0):
    import torch

    from experiments.imagenet_vit_pnc import pnc_core as pc
    from experiments.imagenet_vit_pnc.vit_adapter import ViTPnCAdapter

    ad = ViTPnCAdapter(device=device)
    torch.manual_seed(seed)
    imgs = torch.randn(batch, 3, 224, 224, device=device)

    with torch.inference_mode():
        ref = ad.model(imgs)
        x_resid = ad.prefix(imgs)
        full_tok = ad.tail(x_resid, cls_only=False)
        cls_only = ad.tail(x_resid, cls_only=True)

    d1 = float((full_tok - ref).abs().max())
    gate("cached_prefix_parity", d1 < 1e-4, f"max|tail - model| = {d1:.3e} (tol 1e-4)")
    d2 = float((cls_only - full_tok).abs().max())
    gate("cls_only_parity", d2 < 1e-5, f"max|cls_only - full_token| = {d2:.3e} (tol 1e-5)")

    # tokens: 1 CLS + 14*14 patches
    n_tok = x_resid.shape[1]
    gate("token_count", n_tok == 197, f"T = {n_tok} (expected 197 = 1 + 14x14)")

    # mutation / restoration
    U = pc.perturbation_basis(seed, K=2)
    dW1 = pc.member_dW1(U, np.array([1.0, 0.0], np.float32), 1.0)
    W1v = ad.W1 + torch.as_tensor(dW1, device=ad.device)
    ad.set_W1_code(W1v)
    with torch.inference_mode():
        moved = ad.model(imgs)
    ad.restore()
    with torch.inference_mode():
        back = ad.model(imgs)
    ok = ad.weights_are_pristine() and torch.equal(back, ref)
    gate("mutation_restore", ok,
         f"pristine={ad.weights_are_pristine()} logits_bitwise_equal={torch.equal(back, ref)}, "
         f"perturbation moved logits by {float((moved - ref).abs().max()):.3e}")

    # streaming stats == dense design
    torch.manual_seed(1)
    y = torch.randn(64, 8, device=device)
    tgt = torch.randn(64, 3, device=device)
    Xa = pc.SufficientStats.augment(y)
    st = pc.SufficientStats(p=9, d=3, device=device).update(Xa, tgt)
    G, C = st.to_cpu_f64()
    Gd = (Xa.T @ Xa).double().cpu().numpy()
    gate("stats_accumulation", np.abs(G - Gd).max() < 1e-9,
         f"max|G_stream - X^T X| = {np.abs(G - Gd).max():.3e}")
    return ad


# --------------------------------------------------------------- numpy gates
def solver_gates(seed=0):
    from experiments.imagenet_vit_pnc import pnc_core as pc

    rng = np.random.RandomState(seed)
    n, p, d, lam = 400, 33, 7, 1e-3
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, d))
    prior = rng.normal(size=(p, d))

    ref = ridge_solve(X, Y, lam, w_prior=prior)
    G, C = X.T @ X, X.T @ Y
    alt = pc.ridge_solve_from_stats(G, C, lam, w_prior=prior)
    gate("solver_parity", np.array_equal(ref, alt),
         f"bitwise equal = {np.array_equal(ref, alt)}, max|diff| = {np.abs(ref - alt).max():.3e}")

    cho, tf, ts = pc.cho_solve_shared(G, C, lam, w_prior=prior)
    rel = np.abs(cho - ref).max() / (np.abs(ref).max() + 1e-30)
    gate("cho_parity", rel < 1e-8,
         f"rel max diff = {rel:.3e}, one factorisation ({tf*1e3:.2f} ms) for all {d} outputs "
         f"({ts*1e3:.2f} ms)")

    # residuals must be recoverable from (G, C, ||Y||^2) alone -- no design matrix
    direct = np.linalg.norm(X @ ref - Y) / np.linalg.norm(Y)
    from_stats = pc.relative_residual(
        {"G": G, "C": C, "yty": float((Y ** 2).sum())}, ref)
    gate("residual_from_stats", abs(direct - from_stats) < 1e-9,
         f"direct {direct:.10f} vs from-stats {from_stats:.10f} "
         f"(diff {abs(direct - from_stats):.2e})")

    idx1 = pc.token_index_matrix("cls+4", [7, 8, 9], 197, seed=3)
    idx2 = pc.token_index_matrix("cls+4", [7, 8, 9], 197, seed=3)
    idx3 = pc.token_index_matrix("cls+4", [7, 8, 9], 197, seed=4)
    ok = (np.array_equal(idx1, idx2) and not np.array_equal(idx1, idx3)
          and (idx1[:, 0] == 0).all() and (idx1[:, 1:] > 0).all()
          and all(len(set(r.tolist())) == len(r) for r in idx1))
    gate("token_sampling", ok,
         f"deterministic={np.array_equal(idx1, idx2)} seed-sensitive={not np.array_equal(idx1, idx3)} "
         f"CLS-always={bool((idx1[:, 0] == 0).all())} no-dupes=True")


def basis_parity_gate(seed=0, K=20, M=5):
    """Gate the reused conventions against Banking77 construct.py (needs jax)."""
    from experiments.imagenet_vit_pnc import pnc_core as pc
    try:
        from experiments.banking77_pnc import construct as b77
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] basis_parity: Banking77 construct unavailable ({type(exc).__name__}: "
              f"{str(exc)[:80]})")
        return

    u_mine, u_ref = pc.perturbation_basis(seed, K), b77.perturbation_basis(seed, K)
    c_mine, c_ref = pc.member_coefficients(seed, M, K), b77.member_coefficients(seed, M, K)
    s_mine = pc.base_scale(u_mine, c_mine, 12.5)
    s_ref = b77.base_scale(u_ref, c_ref, 12.5)
    ok = (np.array_equal(u_mine, u_ref) and np.array_equal(c_mine, c_ref) and s_mine == s_ref)
    gate("basis_parity", ok,
         f"basis={np.array_equal(u_mine, u_ref)} coeffs={np.array_equal(c_mine, c_ref)} "
         f"scale={s_mine == s_ref} (D_flat={pc.D_FLAT} == {b77.D_FLAT})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.only == "basis_parity":
        basis_parity_gate()
    else:
        solver_gates()
        basis_parity_gate()
        torch_gates(device=args.device)

    n_fail = sum(1 for r in RESULTS if not r["pass"])
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} gates passed")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(RESULTS, indent=2))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
