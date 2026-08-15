"""Driver for memory-bounded categorical SCOD: validation gates, q and N studies, scoring.

Stages, in the order the spec requires them:

  validate  §10  synthetic exact-vs-MC Fisher gate, analytic-vs-autograd gradient gate,
                 and the ViT-head analytic gate
  qsweep    §14  MC-label convergence on ID data only
  nsweep    §15  calibration-size convergence on ID data only
  fit       §11-13  fit the chosen scopes and record memory/rank/fallbacks
  score     §16  evaluate on the six evaluation sets and on ID misclassification
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from . import full_cache as fc
from . import fu_common as F
from . import scod_ll as S
from .memprobe import GIB, cpu_peak_rss_gib, reset_cuda
from .vit_adapter import ViTPnCAdapter


# ------------------------------------------------------------------ helpers
def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between the column spaces of A and B."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def id_features(ad, n: int):
    corr = fc.load_cache(F.SRC / "raw" / "cache_correction.npz")
    return torch.as_tensor(corr["x_resid_cls"][:n], device=ad.device, dtype=ad.dtype)


# ================================================================ §10 gates
def gate_synthetic(seed: int = 0) -> dict:
    """Exact categorical Fisher vs MC, on a model small enough to enumerate."""
    torch.manual_seed(seed)
    d_in, k, n = 8, 6, 256
    W = torch.randn(k, d_in, dtype=torch.float64)
    b = torch.randn(k, dtype=torch.float64)
    Xtr = torch.randn(n, d_in, dtype=torch.float64)
    Xte = torch.randn(64, d_in, dtype=torch.float64)
    npar = k * d_in + k

    def jac(x):                      # (k, npar): d logits / d params
        J = torch.zeros(k, npar, dtype=torch.float64)
        for c in range(k):
            J[c, c * d_in:(c + 1) * d_in] = x
            J[c, k * d_in + c] = 1.0
        return J

    def exact_F(X):
        A = torch.zeros(npar, npar, dtype=torch.float64)
        for x in X:
            p = torch.softmax(W @ x + b, -1)
            Fi = torch.diag(p) - torch.outer(p, p)
            J = jac(x)
            A += J.T @ Fi @ J
        return A / len(X)

    def mc_F(X, q, s):
        g = torch.Generator().manual_seed(s)
        A = torch.zeros(npar, npar, dtype=torch.float64)
        for x in X:
            p = torch.softmax(W @ x + b, -1)
            ys = torch.multinomial(p.float(), q, replacement=True, generator=g)
            J = jac(x)
            for y in ys:
                e = -p.clone()
                e[y] += 1.0
                gv = J.T @ e
                A += torch.outer(gv, gv) / q
        return A / len(X)

    Fex = exact_F(Xtr)
    de, Ve = torch.linalg.eigh(Fex)
    r = 5
    rows = []
    for q in (1, 2, 4, 8, 32):
        errs, angs, sps = [], [], []
        for s in range(3):
            Fmc = mc_F(Xtr, q, 1000 + s)
            errs.append(float(torch.norm(Fmc - Fex) / torch.norm(Fex)))
            dm, Vm = torch.linalg.eigh(Fmc)
            angs.append(float(principal_angles(Ve[:, -r:].numpy(),
                                               Vm[:, -r:].numpy()).max()))
            # SCOD posterior_pred score on held-out points, exact eigendecomposition
            def sc(dv, Vv):
                sca = torch.sqrt(torch.clamp(dv[-r:], min=0)
                                 / (torch.clamp(dv[-r:], min=0) + 1 / (2 * S.MEPS)))
                o = []
                for x in Xte:
                    p = torch.softmax(W @ x + b, -1)
                    L = jac(x).T @ torch.linalg.cholesky(
                        torch.diag(p) - torch.outer(p, p)
                        + 1e-12 * torch.eye(k, dtype=torch.float64))
                    pr = sca[:, None] * (Vv[:, -r:].T @ L)
                    o.append(float(torch.sqrt(torch.clamp(
                        (L ** 2).sum() - (pr ** 2).sum(), min=0))))
                return np.array(o)
            sps.append(float(spearmanr(sc(de, Ve), sc(dm, Vm)).statistic))
        rows.append({"q": q, "frob_rel_error_mean": float(np.mean(errs)),
                     "max_principal_angle_deg_mean": float(np.mean(angs)),
                     "score_spearman_vs_exact_mean": float(np.mean(sps)),
                     "score_spearman_min": float(np.min(sps))})
        print(f"    q={q:<3} Frob rel err {rows[-1]['frob_rel_error_mean']:.4f}   "
              f"max angle {rows[-1]['max_principal_angle_deg_mean']:6.2f} deg   "
              f"score Spearman {rows[-1]['score_spearman_vs_exact_mean']:.4f}")
    return {"n_params": npar, "n_classes": k, "n_train": n, "leading_r": r,
            "exact_top_eigs": de[-r:].tolist(), "rows": rows}


def gate_autograd(ad, scope: str, n: int = 4, q: int = 2) -> dict:
    """Analytic grad_w log p(y|x) vs autograd, on the real ViT tail."""
    sg = S.ScopeGrad(ad, scope)
    X = id_features(ad, n)
    fw = sg.forward(X)
    gen = torch.Generator(device=ad.device).manual_seed(3)
    ys = S.sample_labels(fw["logp"], q, gen)
    out = torch.zeros(sg.n_params, n * q, device=ad.device)
    sg.grad_columns(fw, ys, out)

    head = sg.head
    # ad.W1 / ad.W2 are transposed *views* of the Linear weights, so autograd has to
    # target the leaves and the resulting gradients are transposed back into code layout.
    leaves = ([(head.weight, False), (head.bias, False)] if scope == "linear"
              else [(ad.mlp1.weight, True), (ad.mlp1.bias, False),
                    (ad.mlp2.weight, True), (ad.mlp2.bias, False),
                    (head.weight, False), (head.bias, False)])
    params = [p for p, _ in leaves]
    prev = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    worst = 0.0
    for i in range(n):
        for j in range(q):
            x = X[i:i + 1]
            h = ad.block.ln_2(x[:, None, :])[:, 0]
            y = torch.nn.functional.gelu(h @ ad.W1 + ad.b1)
            z = y @ ad.W2 + ad.b2
            phi = ad.enc.ln((x + z)[:, None, :])[:, 0]
            lp = torch.log_softmax(head(phi), -1)[0, ys[i, j]]
            gs = torch.autograd.grad(lp, params, retain_graph=False)
            # analytic layout: W1, b1, W2, b2, head_W, head_b (head only for `linear`)
            ref = torch.cat([(g.T if t else g).reshape(-1)
                             for g, (_, t) in zip(gs, leaves)])
            got = out[:, i * q + j]
            worst = max(worst, float((ref - got).norm() / (ref.norm() + 1e-30)))
    for p, was in zip(params, prev):
        p.requires_grad_(was)
    del out
    reset_cuda()
    return {"scope": scope, "n_checked": n * q, "max_rel_error": worst,
            "ok": worst < 1e-4}


@torch.no_grad()
def gate_vit_head(ad, n: int = 512, qs=(1, 2, 4, 8, 32)) -> dict:
    """||L||_F^2 for the head has the closed form (||phi||^2+1)(1-||p||^2).

    This is an independent analytic Fisher quantity — no sketch, no autograd — so it
    isolates the MC estimator on the real 1000-class ViT head.
    """
    sg = S.ScopeGrad(ad, "linear")
    X = id_features(ad, n)
    fw = sg.forward(X)
    p = torch.softmax(fw["logits"], -1).double()
    exact = ((fw["phi"].double() ** 2).sum(-1) + 1.0) * (1.0 - (p ** 2).sum(-1))
    rows = []
    for q in qs:
        est = []
        for s in range(3):
            gen = torch.Generator(device=ad.device).manual_seed(500 + s)
            ys = S.sample_labels(fw["logp"], q, gen)
            acc = torch.zeros(n, dtype=torch.float64, device=ad.device)
            for j in range(q):
                sv = -p.clone()
                sv[torch.arange(n, device=p.device), ys[:, j]] += 1.0
                acc += (sv ** 2).sum(-1) * ((fw["phi"].double() ** 2).sum(-1) + 1.0) / q
            est.append(acc)
        E = torch.stack(est)
        rows.append({"q": q,
                     "mean_rel_error": float(((E.mean(0) - exact).abs()
                                              / exact).mean()),
                     "pooled_rel_error": float((E.mean() - exact.mean()).abs()
                                               / exact.mean()),
                     "spearman_vs_exact": float(spearmanr(
                         exact.cpu().numpy(), E.mean(0).cpu().numpy()).statistic)})
        print(f"    q={q:<3} pooled rel err {rows[-1]['pooled_rel_error']:.5f}   "
              f"per-example rel err {rows[-1]['mean_rel_error']:.4f}   "
              f"Spearman {rows[-1]['spearman_vs_exact']:.4f}")
    return {"n": n, "identity": "||L||_F^2 = (||phi||^2 + 1)(1 - ||p||^2)", "rows": rows}


def stage_validate(args):
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    out = {}
    print("\n=== Gate 1: synthetic exact vs MC categorical Fisher ===")
    out["synthetic"] = gate_synthetic()
    print("\n=== Gate 2: analytic vs autograd per-example gradients (real ViT tail) ===")
    out["autograd"] = {}
    for scope in S.SCOPES:
        g = gate_autograd(ad, scope)
        out["autograd"][scope] = g
        print(f"    {scope:<8} max rel error {g['max_rel_error']:.3e} -> "
              f"{'OK' if g['ok'] else 'FAIL'}")
    print("\n=== Gate 3: ViT 1000-class head, analytic ||L||_F^2 vs MC ===")
    out["vit_head"] = gate_vit_head(ad)
    ok = all(v["ok"] for v in out["autograd"].values())
    out["all_gates_ok"] = bool(ok)
    F.write_json(F.OUT / "metrics" / "scod_validation.json", out)
    print(f"\nwrote {F.OUT/'metrics'/'scod_validation.json'}  (gradient gates "
          f"{'OK' if ok else 'FAILED'})")
    if not ok:
        raise SystemExit("gradient gate failed; not proceeding to the large sketch")


# ================================================================ §14 q sweep
def stage_qsweep(args):
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    n_cal, n_eval = 4096, 4096
    Xc = id_features(ad, n_cal)
    Xe = id_features(ad, 32768)[-n_eval:]
    res = {"scope": "linear", "n_cal": n_cal, "n_eval": n_eval, "seeds": [0, 1, 2],
           "k": S.K_EIGS, "T": S.T_SKETCH, "runs": []}
    store = {}
    for q in (1, 2, 4, 8):
        for s in (0, 1, 2):
            f = S.fit_scod(ad, "linear", Xc, n_cal, S.K_EIGS, S.T_SKETCH, q,
                           seed=S.SKETCH_SEED + 13 * s, log_every=0)
            sc = S.score_scod(ad, "linear", S.Projector(f["eigs"], f["basis"]), Xe, q,
                              n_eigs=S.K_EIGS, seed=S.SKETCH_SEED + 991 + s)
            store[(q, s)] = {"eigs": f["eigs"].numpy(), "basis": f["basis"].numpy(),
                             "score": sc}
            print(f"  q={q} seed={s}: fit {f['fit_seconds']:.0f}s  "
                  f"score mean {sc.mean():.4g}", flush=True)
    ref = store[(8, 0)]
    for q in (1, 2, 4, 8):
        sp = [float(spearmanr(store[(q, s)]["score"], ref["score"]).statistic)
              for s in (0, 1, 2)]
        ang = [float(principal_angles(ref["basis"][:, -S.K_EIGS:],
                                      store[(q, s)]["basis"][:, -S.K_EIGS:]).max())
               for s in (0, 1, 2)]
        cv = [float(store[(q, s)]["score"].std() / store[(q, s)]["score"].mean())
              for s in (0, 1, 2)]
        across = np.stack([store[(q, s)]["score"] for s in (0, 1, 2)])
        res["runs"].append({
            "q": q, "spearman_vs_q8_mean": float(np.mean(sp)),
            "spearman_vs_q8_min": float(np.min(sp)),
            "max_principal_angle_deg": float(np.mean(ang)),
            "score_cv": float(np.mean(cv)),
            "across_seed_score_cv": float(np.mean(across.std(0) / across.mean(0))),
            "top5_eigs": store[(q, 0)]["eigs"][-5:].tolist()})
        r = res["runs"][-1]
        print(f"  q={q}: Spearman vs q=8 {r['spearman_vs_q8_mean']:.4f} "
              f"(min {r['spearman_vs_q8_min']:.4f})  angle "
              f"{r['max_principal_angle_deg']:.2f} deg  across-seed CV "
              f"{r['across_seed_score_cv']:.4f}")
    ok = [r["q"] for r in res["runs"] if r["spearman_vs_q8_mean"] >= 0.99]
    res["chosen_q"] = int(min(ok)) if ok else 8
    res["rule"] = "smallest q with Spearman >= 0.99 vs q=8; ID data only"
    print(f"\n  CHOSEN q = {res['chosen_q']}")
    F.write_json(F.OUT / "metrics" / "scod_qsweep.json", res)


# ================================================================ §15 N sweep
def stage_nsweep(args):
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    q = json.loads((F.OUT / "metrics" / "scod_qsweep.json").read_text())["chosen_q"]
    Xall = id_features(ad, S.N_CAL)
    Xe = Xall[-4096:]
    res = {"scope": "linear", "q": q, "rows": []}
    prev = None
    for n in (4096, 8192, 16384, 32768):
        f = S.fit_scod(ad, "linear", Xall[:n], n, S.K_EIGS, S.T_SKETCH, q, log_every=0)
        sc = S.score_scod(ad, "linear", S.Projector(f["eigs"], f["basis"]), Xe, q,
                          n_eigs=S.K_EIGS)
        row = {"n_cal": n, "fit_seconds": f["fit_seconds"],
               "peak_rss_gib": f["peak_rss_gib"],
               "top5_eigs": f["eigs"][-5:].tolist()}
        if prev is not None:
            row["spearman_vs_previous"] = float(spearmanr(sc, prev["sc"]).statistic)
            row["max_principal_angle_vs_previous_deg"] = float(principal_angles(
                prev["basis"][:, -S.K_EIGS:],
                f["basis"][:, -S.K_EIGS:].numpy()).max())
        res["rows"].append(row)
        print(f"  n={n:>6,}: fit {f['fit_seconds']:.0f}s  "
              f"Spearman vs previous {row.get('spearman_vs_previous', float('nan')):.4f}",
              flush=True)
        prev = {"sc": sc, "basis": f["basis"].numpy()}
    res["selected_n_cal"] = S.N_CAL
    res["rule"] = "largest computationally feasible; ID convergence reported, not used to pick"
    F.write_json(F.OUT / "metrics" / "scod_nsweep.json", res)


# ============================================================ §11-13, §16
def stage_fit(args):
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    qs = json.loads((F.OUT / "metrics" / "scod_qsweep.json").read_text())
    q = qs["chosen_q"]
    scopes = args.scopes.split(",") if args.scopes else S.SCOPES
    Xc = id_features(ad, S.N_CAL)
    caches = {}
    for n in F.SETS:
        c = fc.load_cache(F.cache_path(n))
        caches[n] = fc.cache_to_gpu(c, ad)
    yval = fc.load_cache(F.cache_path("val50k"))["labels"]
    base_pred = np.load(F.SRC / "raw" / "base_val_logits.npy").argmax(-1)

    path = F.OUT / "metrics" / "scod_results.json"
    results = json.loads(path.read_text()) if path.exists() else {"q": q, "scopes": {}}
    for scope in scopes:
        k, T, fb = S.K_EIGS, S.T_SKETCH, None
        need = 2 * S.ScopeGrad(ad, scope).n_params * T * 4 / GIB
        print(f"\n=== SCOD-{scope}: projected sketch+operators {need:.2f} GiB "
              f"(ceiling {S.RAM_CEILING_GIB} GiB) ===")
        if need > S.RAM_CEILING_GIB * 0.75:
            k, T, fb = S.K_FALLBACK, S.T_FALLBACK, "measured projection exceeds guard"
            print(f"  falling back to k={k}, T={T}: {fb}")
        t0 = time.perf_counter()
        f = S.fit_scod(ad, scope, Xc, S.N_CAL, k, T, q)
        proj = S.Projector(f["eigs"], f["basis"])
        n_eigs = min(k, f["eigs"].numel())
        t1 = time.perf_counter()
        sc = {}
        for n in F.SETS:
            sc[n] = S.score_scod(ad, scope, proj, caches[n], q, n_eigs=n_eigs)
            print(f"    scored {n:<12} {len(sc[n]):>7,}", flush=True)
        eval_s = time.perf_counter() - t1
        om = F.ood_metrics(sc["val50k"], {d: sc[d] for d in F.DS})
        err = (base_pred != yval).astype(int)
        from sklearn.metrics import average_precision_score, roc_auc_score
        results["scopes"][f"SCOD-{scope}"] = {
            "scope": scope, "requested_k": S.K_EIGS, "requested_T": S.T_SKETCH,
            "actual_k": k, "actual_T": T, "fallback_reason": fb,
            "k_sketch": f["k_sketch"], "l_sketch": f["l_sketch"],
            "recovered_rank": f["recovered_rank"], "n_eigs_used": n_eigs,
            "q": q, "n_cal": S.N_CAL, "n_params": f["n_params"],
            "fit_seconds": f["fit_seconds"], "basis_seconds": f["basis_seconds"],
            "eval_seconds": eval_s, "total_seconds": time.perf_counter() - t0,
            "peak_rss_gib": max(f["peak_rss_gib"], cpu_peak_rss_gib()),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / GIB,
            "storage_mib": f["storage_mib"],
            "top_eigs": f["eigs"][-10:].tolist(),
            **om,
            "id_error_auroc": float(roc_auc_score(err, sc["val50k"])),
            "id_error_aupr": float(average_precision_score(err, sc["val50k"])),
            "mean_score_correct": float(sc["val50k"][err == 0].mean()),
            "mean_score_incorrect": float(sc["val50k"][err == 1].mean()),
            "OOD data accessed before any choice": "NO"}
        np.savez_compressed(F.OUT / "predictions" / f"scod_{scope}_scores.npz",
                            **{n: sc[n].astype(np.float32) for n in F.SETS})
        r = results["scopes"][f"SCOD-{scope}"]
        print(f"  SCOD-{scope}: Near {r['near_auroc']*100:.2f}  Far {r['far_auroc']*100:.2f}"
              f"  ID-error {r['id_error_auroc']*100:.2f}  peak RSS "
              f"{r['peak_rss_gib']:.1f} GiB  {r['total_seconds']/60:.1f} min")
        F.write_json(path, results)
        del f, proj, sc
        reset_cuda()
    print(f"\nwrote {path}")


# ------------------------------------------------- exact categorical SCOD (§10-§16)
def stage_exactvalidate(args):
    from . import scod_exact as E
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    out = {"note": "gates on the exact categorical factorisation replacing MC Fisher"}
    for scope in (args.scopes.split(",") if args.scopes else ["linear", "ffn"]):
        g = E.validate(ad, scope)
        out[scope] = g
        print(f"  {scope:<8} adjoint {g['adjoint_max_rel_error']:.3e}   "
              f"A-vs-autograd {g['A_column_vs_autograd_max_rel_error']:.3e}   "
              f"norm {g['normsq_max_rel_error']:.3e}  -> "
              f"{'OK' if g['ok'] else 'FAIL'}")
    out["all_ok"] = all(v["ok"] for k, v in out.items() if isinstance(v, dict))
    F.write_json(F.OUT / "metrics" / "scod_exact_validation.json", out)
    print(f"\nwrote {F.OUT/'metrics'/'scod_exact_validation.json'}")
    if not out["all_ok"]:
        raise SystemExit("exact-factorisation gate failed; not fitting")


def stage_exactfit(args):
    from . import scod_exact as E
    from sklearn.metrics import average_precision_score, roc_auc_score
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    Xc = id_features(ad, S.N_CAL)
    caches = {n: fc.cache_to_gpu(fc.load_cache(F.cache_path(n)), ad) for n in F.SETS}
    yval = fc.load_cache(F.cache_path("val50k"))["labels"]
    base_pred = np.load(F.SRC / "raw" / "base_val_logits.npy").argmax(-1)
    err = (base_pred != yval).astype(int)

    path = F.OUT / "metrics" / "scod_results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results.setdefault("fisher", "exact categorical (no MC)")
    results.setdefault("scopes", {})
    for scope in (args.scopes.split(",") if args.scopes else ["linear", "ffn"]):
        k, T, fb = S.K_EIGS, S.T_SKETCH, None
        need = 2 * E.SCOPES[scope](ad).n_params * T * 4 / GIB
        print(f"\n=== SCOD-{scope} (exact): sketch+operators {need:.2f} GiB "
              f"(ceiling {S.RAM_CEILING_GIB} GiB) ===")
        if need > S.RAM_CEILING_GIB * 0.75:
            k, T, fb = S.K_FALLBACK, S.T_FALLBACK, "projection exceeds RAM guard"
            print(f"  falling back to k={k}, T={T}: {fb}")
        t0 = time.perf_counter()
        f = E.fit(ad, scope, Xc, k, T)
        proj = S.Projector(f["eigs"], f["basis"])
        n_eigs = min(k, f["eigs"].numel())
        t1 = time.perf_counter()
        sc = {}
        for n in F.SETS:
            sc[n] = E.score(ad, scope, proj, caches[n], n_eigs)
            print(f"    scored {n:<12} {len(sc[n]):>7,}", flush=True)
        eval_s = time.perf_counter() - t1
        om = F.ood_metrics(sc["val50k"], {d: sc[d] for d in F.DS})
        results["scopes"][f"SCOD-{scope}"] = {
            "scope": scope, "fisher": "exact categorical", "requested_k": S.K_EIGS,
            "requested_T": S.T_SKETCH, "actual_k": k, "actual_T": T,
            "fallback_reason": fb, "k_sketch": f["k_sketch"],
            "l_sketch": f["l_sketch"], "recovered_rank": f["recovered_rank"],
            "n_eigs_used": n_eigs, "q": None, "n_cal": f["n_cal"],
            "n_params": f["n_params"], "fit_seconds": f["fit_seconds"],
            "basis_seconds": f["basis_seconds"], "eval_seconds": eval_s,
            "total_seconds": time.perf_counter() - t0,
            "peak_rss_gib": max(f["peak_rss_gib"], cpu_peak_rss_gib()),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / GIB,
            "storage_mib": f["storage_mib"], "top_eigs": f["eigs"][-10:].tolist(),
            **om,
            "id_error_auroc": float(roc_auc_score(err, sc["val50k"])),
            "id_error_aupr": float(average_precision_score(err, sc["val50k"])),
            "mean_score_correct": float(sc["val50k"][err == 0].mean()),
            "mean_score_incorrect": float(sc["val50k"][err == 1].mean()),
            "OOD data accessed before any choice": "NO"}
        np.savez_compressed(F.OUT / "predictions" / f"scod_{scope}_scores.npz",
                            **{n: sc[n].astype(np.float32) for n in F.SETS})
        r = results["scopes"][f"SCOD-{scope}"]
        print(f"  SCOD-{scope}: Near {r['near_auroc']*100:.2f}  "
              f"Far {r['far_auroc']*100:.2f}  ID-error {r['id_error_auroc']*100:.2f}  "
              f"peak RSS {r['peak_rss_gib']:.1f} GiB  {r['total_seconds']/60:.1f} min")
        F.write_json(path, results)
        del f, proj, sc
        reset_cuda()
    print(f"\nwrote {path}")


def stage_exactnsweep(args):
    """§15 — ID-only calibration-size convergence for the exact estimator."""
    from . import scod_exact as E
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    scope = args.scopes or "linear"
    Xall = id_features(ad, S.N_CAL)
    Xe = Xall[-4096:]
    res = {"scope": scope, "fisher": "exact categorical", "rows": []}
    prev = None
    for n in (4096, 8192, 16384, 32768):
        f = E.fit(ad, scope, Xall[:n], S.K_EIGS, S.T_SKETCH, log_every=0)
        sc = E.score(ad, scope, S.Projector(f["eigs"], f["basis"]), Xe, S.K_EIGS)
        row = {"n_cal": n, "fit_seconds": f["fit_seconds"],
               "peak_rss_gib": f["peak_rss_gib"],
               "top5_eigs": f["eigs"][-5:].tolist()}
        if prev is not None:
            row["spearman_vs_previous"] = float(spearmanr(sc, prev["sc"]).statistic)
            row["max_principal_angle_vs_previous_deg"] = float(principal_angles(
                prev["basis"][:, -S.K_EIGS:], f["basis"][:, -S.K_EIGS:].numpy()).max())
        res["rows"].append(row)
        print(f"  n={n:>6,}: fit {f['fit_seconds']:.0f}s  Spearman vs previous "
              f"{row.get('spearman_vs_previous', float('nan')):.4f}  angle "
              f"{row.get('max_principal_angle_vs_previous_deg', float('nan')):.1f} deg",
              flush=True)
        prev = {"sc": sc, "basis": f["basis"].numpy()}
        del f
        reset_cuda()
    res["selected_n_cal"] = S.N_CAL
    res["rule"] = ("largest computationally feasible, matching the P&C/Mahalanobis pool; "
                   "convergence reported, never used to pick")
    F.write_json(F.OUT / "metrics" / "scod_exact_nsweep.json", res)
    print(f"\nwrote {F.OUT/'metrics'/'scod_exact_nsweep.json'}")


# ------------------------------------------------- §11 SCOD-last-block (image-driven)
def _correction_image_iter(batch: int = 16):
    """Correction-pool images, grouped by parquet row group (order is irrelevant to the
    sketch, which is a sum, and this avoids the row-group thrashing that stalled §20)."""
    from collections import defaultdict
    from .full_data import ImageNetShards, shard_paths
    rows = np.load(F.SRC / "splits" / "correction_rows.npy")
    ds = ImageNetShards(shard_paths("train"))
    groups = defaultdict(list)
    for r in rows:
        s, rg, j = ds._locate(int(r))
        groups[(s, rg)].append(j)

    def it():
        buf = []
        for key in sorted(groups):
            imgs_raw, _ = ds._row_group(*key)
            for img in ds._pool.map(ds._decode,
                                    [imgs_raw[j]["bytes"] for j in groups[key]]):
                buf.append(img)
                if len(buf) == batch:
                    yield torch.stack(buf)
                    buf = []
            del imgs_raw
        if buf:
            yield torch.stack(buf)
    return it, len(rows)


def _eval_image_iter(name: str, batch: int = 16):
    from . import full_ood as fo
    from .full_data import ImageNetShards, shard_paths
    if name == "val50k":
        ds = ImageNetShards(shard_paths("val"))
        idx = np.arange(len(ds))

        def it():
            for imgs, _, _ in ds.iter_batches(idx, batch):
                yield imgs
        return it, len(idx)
    d = fo.OODZip(name)

    def it():
        for imgs, *_ in d.iter_batches(batch):
            yield imgs
    return it, len(d)


def stage_fitlast(args):
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    q = json.loads((F.OUT / "metrics" / "scod_qsweep.json").read_text())["chosen_q"]
    k, T, fb = S.K_EIGS, S.T_SKETCH, None
    need = 2 * S.LastBlockScope(ad).n_params * T * 4 / GIB
    print(f"=== SCOD-last-block: projected sketch+operators {need:.2f} GiB "
          f"(ceiling {S.RAM_CEILING_GIB} GiB) ===")
    if need > S.RAM_CEILING_GIB * 0.75:
        k, T, fb = S.K_FALLBACK, S.T_FALLBACK, "measured projection exceeds guard"
        print(f"  falling back to k={k}, T={T}: {fb}")
    t0 = time.perf_counter()
    it, n = _correction_image_iter()
    f = S.fit_scod_lastblock(ad, it, n, k, T, q)
    proj = S.Projector(f["eigs"], f["basis"])
    n_eigs = min(k, f["eigs"].numel())
    t1 = time.perf_counter()
    sc = {}
    for name in F.SETS:
        eit, en = _eval_image_iter(name)
        sc[name] = S.score_scod_lastblock(ad, proj, eit, en, q, n_eigs)
        print(f"    scored {name:<12} {len(sc[name]):>7,}", flush=True)
    eval_s = time.perf_counter() - t1

    yval = fc.load_cache(F.cache_path("val50k"))["labels"]
    base_pred = np.load(F.SRC / "raw" / "base_val_logits.npy").argmax(-1)
    err = (base_pred != yval).astype(int)
    from sklearn.metrics import average_precision_score, roc_auc_score
    om = F.ood_metrics(sc["val50k"], {d: sc[d] for d in F.DS})
    path = F.OUT / "metrics" / "scod_results.json"
    results = json.loads(path.read_text()) if path.exists() else {"q": q, "scopes": {}}
    results["scopes"]["SCOD-last_block"] = {
        "scope": "last_block", "requested_k": S.K_EIGS, "requested_T": S.T_SKETCH,
        "actual_k": k, "actual_T": T, "fallback_reason": fb,
        "k_sketch": f["k_sketch"], "l_sketch": f["l_sketch"],
        "recovered_rank": f["recovered_rank"], "n_eigs_used": n_eigs,
        "q": q, "n_cal": f["n_cal"], "n_params": f["n_params"],
        "fit_seconds": f["fit_seconds"], "basis_seconds": f["basis_seconds"],
        "eval_seconds": eval_s, "total_seconds": time.perf_counter() - t0,
        "peak_rss_gib": max(f["peak_rss_gib"], cpu_peak_rss_gib()),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / GIB,
        "storage_mib": f["storage_mib"], "top_eigs": f["eigs"][-10:].tolist(),
        "gradient_source": "autograd on encoder_layer_11 + encoder.ln + head, "
                           "per example (attention mixes tokens, so the CLS-only "
                           "cache and the analytic route do not apply)",
        **om,
        "id_error_auroc": float(roc_auc_score(err, sc["val50k"])),
        "id_error_aupr": float(average_precision_score(err, sc["val50k"])),
        "mean_score_correct": float(sc["val50k"][err == 0].mean()),
        "mean_score_incorrect": float(sc["val50k"][err == 1].mean()),
        "OOD data accessed before any choice": "NO"}
    np.savez_compressed(F.OUT / "predictions" / "scod_last_block_scores.npz",
                        **{n: sc[n].astype(np.float32) for n in F.SETS})
    r = results["scopes"]["SCOD-last_block"]
    print(f"  SCOD-last-block: Near {r['near_auroc']*100:.2f}  "
          f"Far {r['far_auroc']*100:.2f}  ID-error {r['id_error_auroc']*100:.2f}  "
          f"peak RSS {r['peak_rss_gib']:.1f} GiB  {r['total_seconds']/60:.1f} min")
    F.write_json(path, results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["validate", "qsweep", "nsweep", "fit", "fitlast",
                             "exactvalidate", "exactfit", "exactnsweep"])
    ap.add_argument("--scopes", default="")
    a = ap.parse_args()
    {"validate": stage_validate, "qsweep": stage_qsweep, "nsweep": stage_nsweep,
     "fit": stage_fit, "fitlast": stage_fitlast,
     "exactvalidate": stage_exactvalidate, "exactfit": stage_exactfit,
     "exactnsweep": stage_exactnsweep}[a.stage](a)


if __name__ == "__main__":
    main()
