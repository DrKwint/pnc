"""Correction stages: token-mode throughput, row budgets, ridge benchmark, member sanity.

Spec sections 10-14 (this module's ``run_correction``) and 15-17 (``run_ensemble``).

The design matrix is never materialised. Every fit goes through streamed sufficient
statistics ``G = X^T X`` / ``C = X^T Z`` accumulated in float32 on the GPU, moved to CPU
and cast to float64 for a single Cholesky solve shared across all 768 output columns.
Residuals are recovered from the same statistics (``pnc_core.relative_residual``), so
held-out evaluation costs one extra streaming pass and no storage.

ID data only: the dataset module cannot load OOD (spec section 21).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import pnc_core as pc
from . import stages_memory as sm
from .data import ImageNetVal, accuracy_gate, split_indices
from .memprobe import (GIB, Recorder, cpu_peak_rss_gib, cpu_rss_gib, probe, reset_cuda,
                       timed)
from .vit_adapter import ViTPnCAdapter

RIDGE = 1e-3                                   # spec section 12
TOKEN_MODES = ["cls", "cls+4", "cls+16", "all"]
# The spec asks for 4096..32768. A pilot showed 4096 rows (rows/dim = 1.33) *degrades*
# held-out ID behaviour -- the fit overfits -- so the grid is extended upward until the
# correction actually helps. Rows are cheap in all-token mode (131072 rows = 666 images),
# so the extension costs seconds; CLS-only is capped by the 40k calibration pool.
ROW_BUDGETS = [4096, 8192, 16384, 32768, 65536, 131072]
CORRECTION_DIM = pc.DHID + 1                   # 3073 with the bias column

BUDGET_COLUMNS = [
    "token_mode", "row_budget", "rows_used", "images_used", "rows_per_image",
    "rows_over_correction_dim", "gram_eig_min", "gram_eig_max", "cond_gram",
    "cond_regularised", "solve_ok", "calib_residual", "heldout_residual",
    "heldout_cls_residual", "heldout_logit_mse", "heldout_logit_max_abs",
    "heldout_top1_agreement", "delta_vs_uncorrected", "solve_time_s", "accum_time_s",
    "images_time_s", "peak_allocated_gib", "notes",
]
RIDGE_COLUMNS = [
    "row_budget", "token_mode", "matrix_dim", "n_outputs", "cpu_rss_before_gib",
    "cpu_peak_rss_gib", "factor_time_s", "solve_time_s", "total_solve_time_s",
    "cond_estimate", "residual", "naive_768_solves_time_s", "speedup_shared_factor",
    "projected_M20_solve_s", "notes",
]


# ---------------------------------------------------------------------------
def _accumulate(ad, ds, indices, mode, dW1, batch, seed, stats=None,
                checkpoints=None, max_rows=None):
    """Stream (G, C) for one token mode. Returns (stats, per-batch timings, snapshots).

    ``checkpoints`` is a sorted list of row counts at which to snapshot (G, C) so a whole
    nested row-budget sweep costs one pass instead of one pass per budget.
    """
    W1v = ad.W1 + dW1
    stats = stats or pc.SufficientStats(device=ad.device)
    snaps, accum_t, fwd_t, n_img = {}, 0.0, 0.0, 0
    pending = sorted(checkpoints or [])
    for imgs, _, ids in ds.iter_batches(indices, batch):
        t0 = time.perf_counter()
        with torch.inference_mode():
            x_resid = ad.prefix(imgs.to(ad.device, ad.dtype))
            h = ad.block.ln_2(x_resid)
            y = torch.nn.functional.gelu(h @ W1v + ad.b1)      # perturbed post-GELU
            z = torch.nn.functional.gelu(h @ ad.W1 + ad.b1) @ ad.W2 + ad.b2   # original out
        torch.cuda.synchronize()
        fwd_t += time.perf_counter() - t0

        idx = pc.token_index_matrix(mode, ids, y.shape[1], seed)
        Yv = pc.gather_tokens(y, idx)
        Z = pc.gather_tokens(z, idx)
        n_img += len(ids)

        # Consume the batch in sub-chunks that stop exactly on each checkpoint, so a
        # "4096-row" fit really is 4096 rows even in all-token mode (3152 rows/batch).
        start, n = 0, Yv.shape[0]
        while start < n:
            stop = n
            if pending:
                stop = min(stop, start + max(0, pending[0] - stats.n_rows))
            if max_rows is not None:
                stop = min(stop, start + max(0, max_rows - stats.n_rows))
            if stop > start:
                t0 = time.perf_counter()
                stats.update(pc.SufficientStats.augment(Yv[start:stop]), Z[start:stop])
                torch.cuda.synchronize()
                accum_t += time.perf_counter() - t0
                start = stop
            while pending and stats.n_rows >= pending[0]:
                snaps[pending[0]] = {**stats.snapshot(), "images_used": n_img,
                                     "accum_time_s": accum_t, "forward_time_s": fwd_t}
                pending.pop(0)
            if max_rows is not None and stats.n_rows >= max_rows:
                break
        del x_resid, h, y, z, Yv, Z
        if max_rows is not None and stats.n_rows >= max_rows:
            break
    return stats, {"accum_time_s": accum_t, "forward_time_s": fwd_t, "images": n_img}, snaps


# ------------------------------------------------------------- section 11
def stage_token_modes(ad, ds, rec, thr, indices, dW1, batch, seed, n_images=512):
    """Throughput of each token-sampling design (spec section 11)."""
    out = []
    subset = indices[:n_images]
    for mode in TOKEN_MODES:
        with probe(rec, "xtx_accumulation", batch_size=batch, dtype="float32",
                   target_block=ad.block_index, member_count=1, token_mode=mode) as r:
            t0 = time.perf_counter()
            stats, tm, _ = _accumulate(ad, ds, subset, mode, dW1, batch, seed)
            wall = time.perf_counter() - t0
            rpi = stats.n_rows / tm["images"]
            r["notes"] = (f"{stats.n_rows} rows from {tm['images']} images "
                          f"({rpi:.1f} rows/img); accum {tm['accum_time_s']:.2f}s of "
                          f"{wall:.2f}s total")
            r["_m"] = (stats, tm, wall, rpi)
        stats, tm, wall, rpi = r["_m"]
        n_batches = int(np.ceil(tm["images"] / batch))
        rec_out = {
            "token_mode": mode, "images": tm["images"], "rows": stats.n_rows,
            "rows_per_image": rpi,
            "rows_per_s": stats.n_rows / wall, "images_per_s": tm["images"] / wall,
            "xtx_accum_s_per_batch": tm["accum_time_s"] / n_batches,
            "forward_s_per_batch": tm["forward_time_s"] / n_batches,
            "wall_s": wall,
            "proj_1000_images_s": wall / tm["images"] * 1000,
            "proj_4000_images_s": wall / tm["images"] * 4000,
            "rows_over_correction_dim": stats.n_rows / CORRECTION_DIM,
            "peak_allocated_gib": r["row"]["peak_allocated_gib"],
            "gpu_stats_mib": stats.nbytes_gpu() / 1024**2,
        }
        out.append(rec_out)
        thr.write(test="xtx_accumulation", batch_size=batch, dtype="float32",
                  token_mode=mode, member_count=1,
                  rows_per_image=round(rpi, 2), rows_per_s=round(rec_out["rows_per_s"], 1),
                  images_per_s=round(rec_out["images_per_s"], 2),
                  median_s="", mean_s="",
                  xtx_accum_s_per_batch=round(rec_out["xtx_accum_s_per_batch"], 5),
                  proj_1000_images_s=round(rec_out["proj_1000_images_s"], 1),
                  proj_4000_images_s=round(rec_out["proj_4000_images_s"], 1),
                  rows_over_correction_dim=round(rec_out["rows_over_correction_dim"], 3),
                  peak_allocated_gib=r["row"]["peak_allocated_gib"],
                  peak_reserved_gib=r["row"]["peak_reserved_gib"],
                  notes=f"{tm['images']} calibration images, G/C on GPU = "
                        f"{rec_out['gpu_stats_mib']:.1f} MiB")
        print(f"  {mode:<8} {rpi:6.1f} rows/img  {rec_out['images_per_s']:6.1f} img/s  "
              f"{rec_out['rows_per_s']:9.0f} rows/s  accum "
              f"{rec_out['xtx_accum_s_per_batch']*1e3:6.1f} ms/batch  "
              f"1k imgs -> {rec_out['proj_1000_images_s']:.0f}s", flush=True)
        del stats
        reset_cuda()
    return out


# ---------------------------------------------------- sections 12 and 13
@torch.inference_mode()
def _cls_logit_probe(ad, ds, indices, dW1, batch):
    """Cache the CLS residual + base logits for a fixed image set.

    The final-block tail is CLS-only and exact, so one (B, 1, 768) slice per image is all
    a fitted Theta needs to be scored in logit space. Returns a closure that maps Theta ->
    logit metrics against the unperturbed base.
    """
    W1v = ad.W1 + dW1
    xs, base, unc = [], [], []
    for imgs, _, _ in ds.iter_batches(indices, batch):
        xr = ad.prefix(imgs.to(ad.device, ad.dtype))[:, :1]
        xs.append(xr)
        base.append(ad.tail(xr, cls_only=True))
        unc.append(ad.tail(xr, W1=W1v, cls_only=True))
    X = torch.cat(xs)
    base = torch.cat(base).double()
    unc = torch.cat(unc).double()

    def score(Theta):
        W2 = torch.as_tensor(Theta[1:], device=ad.device, dtype=ad.dtype)
        b2 = torch.as_tensor(Theta[0], device=ad.device, dtype=ad.dtype)
        cor = ad.tail(X, W1=W1v, W2=W2, b2=b2, cls_only=True).double()
        return {"logit_mse": float(((cor - base) ** 2).mean()),
                "logit_max_abs": float((cor - base).abs().max()),
                "top1_agreement": float((cor.argmax(-1) == base.argmax(-1)).float().mean())}

    baseline = {"uncorrected_logit_mse": float(((unc - base) ** 2).mean()),
                "uncorrected_logit_max_abs": float((unc - base).abs().max()),
                "uncorrected_top1_agreement": float(
                    (unc.argmax(-1) == base.argmax(-1)).float().mean()),
                "n_images": int(X.shape[0])}
    return score, baseline


def stage_row_budgets(ad, ds, rec, budget_rec, ridge_rec, calib, heldout, dW1, batch,
                      seed, modes=("cls", "cls+16", "all"), probe_images=None):
    """Fit at each row budget and score every fit on a COMMON held-out yardstick.

    Each mode's own-token residual is not comparable across modes (all-token residual
    averages over 197 rows/image, CLS-only over 1). So every fitted Theta is additionally
    scored on (a) held-out CLS-token residual and (b) held-out CLS logit MSE against the
    unperturbed model -- the quantities that actually decide ID preservation.
    """
    Theta0 = ad.theta().detach().double().cpu().numpy()

    # Common yardsticks, computed once and reused by every (mode, budget) fit.
    ho_cls_stats, _, _ = _accumulate(ad, ds, heldout, "cls", dW1, batch, seed)
    ho_cls = ho_cls_stats.snapshot()
    uncorrected_cls = pc.relative_residual(ho_cls, Theta0)
    score, probe_base = _cls_logit_probe(ad, ds, probe_images, dW1, batch)
    print(f"  common yardstick: {ho_cls['n_rows']} held-out CLS rows; uncorrected CLS "
          f"residual {uncorrected_cls:.4f}, uncorrected logit MSE "
          f"{probe_base['uncorrected_logit_mse']:.4e} over {probe_base['n_images']} images")
    del ho_cls_stats
    reset_cuda()

    results = []
    for mode in modes:
        rows_per_img = 1 if mode == "cls" else (17 if mode == "cls+16" else 197)
        budgets = [b for b in ROW_BUDGETS if b <= rows_per_img * len(calib)]
        if not budgets:
            continue
        need_imgs = int(np.ceil(max(budgets) / rows_per_img))
        print(f"\n  -- token mode {mode}: {rows_per_img} rows/img, "
              f"{need_imgs} images for {max(budgets)} rows")
        stats, tm, snaps = _accumulate(ad, ds, calib[:need_imgs], mode, dW1, batch, seed,
                                       checkpoints=budgets, max_rows=max(budgets))
        # one held-out pass, reused by every budget for this mode
        ho_stats, ho_tm, _ = _accumulate(ad, ds, heldout, mode, dW1, batch, seed)
        ho = ho_stats.snapshot()
        uncorrected = pc.relative_residual(ho, Theta0)      # what the member does with no fix

        for b in budgets:
            snap = snaps.get(b)
            if snap is None:
                continue
            rss0 = cpu_rss_gib()
            cond = pc.condition_estimate(snap["G"], RIDGE)
            t0 = time.perf_counter()
            Theta, t_fac, t_sol = pc.cho_solve_shared(snap["G"], snap["C"], RIDGE,
                                                      w_prior=Theta0)
            t_total = time.perf_counter() - t0
            ok = bool(np.isfinite(Theta).all())
            calib_res = pc.relative_residual(snap, Theta)
            ho_res = pc.relative_residual(ho, Theta)
            ho_cls_res = pc.relative_residual(ho_cls, Theta)      # common across modes
            lg = score(Theta)                                      # common logit yardstick

            # naive comparison: 768 independent solves vs one shared factorisation
            t0 = time.perf_counter()
            A = snap["G"] + RIDGE * np.eye(snap["G"].shape[0])
            rhs = snap["C"] + RIDGE * Theta0
            for j in range(8):                               # time 8, scale to 768
                np.linalg.solve(A, rhs[:, j])
            t_naive = (time.perf_counter() - t0) / 8 * pc.DIN

            row = {
                "token_mode": mode, "row_budget": b, "rows_used": snap["n_rows"],
                "images_used": snap["images_used"], "rows_per_image": rows_per_img,
                "rows_over_correction_dim": round(snap["n_rows"] / CORRECTION_DIM, 3),
                **{k: round(v, 6) if np.isfinite(v) else v for k, v in cond.items()},
                "solve_ok": ok, "calib_residual": round(calib_res, 6),
                "heldout_residual": round(ho_res, 6),
                "heldout_cls_residual": round(ho_cls_res, 6),
                "heldout_logit_mse": f"{lg['logit_mse']:.6e}",
                "heldout_logit_max_abs": round(lg["logit_max_abs"], 5),
                "heldout_top1_agreement": round(lg["top1_agreement"], 5),
                "delta_vs_uncorrected": round(uncorrected - ho_res, 6),
                "solve_time_s": round(t_total, 4),
                "accum_time_s": round(snap["accum_time_s"], 3),
                "images_time_s": round(snap["forward_time_s"], 2),
                "peak_allocated_gib": "",
                "notes": f"same-mode uncorrected held-out residual {uncorrected:.4f}; "
                         f"common CLS yardstick: uncorrected residual {uncorrected_cls:.4f}, "
                         f"uncorrected logit MSE {probe_base['uncorrected_logit_mse']:.3e}, "
                         f"uncorrected top-1 agreement "
                         f"{probe_base['uncorrected_top1_agreement']:.4f}; "
                         f"lambda={RIDGE}; original-centred",
            }
            budget_rec.write(**row)
            ridge_rec.write(
                row_budget=b, token_mode=mode, matrix_dim=f"{CORRECTION_DIM}x{CORRECTION_DIM}",
                n_outputs=pc.DIN, cpu_rss_before_gib=round(rss0, 3),
                cpu_peak_rss_gib=round(cpu_peak_rss_gib(), 3),
                factor_time_s=round(t_fac, 4), solve_time_s=round(t_sol, 4),
                total_solve_time_s=round(t_total, 4),
                cond_estimate=f"{cond['cond_regularised']:.4g}",
                residual=round(calib_res, 6),
                naive_768_solves_time_s=round(t_naive, 4),
                speedup_shared_factor=round(t_naive / max(t_total, 1e-9), 1),
                projected_M20_solve_s=round(t_total * 20, 2),
                notes="one Cholesky shared across all 768 output columns")
            row["_common_uncorrected"] = {
                "cls_residual": uncorrected_cls, **probe_base}
            results.append(row)
            print(f"     budget {b:>6}: rows/dim={row['rows_over_correction_dim']:6.2f}  "
                  f"cond={cond['cond_regularised']:.2e}  calib={calib_res:.4f}  "
                  f"CLS-res={ho_cls_res:.4f}  logitMSE={lg['logit_mse']:.3e}  "
                  f"agree={lg['top1_agreement']:.4f}  "
                  f"[{snap['images_used']} imgs, {snap['forward_time_s']:.0f}s]  "
                  f"solve {t_total:.2f}s", flush=True)
        del stats, ho_stats
        reset_cuda()
    return results


# ------------------------------------------------------------- section 14
def stage_corrected_member(ad, ds, rec, calib, heldout, bench, U, coeffs, scale, batch,
                           seed, mode="cls+16", row_budget=16384):
    """Build one complete member end-to-end and compare logits on held-out ID images."""
    Theta0 = ad.theta().detach().double().cpu().numpy()
    dW1 = torch.as_tensor(pc.member_dW1(U, coeffs[0], scale), device=ad.device,
                          dtype=ad.dtype)
    rows_per_img = 17 if mode == "cls+16" else 1
    need = int(np.ceil(row_budget / rows_per_img))

    rss0 = cpu_rss_gib()
    with probe(rec, "member_construction", batch_size=batch, dtype="float32",
               target_block=ad.block_index, member_count=1, token_mode=mode) as r:
        t0 = time.perf_counter()
        stats, tm, _ = _accumulate(ad, ds, calib[:need], mode, dW1, batch, seed,
                                   max_rows=row_budget)
        snap = {**stats.snapshot(), "images_used": tm["images"]}
        t_accum = time.perf_counter() - t0
        t1 = time.perf_counter()
        Theta, t_fac, t_sol = pc.cho_solve_shared(snap["G"], snap["C"], RIDGE, w_prior=Theta0)
        t_solve = time.perf_counter() - t1
        r["notes"] = (f"{snap['n_rows']} rows from {snap['images_used']} images; "
                      f"accumulate {t_accum:.1f}s + solve {t_solve:.2f}s")
    construction_s = t_accum + t_solve

    W2c = torch.as_tensor(Theta[1:], device=ad.device, dtype=ad.dtype)
    b2c = torch.as_tensor(Theta[0], device=ad.device, dtype=ad.dtype)
    W1v = ad.W1 + dW1

    base_l, unc_l, cor_l, labels = [], [], [], []
    with probe(rec, "corrected_member_inference", batch_size=batch, dtype="float32",
               target_block=ad.block_index, member_count=1, token_mode=mode) as r:
        for imgs, labs, _ in ds.iter_batches(bench, batch):
            with torch.inference_mode():
                xr = ad.prefix(imgs.to(ad.device, ad.dtype))
                base_l.append(ad.tail(xr).cpu())
                unc_l.append(ad.tail(xr, W1=W1v).cpu())
                cor_l.append(ad.tail(xr, W1=W1v, W2=W2c, b2=b2c).cpu())
            labels.append(labs)
            del xr
        r["notes"] = f"{len(bench)} held-out ID images"

    base = torch.cat(base_l).double()
    unc = torch.cat(unc_l).double()
    cor = torch.cat(cor_l).double()
    labels = torch.cat(labels)
    agree = lambda a, b: float((a.argmax(-1) == b.argmax(-1)).float().mean())
    acc = lambda a: float((a.argmax(-1) == labels).float().mean())
    out = {
        "token_mode": mode, "row_budget": row_budget, "rows_used": snap["n_rows"],
        "images_used": snap["images_used"], "n_eval_images": len(bench),
        "construction_time_s": construction_s,
        "accumulate_time_s": t_accum, "solve_time_s": t_solve,
        "factor_time_s": t_fac, "cho_solve_time_s": t_sol,
        "peak_gpu_gib": r["row"]["peak_allocated_gib"],
        "cpu_rss_before_gib": rss0, "cpu_peak_rss_gib": cpu_peak_rss_gib(),
        "base_top1": acc(base), "uncorrected_top1": acc(unc), "corrected_top1": acc(cor),
        "agreement_base_vs_uncorrected": agree(base, unc),
        "agreement_base_vs_corrected": agree(base, cor),
        "uncorrected_logit_mse": float(((unc - base) ** 2).mean()),
        "corrected_logit_mse": float(((cor - base) ** 2).mean()),
        "uncorrected_logit_max_abs": float((unc - base).abs().max()),
        "corrected_logit_max_abs": float((cor - base).abs().max()),
        "all_finite": bool(torch.isfinite(cor).all()),
    }
    out["mse_reduction_factor"] = (out["uncorrected_logit_mse"]
                                   / max(out["corrected_logit_mse"], 1e-30))
    member = {"coeff": coeffs[0], "W2": W2c, "b2": b2c}
    del stats
    reset_cuda()
    return out, member, Theta


# --------------------------------------------------- cached-prefix construction
def stage_cached_h_construction(ad, ds, rec, calib, U, coeffs, scale, batch, seed,
                                mode="cls+16", row_budget=8192, M=20):
    """Cache (h, z) once per target block, then build M members without re-running the prefix.

    For a fixed target block the FFN input h and the original output z do not depend on
    the member or the scale -- only the perturbed post-GELU y does. Caching the *sampled*
    rows costs 2 x row_budget x 768 x 4 B (48 MiB at 8192 rows) and removes the ViT prefix
    from the per-member inner loop entirely. This is the difference between the naive
    construction cost and the one the real experiment should pay.
    """
    Theta0 = ad.theta().detach().double().cpu().numpy()
    rows_per_img = 17 if mode == "cls+16" else (1 if mode == "cls" else 197)
    need = int(np.ceil(row_budget / rows_per_img))

    with probe(rec, "cache_h_z_once", batch_size=batch, dtype="float32",
               target_block=ad.block_index, member_count=M, token_mode=mode) as r:
        H, Z = [], []
        t0 = time.perf_counter()
        for imgs, _, ids in ds.iter_batches(calib[:need], batch):
            with torch.inference_mode():
                x_resid = ad.prefix(imgs.to(ad.device, ad.dtype))
                h = ad.block.ln_2(x_resid)
                z = torch.nn.functional.gelu(h @ ad.W1 + ad.b1) @ ad.W2 + ad.b2
            idx = pc.token_index_matrix(mode, ids, h.shape[1], seed)
            H.append(pc.gather_tokens(h, idx))
            Z.append(pc.gather_tokens(z, idx))
            del x_resid, h, z
        H = torch.cat(H)[:row_budget]
        Z = torch.cat(Z)[:row_budget]
        torch.cuda.synchronize()
        t_cache = time.perf_counter() - t0
        r["notes"] = (f"cached {H.shape[0]} rows of h and z from {need} images in "
                      f"{t_cache:.1f}s = "
                      f"{(H.numel()+Z.numel())*4/1024**2:.1f} MiB on GPU")
    cache_mib = (H.numel() + Z.numel()) * 4 / 1024**2

    per_member, thetas = [], []
    with probe(rec, "member_from_cached_h", batch_size="", dtype="float32",
               target_block=ad.block_index, member_count=M, token_mode=mode) as r:
        for m in range(M):
            t0 = time.perf_counter()
            dW1 = torch.as_tensor(pc.member_dW1(U, coeffs[m], scale), device=ad.device,
                                  dtype=ad.dtype)
            with torch.inference_mode():
                y = torch.nn.functional.gelu(H @ (ad.W1 + dW1) + ad.b1)
                X = pc.SufficientStats.augment(y)
                G = (X.T @ X).double().cpu().numpy()
                C = (X.T @ Z).double().cpu().numpy()
            torch.cuda.synchronize()
            t_gpu = time.perf_counter() - t0
            t1 = time.perf_counter()
            Theta, t_fac, t_sol = pc.cho_solve_shared(G, C, RIDGE, w_prior=Theta0)
            t_cpu = time.perf_counter() - t1
            per_member.append({"member": m, "gpu_s": t_gpu, "cpu_solve_s": t_cpu,
                               "factor_s": t_fac, "total_s": t_gpu + t_cpu,
                               "finite": bool(np.isfinite(Theta).all())})
            thetas.append(Theta.astype(np.float32))
            del dW1, y, X
        r["notes"] = (f"{M} members from cached h: median "
                      f"{np.median([p['total_s'] for p in per_member]):.2f}s/member")

    tot = [p["total_s"] for p in per_member]
    out = {"mode": mode, "row_budget": int(H.shape[0]), "images_cached": need,
           "cache_time_s": t_cache, "cache_gpu_mib": cache_mib,
           "per_member": per_member,
           "median_member_s": float(np.median(tot)),
           "mean_member_s": float(np.mean(tot)),
           "median_gpu_s": float(np.median([p["gpu_s"] for p in per_member])),
           "median_cpu_solve_s": float(np.median([p["cpu_solve_s"] for p in per_member])),
           "total_M_members_s": float(np.sum(tot)),
           "peak_allocated_gib": r["row"]["peak_allocated_gib"],
           "all_finite": all(p["finite"] for p in per_member)}
    del H, Z
    reset_cuda()
    return out, thetas


# ---------------------------------------------------------------------------
def run_correction(args, OUT: Path, throughput_columns):
    from .run_preflight import load_state, save_state
    st = load_state()
    batch = st.get("safe_batch", 16)
    seed = args.seed

    rec = Recorder(OUT / "memory_preflight.csv")
    thr = Recorder(OUT / "throughput_preflight.csv", throughput_columns)
    budget_rec = Recorder(OUT / "correction_row_budget.csv", BUDGET_COLUMNS)
    ridge_rec = Recorder(OUT / "ridge_benchmark.csv", RIDGE_COLUMNS)

    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    ds = ImageNetVal()
    sp = split_indices(len(ds), n_calib=args.n_calib, n_heldout=args.n_heldout, n_bench=512)
    print(f"dataset: {len(ds)} ImageNet-1k val images across {len(ds.paths)} shards; "
          f"calib {len(sp.calib)} / held-out {len(sp.heldout)} / bench {len(sp.bench)}")
    print(f"safe batch = {batch}\n")

    U = pc.perturbation_basis(seed, args.K)
    coeffs = pc.member_coefficients(seed, args.M, args.K)
    scale = pc.base_scale(U, coeffs, float(ad.W1.norm()))
    dW1 = torch.as_tensor(pc.member_dW1(U, coeffs[0], scale), device=ad.device,
                          dtype=ad.dtype)

    print("== section 11: token-sampling throughput ==")
    modes = stage_token_modes(ad, ds, rec, thr, sp.calib, dW1, batch, seed, n_images=512)

    print("\n== sections 12/13: row budgets, conditioning, held-out ID residual ==")
    budgets = stage_row_budgets(ad, ds, rec, budget_rec, ridge_rec, sp.calib,
                                sp.heldout[:1024], dW1, batch, seed,
                                probe_images=sp.heldout[1024:1536])

    print("\n== section 14: one complete corrected member ==")
    member_out, member, Theta = stage_corrected_member(
        ad, ds, rec, sp.calib, sp.heldout, sp.bench, U, coeffs, scale, batch, seed)
    print(f"  construction {member_out['construction_time_s']:.1f}s "
          f"(accumulate {member_out['accumulate_time_s']:.1f}s + solve "
          f"{member_out['solve_time_s']:.2f}s)")
    print(f"  top-1: base {member_out['base_top1']:.4f} | uncorrected "
          f"{member_out['uncorrected_top1']:.4f} | corrected {member_out['corrected_top1']:.4f}")
    print(f"  agreement with base: uncorrected {member_out['agreement_base_vs_uncorrected']:.4f} "
          f"-> corrected {member_out['agreement_base_vs_corrected']:.4f}")
    print(f"  logit MSE: uncorrected {member_out['uncorrected_logit_mse']:.4e} -> corrected "
          f"{member_out['corrected_logit_mse']:.4e} "
          f"({member_out['mse_reduction_factor']:.1f}x reduction)")

    sm.write_json(OUT / "raw" / "stage_correction.json",
                  {"token_modes": modes, "row_budgets": budgets, "member": member_out,
                   "safe_batch": batch, "ridge": RIDGE, "seed": seed})
    np.savez_compressed(OUT / "raw" / "member0_correction.npz",
                        Theta=Theta.astype(np.float32), coeff=coeffs[0], scale=scale)
    save_state(correction_done=True, member_construction_s=member_out["construction_time_s"],
               member_accumulate_s=member_out["accumulate_time_s"],
               member_solve_s=member_out["solve_time_s"],
               numerical_scale=scale)
    print(f"\nwrote {OUT/'raw'/'stage_correction.json'}")


ID_PRESERVATION_COLUMNS = [
    "token_mode", "row_budget", "rows_over_correction_dim", "ridge_lambda",
    "lambda_per_row", "cond_regularised", "n_eval_images",
    "uncorrected_mean_mse", "corrected_mean_mse",
    "uncorrected_median_mse", "corrected_median_mse", "median_improvement_x",
    "corrected_p90_mse", "corrected_p99_mse", "corrected_max_mse",
    "uncorrected_p99_mse", "uncorrected_max_mse",
    "frac_images_improved", "uncorrected_top1_agreement", "corrected_top1_agreement",
    "heldout_cls_residual", "solve_time_s", "notes",
]


@torch.inference_mode()
def _cache_cls_eval(ad, ds, indices, dW1, batch):
    """Cache CLS residual, base logits and uncorrected logits for a held-out ID set."""
    W1v = ad.W1 + dW1
    xs, base, unc = [], [], []
    for imgs, _, _ in ds.iter_batches(indices, batch):
        x = ad.prefix(imgs.to(ad.device, ad.dtype))[:, :1]
        xs.append(x)
        base.append(ad.tail(x, cls_only=True))
        unc.append(ad.tail(x, W1=W1v, cls_only=True))
    return torch.cat(xs), torch.cat(base).double(), torch.cat(unc).double(), W1v


def stage_id_preservation(ad, ds, rec, calib, heldout, dW1, batch, seed, out_csv,
                          budgets=(8192, 16384, 32768), lambdas=(1e-3, 1e0, 1e2, 1e4),
                          n_eval=2048, mode="cls"):
    """Per-image ID-preservation distribution vs row budget and ridge strength.

    A mean logit MSE over a few hundred images is not a usable yardstick here: the
    corrected model's per-image error is heavy-tailed, so the mean is set by a handful of
    outliers and flips sign depending on which images the sample happens to contain. This
    stage reports the full distribution over ``n_eval`` held-out ID images, and sweeps the
    ridge because lambda=1e-3 in the *summed* objective is effectively no regularisation
    at these row counts (lambda/n ~ 3e-8).
    """
    Theta0 = ad.theta().detach().double().cpu().numpy()
    rec_csv = Recorder(out_csv, ID_PRESERVATION_COLUMNS)

    eval_idx = np.asarray(heldout)[:n_eval]
    X, base, unc, W1v = _cache_cls_eval(ad, ds, eval_idx, dW1, batch)
    mse_unc = ((unc - base) ** 2).mean(-1).cpu().numpy()
    agree_unc = float((unc.argmax(-1) == base.argmax(-1)).double().mean())
    print(f"  eval on {len(eval_idx)} held-out ID images; uncorrected: mean "
          f"{mse_unc.mean():.3e}, median {np.median(mse_unc):.3e}, "
          f"p99 {np.percentile(mse_unc, 99):.3e}, top-1 agreement {agree_unc:.4f}")

    rows_per_img = 1 if mode == "cls" else (17 if mode == "cls+16" else 197)
    need = int(np.ceil(max(budgets) / rows_per_img))
    stats, tm, snaps = _accumulate(ad, ds, calib[:need], mode, dW1, batch, seed,
                                   checkpoints=list(budgets), max_rows=max(budgets))
    ho_stats, _, _ = _accumulate(ad, ds, np.asarray(heldout)[n_eval:n_eval + 1024], mode,
                                 dW1, batch, seed)
    ho = ho_stats.snapshot()
    del stats, ho_stats
    reset_cuda()

    results = []
    for b in budgets:
        snap = snaps.get(b)
        if snap is None:
            continue
        for lam in lambdas:
            t0 = time.perf_counter()
            Theta, _, _ = pc.cho_solve_shared(snap["G"], snap["C"], lam, w_prior=Theta0)
            t_solve = time.perf_counter() - t0
            W2c = torch.as_tensor(Theta[1:], device=ad.device, dtype=ad.dtype)
            b2c = torch.as_tensor(Theta[0], device=ad.device, dtype=ad.dtype)
            cor = ad.tail(X, W1=W1v, W2=W2c, b2=b2c, cls_only=True).double()
            mse = ((cor - base) ** 2).mean(-1).cpu().numpy()
            cond = pc.condition_estimate(snap["G"], lam)
            row = {
                "token_mode": mode, "row_budget": b,
                "rows_over_correction_dim": round(b / CORRECTION_DIM, 3),
                "ridge_lambda": lam, "lambda_per_row": f"{lam / b:.3e}",
                "cond_regularised": f"{cond['cond_regularised']:.4g}",
                "n_eval_images": len(mse),
                "uncorrected_mean_mse": f"{mse_unc.mean():.6e}",
                "corrected_mean_mse": f"{mse.mean():.6e}",
                "uncorrected_median_mse": f"{np.median(mse_unc):.6e}",
                "corrected_median_mse": f"{np.median(mse):.6e}",
                "median_improvement_x": round(float(np.median(mse_unc) / np.median(mse)), 3),
                "corrected_p90_mse": f"{np.percentile(mse, 90):.6e}",
                "corrected_p99_mse": f"{np.percentile(mse, 99):.6e}",
                "corrected_max_mse": f"{mse.max():.6e}",
                "uncorrected_p99_mse": f"{np.percentile(mse_unc, 99):.6e}",
                "uncorrected_max_mse": f"{mse_unc.max():.6e}",
                "frac_images_improved": round(float((mse < mse_unc).mean()), 4),
                "uncorrected_top1_agreement": round(agree_unc, 5),
                "corrected_top1_agreement": round(
                    float((cor.argmax(-1) == base.argmax(-1)).double().mean()), 5),
                "heldout_cls_residual": round(pc.relative_residual(ho, Theta), 6),
                "solve_time_s": round(t_solve, 4),
                "notes": f"{snap['images_used']} calibration images; original-centred",
            }
            rec_csv.write(**row)
            results.append(row)
            print(f"    rows {b:>6} lam {lam:>8.0e}: median {np.median(mse):.3e} "
                  f"({row['median_improvement_x']:>5.2f}x)  mean {mse.mean():.3e}  "
                  f"p99 {np.percentile(mse, 99):.3e}  improved "
                  f"{100*row['frac_images_improved']:.1f}%  agree "
                  f"{row['corrected_top1_agreement']:.4f}  CLSres "
                  f"{row['heldout_cls_residual']:.4f}", flush=True)
            del W2c, b2c, cor
    del X, base, unc
    reset_cuda()
    return {"uncorrected": {"mean": float(mse_unc.mean()),
                            "median": float(np.median(mse_unc)),
                            "p99": float(np.percentile(mse_unc, 99)),
                            "max": float(mse_unc.max()),
                            "top1_agreement": agree_unc,
                            "n_eval_images": len(mse_unc)},
            "grid": results}


def run_quality(args, OUT: Path):
    """Driver for the ID-preservation distribution study."""
    from .run_preflight import load_state
    st = load_state()
    batch = st.get("safe_batch", 16)
    rec = Recorder(OUT / "memory_preflight.csv")
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    ds = ImageNetVal()
    sp = split_indices(len(ds), n_calib=args.n_calib, n_heldout=args.n_heldout, n_bench=512)
    U = pc.perturbation_basis(args.seed, args.K)
    coeffs = pc.member_coefficients(args.seed, args.M, args.K)
    scale = pc.base_scale(U, coeffs, float(ad.W1.norm()))
    dW1 = torch.as_tensor(pc.member_dW1(U, coeffs[0], scale), device=ad.device,
                          dtype=ad.dtype)
    print("== ID preservation: per-image distribution vs row budget and ridge ==")
    out = stage_id_preservation(ad, ds, rec, sp.calib, sp.heldout, dW1, batch, args.seed,
                                OUT / "id_preservation.csv", mode=args.token_mode)
    sm.write_json(OUT / "raw" / "stage_id_preservation.json", out)
    print(f"\nwrote {OUT/'id_preservation.csv'}")


def run_member_only(args, OUT: Path):
    """Section 14 alone, at a chosen token mode / row budget."""
    from .run_preflight import load_state, save_state
    st = load_state()
    batch = st.get("safe_batch", 16)
    rec = Recorder(OUT / "memory_preflight.csv")
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    ds = ImageNetVal()
    sp = split_indices(len(ds), n_calib=args.n_calib, n_heldout=args.n_heldout, n_bench=512)
    U = pc.perturbation_basis(args.seed, args.K)
    coeffs = pc.member_coefficients(args.seed, args.M, args.K)
    scale = pc.base_scale(U, coeffs, float(ad.W1.norm()))

    print(f"== section 14: one complete corrected member "
          f"({args.token_mode}, {args.row_budget} rows) ==")
    out, member, Theta = stage_corrected_member(
        ad, ds, rec, sp.calib, sp.heldout, sp.bench, U, coeffs, scale, batch, args.seed,
        mode=args.token_mode, row_budget=args.row_budget)
    print(f"  construction {out['construction_time_s']:.1f}s "
          f"(accumulate {out['accumulate_time_s']:.1f}s over {out['images_used']} images "
          f"+ solve {out['solve_time_s']:.2f}s)")
    print(f"  peak GPU {out['peak_gpu_gib']:.3f} GiB | peak CPU RSS "
          f"{out['cpu_peak_rss_gib']:.2f} GiB")
    print(f"  top-1 on {out['n_eval_images']} held-out ID images: base {out['base_top1']:.4f} "
          f"| uncorrected {out['uncorrected_top1']:.4f} | corrected {out['corrected_top1']:.4f}")
    print(f"  agreement with base: uncorrected {out['agreement_base_vs_uncorrected']:.4f} "
          f"-> corrected {out['agreement_base_vs_corrected']:.4f}")
    print(f"  logit MSE vs base: uncorrected {out['uncorrected_logit_mse']:.4e} -> corrected "
          f"{out['corrected_logit_mse']:.4e} ({out['mse_reduction_factor']:.2f}x)")
    sm.write_json(OUT / "raw" / "stage_member.json", out)
    np.savez_compressed(OUT / "raw" / "member0_correction.npz",
                        Theta=Theta.astype(np.float32), coeff=coeffs[0], scale=scale)
    save_state(member_construction_s=out["construction_time_s"],
               member_accumulate_s=out["accumulate_time_s"],
               member_solve_s=out["solve_time_s"])
    print(f"\nwrote {OUT/'raw'/'stage_member.json'}")


def run_ensemble(args, OUT: Path, throughput_columns):
    """Sections 15-17: shared-prefix M sweep, compact storage, optional fp16."""
    from .run_preflight import load_state, save_state
    st = load_state()
    batch = st.get("safe_batch", 16)
    seed = args.seed

    rec = Recorder(OUT / "memory_preflight.csv")
    thr = Recorder(OUT / "throughput_preflight.csv", throughput_columns)
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    ds = ImageNetVal()
    sp = split_indices(len(ds), n_calib=args.n_calib, n_heldout=args.n_heldout, n_bench=512)

    U = pc.perturbation_basis(seed, args.K)
    coeffs = pc.member_coefficients(seed, args.M, args.K)
    scale = pc.base_scale(U, coeffs, float(ad.W1.norm()))
    Theta0 = ad.theta().detach().double().cpu().numpy()

    mode, budget = args.token_mode, args.row_budget
    rows_per_img = 17 if mode == "cls+16" else (1 if mode == "cls" else 197)
    n_img = int(np.ceil(budget / rows_per_img))

    # Naive construction cost: re-run the ViT prefix for every member. Timed on a couple of
    # members only -- at the recommended CLS budget this is ~2 min each, and the point is
    # the per-member figure, not 20 repetitions of it.
    print(f"== naive construction cost ({mode}, {budget} rows = {n_img} images/member) ==")
    t_build = []
    for m in range(args.naive_members):
        dW1 = torch.as_tensor(pc.member_dW1(U, coeffs[m], scale), device=ad.device,
                              dtype=ad.dtype)
        t0 = time.perf_counter()
        stats, _, _ = _accumulate(ad, ds, sp.calib[:n_img], mode, dW1, batch, seed,
                                  max_rows=budget)
        snap = stats.snapshot()
        pc.cho_solve_shared(snap["G"], snap["C"], RIDGE, w_prior=Theta0)
        t_build.append(time.perf_counter() - t0)
        del stats, dW1
        reset_cuda()
        print(f"  member {m:>2}: {t_build[-1]:.1f}s (prefix re-run)", flush=True)

    # Real construction: cache (h, z) once, then every member is FFN + G/C + one solve.
    print(f"\n== cached-(h,z) construction: M={args.M} members, prefix paid once ==")
    cached, thetas = stage_cached_h_construction(ad, ds, rec, sp.calib, U, coeffs, scale,
                                                 batch, seed, mode=mode, row_budget=budget,
                                                 M=args.M)
    print(f"  cache {cached['row_budget']} rows from {cached['images_cached']} images: "
          f"{cached['cache_time_s']:.1f}s, {cached['cache_gpu_mib']:.1f} MiB GPU")
    print(f"  per member: {cached['median_member_s']:.2f}s "
          f"(GPU accum {cached['median_gpu_s']:.2f}s + CPU solve "
          f"{cached['median_cpu_solve_s']:.2f}s) vs naive {np.mean(t_build):.1f}s "
          f"-> {np.mean(t_build)/cached['median_member_s']:.0f}x faster\n")

    members = [{"coeff": coeffs[m],
                "W2": torch.as_tensor(th[1:], device=ad.device, dtype=ad.dtype),
                "b2": torch.as_tensor(th[0], device=ad.device, dtype=ad.dtype)}
               for m, th in enumerate(thetas)]

    print("== section 15: shared-prefix sequential inference, corrections resident ==")
    res_resident = sm.stage_ensemble(ad, rec, thr, members, U, scale, batch,
                                     m_list=(1, 5, 20), stream=False)
    print("\n  same, with the M perturbed W1 precomputed once (what the real run should do)")
    res_precomp = sm.stage_ensemble(ad, rec, thr, members, U, scale, batch,
                                    m_list=(1, 5, 20), stream=False, precompute=True)
    print("\n== section 16: streaming corrections from CPU ==")
    cpu_members = [{"coeff": m["coeff"], "W2": m["W2"].cpu(), "b2": m["b2"].cpu()}
                   for m in members]
    res_stream = sm.stage_ensemble(ad, rec, thr, cpu_members, U, scale, batch,
                                   m_list=(1, 5, 20), stream=True)
    storage = sm.stage_storage(members, U, OUT, K=args.K)
    print(f"\n  compact state: basis {storage['shared_basis_mib']:.1f} MiB + "
          f"{storage['corrected_W2_total_mib']:.1f} MiB corrections = "
          f"{storage['cpu_total_mib']:.1f} MiB CPU, {storage['disk_npz_mib']:.1f} MiB on disk "
          f"(vs {storage['twenty_full_models_gib']:.2f} GiB for 20 full models)")

    fp16 = stage_fp16(rec, thr, batch, args)

    sm.write_json(OUT / "raw" / "stage_ensemble.json",
                  {"member_build_times_s": t_build,
                   "mean_member_build_s": float(np.mean(t_build)),
                   "cached_construction": cached,
                   "resident": res_resident, "resident_precomputed": res_precomp,
                   "streaming": res_stream,
                   "storage": storage, "fp16": fp16})
    save_state(ensemble_done=True, mean_member_build_s=float(np.mean(t_build)))
    print(f"\nwrote {OUT/'raw'/'stage_ensemble.json'}")


def stage_fp16(rec, thr, batch, args):
    """Section 17: optional fp16 inference check. Never used for G/C or the ridge solve."""
    print("\n== section 17: optional fp16 inference (float32 remains the primary path) ==")
    try:
        ad16 = ViTPnCAdapter(device="cuda", dtype=torch.float16)
        ad32 = ViTPnCAdapter(device="cuda", dtype=torch.float32)
        x32 = torch.randn(batch, 3, 224, 224, device="cuda")
        with probe(rec, "base_inference_fp16", batch_size=batch, dtype="float16",
                   target_block="n/a", member_count=1, token_mode="n/a") as r:
            t = timed(lambda: ad16.model(x32.half()), warmup=3, iters=15)
            r["notes"] = f"median {t['median_s']*1e3:.1f} ms, {batch/t['median_s']:.1f} img/s"
        with torch.inference_mode():
            l16 = ad16.model(x32.half()).float()
            l32 = ad32.model(x32)
        out = {"images_per_s": batch / t["median_s"],
               "peak_allocated_gib": r["row"]["peak_allocated_gib"],
               "max_abs_logit_diff_vs_fp32": float((l16 - l32).abs().max()),
               "top1_agreement_vs_fp32": float(
                   (l16.argmax(-1) == l32.argmax(-1)).float().mean()),
               "all_finite": bool(torch.isfinite(l16).all())}
        print(f"  fp16 {out['images_per_s']:.1f} img/s, peak {out['peak_allocated_gib']:.3f} GiB, "
              f"max|dlogit| vs fp32 {out['max_abs_logit_diff_vs_fp32']:.3e}, "
              f"top-1 agreement {out['top1_agreement_vs_fp32']:.3f}")
        thr.write(test="base_inference_fp16", batch_size=batch, dtype="float16",
                  token_mode="n/a", member_count=1, rows_per_image="", rows_per_s="",
                  images_per_s=round(out["images_per_s"], 2),
                  median_s=round(t["median_s"], 6), mean_s=round(t["mean_s"], 6),
                  peak_allocated_gib=r["row"]["peak_allocated_gib"],
                  peak_reserved_gib=r["row"]["peak_reserved_gib"],
                  notes="fp16 inference only; G/C and ridge stay float32/float64")
        del ad16, ad32, x32
        reset_cuda()
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"  fp16 test failed ({type(exc).__name__}: {str(exc)[:120]})")
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
