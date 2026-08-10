#!/usr/bin/env python3
"""Phase 1 / Section 4: local conv2 residual -> final logits/probabilities bridge.

For the anchor P&C ensemble (s3b0, seed 0), per eval image and member:
  * exact conv2 residual R_v(x) (validated patch formula) = block-output deviation,
  * actual logit change  d_ell_v(x) = ell_v(x) - ell_0(x),
  * first-order downstream prediction  J_down(x) . R_v(x)  via jax.jvp through the
    remaining blocks (frozen BN) + pool + fc,
  * candidate local-residual statistics: Frobenius, mean-patch, max-patch, p95-patch,
    channelwise, GAP-residual, and J-projected (||J.R_v||).

Reports (4.2) cosine(d_ell, J.R_v), relative magnitude error, Spearman(||R_v||, ||d_ell||);
and (4.3) which local statistic best predicts per-image logit-cov-trace / MI / predictive entropy.
Writes local_to_output_raw.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx
from scipy import stats
import pickle

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO))
from models import PreActResNet18
from cifar_tasks import (_load_cifar_openood_context, _build_single_block_pnc_ensemble,
                         compute_cifar_block_preacts, CIFAROpenOODPnC)
from pnc import extract_patches, flatten_conv_kernel_to_patches

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0; CALIB = 1024; CHUNK = 256; PHI_BATCH = 128
N_EVAL = 64; M_CHECK = 8; SCALE = 25.0; LAM = 1e-3; K = 20; SI, BI = 3, 0


def spearman(a, b):
    if len(a) < 3: return float("nan")
    r, _ = stats.spearmanr(a, b); return float(r)


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE], n_directions=K,
                           n_perturbations=50, subset_size=CALIB, chunk_size=CHUNK, target_stage_idx=SI,
                           target_block_idx=BI, random_directions=True, seed=SEED, lambda_reg=LAM,
                           posthoc_calibrate=True, bootstrap_frac=0.0)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])
    stages = [model.stage1, model.stage2, model.stage3, model.stage4]
    blk = stages[SI][BI]
    w2_orig = np.asarray(blk.conv2.kernel[...], np.float32)
    Theta = np.asarray(flatten_conv_kernel_to_patches(jnp.asarray(w2_orig)), np.float64)
    cout = w2_orig.shape[-1]

    ens, _ = _build_single_block_pnc_ensemble(nnx.clone(model), x_tr, target_stage_idx=SI, target_block_idx=BI,
        n_directions=K, n_perturbations=50, perturbation_scale=SCALE, subset_size=CALIB, chunk_size=CHUNK,
        lambda_reg=LAM, random_directions=True, seed=SEED, bootstrap_frac=0.0, bootstrap_seed=SEED)

    def U_f32(w1, h):
        out = jax.nn.relu(blk.bn1(jnp.asarray(h, jnp.float32), use_running_average=True))
        y = jax.lax.conv_general_dilated(lhs=out, rhs=jnp.asarray(w1, jnp.float32).transpose(3, 2, 0, 1),
            window_strides=tuple(blk.conv1.strides), padding="SAME", dimension_numbers=("NHWC", "OIHW", "NHWC"))
        return jax.nn.relu(blk.bn2(y, use_running_average=True))

    # downstream function: block OUTPUT (post add) -> logits. Runs remaining blocks in this stage,
    # then later stages, final_bn, relu, GAP, fc. (frozen BN)
    def downstream(hb):
        h = hb
        for j in range(BI + 1, len(stages[SI])):
            h = stages[SI][j](h, use_running_average=True)
        for s in range(SI + 1, 4):
            for b2 in range(len(stages[s])):
                h = stages[s][b2](h, use_running_average=True)
        h = model.final_bn(h, use_running_average=True); h = jax.nn.relu(h)
        h = jnp.mean(h, axis=(1, 2)); return model.fc(h)

    @nnx.jit
    def base_block_out_and_logits(x):
        # original block output (post add) and base logits
        h = model.stem(x)
        for s in range(4):
            for b2 in range(len(stages[s])):
                if s == SI and b2 == BI:
                    hb = stages[s][b2](h, use_running_average=True)   # original block output
                    return hb, downstream(hb)
                h = stages[s][b2](h, use_running_average=True)

    def jvp_down(hb0, tangent):
        _, jz = jax.jvp(downstream, (hb0,), (tangent,))
        return np.asarray(jz, np.float64)

    rng = np.random.RandomState(7)
    def take(x):
        return np.asarray(x)[rng.choice(len(x), min(N_EVAL, len(x)), replace=False)]
    evalsets = {"id_test": ("id", take(benchmark["id_test"]["inputs"])),
                "cifar100": ("near", take(benchmark["near_ood"]["cifar100"]["inputs"])),
                "svhn": ("far", take(benchmark["far_ood"]["svhn"]["inputs"]))}

    results = {}
    for name, (regime, X) in evalsets.items():
        h_in = compute_cifar_block_preacts(model, jnp.asarray(X), CHUNK, SI, BI, jnp.asarray(blk.conv1.kernel[...]))[0][0]
        hb0, base_logits = base_block_out_and_logits(jnp.asarray(X))
        hb0 = np.asarray(hb0, np.float64); base_logits = np.asarray(base_logits, np.float64)
        phi0 = np.asarray(extract_patches(U_f32(np.asarray(blk.conv1.kernel[...]), h_in), k=3, strides=1), np.float64)
        N = len(X); Ho = hb0.shape[1]; Wo = hb0.shape[2]

        # all member logits once (M=50) for disagreement targets; first M_CHECK for the bridge
        all_logits = np.asarray(ens.predict(jnp.asarray(X)), np.float64)   # (M,N,C)
        stat_names = ["frob", "meanpatch", "maxpatch", "p95patch", "chan", "gap", "jproj"]
        per_img = {s: np.zeros((min(M_CHECK, len(ens.members_w1)), N)) for s in stat_names}
        cos_list, relmag_list = [], []
        dl_norm = np.zeros((min(M_CHECK, len(ens.members_w1)), N))
        Rfrob = np.zeros_like(dl_norm)
        for mi in range(min(M_CHECK, len(ens.members_w1))):
            w1p = np.asarray(ens.members_w1[mi], np.float32)
            w2c = np.asarray(ens.members_w2[mi][0], np.float64); b2c = np.asarray(ens.members_w2[mi][1], np.float64)
            phiv = np.asarray(extract_patches(U_f32(w1p, h_in), k=3, strides=1), np.float64)
            onse = np.ones((phiv.shape[0], 1))
            Theta_v_aug = np.concatenate([b2c, np.asarray(w2c).transpose(2, 0, 1, 3).reshape(-1, cout)], axis=0)
            Rflat = np.concatenate([onse, phiv], 1) @ Theta_v_aug - phi0 @ Theta   # (N*Ho*Wo, cout)
            R = Rflat.reshape(N, Ho, Wo, cout)
            # member logits (exact) via ensemble forward
            dl = all_logits[mi] - base_logits                                       # actual logit change
            # first-order downstream on R (batched jvp over images)
            jz = jvp_down(jnp.asarray(hb0, jnp.float32), jnp.asarray(R, jnp.float32))
            for j in range(N):
                a = dl[j]; b = jz[j]
                cos_list.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))
                relmag_list.append(float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30)))
                dl_norm[mi, j] = np.linalg.norm(a); Rfrob[mi, j] = np.linalg.norm(R[j])
                Rj = R[j]
                pnorm = np.linalg.norm(Rj.reshape(Ho * Wo, cout), axis=1)  # per-patch norm
                per_img["frob"][mi, j] = np.linalg.norm(Rj)
                per_img["meanpatch"][mi, j] = pnorm.mean()
                per_img["maxpatch"][mi, j] = pnorm.max()
                per_img["p95patch"][mi, j] = np.percentile(pnorm, 95)
                per_img["chan"][mi, j] = np.linalg.norm(np.linalg.norm(Rj.reshape(Ho * Wo, cout), axis=0))
                per_img["gap"][mi, j] = np.linalg.norm(Rj.mean(axis=(0, 1)))
                per_img["jproj"][mi, j] = np.linalg.norm(jz[j])

        allm = all_logits  # (M,N,C), full ensemble
        # per-image ensemble disagreement targets
        logit_cov_tr = np.array([np.trace(np.cov(allm[:, j, :].T)) for j in range(N)])
        p = softmax(allm); mp = p.mean(0); eps = 1e-12
        pred_ent = -(mp * np.log(mp + eps)).sum(-1)
        mi_arr = pred_ent - (-(p * np.log(p + eps)).sum(-1).mean(0))
        # correlate each local statistic (member-averaged) with targets
        stat_corr = {}
        for s in stat_names:
            sm = per_img[s].mean(0)  # member-mean per image
            stat_corr[s] = {"spearman_logit_cov_tr": spearman(sm, logit_cov_tr),
                            "spearman_MI": spearman(sm, mi_arr),
                            "spearman_pred_entropy": spearman(sm, pred_ent)}
        results[name] = {
            "regime": regime, "n_eval": N, "m_check": int(min(M_CHECK, len(ens.members_w1))),
            "downstream_cosine_mean": float(np.mean(cos_list)), "downstream_cosine_median": float(np.median(cos_list)),
            "downstream_relmag_mean": float(np.mean(relmag_list)), "downstream_relmag_median": float(np.median(relmag_list)),
            "spearman_Rfrob_vs_dlogit": spearman(Rfrob.mean(0), dl_norm.mean(0)),
            "stat_vs_disagreement_spearman": stat_corr,
        }
        print(f"[l2o] {name:9} cos(dl,J.R)={results[name]['downstream_cosine_median']:.3f} "
              f"relmag={results[name]['downstream_relmag_median']:.2f} "
              f"Sp(||R||,||dl||)={results[name]['spearman_Rfrob_vs_dlogit']:.2f} | "
              f"best stat->MI: " + max(stat_corr, key=lambda s: abs(stat_corr[s]['spearman_MI'])), flush=True)

    meta = {"seed": SEED, "block": f"s{SI}b{BI}", "scale": SCALE, "calib": CALIB, "m_check": M_CHECK,
            "n_eval": N_EVAL, "runtime_sec": time.time() - t0}
    json.dump({"meta": meta, "results": results}, open(OUT / "local_to_output_raw.json", "w"), indent=2)
    print(f"[l2o] wrote local_to_output_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
