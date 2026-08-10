#!/usr/bin/env python3
"""Phase 1 (Sections 1.2-1.4): exact convolutional finite-scale identity validation.

Stage-A smoke test: base seed-0 checkpoint, blocks {s3b0, s3b1}, a few members,
small calibration + eval sets, ID + a couple OOD datasets.

Feature extraction runs in the network's NATIVE float32 (so the identity is tested
against the activations the network actually uses); all reference linear algebra
(ridge solve, closed-form C1, residual formulas) is done in numpy float64.

Tests
  1.2 patch operator: extract_patches(U(x)) @ flatten(w2_orig)  ==  real conv2(U(x))   [pre-residual]
  1.3 correction identity C1:  implemented (w2,b)  vs  independent f64 ridge  vs  closed-form
         Theta_v = Theta - G_v^{-1} X_v^T dX_v Theta
  1.4 residual identity C2-C4:  real network conv2 residual  vs  patch formula [1,Phi_v]Theta_v - Phi Theta
         and vs  G_v(x) Theta   (closed-form C4, f64)
Also confirms block-output residual (post add) == conv2 residual (shortcut cancels): checked separately.
Writes theorem_validation_raw.json.
"""
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, json, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import pickle

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO); sys.path.insert(0, str(REPO))
from models import PreActResNet18
from cifar_tasks import (_load_cifar_openood_context, _build_single_block_pnc_ensemble,
                         compute_cifar_block_preacts, CIFAROpenOODPnC)
from pnc import extract_patches

OUT = REPO / "results" / "neurips_2026_rebuttal" / "cifar"
SEED = 0
CALIB = 1024         # submitted operating point (overdetermined at s3: 16384 rows > p=4609)
CHUNK = 256
PHI_BATCH = 128      # image batch for patch extraction (GPU memory)
N_EVAL = 64
N_MEMBERS_CHECK = 3
BLOCKS = [(3, 0), (3, 1)]
LAM = 1e-3
SCALE = 25.0
K = 20


def relerr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def flatten_kernel_np(w):
    """HWIO -> (Cin*kh*kw, Cout) matching extract_patches order (Cin, kh, kw)."""
    return np.asarray(w, np.float64).transpose(2, 0, 1, 3).reshape(-1, w.shape[-1])


def unflatten_kernel_np(wf, shape):
    kh, kw, cin, cout = shape
    return wf.reshape(cin, kh, kw, cout).transpose(1, 2, 0, 3)


def main():
    t0 = time.time()
    task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[SCALE],
                           n_directions=K, n_perturbations=50, subset_size=CALIB, chunk_size=CHUNK,
                           target_stage_idx=3, target_block_idx=0, random_directions=True,
                           seed=SEED, lambda_reg=LAM, posthoc_calibrate=True, bootstrap_frac=0.0)
    benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(task)
    model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(SEED))
    with open(task.input().path, "rb") as f:
        nnx.update(model, pickle.load(f)["state"])

    rng = np.random.RandomState(7)
    def take(x):
        return np.asarray(x)[rng.choice(len(x), min(N_EVAL, len(x)), replace=False)]
    evalsets = {"id_test": take(benchmark["id_test"]["inputs"]),
                "cifar100": take(benchmark["near_ood"]["cifar100"]["inputs"]),
                "svhn": take(benchmark["far_ood"]["svhn"]["inputs"])}

    results = {}
    for (si, bi) in BLOCKS:
        name = f"s{si}b{bi}"
        stages = [model.stage1, model.stage2, model.stage3, model.stage4]
        blk = stages[si][bi]
        w1_f32 = np.asarray(blk.conv1.kernel[...], np.float32)
        w2_f32 = np.asarray(blk.conv2.kernel[...], np.float32)
        Theta = flatten_kernel_np(w2_f32)                       # (p-1, Cout) f64
        cout = w2_f32.shape[-1]

        def U_f32(w1, h):  # native float32 post-bn2-relu feature map
            out = jax.nn.relu(blk.bn1(jnp.asarray(h, jnp.float32), use_running_average=True))
            y = jax.lax.conv_general_dilated(lhs=out, rhs=jnp.asarray(w1, jnp.float32).transpose(3, 2, 0, 1),
                window_strides=tuple(blk.conv1.strides), padding="SAME",
                dimension_numbers=("NHWC", "OIHW", "NHWC"))
            return jax.nn.relu(blk.bn2(y, use_running_average=True))
        def Phi(w1, h):
            outs = []
            for i in range(0, len(h), PHI_BATCH):
                outs.append(np.asarray(extract_patches(U_f32(w1, h[i:i+PHI_BATCH]), k=3, strides=1), np.float64))
            return np.concatenate(outs, axis=0)
        def conv2_real(U):  # real network conv2 (float32) -> f64
            return np.asarray(jax.lax.conv_general_dilated(
                lhs=jnp.asarray(U, jnp.float32), rhs=jnp.asarray(w2_f32.transpose(3, 2, 0, 1)),
                window_strides=(1, 1), padding="SAME", dimension_numbers=("NHWC", "OIHW", "NHWC")), np.float64)

        rec = {"block": name, "stage": si, "block_idx": bi, "p": int(Theta.shape[0] + 1), "C_out": int(cout)}

        # --- 1.2 patch operator ---
        h_probe = compute_cifar_block_preacts(model, jnp.asarray(evalsets["id_test"][:16]), CHUNK, si, bi,
                                              jnp.asarray(w1_f32))[0][0]
        U0 = U_f32(w1_f32, h_probe)
        conv_real = conv2_real(np.asarray(U0))
        conv_patch = (np.asarray(extract_patches(U0, k=3, strides=1), np.float64) @ Theta).reshape(conv_real.shape)
        rec["patch_op_relerr"] = relerr(conv_patch, conv_real)
        rec["patch_op_maxabs"] = float(np.max(np.abs(conv_patch - conv_real)))

        # --- build anchor ensemble (bootstrap off for a clean identity check) ---
        ens, _ = _build_single_block_pnc_ensemble(
            nnx.clone(model), x_tr, target_stage_idx=si, target_block_idx=bi,
            n_directions=K, n_perturbations=50, perturbation_scale=SCALE, subset_size=CALIB,
            chunk_size=CHUNK, lambda_reg=LAM, random_directions=True, seed=SEED,
            bootstrap_frac=0.0, bootstrap_seed=SEED)
        cal_h = np.concatenate([np.asarray(c) for c in ens.chunks], axis=0)
        T_orig = np.concatenate([np.asarray(t) for t in ens.T_orig_chunks], axis=0).reshape(-1, cout).astype(np.float64)
        X = Phi(w1_f32, cal_h)
        ones = np.ones((X.shape[0], 1))
        Xaug = np.concatenate([ones, X], axis=1)
        Theta_aug = np.concatenate([np.zeros((1, cout)), Theta], axis=0)   # orig augmented params (bias 0)

        member_recs = []
        for mi in range(min(N_MEMBERS_CHECK, len(ens.members_w1))):
            w1p = np.asarray(ens.members_w1[mi], np.float32)
            w2_code = np.asarray(ens.members_w2[mi][0], np.float64)
            b2_code = np.asarray(ens.members_w2[mi][1], np.float64)
            Xv = Phi(w1p, cal_h); Xvaug = np.concatenate([ones, Xv], axis=1)
            dXv = Xvaug - Xaug
            Dm = Xvaug.shape[1]
            Gv = Xvaug.T @ Xvaug + LAM * np.eye(Dm)
            # (a) f64 solve of the CODE's actual system (target = network T_orig) -> precision/conditioning check
            R_code = T_orig - Xv @ Theta
            Theta_delta_code = np.linalg.solve(Gv, Xvaug.T @ R_code)
            w2_ref = np.asarray(w2_f32, np.float64) + unflatten_kernel_np(Theta_delta_code[1:], w2_f32.shape)
            b2_ref = Theta_delta_code[0:1]
            cond_Gv = float(np.linalg.cond(Gv))
            # (b) closed-form C1 with self-consistent recomputed target XTheta -> exact-algebra check
            Theta_v_cf = Theta_aug - np.linalg.solve(Gv, Xvaug.T @ (dXv @ Theta_aug))
            R_alg = -(dXv @ Theta_aug)                                  # = XTheta - XvTheta (recomputed)
            Theta_delta_alg = np.linalg.solve(Gv, Xvaug.T @ R_alg)
            algebra_selfcheck = relerr(Theta_delta_alg, Theta_v_cf - Theta_aug)
            Theta_v_code_aug = np.concatenate([b2_code, flatten_kernel_np(w2_code)], axis=0)

            resid = {}
            for dname, X_eval in evalsets.items():
                h_e = compute_cifar_block_preacts(model, jnp.asarray(X_eval), CHUNK, si, bi,
                                                  jnp.asarray(w1_f32))[0][0]
                Ue0 = U_f32(w1_f32, h_e); Uev = U_f32(w1p, h_e)
                phi_e = np.asarray(extract_patches(Ue0, k=3, strides=1), np.float64)
                phiv_e = np.asarray(extract_patches(Uev, k=3, strides=1), np.float64)
                onse = np.ones((phiv_e.shape[0], 1))
                phiv_aug = np.concatenate([onse, phiv_e], 1)
                # real network residual (corrected conv2 + bias) - (orig conv2)
                real_corr = np.asarray(jax.lax.conv_general_dilated(
                    lhs=jnp.asarray(Uev, jnp.float32), rhs=jnp.asarray(w2_code.transpose(3, 2, 0, 1), jnp.float32),
                    window_strides=(1, 1), padding="SAME",
                    dimension_numbers=("NHWC", "OIHW", "NHWC")), np.float64) + b2_code.reshape(1, 1, 1, cout)
                real_orig = conv2_real(np.asarray(Ue0))
                R_real = real_corr - real_orig
                # patch formula with CODE params vs real network forward (float32 precision check)
                R_patch = (phiv_aug @ Theta_v_code_aug - phi_e @ Theta).reshape(R_real.shape)
                # closed-form C4 vs patch-formula with self-consistent closed-form params (exact-algebra check)
                dphiv = phiv_aug - np.concatenate([onse, phi_e], 1)
                R_c4 = (dphiv @ Theta_aug - phiv_aug @ np.linalg.solve(Gv, Xvaug.T @ (dXv @ Theta_aug))).reshape(R_real.shape)
                R_patch_cf = (phiv_aug @ Theta_v_cf - phi_e @ Theta).reshape(R_real.shape)
                N = R_real.shape[0]
                perimg = [relerr(R_patch[j], R_real[j]) for j in range(N)]
                resid[dname] = {
                    "residual_vs_realforward_relerr": relerr(R_patch, R_real),
                    "residual_vs_realforward_maxabs": float(np.max(np.abs(R_patch - R_real))),
                    "C4_vs_patch_algebra_relerr": relerr(R_c4, R_patch_cf),
                    "perimg_rel_median": float(np.median(perimg)),
                    "perimg_rel_max": float(np.max(perimg)),
                    "n_eval": int(N),
                }
            member_recs.append({
                "member": mi,
                "C1_code_vs_f64_kernel_relerr": relerr(w2_code, w2_ref),
                "C1_code_vs_f64_bias_relerr": relerr(b2_code, b2_ref),
                "C1_ridge_vs_closedform_algebra_relerr": algebra_selfcheck,
                "cond_Gv": cond_Gv,
                "residual_identity": resid,
            })
        rec["members"] = member_recs
        rec["calib_rows"] = int(Xaug.shape[0])
        rec["nominal_n_over_p"] = float(Xaug.shape[0] / rec["p"])
        results[name] = rec
        m0 = member_recs[0]
        print(f"[thm] {name}: p={rec['p']} rows={rec['calib_rows']} n/p={rec['nominal_n_over_p']:.2f} "
              f"cond(Gv)={m0['cond_Gv']:.1e} | patch_op={rec['patch_op_relerr']:.1e} | "
              f"C1 code-vs-f64={m0['C1_code_vs_f64_kernel_relerr']:.1e} algebra={m0['C1_ridge_vs_closedform_algebra_relerr']:.1e} | "
              f"resid(id)={m0['residual_identity']['id_test']['residual_vs_realforward_relerr']:.1e} "
              f"C4alg={m0['residual_identity']['id_test']['C4_vs_patch_algebra_relerr']:.1e}", flush=True)

    meta = {"seed": SEED, "calib_images": CALIB, "n_eval_per_dataset": N_EVAL,
            "members_checked": N_MEMBERS_CHECK, "blocks": [f"s{s}b{b}" for s, b in BLOCKS],
            "lambda": LAM, "scale": SCALE, "K": K, "feature_dtype": "float32(native)",
            "reference_dtype": "float64(numpy)", "runtime_sec": time.time() - t0}
    json.dump({"meta": meta, "results": results}, open(OUT / "theorem_validation_raw.json", "w"), indent=2)
    print(f"[thm] wrote theorem_validation_raw.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
