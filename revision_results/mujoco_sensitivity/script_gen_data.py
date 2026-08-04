"""Generate id_train/id_eval/ood_far npz for the suite-wide Far-OOD study.

Writes results/<env>/data_<tier>_seed<S>_steps10000.npz matching the harness path
convention. id from Minari expert; ood_far = live random-action rollout. Skips
tiers that already exist. Far-focused study needs only these 3 tiers.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))
from data import load_minari_transitions  # noqa: E402

STEPS = 10000
# tier -> regime passed to loader (id_train/id_eval both use expert, disjoint by seed slice)
TIERS = {"id_train": "id", "id_eval": "id", "ood_far": "ood_far"}


def gen(env, seed):
    base = _ROOT / "results" / env
    base.mkdir(parents=True, exist_ok=True)
    for i, (tier, regime) in enumerate(TIERS.items()):
        out = base / f"data_{tier}_seed{seed}_steps{STEPS}.npz"
        if out.exists():
            print(f"  [skip] {out.name}"); continue
        inp, tgt, meta = load_minari_transitions(env, regime, n_steps=STEPS, seed=seed + i)
        np.savez(out, inputs=np.asarray(inp), targets=np.asarray(tgt))
        meta.update(environment=env, regime=tier, seed=seed + i)
        (Path(str(out) + ".json")).write_text(json.dumps(meta, indent=2))
        print(f"  [wrote] {out.name}  inputs={np.asarray(inp).shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True); ap.add_argument("--seeds", default="0,10,42")
    a = ap.parse_args()
    for s in [int(x) for x in a.seeds.split(",")]:
        print(f"=== {a.env} seed {s} ===")
        gen(a.env, s)
