#!/usr/bin/env bash
# Run benchmark_inference_cost.py once per method as a separate subprocess.
# This avoids GPU memory accumulation across methods that caused segfaults
# when running everything in a single Python process.
set -euo pipefail

PYTHON=".venv/bin/python"
OUT_DIR="results/cifar10/inference_cost"
mkdir -p "$OUT_DIR"

METHODS=(single_model mahalanobis react mc_dropout deep_ensemble swag llla epinet pnc_single pnc_multi)

for METHOD in "${METHODS[@]}"; do
  echo "============================================================"
  echo "  Inference cost: $METHOD"
  echo "============================================================"
  $PYTHON scripts/benchmark_inference_cost.py --method "$METHOD" --out-dir "$OUT_DIR"
done

echo
echo "All methods done. Aggregating into single JSON..."
$PYTHON -c "
import json
from pathlib import Path
out_dir = Path('$OUT_DIR')
combined = {'methods': {}}
meta_keys = ('n_bench_samples', 'batch_size', 'seed')
for f in sorted(out_dir.glob('inference_cost_*.json')):
    with open(f) as fp:
        d = json.load(fp)
    for k in meta_keys:
        if k in d:
            combined.setdefault(k, d[k])
    combined['methods'].update(d.get('methods', {}))
out = Path('results/cifar10/inference_cost.json')
with open(out, 'w') as fp:
    json.dump(combined, fp, indent=2)
print(f'Wrote {out}')
print()
print('| Method | Train cost (×) | Fwd passes | Warm ms/sample | Cold warmup (s) |')
print('|---|---:|---:|---:|---:|')
for name, r in combined['methods'].items():
    print(f\"| {name} | {r['train_cost_factor']:.2f}× | {r['n_forward_passes']} | {r['warm_per_sample_ms']:.2f} | {r['cold_warmup_s']:.2f} |\")
"

echo "Done."
