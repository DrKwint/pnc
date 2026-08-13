# Split protocol

Every construction, selection and calibration operation in this experiment draws from
ImageNet-1K **training** data. The official 50,000-image validation split is not read until
final evaluation, and never influences any hyperparameter.

## Pools

| pool | manifest | n | classes | per class | use |
|---|---|---|---|---|---|
| correction | `correction_32768.csv` | 32,768 | 1000 | 32–33 | fit the corrected W2 |
| selection | `selection_8192.csv` | 8,192 | 1000 | 8–9 | ID-only hyperparameter search |
| temperature | `temperature_8192.csv` | 8,192 | 1000 | 8–9 | shared scalar temperature |

The three pools are **pairwise disjoint by construction** (verified in
`full_validate.py::split_integrity`: 0 overlapping rows in all three pairs). A configuration
may use either 16,384 or 32,768 correction rows, drawn deterministically as a prefix of the
32,768-image correction pool.

## Determinism

Split seed: **20260813**.

1. The 28 cached training shards are indexed (label + original ImageNet filename for all
   122,024 rows; no image bytes are decoded).
2. Per-pool per-class counts are `floor(n/1000)`, with the remainder distributed over a
   seeded permutation of the 1000 class ids — so the extra images do not land systematically
   on low class indices.
3. Within each class, rows are permuted with `RandomState((seed*1000 + class) mod 2^32-1)`
   and the three pools are carved from disjoint consecutive segments of that permutation, in
   a fixed pool order.

Re-running `full_splits.py` on the same shards reproduces the manifests byte-for-byte.

## Manifest columns

`global_row, imagenet_path, class, split_seed`

`imagenet_path` is the original ILSVRC filename as distributed (e.g.
`n01440764_10026_n01440764.JPEG`), so a pool can be reconstructed against any copy of
ImageNet, not only this cache. `global_row` indexes the concatenated shard order used here.

## Checksums

| manifest | SHA-256 |
|---|---|
| `correction_32768.csv` | `08affc7e865e58cee7b3aa110e83072e861760b97894ee7d087c6c02d160b64d` |
| `selection_8192.csv` | `71edce32d39bd6a17a2a56d0993ba73d05fb05278764710428d38de8529078dd` |
| `temperature_8192.csv` | `3271a90afd4a7c3aebd6936e4083ef92dd69876e63416eb9bafb7e95e3e23ee2` |

Also recorded in `splits_summary.json`.

## Source

`ILSVRC/imagenet-1k` on the HuggingFace Hub — the gated, authentic ILSVRC-2012 JPEGs
(~0.63 bytes/pixel, median short side 375). 28 of 294 training shards were downloaded;
shards are shuffled rather than class-sorted, so this is a uniform sample of the training
set covering all 1000 classes with ≥50 images each.

The preflight's public mirror (`evanarlian/imagenet_1k_resized_256`) was **not** used here:
it re-encodes at ~0.26 bytes/pixel with the short side pre-scaled to 256, costing 1.85 pp
of base top-1, and would have left ID images measurably more compressed than the OOD sets —
a confound a detector could exploit.

## OOD

No OOD data appears in any pool above, and none was read before `selected_config.json` was
frozen and committed. OOD membership comes from OpenOOD's own `imglist` files rather than
from archive contents; see `../MANIFEST.md`.
