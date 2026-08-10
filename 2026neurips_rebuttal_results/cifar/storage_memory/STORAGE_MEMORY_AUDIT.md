# CIFAR storage & memory — recovered & classified

Quantities kept separate: **persistent disk** vs **resident GPU** vs **peak GPU** vs **temporary workspace** vs **materialized-ensemble** size. Classes: MEASURED / DERIVED / ESTIMATED / UPPER_BOUND / MISSING.

## Persistent disk
| Quantity | Value | Class | Source |
|---|---|---|---|
| Base checkpoint (.pkl) | 44,729,016 B = 42.66 MiB | MEASURED | `ls -l results/cifar10/preact_resnet18_..._seed0.pkl` |
| P&C persisted on disk | 42.66 MiB (no P&C ckpt — rebuilt each eval from base) | MEASURED/DERIVED | `CIFAR_EFFICIENCY_ACCOUNTING.md` §3.3 |
| P&C materialized 50 members (analytic) | 675.1 MiB (SUPERSEDED) / compact conv2-only 450.1 MiB | DERIVED | §3.3 |
| Deep Ensemble n=5 / n=50 disk | 213.3 MiB / 2,132.9 MiB | DERIVED (n×base) | §3.2 |
| SWAG checkpoint | 980.27 MiB (22.98× base) | MEASURED | §3.1 |
| SCOD serialized sketch | 893,776,858 B ≈ 852 MiB | MEASURED | `scod_profile.json` |
| Epinet head | 0.25 MiB | MEASURED | §3.1 |

## Resident / peak GPU (MEASURED — separate `peak_memory/` run, 2026-07-26, PREALLOCATE=false, 1 run seed0)
| Method | resident MiB | peak MiB | Source (`peak_memory/`) |
|---|---|---|---|
| Single model | — | — | `peak_memory_single_model.json` |
| **P&C s3b0 anchor** | 1,062.7 | 4,332.2 | `peak_memory_pnc_anchor_s3b0.json` |
| P&C s3b1 | 1,329.0 | — | `peak_memory_pnc_single_s3b1.json` |
| P&C multi-block | 4,737.2 | 6,063.3 | `peak_memory_pnc_multi.json` |
| Deep Ensemble n=5 | 226.9 | 2,053.5 | `peak_memory_deep_ensemble.json` |
| Deep Ensemble n=50 | 2,250.3 | 4,071.8 | `peak_memory_deep_ensemble_50.json` |
| SWAG | 1,070.7 | 5,360.1 | `peak_memory_swag.json` |
| LLLA | 245.0 | 2,096.7 (build 1,922.9) | `peak_memory_llla.json` |
| SCOD (online) | 7,308–7,310 | 7,308–7,310 | `scod_profile.json` (online block) |

## SCOD build memory preflight (ESTIMATE — labeled `est_`)
est_peak_gpu 4.6 GB, est_peak_cpu 7.14 GB, est_sketch 5.54 GB — `scod_profile.json` + `seed{0,1,2}_memory_preflight.json`. SCOD online peak GPU (7,308 MiB) IS measured.

## Notes / caveats
- Peak/resident GPU memory **WAS measured** in the dedicated `peak_memory/` run for 11 methods, even though `efficiency_cifar_table.md`'s own footer disclaims it ("Peak GPU memory was not captured in the cached run"). The separate run is authoritative; the table simply wasn't re-rendered.
- MISSING: P&C reconstruction-time-vs-memory tradeoff line-items; LLLA internal memory split. Temporary-workspace sizes are not separated from peak in the per-method JSONs (peak includes workspace).
- Environment: RTX 5060, 8,151 MiB total (6,112.9 MiB visible under PREALLOCATE=false), WSL2, JAX 0.9.1/Flax 0.12.3.
