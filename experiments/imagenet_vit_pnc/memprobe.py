"""CUDA/CPU memory + timing probes with incremental, OOM-tolerant CSV recording.

Every preflight measurement goes through :func:`probe`, which resets CUDA peak
counters, times the block, and appends one row to ``memory_preflight.csv`` even
when the body raises. An OOM is recorded as ``status=OOM`` and the sweep
continues (spec section 5: "An OOM must be recorded rather than crashing the
entire sweep").
"""
from __future__ import annotations

import csv
import gc
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import torch

GIB = 1024.0 ** 3

COLUMNS = [
    "test", "batch_size", "dtype", "target_block", "member_count", "token_mode",
    "current_allocated_gib", "peak_allocated_gib", "peak_reserved_gib",
    "nvsmi_used_gib", "cpu_rss_gib", "wall_time_s", "status", "notes",
]


def reset_cuda() -> None:
    """Drop cached blocks and zero the peak counters before a measurement."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cuda_stats() -> dict:
    if not torch.cuda.is_available():
        return dict.fromkeys(
            ("current_allocated_gib", "peak_allocated_gib", "peak_reserved_gib"), 0.0)
    torch.cuda.synchronize()
    return {
        "current_allocated_gib": torch.cuda.memory_allocated() / GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }


def nvsmi_used_gib() -> float:
    """Whole-GPU used memory per nvidia-smi (includes other processes; -1 on failure)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return -1.0


def cpu_rss_gib() -> float:
    """Resident set size of this process, from /proc (psutil-free)."""
    try:
        with open(f"/proc/{os.getpid()}/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / GIB
    except Exception:
        return -1.0


def cpu_peak_rss_gib() -> float:
    """Peak RSS (VmHWM) of this process, in GiB; -1 if unavailable."""
    try:
        for line in open(f"/proc/{os.getpid()}/status"):
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return -1.0


class Recorder:
    """Append-only CSV writer; flushes each row so partial sweeps survive a crash."""

    def __init__(self, path: str | Path, columns=COLUMNS):
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as fh:
                csv.DictWriter(fh, self.columns).writeheader()

    def write(self, **row) -> dict:
        full = {c: row.get(c, "") for c in self.columns}
        extra = set(row) - set(self.columns)
        if extra:
            raise KeyError(f"unknown columns {sorted(extra)} for {self.path.name}")
        with self.path.open("a", newline="") as fh:
            csv.DictWriter(fh, self.columns).writerow(full)
        return full


@contextmanager
def probe(recorder: Recorder, test: str, **meta):
    """Measure one test. Yields a dict the body may update with ``notes``/extra fields.

    Records ``status`` = ok | OOM | ERROR. OOM and other exceptions are swallowed
    so a sweep can continue; the caller inspects ``result['status']`` to decide
    whether to stop increasing batch size.
    """
    reset_cuda()
    result: dict = {"status": "ok", "notes": ""}
    t0 = time.perf_counter()
    try:
        yield result
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "OOM"
        result["notes"] = (result.get("notes", "") + f" | {type(exc).__name__}").strip(" |")
    except RuntimeError as exc:
        oom = "out of memory" in str(exc).lower()
        result["status"] = "OOM" if oom else "ERROR"
        result["notes"] = (result.get("notes", "") + f" | {str(exc)[:160]}").strip(" |")
    except Exception as exc:  # noqa: BLE001 - preserve the measurement, not the traceback
        result["status"] = "ERROR"
        result["notes"] = (result.get("notes", "") + f" | {type(exc).__name__}: "
                           f"{str(exc)[:140]}").strip(" |")
    finally:
        wall = time.perf_counter() - t0
        stats = cuda_stats()
        row = {**meta, **stats, "test": test,
               "nvsmi_used_gib": round(nvsmi_used_gib(), 4),
               "cpu_rss_gib": round(cpu_rss_gib(), 4),
               "wall_time_s": round(wall, 5),
               "status": result["status"], "notes": result.get("notes", "")}
        for key in ("current_allocated_gib", "peak_allocated_gib", "peak_reserved_gib"):
            row[key] = round(row[key], 4)
        # let the body override measured fields (e.g. a per-forward wall time)
        for key in list(result):
            if key in recorder.columns and key not in ("status", "notes"):
                row[key] = result[key]
        recorder.write(**row)
        result["row"] = row
        if result["status"] == "OOM":
            reset_cuda()


def timed(fn, warmup: int = 3, iters: int = 20):
    """Median/mean wall time of ``fn`` in seconds, with CUDA sync around each call."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    n = len(times)
    median = times[n // 2] if n % 2 else 0.5 * (times[n // 2 - 1] + times[n // 2])
    return {"median_s": median, "mean_s": sum(times) / n,
            "min_s": times[0], "max_s": times[-1], "iters": n}
