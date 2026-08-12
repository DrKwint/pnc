"""Record the exact machine / library / checkpoint inventory (spec sections 2 and 4)."""
from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch
import torchvision

from .vit_adapter import WEIGHT_ENUM, ViTPnCAdapter, checkpoint_info


def _run(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


def _first_match(text: str, pattern: str, default: str = "<unknown>") -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else default


def collect(load_model: bool = True) -> str:
    smi = _run("nvidia-smi")
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    meminfo = _run("grep MemTotal /proc/meminfo")
    cpuinfo = _run("grep -m1 'model name' /proc/cpuinfo")

    out = ["# ViT-B/16 P&C TITAN X preflight -- environment inventory", ""]
    out += ["## Hardware"]
    if props:
        out += [
            f"GPU model             : {props.name}",
            f"total VRAM            : {props.total_memory / 1024**3:.2f} GiB "
            f"({props.total_memory} bytes)",
            f"compute capability    : sm_{props.major}{props.minor}",
            f"multiprocessors       : {props.multi_processor_count}",
        ]
    else:
        out += ["GPU model             : <no CUDA device>"]
    out += [
        f"CPU model             : {_first_match(cpuinfo, r'model name\s*:\s*(.+)')}",
        f"CPU logical cores     : {_run('nproc')}",
        f"system RAM            : {_first_match(meminfo, r'MemTotal:\s*(.+)')}",
        "",
        "## Drivers / toolkit",
        f"NVIDIA driver         : "
        f"{_run('nvidia-smi --query-gpu=driver_version --format=csv,noheader')}",
        f"driver CUDA version   : {_first_match(smi, r'CUDA Version:\s*([0-9.]+)')}",
        f"torch CUDA runtime    : {torch.version.cuda}",
        f"torch cuDNN           : {torch.backends.cudnn.version()}",
        f"torch arch list       : "
        f"{torch.cuda.get_arch_list() if torch.cuda.is_available() else 'n/a'}",
        "",
        "## Python stack",
        f"Python                : {sys.version.split()[0]} ({platform.python_implementation()})",
        f"interpreter           : {sys.executable}",
        f"PyTorch               : {torch.__version__}",
        f"torchvision           : {torchvision.__version__}",
        f"numpy                 : {__import__('numpy').__version__}",
        f"scipy                 : {__import__('scipy').__version__}",
        f"platform              : {platform.platform()}",
        "",
        "## torch.cuda.get_device_properties(0)",
        f"{props}",
        "",
        "## nvidia-smi",
        "```",
        smi,
        "```",
    ]

    if load_model:
        ad = ViTPnCAdapter(device="cuda" if torch.cuda.is_available() else "cpu")
        info = checkpoint_info(ad.model)
        blk = f"encoder.layers.encoder_layer_{ad.block_index}"
        out += [
            "",
            "## Pinned checkpoint (spec section 3)",
            f"weight enum           : {info.weight_enum}",
            f"url                   : {info.url}",
            f"cache path            : {info.cache_path}",
            f"SHA-256               : {info.sha256}",
            f"parameters            : {info.n_params:,}",
            f"classes               : {info.n_classes}",
            f"preprocessing         : {info.transform}",
            f"reported metrics      : {WEIGHT_ENUM.meta['_metrics']['ImageNet-1K']}",
            "",
            "## P&C target (spec section 4)",
            f"target block          : {blk}",
            f"W1 [PERTURB]          : {blk}.mlp.0  nn.Linear weight "
            f"{tuple(ad.mlp1.weight.shape)} -> code layout {tuple(ad.W1.shape)}",
            f"activation            : {blk}.mlp.1  {type(ad.block.mlp[1]).__name__}"
            f"(approximate={getattr(ad.block.mlp[1], 'approximate', 'n/a')})",
            f"W2 [CORRECT]          : {blk}.mlp.3  nn.Linear weight "
            f"{tuple(ad.mlp2.weight.shape)} -> code layout {tuple(ad.W2.shape)}",
            f"||W1||_F              : {float(ad.W1.norm()):.4f}",
            f"||W2||_F              : {float(ad.W2.norm()):.4f}",
            f"correction dimension  : {ad.W2.shape[0]} + 1 bias = {ad.W2.shape[0] + 1}",
        ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/neurips_2026_rebuttal/imagenet_vit_preflight/"
                                     "environment.txt")
    args = ap.parse_args()
    text = collect()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
