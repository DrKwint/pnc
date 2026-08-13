"""OpenOOD ImageNet-1k OOD sets, using OpenOOD's canonical image lists (spec §19).

Images come from the ``torch-uncertainty`` mirrors of the OpenOOD distributions, and the
membership of every benchmark split is taken from OpenOOD's own ``imglist`` files
(``torch-uncertainty/ood-datasets-splits``) rather than from whatever happens to be in a
zip. That matters: the OpenImage-O archive holds 17,632 images, of which OpenOOD's *test*
split is 15,869 and 1,763 belong to a validation split this experiment never touches
(we do not tune on OOD at all). Canonical test sizes:

    SSB-hard      49,000     near
    NINCO          5,879     near
    iNaturalist   10,000     far
    Textures       5,160     far
    OpenImage-O   15,869     far
    total         85,908

Images are read straight out of the zips — no extraction, no second copy on disk.
"""
from __future__ import annotations

import glob
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .vit_adapter import WEIGHT_ENUM

SPLITS_REPO = "torch-uncertainty/ood-datasets-splits"
DATASETS = {
    # key: (hf repo, zip name, OOD group)
    "ssb_hard":    ("SSB_hard",    "ssb_hard.zip",    "near"),
    "ninco":       ("Ninco",       "ninco.zip",       "near"),
    "inaturalist": ("inaturalist", "inaturalist.zip", "far"),
    "textures":    ("Texture",     "texture.zip",     "far"),
    "openimage_o": ("Openimage-O", "openimage_o.zip", "far"),
}
CANONICAL_N = {"ssb_hard": 49000, "ninco": 5879, "inaturalist": 10000,
               "textures": 5160, "openimage_o": 15869}


def _zip_path(repo: str, fn: str) -> str:
    from huggingface_hub import hf_hub_download
    cache = glob.glob(str(Path.home() / ".cache/huggingface/hub"
                          / f"datasets--torch-uncertainty--{repo}/snapshots/*/{fn}"))
    return cache[0] if cache else hf_hub_download(f"torch-uncertainty/{repo}", fn,
                                                  repo_type="dataset")


def canonical_list(key: str) -> list[str]:
    """OpenOOD's own test image list for one OOD dataset."""
    from huggingface_hub import hf_hub_download
    z = zipfile.ZipFile(hf_hub_download(SPLITS_REPO, "splits.zip", repo_type="dataset"))
    raw = z.read(f"splits/imagenet1k/test_{key}.txt").decode().splitlines()
    out = []
    for ln in raw:
        ln = ln.rstrip()
        if not ln:
            continue
        # lines are "<path> <label>", and 199 NINCO paths contain spaces
        # ("NINCO_OOD_classes/Caracal caracal caracal/..."), so split off only the
        # trailing label rather than at the first space.
        head, _, tail = ln.rpartition(" ")
        out.append(head if head and tail.lstrip("-").isdigit() else ln)
    return out


class OODZip:
    """Reads one OOD dataset's canonical test images directly from its zip archive."""

    def __init__(self, key: str, workers: int = 8):
        repo, fn, group = DATASETS[key]
        self.key, self.group = key, group
        self.zip_path = _zip_path(repo, fn)
        self._zf = zipfile.ZipFile(self.zip_path)
        self._names = set(self._zf.namelist())
        self.wanted = canonical_list(key)
        self._lower = None                      # lazily built basename index
        self.entries = [self._resolve(p) for p in self.wanted]
        missing = [w for w, e in zip(self.wanted, self.entries) if e is None]
        if missing:
            raise FileNotFoundError(
                f"{key}: {len(missing)}/{len(self.wanted)} canonical images not in "
                f"{Path(self.zip_path).name}; first missing {missing[:3]}")
        if len(self.entries) != CANONICAL_N[key]:
            raise ValueError(f"{key}: {len(self.entries)} images, expected "
                             f"{CANONICAL_N[key]}")
        self.transform = WEIGHT_ENUM.transforms()
        self._pool = ThreadPoolExecutor(max_workers=workers)

    def _resolve(self, p: str) -> str | None:
        """Map an OpenOOD list path onto this archive's entry names."""
        # NINCO lists are rooted at NINCO_OOD_classes/ and spell class directories with
        # spaces ("Caracal caracal caracal"); the archive drops the root and uses
        # underscores. Try the plain path, the archive's prefixes, and both spellings.
        bases = [p]
        if p.startswith("NINCO_OOD_classes/"):
            bases.append(p[len("NINCO_OOD_classes/"):])
        bases += [b.replace(" ", "_") for b in list(bases) if " " in b]
        cands = []
        for b in bases:
            cands += [b, f"images/{b}"]
        for c in cands:
            if c in self._names:
                return c
        # last resort: basename match, extension- and separator-insensitive
        if self._lower is None:
            self._lower = {}
            for n in self._names:
                if not n.endswith("/"):
                    self._lower.setdefault(Path(n).stem.lower().replace(" ", "_"), n)
        return self._lower.get(Path(p).stem.lower().replace(" ", "_"))

    def __len__(self) -> int:
        return len(self.entries)

    def _decode(self, name: str) -> torch.Tensor:
        with self._zf.open(name) as fh:
            blob = fh.read()
        return self.transform(Image.open(io.BytesIO(blob)).convert("RGB"))

    def iter_batches(self, batch_size: int, indices=None):
        idx = np.arange(len(self.entries)) if indices is None else np.asarray(indices)
        for s in range(0, len(idx), batch_size):
            chunk = idx[s:s + batch_size]
            imgs = list(self._pool.map(self._decode, [self.entries[i] for i in chunk]))
            yield torch.stack(imgs), torch.full((len(chunk),), -1, dtype=torch.long), chunk

    def ids(self, indices=None) -> list[str]:
        idx = range(len(self.wanted)) if indices is None else indices
        return [self.wanted[i] for i in idx]


def verify_all() -> dict:
    """Check every dataset resolves 100% of its canonical list. Cheap, no decoding."""
    out = {}
    for key in DATASETS:
        d = OODZip(key)
        out[key] = {"n": len(d), "expected": CANONICAL_N[key], "group": d.group,
                    "zip": Path(d.zip_path).name}
        print(f"  {key:<13} {len(d):>6} / {CANONICAL_N[key]:<6} {d.group:<5} OK", flush=True)
    return out


if __name__ == "__main__":
    verify_all()
