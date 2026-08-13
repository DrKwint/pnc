"""ViT-B/16 P&C adapter: pinned checkpoint, cached prefix, and per-member FFN tail.

torchvision's ``EncoderBlock.forward`` is

    x = x_in + dropout(self_attention(ln_1(x_in)))      # post-attention residual
    out = x + mlp(ln_2(x))                              # FFN + residual

and ``VisionTransformer.forward`` finishes with ``encoder.ln`` -> ``x[:, 0]`` -> ``heads``.
The P&C target is the final block's FFN:

    h = ln_2(x)                    FFN input            (B, T, 768)
    y = gelu(h @ W1 + b1)          post-GELU            (B, T, 3072)   [W1 PERTURBED]
    z = y @ W2 + b2                FFN output           (B, T, 768)    [W2 CORRECTED]

Weights are exposed in the *code layout* used by the Banking77 DistilBERT
experiment (``y = h @ W1 + b1``, i.e. the transpose of ``nn.Linear.weight``) so the
existing basis / ridge conventions apply unchanged.

Prefix/tail split: the prefix runs the patch embedding and every block up to the
target block's post-attention residual; the tail runs the mutated FFN and the
remaining network. Everything after the final block's FFN is token-wise
(``encoder.ln``) followed by a CLS slice, so for the *final* block the tail only
needs the CLS row -- ``cls_only=True``. That shortcut is verified against the
full-token tail in ``validate.py`` and is NOT valid for earlier blocks, where
later attention layers mix tokens.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16

# Pinned explicitly -- never ViT_B_16_Weights.DEFAULT (spec section 3).
WEIGHT_ENUM = ViT_B_16_Weights.IMAGENET1K_V1
DIN, DHID = 768, 3072
N_BLOCKS = 12


@dataclass
class CheckpointInfo:
    weight_enum: str
    url: str
    cache_path: str
    sha256: str
    n_params: int
    n_classes: int
    transform: str


def checkpoint_info(model) -> CheckpointInfo:
    url = WEIGHT_ENUM.url
    cache = Path(torch.hub.get_dir()) / "checkpoints" / Path(url).name
    sha = "unavailable"
    if cache.exists():
        h = hashlib.sha256()
        with cache.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
    return CheckpointInfo(
        weight_enum=f"ViT_B_16_Weights.{WEIGHT_ENUM.name}",
        url=url,
        cache_path=str(cache),
        sha256=sha,
        n_params=sum(p.numel() for p in model.parameters()),
        n_classes=len(WEIGHT_ENUM.meta["categories"]),
        transform=str(WEIGHT_ENUM.transforms()).replace("\n", " "),
    )


class ViTPnCAdapter:
    """Base forward, activation capture, and member forward for one FFN target."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32,
                 block_index: int = N_BLOCKS - 1):
        self.device = torch.device(device)
        self.dtype = dtype
        self.block_index = block_index
        model = vit_b_16(weights=WEIGHT_ENUM)
        model.eval().to(self.device, dtype=dtype)
        for p in model.parameters():           # no gradients anywhere (spec section 3)
            p.requires_grad_(False)
        self.model = model
        self.enc = model.encoder
        self.block = self.enc.layers[block_index]
        self.mlp1, self.mlp2 = self.block.mlp[0], self.block.mlp[3]
        assert tuple(self.mlp1.weight.shape) == (DHID, DIN), self.mlp1.weight.shape
        assert tuple(self.mlp2.weight.shape) == (DIN, DHID), self.mlp2.weight.shape
        self.is_final = block_index == N_BLOCKS - 1
        # pristine copies for exact restoration after in-place mutation tests
        self._pristine = {
            "w1": self.mlp1.weight.detach().clone(), "b1": self.mlp1.bias.detach().clone(),
            "w2": self.mlp2.weight.detach().clone(), "b2": self.mlp2.bias.detach().clone(),
        }

    # ---- original target parameters, in code layout (y = h @ W1 + b1) ----
    @property
    def W1(self) -> torch.Tensor:                     # (768, 3072)
        return self.mlp1.weight.T

    @property
    def b1(self) -> torch.Tensor:                     # (3072,)
        return self.mlp1.bias

    @property
    def W2(self) -> torch.Tensor:                     # (3072, 768)
        return self.mlp2.weight.T

    @property
    def b2(self) -> torch.Tensor:                     # (768,)
        return self.mlp2.bias

    def theta(self) -> torch.Tensor:
        """[b2; W2] -> (3073, 768): the ridge prior, bias-first as in Banking77."""
        return torch.cat([self.b2[None, :], self.W2], 0)

    # ---- prefix: patch embed + blocks 0..k, up to the post-attention residual ----
    @torch.inference_mode()
    def prefix(self, images: torch.Tensor) -> torch.Tensor:
        """images (B,3,224,224) -> post-attention residual x at the target block (B,T,768)."""
        m = self.model
        x = m._process_input(images)
        x = torch.cat([m.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.enc.dropout(x + self.enc.pos_embedding)
        for i in range(self.block_index):
            x = self.enc.layers[i](x)
        blk = self.block
        a = blk.ln_1(x)
        a, _ = blk.self_attention(a, a, a, need_weights=False)
        return x + blk.dropout(a)

    # ---- tail: mutated FFN + rest of the network ----
    @torch.inference_mode()
    def tail(self, x_resid: torch.Tensor, W1=None, b1=None, W2=None, b2=None,
             cls_only: bool | None = None) -> torch.Tensor:
        """Logits from the cached residual. Member params override the originals."""
        if cls_only is None:
            cls_only = self.is_final
        if cls_only:
            if not self.is_final:
                raise ValueError("cls_only is only valid for the final block")
            x_resid = x_resid[:, :1]
        W1 = self.W1 if W1 is None else W1
        b1 = self.b1 if b1 is None else b1
        W2 = self.W2 if W2 is None else W2
        b2 = self.b2 if b2 is None else b2
        h = self.block.ln_2(x_resid)
        y = F.gelu(h @ W1 + b1)                       # exact-erf, matches nn.GELU()
        out = x_resid + (y @ W2 + b2)
        for i in range(self.block_index + 1, N_BLOCKS):
            out = self.enc.layers[i](out)
        return self.model.heads(self.enc.ln(out)[:, 0])

    @torch.inference_mode()
    def base_forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.tail(self.prefix(images))

    @torch.inference_mode()
    def features(self, x_resid: torch.Tensor, W1=None, b1=None, W2=None, b2=None,
                 cls_only: bool | None = None) -> torch.Tensor:
        """Penultimate CLS feature (after ``encoder.ln``, before ``heads``) -> (B, 768).

        This is the representation ReAct clips, and ``heads(features(...))`` reproduces
        :meth:`tail` exactly.
        """
        if cls_only is None:
            cls_only = self.is_final
        if cls_only:
            if not self.is_final:
                raise ValueError("cls_only is only valid for the final block")
            x_resid = x_resid[:, :1]
        W1 = self.W1 if W1 is None else W1
        b1 = self.b1 if b1 is None else b1
        W2 = self.W2 if W2 is None else W2
        b2 = self.b2 if b2 is None else b2
        h = self.block.ln_2(x_resid)
        out = x_resid + (F.gelu(h @ W1 + b1) @ W2 + b2)
        for i in range(self.block_index + 1, N_BLOCKS):
            out = self.enc.layers[i](out)
        return self.enc.ln(out)[:, 0]

    @torch.inference_mode()
    def head(self, feats: torch.Tensor) -> torch.Tensor:
        """Classifier head applied to penultimate features -> (B, 1000) logits."""
        return self.model.heads(feats)

    @torch.inference_mode()
    def ffn_triplet(self, x_resid: torch.Tensor, W1=None, b1=None):
        """(h, y, z) at the target FFN: h=ln_2(x), y=gelu(h@W1+b1), z=y@W2+b2 (original W2)."""
        W1 = self.W1 if W1 is None else W1
        b1 = self.b1 if b1 is None else b1
        h = self.block.ln_2(x_resid)
        y = F.gelu(h @ W1 + b1)
        return h, y, y @ self.W2 + self.b2

    # ---- in-place mutation / exact restoration (spec section 9) ----
    def set_W1_code(self, W1_code: torch.Tensor) -> None:
        """Write a code-layout (768,3072) W1 into the live module."""
        self.mlp1.weight.copy_(W1_code.T)

    def set_W2_code(self, W2_code: torch.Tensor, b2: torch.Tensor | None = None) -> None:
        self.mlp2.weight.copy_(W2_code.T)
        if b2 is not None:
            self.mlp2.bias.copy_(b2)

    def restore(self) -> None:
        """Restore the pristine checkpoint weights exactly (bitwise)."""
        self.mlp1.weight.copy_(self._pristine["w1"])
        self.mlp1.bias.copy_(self._pristine["b1"])
        self.mlp2.weight.copy_(self._pristine["w2"])
        self.mlp2.bias.copy_(self._pristine["b2"])

    def weights_are_pristine(self) -> bool:
        return (torch.equal(self.mlp1.weight, self._pristine["w1"])
                and torch.equal(self.mlp1.bias, self._pristine["b1"])
                and torch.equal(self.mlp2.weight, self._pristine["w2"])
                and torch.equal(self.mlp2.bias, self._pristine["b2"]))
