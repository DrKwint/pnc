"""DistilBERT P&C adapter: cached-prefix + final-FFN tail (JAX, CPU-first).

The final transformer block (layer 5) is:
    sa_out = attention(x)                          # x = layer-4 output (prefix)
    h      = sa_layer_norm(sa_out + x)             # FFN INPUT (cached, token 0)
    z      = lin2(gelu(lin1(h)))                   # FFN output  (P&C target)
    out    = output_layer_norm(z + h)              # residual
    logits = classifier(relu(pre_classifier(out[:,0])))

Only token 0 reaches the head and no later block mixes tokens, so we cache h[:,0]
once per batch and each P&C member re-runs only the cheap FFN+head tail on it.
P&C perturbs lin1.kernel, recomputes y=gelu(lin1_v(h)), and corrects (lin2.kernel,
lin2.bias) to reproduce the original z on ID calibration activations.

All forward details (q/sqrt(64), -1e30 mask, exact-erf gelu, LayerNorm eps=1e-12)
mirror transformers.models.distilbert.modeling_flax_distilbert exactly; the
cached_tail_parity test verifies logits match the full model to < 1e-5.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes

CKPT = Path("results/banking77_distilbert_pnc/checkpoint")
L5 = "distilbert.transformer.layer.5"
N_HEADS, DIM = 12, 768
DPH = DIM // N_HEADS       # 64


def load_params():
    return from_bytes(None, (CKPT / "flax_params.msgpack").read_bytes())


def g(params, path):
    d = params
    for k in path.split("."):
        d = d[k]
    return jnp.asarray(d)


def _ln(x, scale, bias, eps=1e-12):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * scale + bias


def layer5_attention(x, mask, P):
    """Replicate layer-5 self-attention exactly. x:(B,T,768) mask:(B,T)->(B,T,768)."""
    B, T, _ = x.shape
    q = x @ g(P, f"{L5}.attention.q_lin.kernel") + g(P, f"{L5}.attention.q_lin.bias")
    k = x @ g(P, f"{L5}.attention.k_lin.kernel") + g(P, f"{L5}.attention.k_lin.bias")
    v = x @ g(P, f"{L5}.attention.v_lin.kernel") + g(P, f"{L5}.attention.v_lin.bias")
    sh = lambda t: t.reshape(B, T, N_HEADS, DPH).transpose(0, 2, 1, 3)
    q = sh(q) / jnp.sqrt(DPH); k = sh(k); v = sh(v)
    scores = q @ k.transpose(0, 1, 3, 2)                       # (B,H,T,T)
    m = mask.reshape(B, 1, 1, T).astype(scores.dtype)
    scores = scores - 1e30 * (1.0 - m)
    w = jax.nn.softmax(scores, axis=-1)
    ctx = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, DIM)
    return ctx @ g(P, f"{L5}.attention.out_lin.kernel") + g(P, f"{L5}.attention.out_lin.bias")


class DistilBertPnCAdapter:
    """Base forward, activation capture, and member forward for the final-FFN P&C."""

    def __init__(self, params=None, model=None):
        self.P = load_params() if params is None else params
        if model is None:
            import jax.numpy as _jnp
            from transformers import AutoConfig, FlaxDistilBertForSequenceClassification
            cfg = AutoConfig.from_pretrained(str(CKPT / "config.json"))
            model = FlaxDistilBertForSequenceClassification(cfg, dtype=_jnp.float32, _do_init=False)
        self.model = model
        # original final-FFN target parameters
        self.W1 = g(self.P, f"{L5}.ffn.lin1.kernel"); self.b1 = g(self.P, f"{L5}.ffn.lin1.bias")
        self.W2 = g(self.P, f"{L5}.ffn.lin2.kernel"); self.b2 = g(self.P, f"{L5}.ffn.lin2.bias")
        self.out_ln_s = g(self.P, f"{L5}.output_layer_norm.scale")
        self.out_ln_b = g(self.P, f"{L5}.output_layer_norm.bias")
        self.pc_w = g(self.P, "pre_classifier.kernel"); self.pc_b = g(self.P, "pre_classifier.bias")
        self.cl_w = g(self.P, "classifier.kernel"); self.cl_b = g(self.P, "classifier.bias")

    # --- prefix: capture token-0 FFN input h0 (cached once per batch) ---
    def capture_h0(self, input_ids, attention_mask):
        out = self.model(jnp.asarray(input_ids), attention_mask=jnp.asarray(attention_mask),
                         params=self.P, output_hidden_states=True, train=False)
        x5 = jnp.asarray(out.hidden_states[-2])               # layer-4 output = layer-5 input
        sa = layer5_attention(x5, jnp.asarray(attention_mask), self.P)
        h = _ln(sa + x5, g(self.P, f"{L5}.sa_layer_norm.scale"), g(self.P, f"{L5}.sa_layer_norm.bias"))
        return h[:, 0]                                        # (B, 768)

    # --- FFN + head tail (per member); member params override lin1/lin2 ---
    def tail(self, h0, W1=None, b1=None, W2=None, b2=None):
        W1 = self.W1 if W1 is None else W1; b1 = self.b1 if b1 is None else b1
        W2 = self.W2 if W2 is None else W2; b2 = self.b2 if b2 is None else b2
        y = jax.nn.gelu(h0 @ W1 + b1, approximate=False)      # (B, 3072)  post-GELU
        z = y @ W2 + b2                                       # (B, 768)   FFN output
        out = _ln(z + h0, self.out_ln_s, self.out_ln_b)       # residual + output LN
        pooled = jax.nn.relu(out @ self.pc_w + self.pc_b)
        return pooled @ self.cl_w + self.cl_b                 # (B, 77) logits

    def base_forward(self, input_ids, attention_mask):
        return self.tail(self.capture_h0(input_ids, attention_mask))

    def block_output(self, h0, W1=None, b1=None, W2=None, b2=None):
        """Layer-5 block output at token 0 (= pre_classifier input), for the head-P&C ablation."""
        W1 = self.W1 if W1 is None else W1; b1 = self.b1 if b1 is None else b1
        W2 = self.W2 if W2 is None else W2; b2 = self.b2 if b2 is None else b2
        y = jax.nn.gelu(h0 @ W1 + b1, approximate=False)
        z = y @ W2 + b2
        return _ln(z + h0, self.out_ln_s, self.out_ln_b)      # (B, 768)

    def head_forward(self, out, pc_w=None, pc_b=None, cl_w=None, cl_b=None):
        """pre_classifier -> ReLU -> classifier; member head params override the defaults."""
        pc_w = self.pc_w if pc_w is None else pc_w; pc_b = self.pc_b if pc_b is None else pc_b
        cl_w = self.cl_w if cl_w is None else cl_w; cl_b = self.cl_b if cl_b is None else cl_b
        return jax.nn.relu(out @ pc_w + pc_b) @ cl_w + cl_b

    def capture_triplet(self, h0, W1=None, b1=None):
        """(h, y, z) at the final FFN: y=gelu(lin1(h)), z=lin2(y). Original lin1 by default."""
        W1 = self.W1 if W1 is None else W1; b1 = self.b1 if b1 is None else b1
        y = jax.nn.gelu(h0 @ W1 + b1, approximate=False)
        z = y @ self.W2 + self.b2
        return h0, y, z
