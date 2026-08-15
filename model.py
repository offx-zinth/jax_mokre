"""Pure-JAX MoRE (Mixture-of-Recursions) model.

Faithful port of ``mokre/`` (PyTorch) to JAX. Params are a plain pytree of
dicts / lists / arrays.

Architecture:
    embed -> first layer (KDA + MoE) -> router -> [recursion block]^Nr
        -> last layer (MoE) -> lm head (tied to embed)
    block layers: KDA / KDA / MLA / KDA, each = attention + MoE FFN.

The KDA recurrence is vectorized with jax.lax.associative_scan (RESEARCH.md)
instead of the torch Python loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from .config import MoREConfig

EPS = 1e-6


# ---------------------------------------------------------------- primitives

def rmsnorm(x, w, eps=EPS):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w


def lin(x, w, b=None):
    """x (..., in) -> (..., out); w (out, in)."""
    if b is not None:
        return jnp.einsum("...i,oi->...o", x, w) + b
    return jnp.einsum("...i,oi->...o", x, w)


def init_linear(key, out, inn, std):
    return jax.random.normal(key, (out, inn)) * std


# ------------------------------------------------------------------- KDA
# Linear diagonal recurrence, vectorized via associative scan:
#   A_t = alpha_t * (1 - beta_t * k_t^2),  B_t = beta_t * k_t * v_t
#   s_t = A_t*s_{t-1} + B_t ; y_t = q_t * s_t
#   (A1,B1) x (A2,B2) = (A1*A2, B2 + A2*B1)

def init_kda(cfg, key):
    k = jax.random.split(key, 6)
    H, NH, NG, D = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    std = cfg.initializer_range
    return {
        "q_w": init_linear(k[0], NH * D, H, std),
        "k_w": init_linear(k[1], NG * D, H, std),
        "v_w": init_linear(k[2], NG * D, H, std),
        "o_w": init_linear(k[3], H, NH * D, std),
        "gate_w": init_linear(k[4], NG * (D + 1), H, std),
        "gate_b": jnp.zeros((NG * (D + 1),)),
        "norm_w": jnp.ones((H,)),
        "init_state": jnp.zeros((1, NG, D)),
    }


def _combine(x, y):
    return (x[0] * y[0], y[1] + y[0] * x[1])


def kda_forward(cfg, p, x, token_mask=None):
    bsz, S, H = x.shape
    NH, NG, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    G = NH // NG

    xn = rmsnorm(x, p["norm_w"], cfg.rms_norm_eps)
    q = lin(xn, p["q_w"]).reshape(bsz, S, NH, D)
    k = lin(xn, p["k_w"]).reshape(bsz, S, NG, D)
    v = lin(xn, p["v_w"]).reshape(bsz, S, NG, D)
    gate = lin(xn, p["gate_w"], p["gate_b"]).reshape(bsz, S, NG, D + 1)
    alpha = jax.nn.sigmoid(gate[..., :-1])
    beta = jax.nn.sigmoid(gate[..., -1:])

    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)

    # GQA: repeat kv heads to query heads
    k = jnp.repeat(k, G, axis=2)
    v = jnp.repeat(v, G, axis=2)
    alpha = jnp.repeat(alpha, G, axis=2)
    beta = jnp.repeat(beta, G, axis=2)

    A = alpha * (1.0 - beta * k * k)
    W = beta * k * v
    if token_mask is not None:
        m = token_mask[..., None, None].astype(A.dtype)  # (B,S,1,1)
        A = m * A + (1.0 - m)
        W = m * W

    accA, accW = jax.lax.associative_scan(_combine, (A, W), axis=1)
    s0 = jnp.repeat(p["init_state"], G, axis=1)        # (1,NH,D)
    s0 = jnp.broadcast_to(s0[:, None, :, :], (bsz, S, NH, D))
    s_t = accA * s0 + accW
    out = s_t * q
    out = out.reshape(bsz, S, NH * D)
    return lin(out, p["o_w"])


# ------------------------------------------------------------------- MLA

def init_mla(cfg, key):
    k = jax.random.split(key, 5)
    H, NH, NG, D = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    qk, vd = cfg.mla_qk_latent_dim, cfg.mla_v_latent_dim
    std = cfg.initializer_range
    return {
        "q_w": init_linear(k[0], NH * D, H, std),
        "kv_w": init_linear(k[1], NG * (qk + vd), H, std),
        "kd_w": init_linear(k[2], D, qk, std),
        "vd_w": init_linear(k[3], D, vd, std),
        "o_w": init_linear(k[4], H, NH * D, std),
        "norm_w": jnp.ones((H,)),
    }


def mla_forward(cfg, p, x, attention_mask=None, token_mask=None):
    B, S, H = x.shape
    NH, NG, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    qk, vd = cfg.mla_qk_latent_dim, cfg.mla_v_latent_dim
    G = NH // NG
    scale = D ** -0.5

    xn = rmsnorm(x, p["norm_w"], cfg.rms_norm_eps)
    q = lin(xn, p["q_w"]).reshape(B, S, NH, D) * scale
    kv = lin(xn, p["kv_w"]).reshape(B, S, NG, qk + vd)
    kc, vc = kv[..., :qk], kv[..., qk:]
    k = lin(kc, p["kd_w"]).reshape(B, S, NG, D)
    v = lin(vc, p["vd_w"]).reshape(B, S, NG, D)
    k = jnp.repeat(k, G, axis=2)
    v = jnp.repeat(v, G, axis=2)

    scores = jnp.einsum("bfhd,bShd->bhfS", q, k)        # (B,NH,Qlen,Slen)
    causal = jnp.triu(jnp.ones((S, S), dtype=bool), 1)
    scores = jnp.where(causal[None, None], -1e30, scores)
    if attention_mask is not None:
        scores = scores + jnp.where(attention_mask[:, None, None, :] == 0, -1e30, 0.0)
    if token_mask is not None:
        scores = scores + jnp.where(token_mask[:, None, None, :] == 0, -1e30, 0.0)
    # max-shift keeps fully-masked rows finite (softmax of all -inf is NaN and
    # its VJP stays NaN even after nan_to_num). Frozen rows get a garbage-but-
    # finite output that the layer's recursion gating discards anyway.
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhfS,bshd->bfhd", probs, v)
    out = out.reshape(B, S, NH * D)
    return lin(out, p["o_w"])


# ------------------------------------------------------------------- MoE
# Expert weights are stacked (E, M, H); all experts computed densely with one
# batched einsum (fine at E=4, top-k=1). Routing mask from lax.top_k.

def init_moe(cfg, key):
    k = jax.random.split(key, 6)
    H, M, E = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts
    std = cfg.initializer_range
    return {
        "router_w": init_linear(k[0], E, H, std),
        "expert_gate_w": init_linear(k[1], E * M, H, std).reshape(E, M, H),
        "expert_up_w": init_linear(k[2], E * M, H, std).reshape(E, M, H),
        "expert_down_w": init_linear(k[3], E * H, M, std).reshape(E, H, M),
        "shared_gate_w": init_linear(k[4], M, H, std),
        "shared_up_w": init_linear(k[4], M, H, std),
        "shared_down_w": init_linear(k[5], H, M, std),
        "norm_w": jnp.ones((H,)),
    }


def moe_forward(cfg, p, x, training=False):
    B, S, H = x.shape
    E, M, k = p["expert_gate_w"].shape[0], p["expert_gate_w"].shape[1], cfg.top_k
    coef = cfg.load_balancing_loss_coef

    xn = rmsnorm(x, p["norm_w"], cfg.rms_norm_eps)
    xf = xn.reshape(-1, H)                              # (N,H)
    N = xf.shape[0]

    logits = xf @ p["router_w"].T                       # (N,E)
    topk_val, topk_idx = jax.lax.top_k(logits, k)       # (N,k)
    w = jax.nn.softmax(topk_val, axis=-1)               # (N,k)

    # per-expert token weight: sum of topk weights selecting that expert
    w_e = jnp.sum(jnp.where(topk_idx[..., None] == jnp.arange(E), w[..., None], 0.0),
                  axis=1)                               # (N,E)

    # experts applied one at a time (same FLOPs as a batched einsum, but only
    # one (N,M) intermediate lives at a time — keeps TPU HBM usage low)
    routed = jnp.zeros_like(xf)
    for e in range(E):
        gate = xf @ p["expert_gate_w"][e].T
        up = xf @ p["expert_up_w"][e].T
        act = jax.nn.silu(gate) * up
        down = act @ p["expert_down_w"][e].T
        routed = routed + w_e[:, e:e + 1] * down

    sgate = jax.nn.silu(xf @ p["shared_gate_w"].T)
    sup = xf @ p["shared_up_w"].T
    shared = (sgate * sup) @ p["shared_down_w"].T

    out = (routed + shared).reshape(B, S, H)

    if training:
        oh = (jax.nn.one_hot(topk_idx, E) / k).sum(axis=1)  # (N,E) frac of selections
        mi = oh.mean(axis=0)
        pi = jax.nn.softmax(logits, axis=-1).mean(axis=0)
        aux = coef * E * jnp.sum(mi * pi)
    else:
        aux = 0.0
    return out, aux


# ------------------------------------------------------------------- router

def init_router(cfg, key):
    k = jax.random.split(key, 2)
    H, R = cfg.hidden_size, cfg.router_hidden_size
    std = cfg.initializer_range
    return {
        "l1_w": init_linear(k[0], R, H, std),
        "l2_w": init_linear(k[1], cfg.max_recursion_depth, R, std),
        "norm_w": jnp.ones((H,)),
    }


def router_forward(cfg, p, x, training=False):
    B, S, H = x.shape
    Nr = p["l2_w"].shape[0]
    coef = cfg.load_balancing_loss_coef
    xn = rmsnorm(x, p["norm_w"], cfg.rms_norm_eps)
    h = jnp.tanh(xn @ p["l1_w"].T)
    logits = h @ p["l2_w"].T                            # (B,S,Nr)
    probs = jax.nn.softmax(logits, axis=-1)
    depths = jnp.argmax(probs, axis=-1) + 1             # 1..Nr
    if training:
        oh = jax.nn.one_hot(depths - 1, Nr).astype(x.dtype)
        f = (Nr / (B * S)) * oh.sum(axis=(0, 1))
        P = probs.mean(axis=(0, 1))
        aux = coef * jnp.sum(f * P)
    else:
        aux = 0.0
    return depths, aux


# ------------------------------------------------------------------- model

def mor_layer(cfg, p, x, layer_type, token_mask=None, attention_mask=None, training=False):
    orig = x
    residual = x
    x = rmsnorm(x, p["input_norm"], cfg.rms_norm_eps)
    if layer_type == "kda":
        attn = kda_forward(cfg, p["attn"], x, token_mask=token_mask)
    else:
        attn = mla_forward(cfg, p["attn"], x, attention_mask=attention_mask,
                           token_mask=token_mask)
    x = residual + attn
    residual = x
    x = rmsnorm(x, p["post_norm"], cfg.rms_norm_eps)
    moe_out, moe_aux = moe_forward(cfg, p["moe"], x, training)
    x = residual + moe_out
    if token_mask is not None:
        m = token_mask[..., None].astype(x.dtype)
        x = m * x + (1.0 - m) * orig
    return x, moe_aux


def init_model(cfg, key):
    keys = jax.random.split(key, 3 + len(cfg.layer_types))
    H = cfg.hidden_size
    std = cfg.initializer_range
    block = []
    for i, lt in enumerate(cfg.layer_types):
        akey, mkey = jax.random.split(keys[i])
        attn = init_kda(cfg, akey) if lt == "kda" else init_mla(cfg, akey)
        block.append({
            "attn": attn,
            "moe": init_moe(cfg, mkey),
            "input_norm": jnp.ones((H,)),
            "post_norm": jnp.ones((H,)),
        })
    return {
        "embed_tokens": init_linear(keys[-3], cfg.vocab_size, H, std),
        "embed_norm": jnp.ones((H,)),
        "first": {
            "attn": init_kda(cfg, keys[-2]),
            "moe": init_moe(cfg, keys[-2]),
            "input_norm": jnp.ones((H,)),
            "post_norm": jnp.ones((H,)),
        },
        "block": block,
        "router": init_router(cfg, keys[-1]),
        "last": {"moe": init_moe(cfg, keys[-1]), "norm": jnp.ones((H,))},
    }


def forward(cfg, params, input_ids, training=False, attention_mask=None, return_hidden=False):
    """input_ids (B,S) -> (logits (B,S,V), aux, depths) or hidden states."""
    B, S = input_ids.shape
    h = params["embed_tokens"][input_ids]
    h = rmsnorm(h, params["embed_norm"], cfg.rms_norm_eps)

    h, a1 = mor_layer(cfg, params["first"], h, "kda", training=training)
    depths, a_router = router_forward(cfg, params["router"], h, training)
    aux = a1 + a_router

    Nr = cfg.max_recursion_depth
    for step in range(1, Nr + 1):
        m = (depths >= step).astype(jnp.float32)
        h_prev = h
        block_aux = 0.0
        for i, lt in enumerate(cfg.layer_types):
            h, laux = mor_layer(cfg, params["block"][i], h, lt,
                                token_mask=m, attention_mask=attention_mask,
                                training=training)
            block_aux = block_aux + laux
        aux = aux + block_aux / Nr
        h = m[..., None] * h + (1.0 - m[..., None]) * h_prev

    # last layer (MoE only)
    residual = h
    h = rmsnorm(h, params["last"]["norm"], cfg.rms_norm_eps)
    moe_out, a_last = moe_forward(cfg, params["last"]["moe"], h, training)
    h = residual + moe_out
    aux = aux + a_last

    if return_hidden:
        return h, aux, depths
    logits = jnp.einsum("bsh,vh->bsv", h, params["embed_tokens"])
    return logits, aux, depths


def count_params(params):
    return sum(x.size for x in jax.tree.leaves(params))
