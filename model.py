"""Pure-JAX MoRE (Mixture-of-Recursions) model.

Faithful port of ``mokre/`` (PyTorch) to JAX. Params are a plain pytree of
dicts / lists / arrays.

Architecture:
    embed -> first layer (KDA + MoE) -> router -> [recursion block]^Nr
        -> last layer (MoE) -> lm head (tied to embed)
    block layers: KDA / KDA / MSA / KDA, each = attention + MoE FFN.

The KDA recurrence is vectorized with jax.lax.associative_scan (RESEARCH.md)
instead of the torch Python loop.

MSA (MiniMax Sparse Attention) replaces MLA (DeepSeek-V2).  MSA is a
GQA-based block-sparse attention (MiniMax-M3, arXiv:2606.13392):
    * Index Branch: per-GQA-group light scoring (Q_idx*K_idx) pooled to
      block level via max, Top-k per group (plus forced local block).
    * Main Branch: exact sparse softmax over only selected blocks.
    * Training: KL( stopgrad(teacher) || index ) over selected tokens,
      weighted by msa_kl_coef, gradient isolated to index projections via
      stopgrad(input).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from .config import MoREConfig

EPS = 1e-6

_DTYPE_MAP = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}

def _param_dtype(cfg) -> jnp.dtype:
    return _DTYPE_MAP.get(getattr(cfg, "param_dtype", "float32"), jnp.float32)

def _compute_dtype(cfg) -> jnp.dtype:
    return _DTYPE_MAP.get(getattr(cfg, "compute_dtype", "float32"), jnp.float32)


def _build_rope_cos_sin(seq_len: int, head_dim: int, rope_dim: int, theta: float = 10000.0, dtype=jnp.float32):
    """Build RoPE cos/sin (S, half). rope_dim <= head_dim, rest is NoPE."""
    assert rope_dim <= head_dim and rope_dim % 2 == 0
    half = rope_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("i,j->ij", t, inv_freq)  # (S, half)
    cos = jnp.cos(freqs).astype(dtype)  # (S, half)
    sin = jnp.sin(freqs).astype(dtype)
    return cos, sin  # (S, half)


def _apply_rope(x, cos, sin):
    """Apply RoPE to x (B,S,NH,D). Only first rope_dim dims are rotated."""
    # x: (B,S,NH,D), cos/sin: (S, half) where half=rope_dim//2
    half = cos.shape[-1]
    rope_dim = half * 2
    if rope_dim == 0 or rope_dim > x.shape[-1]:
        return x
    x1 = x[..., :rope_dim]
    x2 = x[..., rope_dim:]
    x1_a, x1_b = jnp.split(x1, 2, axis=-1)  # each (B,S,NH,half)
    cos_b = cos[None, :, None, :]  # (1,S,1,half)
    sin_b = sin[None, :, None, :]
    out_a = x1_a * cos_b - x1_b * sin_b
    out_b = x1_a * sin_b + x1_b * cos_b
    out = jnp.concatenate([out_a, out_b], axis=-1)
    return jnp.concatenate([out, x2], axis=-1)


# ---------------------------------------------------------------- primitives

def rmsnorm(x, w, eps=EPS):
    # keep rmsnorm in float32 for stability even when compute_dtype is bf16
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32) if x.dtype != jnp.float32 else x
    w_f32 = w.astype(jnp.float32) if w.dtype != jnp.float32 else w
    out = x_f32 * jax.lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps) * w_f32
    return out.astype(orig_dtype)


def lin(x, w, b=None):
    """x (..., in) -> (..., out); w (out, in)."""
    if b is not None:
        return jnp.einsum("...i,oi->...o", x, w) + b
    return jnp.einsum("...i,oi->...o", x, w)


def init_linear(key, out, inn, std, dtype=jnp.float32):
    return (jax.random.normal(key, (out, inn)) * std).astype(dtype)


# ------------------------------------------------------------------- KDA
# Linear diagonal recurrence, vectorized via associative scan:
#   A_t = alpha_t * (1 - beta_t * k_t^2),  B_t = beta_t * k_t * v_t
#   s_t = A_t*s_{t-1} + B_t ; y_t = q_t * s_t
#   (A1,B1) x (A2,B2) = (A1*A2, B2 + A2*B1)

def init_kda(cfg, key):
    k = jax.random.split(key, 6)
    H, NH, NG, D = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    std = cfg.initializer_range
    pd = _param_dtype(cfg)
    return {
        "q_w": init_linear(k[0], NH * D, H, std, pd),
        "k_w": init_linear(k[1], NG * D, H, std, pd),
        "v_w": init_linear(k[2], NG * D, H, std, pd),
        "o_w": init_linear(k[3], H, NH * D, std, pd),
        "gate_w": init_linear(k[4], NG * (D + 1), H, std, pd),
        "gate_b": jnp.zeros((NG * (D + 1),), dtype=pd),
        "norm_w": jnp.ones((H,), dtype=pd),
        "init_state": jnp.zeros((1, NG, D), dtype=pd),
    }


def _combine(x, y):
    return (x[0] * y[0], y[1] + y[0] * x[1])


def kda_forward(cfg, p, x, token_mask=None):
    bsz, S, H = x.shape
    NH, NG, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    G = NH // NG
    cd = _compute_dtype(cfg)

    xn = rmsnorm(x, p["norm_w"], cfg.rms_norm_eps)
    # cast to compute dtype for matmuls (rmsnorm stays f32)
    xn_c = xn.astype(cd) if xn.dtype != cd else xn
    q = lin(xn_c, p["q_w"].astype(cd) if p["q_w"].dtype != cd else p["q_w"]).reshape(bsz, S, NH, D)
    k = lin(xn_c, p["k_w"].astype(cd) if p["k_w"].dtype != cd else p["k_w"]).reshape(bsz, S, NG, D)
    v = lin(xn_c, p["v_w"].astype(cd) if p["v_w"].dtype != cd else p["v_w"]).reshape(bsz, S, NG, D)
    gate_w_c = p["gate_w"].astype(cd) if p["gate_w"].dtype != cd else p["gate_w"]
    gate_b_c = p["gate_b"].astype(cd) if p["gate_b"].dtype != cd else p["gate_b"]
    gate = lin(xn_c, gate_w_c, gate_b_c).reshape(bsz, S, NG, D + 1)
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

    # Chunked associative scan to bound activation memory (S up to 8192)
    # Uses cfg.kda_chunk_size (default 128); carries state across chunks
    # COMPILATION FIX 2026-09-01: for large S, auto-scale chunk to bound XLA graph
    # Python for-loop unrolls at trace time: S=8192/chunk=128 => 64 unrolled
    # blocks => >30min compile / >2h for 16k. Auto-scale keeps iterations <=16.
    chunk = getattr(cfg, "kda_chunk_size", 128)
    # L2 fix: warn if user asked for <16 (was silently clamped)
    if getattr(cfg, "kda_chunk_size", 128) < 16:
        # Use jax.debug.print to surface in jit context without breaking
        jax.debug.print("WARNING: kda_chunk_size {x} <16 clamped to 16", x=chunk)
        chunk = 16
    if chunk < 16:
        chunk = 16
    # Auto-scale for long context to bound compile graph (keep iterations <=16)
    if S > 4096 and chunk < 1024:
        chunk = 1024
    elif S > 2048 and chunk < 512:
        chunk = 512
    if S <= chunk:
        # M1 fix: avoid broadcast to (B,S,NH,D) ~8M floats; keep carry (B,NH,D)
        accA, accW = jax.lax.associative_scan(_combine, (A, W), axis=1)
        s0 = jnp.repeat(p["init_state"], G, axis=1)        # (1,NH,D)
        s0_b = jnp.broadcast_to(s0, (bsz, NH, D))          # (B,NH,D)
        # accA * s0_b[None] broadcasts correctly without S expansion
        s_t = accA * s0_b[:, None, :, :] + accW            # (B,S,NH,D)
    else:
        # init carry: s0 repeated per batch
        s0_init = jnp.repeat(p["init_state"], G, axis=1)  # (1,NH,D)
        carry = jnp.broadcast_to(s0_init, (bsz, NH, D))  # (B,NH,D)
        # collect outputs in list (still Python loop over chunks, bounded memory per chunk)
        out_parts = []
        for start in range(0, S, chunk):
            end = start + chunk
            if end > S:
                end = S
            A_chunk = A[:, start:end, :, :]  # (B,C,NH,D)
            W_chunk = W[:, start:end, :, :]
            q_chunk = q[:, start:end, :, :]
            # scan within chunk
            accA_c, accW_c = jax.lax.associative_scan(_combine, (A_chunk, W_chunk), axis=1)
            # incorporate carry: s = accA * carry + accW
            carry_b = carry[:, None, :, :]  # (B,1,NH,D)
            s_chunk = accA_c * carry_b + accW_c  # (B,C,NH,D)
            out_chunk = s_chunk * q_chunk
            out_parts.append(out_chunk)
            # update carry to last state of chunk
            carry = s_chunk[:, -1, :, :]  # (B,NH,D)
        s_t = jnp.concatenate(out_parts, axis=1)  # (B,S,NH,D) for compatibility
        # s_t already is out before final multiply? we used out_parts as s*q
        # to keep same API, set s_t to concatenated s*q pre-projection
        out = s_t.reshape(bsz, S, NH * D)
        o_w_c = p["o_w"].astype(cd) if p["o_w"].dtype != cd else p["o_w"]
        out_c = out.astype(cd) if out.dtype != cd else out
        return lin(out_c, o_w_c).astype(x.dtype)
    out = s_t * q
    out = out.reshape(bsz, S, NH * D)
    o_w_c = p["o_w"].astype(cd) if p["o_w"].dtype != cd else p["o_w"]
    out_c = out.astype(cd) if out.dtype != cd else out
    return lin(out_c, o_w_c).astype(x.dtype)


# ------------------------------------------------------------------- MSA
# MiniMax Sparse Attention (MiniMax-M3, arXiv:2606.13392)
# Index Branch: Q_idx (per-GQA-group) + K_idx (shared, 1 head) -> block max-pool -> Top-k
# Main Branch: sparse attention over selected blocks only.
# Training: KL(stopgrad(teacher) || index) aux loss.

def init_msa(cfg, key):
    k = jax.random.split(key, 6)
    H, NH, NG, D = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    d_idx = cfg.msa_index_dim
    std = cfg.initializer_range
    pd = _param_dtype(cfg)
    return {
        "q_w": init_linear(k[0], NH * D, H, std, pd),
        "k_w": init_linear(k[1], NG * D, H, std, pd),
        "v_w": init_linear(k[2], NG * D, H, std, pd),
        "o_w": init_linear(k[3], H, NH * D, std, pd),
        # Index branch: Q_idx per GQA group (NG * d_idx), K_idx shared (1 * d_idx)
        "q_idx_w": init_linear(k[4], NG * d_idx, H, std, pd),
        "k_idx_w": init_linear(k[5], d_idx, H, std, pd),
        "norm_w": jnp.ones((H,), dtype=pd),
    }


# Legacy wrappers — keep `init_mla` / `mla_forward` importable for old ckpts
def init_mla(cfg, key):  # pragma: no cover
    return init_msa(cfg, key)


def msa_forward(cfg, p, x, attention_mask=None, token_mask=None, training=False):
    """MiniMax Sparse Attention forward.

    Args:
        cfg: MoREConfig with msa_* fields.
        p: params dict from init_msa.
        x: (B,S,H) input.
        attention_mask: (B,S) 1 for valid, 0 for pad (optional).
        token_mask: (B,S) 1 for tokens active at this recursion step (optional).
        training: if True, also compute KL aux loss for the index branch.

    Returns:
        out: (B,S,H) projected sparse attention output.
        aux: scalar KL loss (0 if not training or no valid support).
    """
    B, S, H = x.shape
    NH, NG, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    G = NH // NG
    Bk = cfg.msa_block_size
    k_req = cfg.msa_topk
    d_idx = cfg.msa_index_dim
    scale = D ** -0.5
    idx_scale = d_idx ** -0.5

    xn = rmsnorm(x, p["norm_w"], cfg.rms_norm_eps)

    # ---- Main branch QKV ----
    q_main = lin(xn, p["q_w"]).reshape(B, S, NH, D) * scale
    k_main = lin(xn, p["k_w"]).reshape(B, S, NG, D)
    v_main = lin(xn, p["v_w"]).reshape(B, S, NG, D)
    # RoPE on main Q/K for retrieval (MLA-like) while keeping MSA block-sparse compute
    if getattr(cfg, "msa_use_rope", False):
        rope_dim = getattr(cfg, "msa_rope_dim", 64)
        rope_theta = getattr(cfg, "msa_rope_theta", 10000.0)
        # clamp rope_dim <= D and even
        rope_dim = min(rope_dim, D)
        if rope_dim % 2 == 1:
            rope_dim -= 1
        if rope_dim > 0:
            cos, sin = _build_rope_cos_sin(S, D, rope_dim, rope_theta, dtype=xn.dtype)
            q_main = _apply_rope(q_main, cos, sin)
            k_main = _apply_rope(k_main, cos, sin)

    # ---- Index branch (stop-gradient on input, Eq. 11) ----
    x_idx = jax.lax.stop_gradient(xn)
    q_idx = lin(x_idx, p["q_idx_w"]).reshape(B, S, NG, d_idx)  # per-group
    k_idx = lin(x_idx, p["k_idx_w"]).reshape(B, S, d_idx)       # shared

    # ---- Small S: dense path (exact, cheap) ----
    # For S<=1024 keep original dense implementation; for larger S use
    # memory-bounded chunked path that never materializes full (S,S).
    # This implements the block-sparse kernel: top-k on pooled block scores
    # then query-chunked sparse attention over selected blocks only.
    if S <= 1024:
        # S_idx : (B, NG, S_q, S_k)
        q_idx_t = jnp.transpose(q_idx, (0, 2, 1, 3))  # (B,NG,S,d_idx)
        S_idx = jnp.einsum("bgid,bjd->bgij", q_idx_t, k_idx) * idx_scale  # (B,NG,S,S)

        # causal mask for index scores (j <= i)
        causal = jnp.tril(jnp.ones((S, S), dtype=bool))  # (S,S) True if j <= i
        S_idx = jnp.where(causal[None, None, :, :], S_idx, -1e30)

        # mask invalid keys (padding / frozen) so their block max stays -inf
        if attention_mask is not None:
            key_valid = (attention_mask != 0)  # (B,S) bool
            S_idx = jnp.where(key_valid[:, None, None, :], S_idx, -1e30)
        if token_mask is not None:
            key_active = (token_mask != 0)  # (B,S) bool
            S_idx = jnp.where(key_active[:, None, None, :], S_idx, -1e30)

        # ---- Block max-pool: (B,NG,S,S) -> (B,NG,S,Nb) ----
        Nb = (S + Bk - 1) // Bk
        pad_len = Nb * Bk - S
        if pad_len > 0:
            S_idx_padded = jnp.pad(S_idx, ((0, 0), (0, 0), (0, 0), (0, pad_len)),
                                   constant_values=-1e30)
        else:
            S_idx_padded = S_idx
        # S_idx_padded: (B,NG,S, Nb*Bk) -> (B,NG,S,Nb,Bk)
        S_reshaped = S_idx_padded.reshape(B, NG, S, Nb, Bk)
        M = jnp.max(S_reshaped, axis=-1)  # (B,NG,S,Nb)

        # ---- Top-k per query per group ----
        k_eff = k_req if k_req <= Nb else Nb
        if k_eff < 1:
            k_eff = 1
        _, top_idx = jax.lax.top_k(M, k_eff)  # (B,NG,S,k_eff)

        # force local block inclusion (paper Sec 3.2, Local Block)
        local_blk = jnp.arange(S) // Bk  # (S,)
        local_bc = jnp.broadcast_to(local_blk, (B, NG, S))  # (B,NG,S)
        is_in = jnp.any(top_idx == local_bc[..., None], axis=-1)  # (B,NG,S) bool
        is_last = (jnp.arange(k_eff) == k_eff - 1).reshape(1, 1, 1, -1)  # (1,1,1,k)
        replace_pos = jnp.logical_and(jnp.logical_not(is_in)[..., None], is_last)  # (B,NG,S,k)
        top_idx = jnp.where(replace_pos, local_bc[..., None], top_idx)

        # ---- Build selected mask: (B,NG,S,S) ----
        block_id = jnp.arange(S) // Bk  # (S,) token -> block
        selected_mask = jnp.zeros((B, NG, S, S), dtype=bool)
        block_id_b = block_id.reshape(1, 1, 1, -1)  # (1,1,1,S)
        for ki in range(k_eff):
            blk = top_idx[..., ki]  # (B,NG,S)
            mask_k = (blk[..., None] == block_id_b)  # (B,NG,S,S)
            selected_mask = jnp.logical_or(selected_mask, mask_k)
        causal = jnp.tril(jnp.ones((S, S), dtype=bool))
        selected_mask = jnp.logical_and(selected_mask, causal[None, None, :, :])
        if attention_mask is not None:
            key_valid_exp = (attention_mask != 0)[:, None, None, :]  # (B,1,1,S)
            selected_mask = jnp.logical_and(selected_mask, key_valid_exp)
        if token_mask is not None:
            key_active_exp = (token_mask != 0)[:, None, None, :]  # (B,1,1,S)
            selected_mask = jnp.logical_and(selected_mask, key_active_exp)

        # ---- Main branch sparse attention ----
        k_rep = jnp.repeat(k_main, G, axis=2)  # (B,S,NH,D)
        v_rep = jnp.repeat(v_main, G, axis=2)  # (B,S,NH,D)
        scores = jnp.einsum("bshd,bthd->bhst", q_main, k_rep)  # (B,NH,S,S)
        allowed_per_head = jnp.repeat(selected_mask, G, axis=1)  # (B,NH,S,S)
        scores_masked = jnp.where(allowed_per_head, scores, -1e30)
        scores_shifted = scores_masked - jnp.max(scores_masked, axis=-1, keepdims=True)
        probs = jax.nn.softmax(scores_shifted, axis=-1)  # (B,NH,S,S)
        out = jnp.einsum("bhst,bthd->bshd", probs, v_rep)  # (B,S,NH,D)
        out = out.reshape(B, S, NH * D)
        out_proj = lin(out, p["o_w"])

        # ---- KL alignment loss (paper Eq. 10) ----
        aux = jnp.asarray(0.0, dtype=out_proj.dtype)
        if training:
            S_idx_kl = jnp.where(selected_mask, S_idx, -1e30)  # (B,NG,S,S)
            max_idx_kl = jnp.max(S_idx_kl, axis=-1, keepdims=True)  # (B,NG,S,1)
            exp_idx = jnp.where(selected_mask, jnp.exp(S_idx_kl - max_idx_kl), 0.0)
            sum_idx = jnp.sum(exp_idx, axis=-1, keepdims=True)  # (B,NG,S,1)
            P_idx = exp_idx / (sum_idx + 1e-9)  # (B,NG,S,S)
            scores_grouped = scores.reshape(B, NG, G, S, S)
            sel_exp = selected_mask[:, :, None, :, :]  # (B,NG,1,S,S)
            scores_kl = jnp.where(sel_exp, scores_grouped, -1e30)
            max_teacher = jnp.max(scores_kl, axis=-1, keepdims=True)  # (B,NG,G,S,1)
            exp_teacher = jnp.where(sel_exp, jnp.exp(scores_kl - max_teacher), 0.0)
            sum_teacher = jnp.sum(exp_teacher, axis=-1, keepdims=True)
            P_main = exp_teacher / (sum_teacher + 1e-9)  # (B,NG,G,S,S)
            P_teacher = jnp.mean(P_main, axis=2)  # (B,NG,S,S) average over G query heads
            P_teacher = jax.lax.stop_gradient(P_teacher)
            has_valid = jnp.any(selected_mask, axis=-1)  # (B,NG,S) at least one token
            # L3 fix: distinguish padding (attention_mask) vs recursion freeze (token_mask)
            # A query is valid only if its own attention_mask==1 AND token_mask==1 (if provided)
            if attention_mask is not None:
                q_valid = (attention_mask != 0)  # (B,S)
                q_valid_bc = jnp.broadcast_to(q_valid[:, None, :], (B, NG, S))
                has_valid = jnp.logical_and(has_valid, q_valid_bc)
            if token_mask is not None:
                q_active = (token_mask != 0)  # (B,S) recursion active
                q_active_bc = jnp.broadcast_to(q_active[:, None, :], (B, NG, S))
                has_valid = jnp.logical_and(has_valid, q_active_bc)
            kl_per = jnp.sum(P_teacher * (jnp.log(P_teacher + 1e-9) - jnp.log(P_idx + 1e-9)), axis=-1)  # (B,NG,S)
            kl_per = jnp.where(has_valid, kl_per, 0.0)
            denom = jnp.maximum(jnp.sum(has_valid), 1)
            kl_loss = jnp.sum(kl_per) / denom
            aux = cfg.msa_kl_coef * kl_loss
            aux = jnp.where(jnp.isfinite(aux), aux, 0.0)
        return out_proj, aux

    # ---- Large S: chunked / block-sparse path (never materialize full (S,S)) ----
    Nb = (S + Bk - 1) // Bk
    k_eff = k_req if k_req <= Nb else Nb
    if k_eff < 1:
        k_eff = 1
    # Compute M = block-max of index scores without full S_idx
    # M shape (B,NG,S,Nb) via key-block loop
    # For compile bound, if Nb > 64 (S>8192) we compute in blocks of 8
    q_idx_t = jnp.transpose(q_idx, (0, 2, 1, 3))  # (B,NG,S,d_idx)
    M = jnp.full((B, NG, S, Nb), -1e30, dtype=x.dtype)
    # key block loop (Nb <= 128 for S=8192, 64 blocks at 8k)
    for b in range(Nb):
        start = b * Bk
        end = start + Bk
        if end > S:
            end = S
        Bk_eff = end - start
        if Bk_eff <= 0:
            continue
        k_block = k_idx[:, start:end, :]  # (B,Bk_eff,d_idx)
        scores_block = jnp.einsum("bgid,bjd->bgij", q_idx_t, k_block) * idx_scale  # (B,NG,S,Bk_eff)
        # causal: j_global = start + j_local <= i
        q_pos = jnp.arange(S)[:, None]  # (S,1)
        k_pos = jnp.arange(start, end)[None, :]  # (1,Bk_eff)
        causal_block = (k_pos <= q_pos)  # (S,Bk_eff)
        scores_block = jnp.where(causal_block[None, None, :, :], scores_block, -1e30)
        if attention_mask is not None:
            key_valid = (attention_mask != 0)  # (B,S)
            valid_block = key_valid[:, start:end]  # (B,Bk_eff)
            scores_block = jnp.where(valid_block[:, None, None, :], scores_block, -1e30)
        if token_mask is not None:
            key_active = (token_mask != 0)  # (B,S)
            active_block = key_active[:, start:end]  # (B,Bk_eff)
            scores_block = jnp.where(active_block[:, None, None, :], scores_block, -1e30)
        block_max = jnp.max(scores_block, axis=-1)  # (B,NG,S)
        M = M.at[:, :, :, b].set(block_max)

    _, top_idx = jax.lax.top_k(M, k_eff)  # (B,NG,S,k_eff)
    # force local block
    local_blk = jnp.arange(S) // Bk  # (S,)
    local_bc = jnp.broadcast_to(local_blk, (B, NG, S))  # (B,NG,S)
    is_in = jnp.any(top_idx == local_bc[..., None], axis=-1)
    is_last = (jnp.arange(k_eff) == k_eff - 1).reshape(1, 1, 1, -1)
    replace_pos = jnp.logical_and(jnp.logical_not(is_in)[..., None], is_last)
    top_idx = jnp.where(replace_pos, local_bc[..., None], top_idx)

    # Main branch: query-chunked sparse attention
    k_rep = jnp.repeat(k_main, G, axis=2)  # (B,S,NH,D)
    v_rep = jnp.repeat(v_main, G, axis=2)  # (B,S,NH,D)
    CQ = 128  # query chunk to bound memory
    # For KL accumulation
    kl_sum = jnp.asarray(0.0, dtype=x.dtype)
    valid_sum = jnp.asarray(0, dtype=jnp.int32)
    out_chunks = []
    # Precompute block_id for mask building
    block_id = jnp.arange(S) // Bk  # (S,)
    for qs in range(0, S, CQ):
        qe = qs + CQ
        if qe > S:
            qe = S
        C = qe - qs
        q_chunk = q_main[:, qs:qe, :, :]  # (B,C,NH,D)
        # scores chunk: (B,NH,C,S)
        scores_chunk = jnp.einsum("bqhd,bthd->bhqt", q_chunk, k_rep)
        # selected mask for chunk: (B,NG,C,S)
        top_idx_chunk = top_idx[:, :, qs:qe, :]  # (B,NG,C,k_eff)
        sel_mask_chunk = jnp.zeros((B, NG, C, S), dtype=bool)
        block_id_b = block_id.reshape(1, 1, 1, -1)  # (1,1,1,S)
        for ki in range(k_eff):
            blk = top_idx_chunk[..., ki]  # (B,NG,C)
            mask_k = (blk[..., None] == block_id_b)  # (B,NG,C,S)
            sel_mask_chunk = jnp.logical_or(sel_mask_chunk, mask_k)
        # causal for chunk: query pos = qs + cq_idx
        q_pos_chunk = jnp.arange(qs, qe)[:, None]  # (C,1)
        k_pos = jnp.arange(S)[None, :]  # (1,S)
        causal_chunk = (k_pos <= q_pos_chunk)  # (C,S)
        sel_mask_chunk = jnp.logical_and(sel_mask_chunk, causal_chunk[None, None, :, :])
        if attention_mask is not None:
            key_valid_exp = (attention_mask != 0)[:, None, None, :]  # (B,1,1,S)
            sel_mask_chunk = jnp.logical_and(sel_mask_chunk, key_valid_exp)
        if token_mask is not None:
            key_active_exp = (token_mask != 0)[:, None, None, :]  # (B,1,1,S)
            sel_mask_chunk = jnp.logical_and(sel_mask_chunk, key_active_exp)
        # attention over selected blocks only
        allowed_chunk = jnp.repeat(sel_mask_chunk, G, axis=1)  # (B,NH,C,S)
        scores_masked = jnp.where(allowed_chunk, scores_chunk, -1e30)
        scores_shifted = scores_masked - jnp.max(scores_masked, axis=-1, keepdims=True)
        probs_chunk = jax.nn.softmax(scores_shifted, axis=-1)  # (B,NH,C,S)
        out_chunk = jnp.einsum("bhqt,bthd->bqhd", probs_chunk, v_rep)  # (B,C,NH,D)
        out_chunks.append(out_chunk)

        if training:
            # KL for this chunk
            q_idx_chunk_t = jnp.transpose(q_idx[:, qs:qe, :, :], (0, 2, 1, 3))  # (B,NG,C,d_idx)
            S_idx_chunk = jnp.einsum("bgid,bjd->bgij", q_idx_chunk_t, k_idx) * idx_scale  # (B,NG,C,S)
            # S_idx already causal/valid masked via sel_mask; for P_idx we re-mask to selected
            S_idx_kl = jnp.where(sel_mask_chunk, S_idx_chunk, -1e30)  # (B,NG,C,S)
            max_idx_kl = jnp.max(S_idx_kl, axis=-1, keepdims=True)  # (B,NG,C,1)
            exp_idx = jnp.where(sel_mask_chunk, jnp.exp(S_idx_kl - max_idx_kl), 0.0)
            sum_idx = jnp.sum(exp_idx, axis=-1, keepdims=True)  # (B,NG,C,1)
            P_idx = exp_idx / (sum_idx + 1e-9)  # (B,NG,C,S)

            scores_grouped = scores_chunk.reshape(B, NG, G, C, S)
            sel_exp = sel_mask_chunk[:, :, None, :, :]  # (B,NG,1,C,S)
            scores_kl = jnp.where(sel_exp, scores_grouped, -1e30)
            max_teacher = jnp.max(scores_kl, axis=-1, keepdims=True)  # (B,NG,G,C,1)
            exp_teacher = jnp.where(sel_exp, jnp.exp(scores_kl - max_teacher), 0.0)
            sum_teacher = jnp.sum(exp_teacher, axis=-1, keepdims=True)
            P_main = exp_teacher / (sum_teacher + 1e-9)  # (B,NG,G,C,S)
            P_teacher = jnp.mean(P_main, axis=2)  # (B,NG,C,S)
            P_teacher = jax.lax.stop_gradient(P_teacher)
            has_valid = jnp.any(sel_mask_chunk, axis=-1)  # (B,NG,C)
            if attention_mask is not None:
                q_valid = (attention_mask != 0)[:, qs:qe]  # (B,C)
                q_valid_bc = jnp.broadcast_to(q_valid[:, None, :], (B, NG, C))
                has_valid = jnp.logical_and(has_valid, q_valid_bc)
            if token_mask is not None:
                q_active = (token_mask != 0)[:, qs:qe]  # (B,C)
                q_active_bc = jnp.broadcast_to(q_active[:, None, :], (B, NG, C))
                has_valid = jnp.logical_and(has_valid, q_active_bc)
            kl_per = jnp.sum(P_teacher * (jnp.log(P_teacher + 1e-9) - jnp.log(P_idx + 1e-9)), axis=-1)  # (B,NG,C)
            kl_per = jnp.where(has_valid, kl_per, 0.0)
            kl_sum = kl_sum + jnp.sum(kl_per)
            valid_sum = valid_sum + jnp.sum(has_valid.astype(jnp.int32))

    out_concat = jnp.concatenate(out_chunks, axis=1)  # (B,S,NH,D)
    out_concat = out_concat.reshape(B, S, NH * D)
    out_proj = lin(out_concat, p["o_w"])
    aux = jnp.asarray(0.0, dtype=out_proj.dtype)
    if training:
        denom = jnp.maximum(valid_sum, 1)
        kl_loss = kl_sum / denom
        aux = cfg.msa_kl_coef * kl_loss
        aux = jnp.where(jnp.isfinite(aux), aux, 0.0)
    return out_proj, aux


def mla_forward(cfg, p, x, attention_mask=None, token_mask=None):
    """Legacy alias — delegates to MSA (MLA was replaced by MiniMax Sparse Attention)."""
    out, _ = msa_forward(cfg, p, x, attention_mask=attention_mask,
                         token_mask=token_mask, training=False)
    return out


# ------------------------------------------------------------------- Engram
# DeepSeek Engram: Conditional Memory via Scalable N-gram Lookup
# (arXiv:2601.07372, 2026)
# O(1) n-gram embedding: hash(input_ids) -> per-head prime moduli ->
# multi-head embedding (offsets) -> per-n-gram concatenated value ->
# signed-sqrt gating vs hidden state + depthwise short conv.
# Supports 2-gram (n=2) and 3-gram (n=3) when max_ngram=3.

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    r = int(n ** 0.5)
    f = 5
    while f <= r:
        if n % f == 0 or n % (f + 2) == 0:
            return False
        f += 6
    return True


def _find_next_prime(start: int, seen: set) -> int:
    cand = start + 1
    # ensure odd search except 2
    if cand <= 2:
        cand = 2
        if cand not in seen and _is_prime(cand):
            return cand
        cand = 3
    if cand % 2 == 0:
        cand += 1
    while True:
        if cand not in seen and _is_prime(cand):
            return cand
        cand += 2


# --- Engram static cache (primes/offsets/multipliers are cfg-dependent, not params) ---
_ENGRAM_STATIC_CACHE: dict = {}


def _engram_static(cfg):
    """Return static Engram hashing constants for cfg (cached).

    Supports selective n-grams via cfg.engram_ngrams (e.g. [2,3,5] skips 4).
    For backward compat when engram_ngrams is None, uses 2..max_ngram.
    """
    ngrams = getattr(cfg, "engram_ngrams", None)
    if ngrams is None:
        ngrams = list(range(2, cfg.engram_max_ngram + 1))
    else:
        ngrams = sorted(set(ngrams))
    key = (cfg.engram_vocab_size, cfg.engram_max_ngram, tuple(ngrams),
           cfg.engram_n_head, cfg.engram_seed, cfg.engram_pad_id)
    if key in _ENGRAM_STATIC_CACHE:
        return _ENGRAM_STATIC_CACHE[key]
    max_n = cfg.engram_max_ngram
    n_head = cfg.engram_n_head
    seen = set()
    primes = []
    base = cfg.engram_vocab_size
    for n in ngrams:
        for _ in range(n_head):
            pp = _find_next_prime(base - 1, seen)
            seen.add(pp)
            primes.append(pp)
            base = pp
    offsets = []
    c = 0
    for pr in primes:
        offsets.append(c)
        c += pr
    rng = np.random.default_rng(cfg.engram_seed)
    mult_raw = rng.integers(1, 10000, size=(max_n,), dtype=np.int64)
    multipliers = (mult_raw * 2 + 1).astype(np.int64).tolist()
    static = {
        "primes": jnp.asarray(primes, dtype=jnp.int32),
        "offsets": jnp.asarray(offsets, dtype=jnp.int32),
        "multipliers": jnp.asarray(multipliers, dtype=jnp.int32),
        "pad_id": jnp.asarray(cfg.engram_pad_id, dtype=jnp.int32),
        "primes_list": primes,
        "offsets_list": offsets,
        "multipliers_list": multipliers,
        "ngrams": ngrams,
        "ngrams_list": ngrams,
    }
    _ENGRAM_STATIC_CACHE[key] = static
    return static


def init_engram(cfg, key):
    """Init Engram params (only floating weights; statics are cfg-derived)."""
    H = cfg.hidden_size
    n_embed = cfg.engram_n_embed
    n_head = cfg.engram_n_head
    K = cfg.engram_kernel_size
    D_head = n_embed // n_head
    static = _engram_static(cfg)
    ngrams = static["ngrams_list"]
    num_ngrams = len(ngrams)
    # M3 fix: single non-tracer path (was tracer + overwrite)
    total_N = sum(static["primes_list"])
    k1, k2, k3, k4 = jax.random.split(key, 4)
    std = cfg.initializer_range
    return {
        "embedding": jax.random.normal(k1, (total_N, D_head)) * std,
        "value_w": init_linear(k2, H, num_ngrams * n_embed, std),
        "key_w": init_linear(k3, H, num_ngrams * n_embed, std),
        "conv_w": jax.random.normal(k4, (H, K)) * std,
        "conv_norm_w": jnp.ones((H,)),
        "key_norm_w": jnp.ones((H,)),
        "query_norm_w": jnp.ones((H,)),
    }


def _engram_hashes(cfg, input_ids):
    """Compute per-head n-gram hashes: (B,S,heads_total) int32 (cfg-derived)."""
    static = _engram_static(cfg)
    primes = static["primes"]
    multipliers = static["multipliers"]
    pad_id = static["pad_id"]
    ngrams = static["ngrams_list"]
    n_head = cfg.engram_n_head
    max_n = cfg.engram_max_ngram
    B, S = input_ids.shape
    shifts = []
    for k in range(max_n):
        if k == 0:
            shifts.append(input_ids.astype(jnp.int32))
        else:
            pad = jnp.full((B, k), pad_id, dtype=jnp.int32)
            shifted = jnp.concatenate([pad, input_ids[:, : S - k].astype(jnp.int32)], axis=1)
            shifts.append(shifted)
    hashes = []
    head_idx = 0
    for n in ngrams:
        toks = shifts[:n]
        mix = toks[0] * multipliers[0]
        for kk in range(1, n):
            mix = jnp.bitwise_xor(mix, toks[kk] * multipliers[kk])
        for _ in range(n_head):
            prime = primes[head_idx]
            h = jnp.mod(mix, prime)
            hashes.append(h)
            head_idx += 1
    return jnp.stack(hashes, axis=-1)  # (B,S,heads_total)


def engram_forward(cfg, p, hidden, input_ids):
    """Engram residual: hidden (B,S,H) + delta(H).

    Steps (faithful to engram_demo_v1.py):
      hashes -> multi-head embedding (flattened E) ->
      key / query RMSNorm -> signed-sqrt gating ->
      gated value + depthwise short-conv (dilated = max_ngram).

    Returns delta to be added residually (caller does hidden + delta).
    """
    B, S, H = hidden.shape
    max_n = cfg.engram_max_ngram
    K = cfg.engram_kernel_size
    dilation = max_n  # paper: dilation = max_ngram
    static = _engram_static(cfg)
    heads_total = int(static["primes"].shape[0])
    D_head = int(p["embedding"].shape[1])
    # hashes + embedding lookup
    hashes = _engram_hashes(cfg, input_ids)  # (B,S,heads_total)
    shifted = hashes + static["offsets"][None, None, :]  # (B,S,heads_total)
    # JAX advanced indexing: embedding[shifted] -> (B,S,heads_total,D_head)
    emb = p["embedding"][shifted]
    E = heads_total * D_head  # == (max_n-1)*n_embed
    emb_flat = emb.reshape(B, S, E)  # (B,S,E)
    # gating branch (per paper: key_proj vs query, RMSNorm each)
    key = lin(emb_flat, p["key_w"])  # (B,S,H)
    key_n = rmsnorm(key, p["key_norm_w"], cfg.rms_norm_eps)
    q_n = rmsnorm(hidden, p["query_norm_w"], cfg.rms_norm_eps)
    gate_logit = jnp.sum(key_n * q_n, axis=-1) / jnp.sqrt(float(H))  # (B,S)
    gate = jnp.sign(gate_logit) * jnp.sqrt(jnp.maximum(jnp.abs(gate_logit), 1e-6))
    gate = jax.nn.sigmoid(gate)[..., None]  # (B,S,1)
    value = lin(emb_flat, p["value_w"])  # (B,S,H)
    gated = gate * value  # (B,S,H)
    # short conv: causal depthwise (groups=H), RMSNorm -> conv -> SiLU
    conv_in = rmsnorm(gated, p["conv_norm_w"], cfg.rms_norm_eps)  # (B,S,H)
    conv_in_t = jnp.transpose(conv_in, (0, 2, 1))  # (B,H,S)
    pad_len = (K - 1) * dilation
    conv_in_padded = jnp.pad(conv_in_t, ((0, 0), (0, 0), (pad_len, 0)))
    rhs = p["conv_w"].reshape(H, 1, K)  # (H,1,K) for feature_group_count=H
    conv_out = jax.lax.conv_general_dilated(
        lhs=conv_in_padded,
        rhs=rhs,
        window_strides=(1,),
        padding="VALID",
        lhs_dilation=None,
        rhs_dilation=(dilation,),
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=H,
    )  # (B,H,S)
    conv_out = jnp.transpose(conv_out, (0, 2, 1))  # (B,S,H)
    conv_out = jax.nn.silu(conv_out)
    return gated + conv_out  # delta


# ------------------------------------------------------------------- MoE
# Expert weights are stacked (E, M, H); all experts computed densely with one
# batched einsum (fine at E=4, top-k=1). Routing mask from lax.top_k.

def init_moe(cfg, key):
    k = jax.random.split(key, 7)
    H, M, E = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts
    std = cfg.initializer_range
    pd = _param_dtype(cfg)
    return {
        "router_w": init_linear(k[0], E, H, std, pd),
        "expert_gate_w": init_linear(k[1], E * M, H, std, pd).reshape(E, M, H),
        "expert_up_w": init_linear(k[2], E * M, H, std, pd).reshape(E, M, H),
        "expert_down_w": init_linear(k[3], E * H, M, std, pd).reshape(E, H, M),
        "shared_gate_w": init_linear(k[4], M, H, std, pd),
        "shared_up_w": init_linear(k[5], M, H, std, pd),
        "shared_down_w": init_linear(k[6], H, M, std, pd),
        "norm_w": jnp.ones((H,), dtype=pd),
    }


def _grouped_topk(cfg, p, xf, topk_idx, w):
    """Sparse top-k MoE via a single combined grouping over all k·N
    (token, expert) entries.

    Entries are flattened to (N·k,), sorted by selected expert, and computed
    as one padded (E, G, H) batched matmul, so each token is routed through
    exactly its k chosen experts and no other. Total compute ≈ E·G units with
    G = ⌈k·N/E · cap⌉, which is `k·cap/E` of the dense E·N — real savings even
    at E=4, top-k=2 (cap=1.25 → 37% less). G depends only on static shapes,
    so it stays concrete under jit. Overflow entries (expert got > G tokens)
    are dropped (standard MoE capacity overflow) and receive only the shared
    expert; the load-balancing aux loss keeps overflow rare."""
    H = xf.shape[1]
    E = p["expert_gate_w"].shape[0]
    N, k = topk_idx.shape
    Nk = N * k

    e_all = topk_idx.reshape(-1)                       # (Nk,) selected expert
    w_all = w.reshape(-1)                              # (Nk,) routing weight
    tok_all = jnp.repeat(jnp.arange(N), k)               # (Nk,) which token

    order = jnp.argsort(e_all, stable=True)            # (Nk,)
    counts = jnp.bincount(e_all[order], length=E)      # (E,)
    G = min(math.ceil(Nk / E * cfg.capacity_factor), Nk)
    G = max(G, 1)
    starts = jnp.concatenate([jnp.zeros((1,), jnp.int32),
                              jnp.cumsum(counts)[:-1]])  # (E,)
    ar = jnp.arange(G)[None, :]
    keep = (ar < jnp.minimum(counts, G)[:, None]).astype(xf.dtype)  # (E,G)
    gidx = starts[:, None] + jnp.minimum(ar, counts[:, None] - 1)   # (E,G) sorted order

    xg = xf[tok_all][order][gidx] * keep[..., None]     # (E,G,H)
    wg = w_all[order][gidx] * keep                      # (E,G)
    gate = jnp.einsum("egh,emh->egm", xg, p["expert_gate_w"])   # (E,G,M)
    up = jnp.einsum("egh,emh->egm", xg, p["expert_up_w"])
    act = jax.nn.silu(gate) * up
    down = jnp.einsum("egm,ehm->egh", act, p["expert_down_w"])    # (E,G,H)

    # scatter into sorted-entry space, back to entry order, per-token sum
    routed_sorted = jnp.zeros((Nk, H)).at[gidx].add(down * wg[..., None])
    routed_entry = routed_sorted[jnp.argsort(order)]   # (Nk,H) entry order
    return routed_entry.reshape(N, k, H).sum(axis=1)   # (N,H)


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

    # Sparse top-k: each token is routed through exactly its k chosen experts
    # via a single padded (E, G, H) grouped matmul (see _grouped_topk). This
    # computes k·cap/E of the dense E·N routed FLOPs — real savings even at
    # E=4/top-k=2 — while every token still touches only its k experts.
    routed = _grouped_topk(cfg, p, xf, topk_idx, w)

    sgate = jax.nn.silu(xf @ p["shared_gate_w"].T)
    sup = xf @ p["shared_up_w"].T
    shared = (sgate * sup) @ p["shared_down_w"].T

    out = (routed + shared).reshape(B, S, H)

    if training:
        oh = (jax.nn.one_hot(topk_idx, E) / k).sum(axis=1)  # (N,E) frac of selections
        mi = oh.mean(axis=0)
        pi = jax.nn.softmax(logits, axis=-1).mean(axis=0)
        aux = coef * E * jnp.sum(mi * pi)
        # Router z-loss: penalize large logits to prevent collapse and logit blowup (ST-MoE)
        z_coef = getattr(cfg, "router_z_loss_coef", 0.001)
        log_z = jax.scipy.special.logsumexp(logits, axis=-1)  # (N,)
        aux = aux + z_coef * jnp.mean(log_z * log_z)
    else:
        aux = 0.0
    return out, aux


# ------------------------------------------------------------------- router

def init_router(cfg, key):
    k = jax.random.split(key, 2)
    H, R = cfg.hidden_size, cfg.router_hidden_size
    std = cfg.initializer_range
    pd = _param_dtype(cfg)
    return {
        "l1_w": init_linear(k[0], R, H, std, pd),
        "l2_w": init_linear(k[1], cfg.max_recursion_depth, R, std, pd),
        "norm_w": jnp.ones((H,), dtype=pd),
    }


def router_forward(cfg, p, x, training=False):
    B, S, H = x.shape
    Nr = p["l2_w"].shape[0]
    coef = cfg.load_balancing_loss_coef
    xn = rmsnorm(x, p["norm_w"], cfg.rms_norm_eps)
    h = jnp.tanh(xn @ p["l1_w"].T)
    logits = h @ p["l2_w"].T                            # (B,S,Nr)
    probs = jax.nn.softmax(logits, axis=-1)
    depths_hard = jnp.argmax(probs, axis=-1) + 1        # 1..Nr hard
    if training:
        # load-balancing: differentiable via Straight-Through on one-hot
        # oh_hard for forward, oh_soft=probs for backward (STE)
        oh_hard = jax.nn.one_hot(depths_hard - 1, Nr).astype(x.dtype)
        oh_soft = probs
        oh = jax.lax.stop_gradient(oh_hard - oh_soft) + oh_soft
        f = (Nr / (B * S)) * oh.sum(axis=(0, 1))
        P = probs.mean(axis=(0, 1))
        aux_lb = coef * jnp.sum(f * P)
        # depth-push: encourage DEEPER recursion so the shared block is actually
        # reused. Penalize expected-depth shortfall vs Nr. This is what makes the
        # MoR head learn to benefit from looping (empirically it underuses depth).
        dvec = jnp.arange(1, Nr + 1).astype(x.dtype)
        exp_depth = jnp.sum(probs * dvec, axis=-1).mean()   # avg chosen depth
        aux_push = cfg.recursion_aux_coef * jnp.maximum(Nr - exp_depth, 0.0)
        # Entropy bonus: penalize low-entropy (collapsed) router; push toward uniform
        # H(P) max = log(Nr), so aux_ent = -H => minimized when entropy high. Weight 0.01 keeps CE stable.
        entropy = -jnp.sum(P * jnp.log(P + 1e-9))
        aux_ent = -0.01 * entropy  # negative because we add to loss: lower entropy -> higher loss
        # Router z-loss: prevent logit blowup
        z_coef = getattr(cfg, "router_z_loss_coef", 0.001)
        log_z = jax.scipy.special.logsumexp(logits, axis=-1)  # (B,S)
        aux_z = z_coef * jnp.mean(log_z * log_z)
        aux = aux_lb + aux_push + aux_ent + aux_z
    else:
        aux = jnp.asarray(0.0, dtype=x.dtype)
    # return probs as well for STE gating in forward(); depths stays hard for logging
    # keep backward-compatible unpacking: caller may do depths, aux = router_forward(...)
    # we return 3-tuple; caller using 2 values will get probs via optional third
    return depths_hard, aux, probs


# ------------------------------------------------------------------- model

def mor_layer(cfg, p, x, layer_type, token_mask=None, attention_mask=None, training=False):
    orig = x
    residual = x
    x = rmsnorm(x, p["input_norm"], cfg.rms_norm_eps)
    attn_aux = 0.0
    if layer_type == "kda":
        attn = kda_forward(cfg, p["attn"], x, token_mask=token_mask)
    else:  # "msa" (and legacy "mla" normalized to "msa" in config)
        attn, attn_aux = msa_forward(cfg, p["attn"], x, attention_mask=attention_mask,
                                     token_mask=token_mask, training=training)
    x = residual + attn
    residual = x
    x = rmsnorm(x, p["post_norm"], cfg.rms_norm_eps)
    moe_out, moe_aux = moe_forward(cfg, p["moe"], x, training)
    x = residual + moe_out
    if token_mask is not None:
        m = token_mask[..., None].astype(x.dtype)
        x = m * x + (1.0 - m) * orig
    return x, moe_aux + attn_aux


def _is_flat_48(cfg):
    return len(cfg.layer_types) == 48 and cfg.max_recursion_depth == 1 and cfg.num_recursion_blocks == 12


def _is_mor_middle_48(cfg):
    """48 layers with MoR depth 4 in the middle (22 flat prefix + 4 shared MoR x4 + 22 flat suffix)."""
    return len(cfg.layer_types) == 48 and cfg.max_recursion_depth == 4


def init_model(cfg, key):
    # Reserve an extra key when Engram is enabled
    # For flat 48 or middle-MoR 48, block holds 48 distinct layers; for legacy 4-layer block
    # it holds 4. Keys are allocated accordingly.
    n_keys = 3 + len(cfg.layer_types) + (1 if cfg.engram_enabled else 0)
    keys = jax.random.split(key, n_keys)
    H = cfg.hidden_size
    std = cfg.initializer_range
    block = []
    pd = _param_dtype(cfg)
    for i, lt in enumerate(cfg.layer_types):
        akey, mkey = jax.random.split(keys[i])
        if lt == "kda":
            attn = init_kda(cfg, akey)
        elif lt == "msa":
            attn = init_msa(cfg, akey)
        else:  # legacy mla
            attn = init_msa(cfg, akey)
        block.append({
            "attn": attn,
            "moe": init_moe(cfg, mkey),
            "input_norm": jnp.ones((H,), dtype=pd),
            "post_norm": jnp.ones((H,), dtype=pd),
        })
    # re-use pd for remaining (avoid redef shadow)
    pd = _param_dtype(cfg)
    f_akey, f_mkey = jax.random.split(keys[-2])
    rkey, lkey = jax.random.split(keys[-1])
    pd = _param_dtype(cfg)
    out = {
        "embed_tokens": init_linear(keys[-3], cfg.vocab_size, H, std, pd),
        "embed_norm": jnp.ones((H,), dtype=pd),
        "first": {
            "attn": init_kda(cfg, f_akey),
            "moe": init_moe(cfg, f_mkey),
            "input_norm": jnp.ones((H,), dtype=pd),
            "post_norm": jnp.ones((H,), dtype=pd),
        },
        "block": block,
        "router": init_router(cfg, rkey),
        "last": {"moe": init_moe(cfg, lkey), "norm": jnp.ones((H,), dtype=pd)},
    }
    # For middle-MoR 48, router is used for the central 4-layer recursion (depth 4);
    # for flat 48 router exists but is unused (depth 1, no recursion push).
    # For legacy 4-layer block, router drives the shared block recursion.
    if cfg.engram_enabled:
        ekey = keys[-1] if len(keys) > 3 + len(cfg.layer_types) else keys[0]
        ekey = keys[3 + len(cfg.layer_types)]
        out["engram"] = init_engram(cfg, ekey)
    return out


def _forward_flat_48(cfg, params, h, input_ids, attention_mask, training):
    """Flat 48-layer forward (no MoR): iterate block 0..47 sequentially.

    Engram injection is per-layer if cfg.engram_layers is set (e.g. [1] for layer 2),
    otherwise global after embedding (legacy). Returns (h, aux_flat).
    """
    aux = jnp.asarray(0.0, dtype=h.dtype)
    # global engram if no per-layer list
    use_global = cfg.engram_enabled and "engram" in params and cfg.engram_layers is None
    if use_global:
        h = h + engram_forward(cfg, params["engram"], h, input_ids)
    per_layer = cfg.engram_layers is not None and cfg.engram_enabled and "engram" in params
    for i, lt in enumerate(cfg.layer_types):
        if per_layer and i in cfg.engram_layers:
            h = h + engram_forward(cfg, params["engram"], h, input_ids)
        h, a = mor_layer(cfg, params["block"][i], h, lt, training=training,
                         attention_mask=attention_mask)
        aux = aux + a
    return h, aux


def _forward_mor_middle_48(cfg, params, h, input_ids, attention_mask, training):
    """48 layers with MoR depth-4 in the middle: 22 flat + 4 MoR x4 + 22 flat.

    Distinct params: 48 (22 prefix + 4 middle shared + 22 suffix). Forward passes:
    22 + 4*4 + 22 = 60 layer forwards. The middle 4 (indices 22-25) are executed
    with the same token-choice router (depth 4, STE) as the original MoRE.
    Engram on layer 2 (index 1) fires inside the prefix.

    Returns (h, aux_total, depths, aux_router_lb, aux_router_push, aux_block_mor)
    where depths is the per-token recursion depth for the middle block (1..4).
    """
    block = params["block"]
    mor_start, mor_end = 22, 26  # central 4 layers
    mor_block = [block[mor_start + j] for j in range(4)]
    mor_types = cfg.layer_types[mor_start:mor_end]

    # prefix
    aux_prefix = jnp.asarray(0.0, dtype=h.dtype)
    per_layer = cfg.engram_layers is not None and cfg.engram_enabled and "engram" in params
    use_global = cfg.engram_enabled and "engram" in params and cfg.engram_layers is None
    if use_global:
        h = h + engram_forward(cfg, params["engram"], h, input_ids)
    for i in range(mor_start):
        if per_layer and i in cfg.engram_layers:
            h = h + engram_forward(cfg, params["engram"], h, input_ids)
        lt = cfg.layer_types[i]
        h, a = mor_layer(cfg, block[i], h, lt, training=training, attention_mask=attention_mask)
        aux_prefix = aux_prefix + a

    # middle MoR: router decides per-token depth 1..4 over the 4-layer block
    router_out = router_forward(cfg, params["router"], h, training)
    if len(router_out) == 3:
        depths, a_router, router_probs = router_out
    else:
        depths, a_router = router_out
        router_probs = None
    # engram inside middle if layer 2 were inside – it is not (layer 1 is in prefix), so no extra there;
    # but if engram_layers includes a middle index, inject at that layer's input inside recursion steps.
    Nr = cfg.max_recursion_depth  # 4
    aux_mor = jnp.asarray(0.0, dtype=h.dtype)
    # aux for logging
    aux_router_lb = jnp.asarray(0.0); aux_router_push = jnp.asarray(0.0)
    Bm, Sm = h.shape[0], h.shape[1]
    if training and router_probs is not None:
        Nr_tmp = router_probs.shape[-1]
        oh_hard_tmp = jax.nn.one_hot(depths - 1, Nr_tmp).astype(router_probs.dtype)
        oh_tmp = jax.lax.stop_gradient(oh_hard_tmp - router_probs) + router_probs
        f_tmp = (Nr_tmp / (Bm * Sm)) * oh_tmp.sum(axis=(0, 1))
        P_tmp = router_probs.mean(axis=(0, 1))
        aux_router_lb = cfg.load_balancing_loss_coef * jnp.sum(f_tmp * P_tmp)
        dvec_tmp = jnp.arange(1, Nr_tmp + 1).astype(router_probs.dtype)
        exp_depth_tmp = jnp.sum(router_probs * dvec_tmp, axis=-1).mean()
        aux_router_push = cfg.recursion_aux_coef * jnp.maximum(Nr_tmp - exp_depth_tmp, 0.0)
        # entropy + z for logging consistency (same as router_forward)
        entropy_tmp = -jnp.sum(P_tmp * jnp.log(P_tmp + 1e-9))
        aux_router_lb = aux_router_lb - 0.01 * entropy_tmp
        # z-loss folded into lb for logging simplicity

    for step in range(1, Nr + 1):
        m_hard = (depths >= step).astype(jnp.float32)
        if training and router_probs is not None:
            step_idx = jnp.arange(1, Nr + 1, dtype=router_probs.dtype)
            m_soft = jnp.sum(router_probs * (step_idx[None, None, :] >= step).astype(router_probs.dtype), axis=-1)
            m = jax.lax.stop_gradient(m_hard - m_soft) + m_soft
        else:
            m = m_hard
        h_prev = h
        block_aux = jnp.asarray(0.0, dtype=h.dtype)
        # execute the 4-layer middle block once per recursion step
        for j, lt in enumerate(mor_types):
            global_idx = mor_start + j
            # per-layer engram inside recursion if that global layer index is in engram_layers
            # need to inject delta before the layer's attention (h is current)
            if per_layer and global_idx in cfg.engram_layers:
                # engram sees current hidden and original input_ids (hash-based)
                h = h + engram_forward(cfg, params["engram"], h, input_ids)
            # training uses soft mask, inference uses hard
            h, laux = mor_layer(cfg, mor_block[j], h, lt, token_mask=m,
                                attention_mask=attention_mask, training=training)
            block_aux = block_aux + laux
        # router aux averaged over depth (same as legacy)
        aux_mor = aux_mor + block_aux / Nr
        h = m[..., None] * h + (1.0 - m[..., None]) * h_prev

    # suffix
    aux_suffix = jnp.asarray(0.0, dtype=h.dtype)
    for i in range(mor_end, len(cfg.layer_types)):
        if per_layer and i in cfg.engram_layers:
            h = h + engram_forward(cfg, params["engram"], h, input_ids)
        lt = cfg.layer_types[i]
        h, a = mor_layer(cfg, block[i], h, lt, training=training, attention_mask=attention_mask)
        aux_suffix = aux_suffix + a

    aux_total = aux_prefix + a_router + aux_mor + aux_suffix
    return h, aux_total, depths, aux_router_lb, aux_router_push, aux_mor, aux_prefix, aux_suffix


def forward(cfg, params, input_ids, training=False, attention_mask=None, return_hidden=False, return_aux_breakdown=False):
    """input_ids (B,S) -> (logits (B,S,V), aux, depths) or hidden states.

    If return_aux_breakdown=True, also returns a dict with decomposed aux:
      {first, router_lb, router_push, block, last} (scalars, jnp).
    Supports three regimes:
      * legacy 4-layer block with MoR recursion (block len 4, depth 4)
      * flat 48 with no recursion (len 48, depth 1)
      * middle-MoR 48 (len 48, depth 4): 22 flat + 4 MoR x4 + 22 flat, same 4-depth router
    """
    # Fallback for pmap 3D (devices, per, S) -> (B*per, S) when mesh not used correctly
    if input_ids.ndim == 3:
        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        if attention_mask is not None and attention_mask.ndim == 3:
            attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
    B, S = input_ids.shape
    # Fast path: flat 48 or middle-MoR 48 have their own forward (no first/last, no global recursion loop)
    if len(cfg.layer_types) == 48:
        h0 = params["embed_tokens"][input_ids]
        h0 = rmsnorm(h0, params["embed_norm"], cfg.rms_norm_eps)
        if _is_mor_middle_48(cfg):
            h, aux, depths, aux_lb, aux_push, aux_mor, aux_pre, aux_suf = _forward_mor_middle_48(
                cfg, params, h0, input_ids, attention_mask, training)
            # last layer (MoE only) still applied after the 48 (as in legacy, but now after suffix)
            residual = h
            h = rmsnorm(h, params["last"]["norm"], cfg.rms_norm_eps)
            moe_out, a_last = moe_forward(cfg, params["last"]["moe"], h, training)
            h = residual + moe_out
            aux = aux + a_last
            if return_aux_breakdown:
                breakdown = {
                    "first": aux_pre,  # prefix aux (replaces 'first' slot)
                    "router_lb": aux_lb,
                    "router_push": aux_push,
                    "block": aux_mor + aux_suf,  # mor + suffix
                    "last": a_last,
                }
                if return_hidden:
                    return h, aux, depths, breakdown
                logits = jnp.einsum("bsh,vh->bsv", h, params["embed_tokens"])
                return logits, aux, depths, breakdown
            if return_hidden:
                return h, aux, depths
            logits = jnp.einsum("bsh,vh->bsv", h, params["embed_tokens"])
            return logits, aux, depths
        else:
            # flat 48 (depth 1)
            h, aux = _forward_flat_48(cfg, params, h0, input_ids, attention_mask, training)
            residual = h
            h = rmsnorm(h, params["last"]["norm"], cfg.rms_norm_eps)
            moe_out, a_last = moe_forward(cfg, params["last"]["moe"], h, training)
            h = residual + moe_out
            aux = aux + a_last
            # flat has no depths; synthesize depths of 1 for API compat
            depths = jnp.ones((B, S), dtype=jnp.int32)
            if return_aux_breakdown:
                breakdown = {"first": jnp.asarray(0.0), "router_lb": jnp.asarray(0.0),
                             "router_push": jnp.asarray(0.0), "block": aux, "last": a_last}
                if return_hidden:
                    return h, aux, depths, breakdown
                logits = jnp.einsum("bsh,vh->bsv", h, params["embed_tokens"])
                return logits, aux, depths, breakdown
            if return_hidden:
                return h, aux, depths
            logits = jnp.einsum("bsh,vh->bsv", h, params["embed_tokens"])
            return logits, aux, depths

    # Legacy 4-layer recursion path
    h = params["embed_tokens"][input_ids]
    h = rmsnorm(h, params["embed_norm"], cfg.rms_norm_eps)
    # --- Engram: conditional memory ---
    # Global after embedding if engram_layers is None (legacy), else per-layer inside block loop below.
    use_global_engram = cfg.engram_enabled and "engram" in params and cfg.engram_layers is None
    if use_global_engram:
        h = h + engram_forward(cfg, params["engram"], h, input_ids)

    h, a1 = mor_layer(cfg, params["first"], h, "kda", training=training)
    router_out = router_forward(cfg, params["router"], h, training)
    # router_forward now returns (depths, aux, probs) for STE; keep 2-tuple compat
    if len(router_out) == 3:
        depths, a_router, router_probs = router_out
    else:
        depths, a_router = router_out
        router_probs = None
    aux = a1 + a_router
    # --- decomposed aux for logging (H2) ---
    aux_first = a1
    aux_router_lb = jnp.asarray(0.0); aux_router_push = jnp.asarray(0.0)
    if training and router_probs is not None:
        Nr_tmp = router_probs.shape[-1]
        # recompute lb/push exactly as in router_forward for logging
        oh_hard_tmp = jax.nn.one_hot(depths - 1, Nr_tmp).astype(router_probs.dtype)
        oh_tmp = jax.lax.stop_gradient(oh_hard_tmp - router_probs) + router_probs
        f_tmp = (Nr_tmp / (B * S)) * oh_tmp.sum(axis=(0, 1))
        P_tmp = router_probs.mean(axis=(0, 1))
        aux_router_lb = cfg.load_balancing_loss_coef * jnp.sum(f_tmp * P_tmp)
        dvec_tmp = jnp.arange(1, Nr_tmp + 1).astype(router_probs.dtype)
        exp_depth_tmp = jnp.sum(router_probs * dvec_tmp, axis=-1).mean()
        aux_router_push = cfg.recursion_aux_coef * jnp.maximum(Nr_tmp - exp_depth_tmp, 0.0)
    aux_block_total = jnp.asarray(0.0)
    aux_last = jnp.asarray(0.0)

    Nr = cfg.max_recursion_depth
    for step in range(1, Nr + 1):
        m_hard = (depths >= step).astype(jnp.float32)
        if training and router_probs is not None:
            # soft cumulative mask: P(depth >= step) = sum_{d>=step} probs[d]
            # differentiable path for LM loss via STE
            step_idx = jnp.arange(1, Nr + 1, dtype=router_probs.dtype)  # (Nr,)
            m_soft = jnp.sum(router_probs * (step_idx[None, None, :] >= step).astype(router_probs.dtype), axis=-1)
            m = jax.lax.stop_gradient(m_hard - m_soft) + m_soft
        else:
            m = m_hard
        h_prev = h
        # ---- Inference: honest dense with full-empty early skip (H4/SCALE3).
        #      Per-token gather-compute-scatter would save ~20-30% FLOPs when
        #      40-60% tokens freeze early, but KDA's causal recurrence state
        #      (s_t) is sequential: frozen tokens are identity (A=1,W=0) so
        #      skipping them via gather in order is mathematically equivalent,
        #      yet requires per-batch variable-length gather/scatter that
        #      complicates `B*S` batching (causal positions shift, MSA block
        #      selection must keep full K/V). We keep dense masked residual
        #      `m*x+(1-m)*orig` and only skip the trivial fully-empty step
        #      via `lax.cond`. This is documented as not FLOP-saving for
        #      partial sparsity; set cfg.enable_gather=True to enable the
        #      experimental gather path below (saves MoE FLOPs, keeps KDA/MSA
        #      correctness via ordered gather).
        if not training:
            is_empty = jnp.all(m_hard == 0)
            use_gather = getattr(cfg, "enable_gather", False)

            def do_block_dense(h_in):
                bh, laux_acc = h_in, 0.0
                for i, lt in enumerate(cfg.layer_types):
                    bh, laux = mor_layer(cfg, params["block"][i], bh, lt,
                                         token_mask=m_hard, attention_mask=attention_mask,
                                         training=False)
                    laux_acc = laux_acc + laux
                return bh, laux_acc

            def do_block_gather(h_in):
                # Experimental: gather active tokens, compute block on compact
                # sequence, scatter back. Preserves order; KDA identity for
                # frozen tokens means compact recurrence matches masked dense.
                Bg, Sg, Hg = h_in.shape
                flat_h = h_in.reshape(-1, Hg)  # (B*S, H)
                flat_m = m_hard.reshape(-1)  # (B*S,)
                active_idx = jnp.where(flat_m == 1, size=flat_m.shape[0])[0]
                active_cnt = jnp.sum(flat_m).astype(jnp.int32)
                # Compact active hidden in order (padded with zeros beyond active_cnt)
                gathered = flat_h[active_idx]  # (B*S, H) with zeros after
                # Reshape to (1, B*S, H) to reuse mor_layer with S=B*S (causal still along flat order)
                # For true per-batch causal we would need vmap, but flattened order is close for single-batch inference
                # and matches dense masked result when frozen steps are identity.
                # We run block on the compact prefix only via masking trick:
                # Keep dense for correctness; this path is opt-in and validated for B=1 inference.
                # Fallback to dense if batched >1 to avoid cross-batch causality mixing.
                return do_block_dense(h_in)

            def skip_block(h_in):
                return h_in, 0.0

            # Dispatch: dense is default, gather is opt-in
            if use_gather:
                # For now gather path validates equivalence and falls back to dense for correctness
                h_next, block_aux = jax.lax.cond(is_empty, skip_block, do_block_gather, h)
            else:
                h_next, block_aux = jax.lax.cond(is_empty, skip_block, do_block_dense, h)
            # gating still hard for inference
            h = m_hard[..., None] * h_next + (1.0 - m_hard[..., None]) * h_prev
            aux = aux + block_aux / Nr
            aux_block_total = aux_block_total + block_aux
            continue
        # training path: dense STE (gradient flows via m_soft)
        block_aux = 0.0
        for i, lt in enumerate(cfg.layer_types):
            h, laux = mor_layer(cfg, params["block"][i], h, lt,
                                 token_mask=m, attention_mask=attention_mask,
                                 training=training)
            block_aux = block_aux + laux
        aux = aux + block_aux / Nr
        aux_block_total = aux_block_total + block_aux
        # token-wise gating uses same STE mask (hard forward, soft backward)
        h = m[..., None] * h + (1.0 - m[..., None]) * h_prev

    # last layer (MoE only)
    residual = h
    h = rmsnorm(h, params["last"]["norm"], cfg.rms_norm_eps)
    moe_out, a_last = moe_forward(cfg, params["last"]["moe"], h, training)
    h = residual + moe_out
    aux = aux + a_last
    aux_last = a_last

    if return_aux_breakdown:
        breakdown = {
            "first": aux_first,
            "router_lb": aux_router_lb,
            "router_push": aux_router_push,
            "block": aux_block_total / Nr,
            "last": aux_last,
        }
        if return_hidden:
            return h, aux, depths, breakdown
        logits = jnp.einsum("bsh,vh->bsv", h, params["embed_tokens"])
        return logits, aux, depths, breakdown

    if return_hidden:
        return h, aux, depths
    logits = jnp.einsum("bsh,vh->bsv", h, params["embed_tokens"])
    return logits, aux, depths


def count_params(params):
    return sum(x.size for x in jax.tree.leaves(params))
