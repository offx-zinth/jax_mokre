"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz) for optax.

Canonical Muon update (Jordan 2024; Moonshot "Muon is Scalable" paper):

    buf = mu * buf + (1 - mu) * g
    d   = (1 - mu) * g + mu * buf          # nesterov
    O   = NewtonSchulz5(d)                 # approx UV^T, singular vals in [~0.7, 1.3]
    upd = -lr * scale * O                  # scale = sqrt(max(1, m/n)) on the
                                           # flattened (m, n) view of the leaf

The update RMS after scaling is ~lr regardless of matrix aspect ratio, so the
same LR schedule as AdamW is a sane default (--muon_lr_scale to tune).

Partitioning follows the papers' guidance:
- Muon: hidden weight matrices (ndim >= 2, both dims > 1), flattened to 2D
  when stacked (e.g. expert weights (E, M, H) -> (E*M, H)). Excluded by name:
  `embed_tokens` (tied embedding + LM head) and `init_state` (KDA recurrence
  state init -- orthogonalizing it has no meaning).
- AdamW: everything else -- embeddings/head, norms/gains/biases, and MoE +
  recursion router weights (routers are excluded by default; pass
  include_routers=True or --muon_on_routers to hand them to Muon too).

Memory notes (10 GB laptop / TPU HBM friendly):
- one momentum buffer per Muon leaf only (~66M of the 85M params);
- m + v Adam moments only for the remaining leaves;
- excluded slots hold scalar sentinels (TPU-friendly; XLA dislikes 0-sized
  dims and they caused `SIGSEGV STACK OVERFLOW` on TpuV5E8), so no memory is
  wasted on masked-out leaves (optax.partition/masked would allocate full-size
  zeros for every branch);
- Newton-Schulz runs in bf16 on TPU / fp32 on CPU with ~3x one leaf's bytes as
  transient buffers, via `lax.fori_loop` to avoid unrolling.
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple

import jax
import jax.numpy as jnp
import optax

# Quintic Newton-Schulz coefficients (Keller Jordan's canonical constants).
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315


def newton_schulz(G, steps=5, dtype=jnp.float32):
    """Approximate UV^T of G via the quintic iteration; sigma(O) in [0.7, 1.3].

    Runs in `dtype` (bf16 on TPU is far cheaper per matmul) and always leads
    with the smaller dimension so the iteration converges fast.
    Uses `lax.fori_loop` to avoid unrolling 5× matmuls per leaf into the HLO
    (large-model TPU compile would stack-overflow otherwise)."""
    a, b, c = NS_A, NS_B, NS_C
    X = G.astype(dtype)
    X = X / (jnp.linalg.norm(X) + 1e-7)
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    def body(i, X_):
        A = X_ @ X_.T
        B = b * A + c * (A @ A)
        return a * X_ + B @ X_

    X = jax.lax.fori_loop(0, steps, body, X)
    if transposed:
        X = X.T
    return X


def _path_name(path: Iterable[Any]) -> str:
    return "/".join(str(getattr(k, "value", k)) for k in path)


def make_muon_mask(params, exclude=("embed_tokens", "init_state"),
                   include_routers=False):
    """Python-bool pytree mirroring `params`: True -> Muon, False -> AdamW.

    Muon gets ndim >= 2 leaves with both dims > 1 whose path avoids `exclude`.
    Router weights (MoE `router_w`, recursion `router/*`) stay on AdamW unless
    include_routers=True."""
    def fn(path, leaf):
        name = _path_name(path)
        if leaf.ndim < 2 or min(leaf.shape) <= 1:
            return False
        if any(ex in name for ex in exclude):
            return False
        if not include_routers and "router" in name:
            return False
        return True
    return jax.tree_util.tree_map_with_path(fn, params)


def muon_param_counts(params, mask):
    """(n_muon_params, n_adam_params) as static Python ints, for logging."""
    n_mu, n_ad = 0, 0
    for leaf, m in zip(jax.tree.leaves(params), jax.tree.leaves(mask)):
        n = 1
        for s in leaf.shape:
            n *= s
        if m:
            n_mu += n
        else:
            n_ad += n
    return n_mu, n_ad


class MuonAdamWState(NamedTuple):
    count: Any        # int32 scalar step counter (adam bias correction)
    mu_mom: Any       # momentum buffer | zero-size sentinel per leaf
    adam_mu: Any      # first moment | zero-size sentinel per leaf
    adam_nu: Any      # second moment | zero-size sentinel per leaf


def muon_adamw(mask, *, momentum=0.95, nesterov=True, ns_steps=5,
               ns_dtype=jnp.float32, muon_lr_scale=1.0, muon_weight_decay=0.0,
               b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01):
    """Muon for mask==True leaves, decoupled AdamW for the rest.

    The returned transform reads the LIVE learning rate from the
    `learning_rate=` kwarg on every update() call (warmup / cosine / NaN
    rollback all reach it), matching adamw mode's inject_hyperparams."""

    def update_fn(grads, state, params=None, learning_rate=None, **_):
        if learning_rate is None:
            raise ValueError("muon_adamw requires learning_rate=... kwarg")
        del _
        if params is None:
            raise ValueError("muon_adamw requires params")
        c = state.count + 1
        lr = learning_rate

        # Every tree has exactly one leaf per param position (zero-size
        # sentinels included), so flat leaf lists align across all of them.
        gs, td = jax.tree_util.tree_flatten(grads)
        ps = jax.tree_util.tree_leaves(params)
        moms = jax.tree_util.tree_leaves(state.mu_mom)
        mus = jax.tree_util.tree_leaves(state.adam_mu)
        nus = jax.tree_util.tree_leaves(state.adam_nu)
        ms = jax.tree_util.tree_leaves(mask)

        ups, bufs, mus_out, nus_out = [], [], [], []
        for g, p, mom, mu, nu, m in zip(gs, ps, moms, mus, nus, ms):
            if m:
                # ---- Muon ----
                buf = momentum * mom + (1.0 - momentum) * g
                d = (1.0 - momentum) * g + momentum * buf if nesterov else buf
                G2 = d.reshape(-1, d.shape[-1])   # stack leading axes -> 2D
                O = newton_schulz(G2, ns_steps, ns_dtype)
                scale = jnp.sqrt(jnp.maximum(1.0, G2.shape[0] / G2.shape[1]))
                upd = -(lr * muon_lr_scale * scale) * O.astype(g.dtype)
                upd = upd.reshape(g.shape) - lr * muon_weight_decay * p
            else:
                # ---- AdamW ----
                buf = mom                          # untouched sentinel
                mu = b1 * mu + (1.0 - b1) * g
                nu = b2 * nu + (1.0 - b2) * (g * g)
                mhat = mu / (1.0 - b1 ** c)
                vhat = nu / (1.0 - b2 ** c)
                upd = -(lr * mhat / (jnp.sqrt(vhat) + eps)) \
                    - lr * weight_decay * p
            ups.append(upd)
            bufs.append(buf)
            mus_out.append(mu)
            nus_out.append(nu)

        updates = jax.tree_util.tree_unflatten(td, ups)
        new_state = MuonAdamWState(
            count=c,
            mu_mom=jax.tree_util.tree_unflatten(td, bufs),
            adam_mu=jax.tree_util.tree_unflatten(td, mus_out),
            adam_nu=jax.tree_util.tree_unflatten(td, nus_out),
        )
        return updates, new_state

    def init_fn(params):
        # Scalar sentinels are TPU-friendly (XLA dislikes 0-sized dims and
        # they caused `SIGSEGV STACK OVERFLOW` on TpuV5E8). The sentinel
        # value is never used arithmetically for the other branch.
        def fresh_muon(p, m):
            return jnp.zeros(p.shape, p.dtype) if m else jnp.zeros((), p.dtype)

        def fresh_adam(p, m):
            return jnp.zeros(p.shape, p.dtype) if not m else jnp.zeros((), p.dtype)

        zero = jnp.zeros((), jnp.int32)
        return MuonAdamWState(
            count=zero,
            mu_mom=jax.tree.map(fresh_muon, params, mask),
            adam_mu=jax.tree.map(fresh_adam, params, mask),
            adam_nu=jax.tree.map(fresh_adam, params, mask),
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
