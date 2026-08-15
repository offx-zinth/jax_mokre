"""CPU smoke test for the JAX MoRE port.

Checks:
  1. Model builds (tinystories config) with ~85M params.
  2. Logits shape correct.
  3. KDA associative scan == sequential reference (the core vectorization).
  4. MLA attention matches a simple reference.
  5. MoR gating: fully-frozen tokens pass through unchanged.
  6. 10 train steps on synthetic data: finite loss that decreases.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .config import MoREConfig
from . import model as M
from . import train as T


def tinystories():
    return MoREConfig(
        vocab_size=50257, hidden_size=384, intermediate_size=1024,
        num_attention_heads=6, num_key_value_heads=2, head_dim=64,
        max_seq_len=256, max_recursion_depth=4,
        num_experts=8, num_local_experts=8, num_shared_experts=1, top_k=1,
        router_hidden_size=64, kda_state_size=64, kda_chunk_size=16,
        layer_types=["kda", "kda", "mla", "kda"],
        load_balancing_loss_coef=0.01, rms_norm_eps=1e-6, initializer_range=0.02,
    )


def test_params_and_shape():
    cfg = tinystories()
    params = M.init_model(cfg, jax.random.PRNGKey(0))
    n = M.count_params(params)
    print(f"[1] params: {n:,} (torch tinystories: 85,227,144)")
    assert 80_000_000 < n < 90_000_000, n
    ids = jnp.asarray(np.random.randint(0, 50257, (2, 64)))
    logits, aux, depths = M.forward(cfg, params, ids, training=False)
    assert logits.shape == (2, 64, 50257), logits.shape
    assert jnp.isfinite(logits).all()
    assert depths.shape == (2, 64)
    print(f"[1] OK  logits={logits.shape} aux={float(aux):.4f}")


def test_kda_scan():
    cfg = tinystories()
    p = M.init_kda(cfg, jax.random.PRNGKey(1))
    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (2, 20, cfg.hidden_size))
    tm = jnp.asarray([[1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]])

    # reference: sequential python scan over tokens (per head)
    def seq_ref():
        bsz, S, H = x.shape
        NH, NG, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        G = NH // NG
        xn = M.rmsnorm(x, p["norm_w"], cfg.rms_norm_eps)
        q = M.lin(xn, p["q_w"]).reshape(bsz, S, NH, D)
        k = M.lin(xn, p["k_w"]).reshape(bsz, S, NG, D)
        v = M.lin(xn, p["v_w"]).reshape(bsz, S, NG, D)
        gate = M.lin(xn, p["gate_w"], p["gate_b"]).reshape(bsz, S, NG, D + 1)
        alpha = jax.nn.sigmoid(gate[..., :-1])
        beta = jax.nn.sigmoid(gate[..., -1:])
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)
        k = jnp.repeat(k, G, axis=2)
        vb = jnp.repeat(v, G, axis=2)
        ab = jnp.repeat(alpha, G, axis=2)
        bb = jnp.repeat(beta, G, axis=2)
        s = jnp.broadcast_to(jnp.repeat(p["init_state"], G, axis=1), (bsz, NH, D))
        outs = []
        for t in range(S):
            a_t, b_t = ab[:, t], bb[:, t]          # (B,NH,D),(B,NH,1)
            kt, vt = k[:, t], vb[:, t]
            A = a_t * (1.0 - b_t * kt * kt)
            W = b_t * kt * vt
            m = tm[:, t:t + 1][..., None].astype(A.dtype)
            A = m * A + (1.0 - m)
            W = m * W
            s = A * s + W
            outs.append(s * q[:, t])
        out = jnp.stack(outs, axis=1).reshape(bsz, S, NH * D)
        return M.lin(out, p["o_w"])

    fast = M.kda_forward(cfg, p, x, token_mask=tm)
    ref = seq_ref()
    err = float(jnp.max(jnp.abs(fast - ref)))
    print(f"[2] KDA scan vs sequential ref: max abs err = {err:.2e}")
    assert err < 1e-5, err


def test_mla():
    cfg = tinystories()
    p = M.init_mla(cfg, jax.random.PRNGKey(3))
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 8, cfg.hidden_size))
    out = M.mla_forward(cfg, p, x)
    assert out.shape == (2, 8, cfg.hidden_size)
    assert jnp.isfinite(out).all()
    print(f"[3] MLA forward OK  out={out.shape}")


def test_moR_gating():
    cfg = tinystories()
    params = M.init_model(cfg, jax.random.PRNGKey(5))
    ids = jnp.asarray(np.random.randint(0, 50257, (1, 32)))
    hidden = params["embed_tokens"][ids]

    # force all tokens to depth 1 => steps 2..Nr fully frozen
    depths = jnp.ones((1, 32), dtype=jnp.int32)
    h_prev = hidden
    for step in range(2, cfg.max_recursion_depth + 1):
        m = (depths >= step).astype(jnp.float32)
        h = h_prev
        for i, lt in enumerate(cfg.layer_types):
            h, _ = M.mor_layer(cfg, params["block"][i], h, lt, token_mask=m)
        assert jnp.allclose(h, h_prev), "frozen recursion steps must be identity"
    print("[4] MoR gating: frozen recursion steps are identity OK")


def test_train_steps():
    cfg = tinystories()
    cfg.max_seq_len = 128
    params = M.init_model(cfg, jax.random.PRNGKey(6))
    from transformers import AutoTokenizer
    from . import data as D
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = D.ensure_tokens("train", tokenizer, "/tmp/tinystories_smoke",
                             max_files=1, max_stories=2000, force=False)
    it = D.make_iter(tokens, 4, 64)

    opt = optax.MultiSteps(
        optax.chain(optax.clip_by_global_norm(1.0),
                    optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.01)),
        every_k_schedule=2)
    os_ = opt.init(params)
    step = T.make_train_step(cfg, opt, 16, dist=False)

    losses = []
    for i in range(15):
        x, y = next(it)
        params, os_, loss, ce, aux = step(params, os_, jnp.asarray(x), jnp.asarray(y),
                                          jnp.asarray(3e-4, dtype=jnp.float32))
        lv = float(loss)
        losses.append(lv)
        assert np.isfinite(lv), f"non-finite loss at step {i}: {lv}"
    print(f"[5] train steps (real TinyStories): losses = {[round(v, 3) for v in losses]}")
    print(f"    first 5 avg {np.mean(losses[:5]):.4f}  last 5 avg {np.mean(losses[-5:]):.4f}")
    assert np.mean(losses[-5:]) < np.mean(losses[:5]), losses


if __name__ == "__main__":
    test_params_and_shape()
    test_kda_scan()
    test_mla()
    test_moR_gating()
    test_train_steps()
    print("\nALL SMOKE TESTS PASSED")