"""Training loop for the JAX MoRE model on TinyStories.

Single/multi-device (JAX devices detected automatically; batch is sharded
data-parallel when >1 device). Writes train.log lines as:
    step=N loss=<finite> aux=<..> lr=<..> tok/s=<..>

Optimizer: Muon (Newton-Schulz orthogonalized momentum) on hidden 2D weight
matrices + AdamW on embeddings/head/norms/routers by default (--optim muon).
--optim adamw restores the previous pure-AdamW behaviour (and matches old
checkpoints' optimizer state).

Usage:
    python -m jax_mokre.train --seq_len 512 --batch_size 16 --total_steps 40000
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import logsumexp

from .config import MoREConfig
from . import model as M
from . import data as D
from .muon import make_muon_mask, muon_adamw, muon_param_counts


def _ce_body(Wt):
    def body(_, batch):
        h, y, w = batch                       # (chunk*B, H), (chunk*B,), (chunk*B,)
        lg = jnp.einsum("nh,vh->nv", h, Wt)   # only this chunk's logits exist
        lse = logsumexp(lg, axis=-1)
        picked = jnp.take_along_axis(lg, y[:, None], axis=-1)[..., 0]
        return None, jnp.sum(w * (lse - picked))
    return body


def chunked_ce(cfg, Wt, hidden, labels, chunk):
    """Mean CE over the tied vocab head.

    Chunks are processed with lax.scan and a rematerialized body, so neither
    the forward nor the backward pass ever materializes more than ONE chunk of
    (chunk*B, V) logits. The naive unrolled loop keeps EVERY chunk's logits
    (and a log_softmax copy) alive until backward — ~3.3 GB at bs16/seq512/
    vocab50k vs ~100 MB here."""
    B, S, H = hidden.shape
    pad = (-S) % chunk
    if pad:
        hidden = jnp.pad(hidden, ((0, 0), (0, pad), (0, 0)))
        labels = jnp.pad(labels, ((0, 0), (0, pad)))
    Sp = hidden.shape[1]
    n = Sp // chunk
    w = jnp.ones((B, Sp), dtype=hidden.dtype)
    if pad:
        w = w.at[:, S:].set(0.0)
    hs = jnp.transpose(hidden, (1, 0, 2)).reshape(n, chunk * B, H)
    ys = jnp.transpose(labels, (1, 0)).reshape(n, chunk * B)
    ws = jnp.transpose(w, (1, 0)).reshape(n, chunk * B)
    _, sums = jax.lax.scan(jax.remat(_ce_body(Wt)), None, (hs, ys, ws))
    return jnp.sum(sums) / (B * S)


def make_loss_fn(cfg, ce_chunk, remat):
    def _fwd(params, x):
        return M.forward(cfg, params, x, training=True, return_hidden=True)
    fwd = jax.remat(_fwd) if remat else _fwd

    def loss_fn(params, x, y):
        hidden, aux, _ = fwd(params, x)
        ce = chunked_ce(cfg, params["embed_tokens"], hidden, y, ce_chunk)
        return ce + aux, (ce, aux)
    return loss_fn


def make_train_step(cfg, opt, ce_chunk, dist, remat=True):
    loss_fn = make_loss_fn(cfg, ce_chunk, remat)

    def step(params, opt_state, x, y, lr):
        (loss, (ce, aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y)
        if dist:
            grads = jax.lax.pmean(grads, axis_name="batch")
        # `lr` is a live hyperparameter (inject_hyperparams below): warmup,
        # cosine decay and NaN recovery actually reach the optimizer.
        updates, opt_state = opt.update(grads, opt_state, params, learning_rate=lr)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, ce, aux

    if dist:
        return jax.pmap(step, axis_name="batch",
                        in_axes=(None, None, 0, 0, None),
                        out_axes=(None, None, 0, 0, 0))
    return jax.jit(step)


def make_gen_step(cfg, window=64):
    """Greedy decode from a FIXED-SHAPE (B, window) context buffer.

    A growing context changes shapes every token and forces XLA to recompile
    per generated token; a fixed rolling window compiles exactly once. The LM
    head runs on the last position only (full-sequence logits are B*S*V)."""
    @jax.jit
    def step(params, ids):
        hidden, _, _ = M.forward(cfg, params, ids, training=False,
                                 return_hidden=True)
        logits = hidden[:, -1] @ params["embed_tokens"].T
        return jnp.argmax(logits, axis=-1)
    return step


def save_state(path, cfg, params, opt_state, step, key, lr_scale=1.0,
               optimizer="adamw"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "config": cfg.__dict__,
            "params": params,
            "opt_state": opt_state,
            "step": step,
            "key": key,
            "lr_scale": lr_scale,
            "optimizer": optimizer,
        }, f)
    print(f"  checkpoint -> {path}")


def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _first(p):
    """pmap leaves with out_axes=(None, ...) are already single-device (replicated)."""
    return p


def schedule(step, lr, lr_min, warmup, total):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= total:
        return lr_min
    t = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * t))


def build_config(args) -> MoREConfig:
    if args.config == "tinystories":
        cfg = MoREConfig(
            vocab_size=50257, hidden_size=384, intermediate_size=1024,
            num_attention_heads=6, num_key_value_heads=2, head_dim=64,
            max_seq_len=args.seq_len, max_recursion_depth=4,
            num_experts=8, num_local_experts=8, num_shared_experts=1, top_k=1,
            router_hidden_size=64, kda_state_size=64, kda_chunk_size=16,
            layer_types=["kda", "kda", "mla", "kda"],
            load_balancing_loss_coef=args.aux_coef, recursion_aux_coef=args.rec_aux_coef,
            rms_norm_eps=1e-6,
            initializer_range=0.02,
        )
    else:
        cfg = MoREConfig(
            vocab_size=50257, hidden_size=1024, intermediate_size=1024,
            num_attention_heads=16, num_key_value_heads=8, head_dim=64,
            max_seq_len=args.seq_len, max_recursion_depth=4,
            num_experts=8, num_local_experts=8, num_shared_experts=1, top_k=2,
            router_hidden_size=128, kda_state_size=64, kda_chunk_size=16,
            layer_types=["kda", "kda", "mla", "kda"],
            load_balancing_loss_coef=args.aux_coef, recursion_aux_coef=args.rec_aux_coef,
            rms_norm_eps=1e-6,
            initializer_range=0.02,
        )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tinystories", choices=["tinystories", "default"])
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--total_steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_min", type=float, default=3e-5)
    ap.add_argument("--warmup_steps", type=int, default=2000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--optim", default="muon", choices=["muon", "adamw"],
                    help="muon: Newton-Schulz Muon on hidden 2D matrices + "
                         "AdamW on embed/router/norms; adamw: pure AdamW")
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--muon_ns_steps", type=int, default=5,
                    help="Newton-Schulz iteration count (5 = canonical)")
    ap.add_argument("--muon_lr_scale", type=float, default=1.0,
                    help="multiplier on --lr for the Muon branch only")
    ap.add_argument("--muon_wd", type=float, default=0.0,
                    help="decoupled weight decay for the Muon branch")
    ap.add_argument("--muon_on_routers", action="store_true",
                    help="give MoE/recursion router weights to Muon instead "
                         "of AdamW (default off, per Muon-paper guidance)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--aux_coef", type=float, default=0.01)
    ap.add_argument("--rec_aux_coef", type=float, default=0.1,
                    help="recursion depth-push aux: pushes router toward deeper loops")
    ap.add_argument("--ce_chunk", type=int, default=32,
                    help="CE head chunk size; peak logits RAM ~ chunk*B*V*4 bytes")
    ap.add_argument("--gen_window", type=int, default=64,
                    help="fixed sampling context window (static shape = one compile)")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--gen_every", type=int, default=1000)
    ap.add_argument("--gen_prompt", type=str, default="Once upon a time,")
    ap.add_argument("--gen_len", type=int, default=32)
    ap.add_argument("--data_dir", type=str, default="./tinystories_data")
    ap.add_argument("--out_dir", type=str, default="./jax_run")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--max_stories", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--stream", action="store_true",
                    help="process the dataset one parquet shard at a time (peak RAM = 1 shard)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-remat", action="store_true",
                    help="disable gradient rematerialization (default: on, bounds HBM)")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fineweb", action="store_true",
                    help="train on local FineWeb-Edu parquet (--fineweb_source)")
    ap.add_argument("--fineweb_source", type=str, default=None,
                    help="folder containing fineweb-edu-dedup-10b train-*.parquet")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")

    devices = jax.devices()
    dist = len(devices) > 1
    print(f"JAX devices: {devices}  dist={dist}")

    cfg = build_config(args)
    print(f"Config: hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
          f"kv={cfg.num_key_value_heads} experts={cfg.num_local_experts} "
          f"topk={cfg.top_k} layers={cfg.layer_types} depth={cfg.max_recursion_depth}")

    rng = jax.random.PRNGKey(args.seed)

    if args.resume and os.path.exists(args.resume):
        st = load_state(args.resume)
        params = st["params"]
        opt_state = st["opt_state"]
        start_step = st["step"] + 1
        rng = st["key"]
        lr_scale = st.get("lr_scale", 1.0)
        saved_opt = st.get("optimizer", "adamw")
        if saved_opt != args.optim:
            # Optimizer states are not interchangeable (muon momentum vs adam
            # m/v). Keep the weights, restart the optimizer from scratch.
            print(f"  !! checkpoint '{args.resume}' was trained with "
                  f"'{saved_opt}' but --optim {args.optim}: discarding stale "
                  f"optimizer state (fresh moments)")
            opt_state = None
        print(f"Resumed from {args.resume} at step {start_step}")
    else:
        params = M.init_model(cfg, rng)
        rng, _ = jax.random.split(rng)
        start_step = 0
        opt_state = None
        lr_scale = 1.0
        print(f"Initialized model: {M.count_params(params):,} params")

    if args.optim == "muon":
        muon_mask = make_muon_mask(params,
                                   include_routers=args.muon_on_routers)
        n_mu, n_ad = muon_param_counts(params, muon_mask)
        ns_dtype = jnp.bfloat16 if devices and devices[0].platform == "tpu" \
            else jnp.float32
        print(f"Optimizer: Muon on {n_mu:,} params + AdamW on {n_ad:,} params "
              f"(momentum={args.muon_momentum}, nesterov, "
              f"ns_steps={args.muon_ns_steps}, ns_dtype={ns_dtype})")
        inner_opt = muon_adamw(
            muon_mask,
            momentum=args.muon_momentum, nesterov=True,
            ns_steps=args.muon_ns_steps, ns_dtype=ns_dtype,
            muon_lr_scale=args.muon_lr_scale, muon_weight_decay=args.muon_wd,
            b1=0.9, b2=0.95, eps=1e-8, weight_decay=args.weight_decay)
    else:
        inner_opt = optax.inject_hyperparams(optax.adamw)(
            learning_rate=args.lr, b1=0.9, b2=0.95, eps=1e-8,
            weight_decay=args.weight_decay)

    base_opt = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        # Both optimizer modes read the live `learning_rate` kwarg each step
        # (warmup, cosine decay and NaN recovery actually reach the optimizer).
        inner_opt,
    )
    opt = optax.MultiSteps(base_opt, every_k_schedule=args.accum)

    if opt_state is None:
        opt_state = opt.init(params)

    step_fn = make_train_step(cfg, opt, args.ce_chunk, dist, remat=not args.no_remat)
    gen_window_len = min(cfg.max_seq_len, max(args.gen_window, 16))
    gen_fn = make_gen_step(cfg, window=gen_window_len)

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    if args.fineweb:
        if not args.fineweb_source:
            raise SystemExit("--fineweb requires --fineweb_source=<folder of train-*.parquet>")
        shards = D.ensure_fineweb_shards(tokenizer, args.data_dir,
                                         args.fineweb_source,
                                         max_shards=args.max_files)
        n_shards = len(shards)
        steps_per_shard = max(int(np.ceil(args.total_steps / n_shards)), 1)
        data_iter = D.stream_iter(shards, args.batch_size, args.seq_len,
                                  steps_per_shard)
        print(f"Dataset: FineWeb-Edu streamed across {n_shards} shard(s), "
              f"{steps_per_shard} steps/shard")
    elif args.synthetic:
        rngd = np.random.default_rng(args.seed)
        tokens = rngd.integers(0, cfg.vocab_size, size=100_000, dtype=np.uint16)
        data_iter = D.make_iter(tokens, args.batch_size, args.seq_len)
        print("Dataset: synthetic")
    else:
        if args.stream:
            shards = D.ensure_shards(args.split, tokenizer, args.data_dir,
                                     max_files=args.max_files,
                                     max_stories=args.max_stories)
            n_shards = len(shards)
            steps_per_shard = max(int(np.ceil(args.total_steps / n_shards)), 1)
            data_iter = D.stream_iter(shards, args.batch_size, args.seq_len,
                                      steps_per_shard)
            print(f"Dataset: {args.split} streamed across {n_shards} shard(s), "
                  f"{steps_per_shard} steps/shard")
        else:
            tokens = D.ensure_tokens(args.split, tokenizer, args.data_dir,
                                     max_files=args.max_files,
                                     max_stories=args.max_stories,
                                     max_tokens=args.max_tokens)
            data_iter = D.make_iter(tokens, args.batch_size, args.seq_len)
            print(f"Dataset: {args.split} tokens={tokens.shape[0]:,}")

    per = max(args.batch_size // len(devices), 1)
    if dist and args.batch_size % len(devices) != 0:
        raise SystemExit(
            f"--batch_size {args.batch_size} must be divisible by "
            f"{len(devices)} device(s); got per-device batch {per}")
    step_tokens = per * args.seq_len * len(devices)

    step = start_step
    losses = []
    last_good = (params, opt_state)
    nan_streak = 0
    t0 = time.time()
    log_f = open(log_path, "a")

    def log(step_, loss, aux, lr_, tok_per_s):
        line = f"step={step_} loss={float(loss):.4f} aux={float(aux):.5f} lr={lr_:.2e} tok/s={tok_per_s:.0f}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    prompt_ids = tokenizer.encode(args.gen_prompt,
                                  add_special_tokens=False)[:gen_window_len]
    pad_id = 50256  # GPT-2 <|endoftext|>; left-padding decays out of the KDA state

    def sample(step_):
        buf = np.full((1, gen_window_len), pad_id, dtype=np.int32)
        buf[0, gen_window_len - len(prompt_ids):] = prompt_ids
        out = []
        for _ in range(args.gen_len):
            nxt = int(np.asarray(gen_fn(params, jnp.asarray(buf)))[0])
            out.append(nxt)
            buf = np.roll(buf, -1, axis=1)
            buf[0, -1] = nxt
        text = tokenizer.decode(prompt_ids + out)
        print(f"  sample@{step_}: {text!r}")
        log_f.write(f"sample@{step_}: {text!r}\n")
        log_f.flush()

    try:
        while step < args.total_steps:
            x, y = next(data_iter)
            if dist:
                x = x.reshape(len(devices), -1, args.seq_len)
                y = y.reshape(len(devices), -1, args.seq_len)
            else:
                x = x[:per]
                y = y[:per]
            x = jnp.asarray(x)
            y = jnp.asarray(y)
            lr = schedule(step, args.lr, args.lr_min, args.warmup_steps,
                          args.total_steps) * lr_scale
            lr = jnp.asarray(lr, dtype=jnp.float32)

            params, opt_state, loss, ce, aux = step_fn(params, opt_state, x, y, lr)

            loss_v = float(np.asarray(loss).mean())
            if np.isfinite(loss_v):
                last_good = (params, opt_state)
                nan_streak = 0
                losses.append(loss_v)
            else:
                # Roll back the bad update; the halved scale now really feeds
                # back into the optimizer via learning_rate=lr.
                params, opt_state = last_good
                nan_streak += 1
                lr_scale = max(lr_scale * 0.5, 1e-3)
                msg = (f"step={step} loss=NaN rolled_back lr_scale={lr_scale:.3g}")
                print(f"  !! {msg}")
                log_f.write(msg + "\n")
                log_f.flush()
                if nan_streak >= 10:
                    raise SystemExit(
                        f"Aborting: {nan_streak} consecutive NaN steps "
                        f"(lr_scale={lr_scale:.3g}); resume from an earlier ckpt.")

            step += 1

            if step % args.log_every == 0 or step == start_step + 1:
                tok_per_s = step_tokens / max(time.time() - t0, 1e-6)
                log(step, loss_v, float(np.asarray(aux).mean()),
                    float(lr), tok_per_s)
                t0 = time.time()

            if step % args.ckpt_every == 0:
                save_state(os.path.join(args.out_dir, f"ckpt_{step}.pkl"), cfg,
                           params, opt_state, step, rng, lr_scale,
                           optimizer=args.optim)

            if step % args.gen_every == 0:
                sample(step)

            if step >= args.total_steps:
                break
    finally:
        log_f.close()

    save_state(os.path.join(args.out_dir, "final.pkl"), cfg,
               params, opt_state, step, rng, lr_scale, optimizer=args.optim)
    tail = losses[-50:]
    avg = np.mean(tail) if tail else float("nan")
    print(f"Done. {step} steps. final avg loss (last {len(tail)}): {avg:.4f}")


if __name__ == "__main__":
    main()
