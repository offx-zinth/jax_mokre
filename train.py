"""Training loop for the JAX MoRE model on TinyStories.

Single/multi-device (JAX devices detected automatically; batch is sharded
data-parallel when >1 device). Writes train.log lines as:
    step=N loss=<finite> aux=<..> lr=<..> tok/s=<..>

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

from .config import MoREConfig
from . import model as M
from . import data as D


def chunked_ce(cfg, Wt, hidden, labels, chunk):
    """Cross-entropy over the vocab head, computed in seq-chunks to bound memory."""
    B, S, H = hidden.shape
    losses = []
    for i in range(0, S, chunk):
        h = hidden[:, i:i + chunk].reshape(-1, H)
        y = labels[:, i:i + chunk].reshape(-1)
        lg = jnp.einsum("nh,vh->nv", h, Wt)
        logp = jax.nn.log_softmax(lg, axis=-1)
        losses.append(-jnp.mean(logp[jnp.arange(y.shape[0]), y]))
    return sum(losses) / len(losses)


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
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, ce, aux

    if dist:
        return jax.pmap(step, axis_name="batch",
                        in_axes=(None, None, 0, 0, None),
                        out_axes=(None, None, 0, 0, 0))
    return jax.jit(step)


def make_gen_step(cfg):
    @jax.jit
    def step(params, ids):
        logits, _, _ = M.forward(cfg, params, ids, training=False)
        return jnp.argmax(logits[:, -1, :], axis=-1)
    return step


def save_state(path, cfg, params, opt_state, step, key):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "config": cfg.__dict__,
            "params": params,
            "opt_state": opt_state,
            "step": step,
            "key": key,
        }, f)
    print(f"  checkpoint -> {path}")


def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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
            load_balancing_loss_coef=args.aux_coef, rms_norm_eps=1e-6,
            initializer_range=0.02,
        )
    else:
        cfg = MoREConfig(
            vocab_size=50257, hidden_size=768, intermediate_size=2048,
            num_attention_heads=12, num_key_value_heads=4, head_dim=64,
            max_seq_len=args.seq_len, max_recursion_depth=4,
            num_experts=8, num_local_experts=8, num_shared_experts=1, top_k=2,
            router_hidden_size=128, kda_state_size=64, kda_chunk_size=32,
            layer_types=["kda", "kda", "mla", "kda"],
            load_balancing_loss_coef=args.aux_coef, rms_norm_eps=1e-6,
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
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--aux_coef", type=float, default=0.01)
    ap.add_argument("--ce_chunk", type=int, default=32)
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
        print(f"Resumed from {args.resume} at step {start_step}")
    else:
        params = M.init_model(cfg, rng)
        rng, _ = jax.random.split(rng)
        start_step = 0
        opt_state = None
        print(f"Initialized model: {M.count_params(params):,} params")

    base_opt = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(args.lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=args.weight_decay),
    )
    opt = optax.MultiSteps(base_opt, every_k_schedule=args.accum)

    if opt_state is None:
        opt_state = opt.init(params)

    step_fn = make_train_step(cfg, opt, args.ce_chunk, dist, remat=not args.no_remat)
    gen_fn = make_gen_step(cfg)

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    if args.synthetic:
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
    step_tokens = per * args.seq_len * len(devices)

    step = start_step
    losses = []
    last_good = (params, opt_state)
    t0 = time.time()

    def log(step_, loss, aux, lr_, tok_per_s):
        line = f"step={step_} loss={float(loss):.4f} aux={float(aux):.5f} lr={lr_:.2e} tok/s={tok_per_s:.0f}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    def sample(step_):
        ids = tokenizer.encode(args.gen_prompt, add_special_tokens=False)
        ids = np.asarray(ids[:16], dtype=np.int32)[None, :]
        out = list(ids[0])
        for _ in range(args.gen_len):
            nxt = np.asarray(gen_fn(params, ids))
            out.append(int(nxt[0]))
            ids = np.concatenate([ids, nxt[:, None]], axis=1)[:, -cfg.max_seq_len:]
        text = tokenizer.decode(out)
        print(f"  sample@{step_}: {text!r}")
        with open(log_path, "a") as f:
            f.write(f"sample@{step_}: {text!r}\n")

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
        lr = schedule(step, args.lr, args.lr_min, args.warmup_steps, args.total_steps)
        lr = jnp.asarray(lr, dtype=jnp.float32)

        params, opt_state, loss, ce, aux = step_fn(params, opt_state, x, y, lr)

        loss_v = float(np.asarray(loss).mean())
        if np.isfinite(loss_v):
            last_good = (params, opt_state)
        else:
            params, opt_state = last_good
            args.lr = max(args.lr * 0.5, 1e-6)
            print(f"  !! NaN at step {step}: loss={loss_v}; lr halved to {args.lr}")
            with open(log_path, "a") as f:
                f.write(f"step={step} loss=NaN lr_halved lr={args.lr:.2e}\n")

        losses.append(loss_v)
        step += 1

        if step % args.log_every == 0 or step == start_step + 1:
            tok_per_s = step_tokens / max(time.time() - t0, 1e-6)
            log(step, loss_v, float(np.asarray(aux).mean()),
                lr, tok_per_s)
            t0 = time.time()

        if step % args.ckpt_every == 0:
            save_state(os.path.join(args.out_dir, f"ckpt_{step}.pkl"), cfg,
                       params if not dist else params[0],
                       opt_state if not dist else opt_state[0], step, rng)

        if step % args.gen_every == 0:
            sample(step)

        if step >= args.total_steps:
            break

    save_state(os.path.join(args.out_dir, "final.pkl"), cfg,
               params if not dist else params[0],
               opt_state if not dist else opt_state[0], step, rng)
    print(f"Done. {step} steps. final avg loss (last 50): {np.mean(losses[-50:]):.4f}")


if __name__ == "__main__":
    main()
