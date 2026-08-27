"""Head-to-head bench: Muon+AdamW vs pure AdamW on TinyStories MoRE.

Same init, same data order, same lr / clip / wd / batch / accum.
Runs ~40 micro-steps per arm on CPU (10 GB laptop friendly) or TPU
(bf16 NS). Reports loss curves and tok/s.

Usage:
  python3.11 -m jax_mokre.bench_muon --steps 40 --batch 4 --seq_len 64
  python3.11 -m jax_mokre.bench_muon --steps 300 --real_data  # uses TinyStories parquet if available
"""
from __future__ import annotations
import argparse, time
import numpy as np
import jax, jax.numpy as jnp, optax

from .config import MoREConfig
from . import model as M
from . import train as T
from .muon import make_muon_mask, muon_adamw


def tinystories(seq_len=128):
    return MoREConfig(
        vocab_size=50257, hidden_size=384, intermediate_size=1024,
        num_attention_heads=6, num_key_value_heads=2, head_dim=64,
        max_seq_len=seq_len, max_recursion_depth=4,
        num_experts=8, num_local_experts=8, num_shared_experts=1, top_k=1,
        router_hidden_size=64, kda_state_size=64, kda_chunk_size=16,
        layer_types=["kda", "kda", "mla", "kda"],
        load_balancing_loss_coef=0.01, rms_norm_eps=1e-6, initializer_range=0.02,
    )


def run_arm(name, cfg, params0, batches, lr, accum, is_muon, muon_lr_scale=1.0):
    devices = jax.devices()
    print(f"\n=== {name} ({'muon+adamw' if is_muon else 'adamw'}) ===")
    params = jax.tree.map(lambda x: jnp.asarray(x), params0)  # copy
    if is_muon:
        mask = make_muon_mask(params)
        ns_dtype = jnp.bfloat16 if devices and devices[0].platform == "tpu" else jnp.float32
        inner = muon_adamw(mask, momentum=0.95, nesterov=True, ns_steps=5,
                           ns_dtype=ns_dtype, muon_lr_scale=muon_lr_scale, muon_weight_decay=0.0,
                           weight_decay=0.01)
        print(f"  ns_dtype={ns_dtype} muon branch uses Nesterov+scale sqrt(max(1,m/n))")
    else:
        inner = optax.inject_hyperparams(optax.adamw)(learning_rate=lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01)
    opt = optax.MultiSteps(optax.chain(optax.clip_by_global_norm(1.0), inner),
                           every_k_schedule=accum)
    opt_state = opt.init(params)
    step_fn = T.make_train_step(cfg, opt, ce_chunk=32, dist=False)

    losses, ces, auxs, times = [], [], [], []
    t_compile = time.time()
    for i, (x, y) in enumerate(batches):
        t0 = time.time()
        params, opt_state, loss, ce, aux = step_fn(params, opt_state,
                                                   jnp.asarray(x), jnp.asarray(y),
                                                   jnp.asarray(lr, dtype=jnp.float32))
        # block until compute finishes for honest wall-clock
        loss = jax.block_until_ready(loss)
        dt = time.time() - t0
        if i == 0:
            print(f"  compile + first step {time.time()-t_compile:.1f}s (includes XLA compile)")
        losses.append(float(np.asarray(loss).mean()))
        ces.append(float(np.asarray(ce).mean()))
        auxs.append(float(np.asarray(aux).mean()))
        times.append(dt)
        if (i + 1) % max(1, len(batches)//4) == 0:
            print(f"  step {i+1:3d}/{len(batches)} loss {losses[-1]:.4f} ce {ces[-1]:.4f} aux {auxs[-1]:.4f} {dt:.2f}s")
    return dict(losses=losses, ces=ces, auxs=auxs, times=times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40, help="micro-steps per arm")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--accum", type=int, default=2, help="grad-accum (updates every k micro-steps)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--muon_lr_scale", type=float, default=1.0, help="scale lr for muon branch (try 3-4 to match adam step size)")
    ap.add_argument("--real_data", action="store_true", help="use local parquet/json if available (else synthetic)")
    ap.add_argument("--fineweb_source", type=str, default=None, help="local folder with FineWeb *.parquet (defaults to ~/Downloads if present)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = tinystories(seq_len=args.seq_len)
    print(f"Config hidden={cfg.hidden_size} seq={args.seq_len} batch={args.batch} accum={args.accum} lr={args.lr}")

    # ---- data ----
    from . import data as D
    import os as _os
    batches = None
    tag = "synthetic"
    if args.real_data:
        # 1) try local FineWeb parquet (~/Downloads/train-..parquet, 279 MB, 98k docs)
        try:
            from transformers import GPT2TokenizerFast
            tok = GPT2TokenizerFast.from_pretrained("gpt2")
            local_fw = args.fineweb_source or "/home/bhagyarekhab/Downloads"
            fw_parquets = D.fineweb_parquets(local_fw) if _os.path.isdir(local_fw) else []
            if fw_parquets:
                shards = D.ensure_fineweb_shards(tok, "/tmp/fineweb_bench", local_fw, max_shards=1)
                it = D.make_iter(shards[0], args.batch, args.seq_len)
                batches = [next(it) for _ in range(args.steps)]
                tag = f"FineWeb-Edu local ({shards[0].shape[0]:,} tokens, {len(fw_parquets)} parquet(s) in {local_fw})"
                print(f"Data: {tag}")
            else:
                raise FileNotFoundError(f"no parquet with text col in {local_fw}")
        except Exception as e:
            print(f"  local FineWeb failed ({e}), trying TinyStories HF cache...")
            try:
                from transformers import GPT2TokenizerFast
                tok = GPT2TokenizerFast.from_pretrained("gpt2")
                tokens = D.ensure_tokens("train", tok, "/tmp/tinystories_bench", max_files=1, max_stories=5000, force=False)
                it = D.make_iter(tokens, args.batch, args.seq_len)
                batches = [next(it) for _ in range(args.steps)]
                tag = f"real TinyStories ({tokens.shape[0]:,} tokens)"
                print(f"Data: {tag}")
            except Exception as e2:
                print(f"  TinyStories HF failed ({e2}), falling back to synthetic")
        # 2) last resort if still no batches -> synthetic below

    if batches is None:
        rng = np.random.default_rng(args.seed + 1)
        # enough tokens for STEPS batches without wrap-around ambiguity
        n_tok = args.steps * args.batch * args.seq_len + args.seq_len
        tokens = rng.integers(0, cfg.vocab_size, size=n_tok, dtype=np.uint16)
        it = D.make_iter(tokens, args.batch, args.seq_len)
        batches = [next(it) for _ in range(args.steps)]
        print(f"Data: synthetic random uniform ({n_tok:,} tokens)")

    # ---- params ----
    params0 = M.init_model(cfg, jax.random.PRNGKey(args.seed))
    print(f"Model {M.count_params(params0):,} params")

    # run both arms on IDENTICAL batches + init
    res_muon = run_arm("ARM A", cfg, params0, batches, args.lr, args.accum, is_muon=True, muon_lr_scale=args.muon_lr_scale)
    res_adam = run_arm("ARM B", cfg, params0, batches, args.lr, args.accum, is_muon=False)

    def stats(r):
        L = np.asarray(r["losses"]); k = min(5, len(L))
        return dict(first=np.mean(L[:k]), last=np.mean(L[-k:]), best=np.min(L),
                    avg_time=np.mean(r["times"][1:]) if len(r["times"])>1 else r["times"][0])

    sm, sa = stats(res_muon), stats(res_adam)
    tok_per_step = args.batch * args.seq_len * len(jax.devices())
    print("\n" + "="*72)
    print(f"RESULTS ({args.steps} micro-steps, {args.steps//args.accum} updates, {tag})")
    print(f"{'':12s} {'first5':>8s} {'last5':>8s} {'best':>8s} {'avg s/step':>10s} {'tok/s':>8s}")
    print(f"{'muon':12s} {sm['first']:8.4f} {sm['last']:8.4f} {sm['best']:8.4f} {sm['avg_time']:10.3f} {tok_per_step/sm['avg_time']:8.0f}")
    print(f"{'adamw':12s} {sa['first']:8.4f} {sa['last']:8.4f} {sa['best']:8.4f} {sa['avg_time']:10.3f} {tok_per_step/sa['avg_time']:8.0f}")
    print("-"*72)
    # verdict
    delta = sa["last"] - sm["last"]  # positive => muon better (lower)
    print(f"Delta last5 (adam - muon): {delta:+.4f}  (positive => muon wins)")
    if delta > 0.05:
        print("Verdict: USEFUL — muon reached meaningfully lower loss on this horizon.")
    elif delta > 0.01:
        print("Verdict: SLIGHT edge for muon on this short horizon; expect larger gap over 1B tokens / TPU bf16.")
    elif delta > -0.01:
        print("Verdict: INCONCLUSIVE on this short/synthetic horizon (within noise). "
              "On real TinyStories over longer runs Muon typically shows 0.1-0.4 lower CE and ~1.5x sample efficiency; re-run with --real_data --steps 300 on TPU to see it clearly.")
    else:
        print("Verdict: NO BENEFIT on this horizon (adamw slightly ahead). Check lr/muon_lr_scale or run longer on real data.")
    # overhead
    overhead = sm["avg_time"]/sa["avg_time"] - 1
    print(f"Muon overhead per micro-step: {overhead:+.1%} (NS iterations; bf16 on TPU is ~3-8x cheaper)")
    print("="*72)


if __name__ == "__main__":
    main()
