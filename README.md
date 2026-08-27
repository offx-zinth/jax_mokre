# MoRE in JAX — TinyStories training on a TPU v5e-1

Pure-JAX port of the MoRE (Mixture-of-Recursions) model from `mokre/`
(PyTorch). Same math, 85,227,144 params at the tinystories config.

Key differences from the torch version:
- **KDA is vectorized** with `jax.lax.associative_scan` over the whole
  sequence (no Python per-token loop). Verified bit-close vs a sequential
  reference (max abs err ~2.8e-9). Math in `RESEARCH.md`.
- **MoE** runs experts in a static loop (same FLOPs, low HBM footprint).
- **MLA** uses a max-shifted finite mask instead of `-inf`+`nan_to_num`, so
  fully-masked (frozen) rows stay finite in both forward and backward.

## Files
- `config.py`   — `MoREConfig` (tinystories defaults)
- `model.py`    — rmsnorm/linear/KDA/MLA/MoE/router/model in pure JAX
- `data.py`     — TinyStories parquet -> GPT-2 tokens (uint16 cache) -> batches
- `muon.py`     — Muon optimizer (Newton-Schulz orthogonalized momentum) — hidden 2D weights get Muon, embed/router/norms get AdamW; 10 GB laptop friendly, bf16 on TPU
- `train.py`    — jitted train step, Muon/AdamW + cosine + grad-accum, ckpt
- `bench_muon.py` — head-to-head bench (same init/data) to check usefulness
- `smoke_test.py` — CPU verification suite (now includes 3 Muon tests)
- `run_tpu.sh`  — TPU launcher

Note: the LR schedule (warmup + cosine) is a live optimizer hyperparameter
— it is applied every step and survives resume via the saved `lr_scale`.
For `--optim adamw` it is plumbed via `inject_hyperparams`; for `--optim muon`
(default) it is read as a live `learning_rate=` kwarg by the Muon+AdamW
partition (warmup, cosine and NaN rollback all reach the optimizer). The
LM-head cross-entropy streams through `lax.scan` with a rematerialized body,
so peak logits memory is one `--ce_chunk` slice (~100 MB), not the whole
`(B·S·V)` tensor (~GBs). Greedy sampling uses a fixed-shape context window
(`--gen_window`, default 64) so XLA compiles once, not once per token.

## Optimizer — Muon (default) and laptop notes

`--optim muon` (default) — MomentUm Orthogonalized by Newton-Schulz:
hidden weight matrices (ndim ≥ 2, both dims > 1, excluding the tied
`embed_tokens` + `init_state`) run Muon; everything else (embeddings,
norms/biases, MoE + recursion routers) stays on decoupled AdamW. This
matches the Muon papers' guidance (routers excluded unless
`--muon_on_routers`) and the Moonshot “Muon is Scalable” recipe.
State is memory-minimal: zero-size sentinels for the other branch, not
`optax.partition`'s full-tree zeros. Newton-Schulz (5 steps, quintic
`3.4445/-4.7750/2.0315`) runs in **bf16 on TPU** and **fp32 on CPU**.

Flags: `--optim {muon,adamw}`, `--muon_momentum 0.95`, `--muon_ns_steps 5`,
`--muon_lr_scale 1.0` (try `2–4` if you want muon updates to match AdamW's
RMS at the same `--lr`), `--muon_wd 0.0`, `--muon_on_routers`.

10 GB laptop / CPU only: Muon works but is **~3.5× slower per micro-step**
than AdamW on CPU (float32 NS matmuls) and needs the same 10 GB budget:
params 340 MB + muon momentum ~260 MB (only on 65.8M muon params) +
Adam m/v ~78 MB (19.4M adam params) + grads/activations; comfortably <10 GB
at `seq 64 / batch 4` for smoke tests but not for 1B-token runs. For daily
dev on the laptop, use `--optim adamw` (or `--synthetic` + `--optim muon`
for a quick smoke). Final 1B-token training is intended for **TPU v5e** where
bf16 NS matmuls are ~8× cheaper and the overhead drops to ~15–20%.

Check that muon is wired: `python3.11 -m jax_mokre.smoke_test` now runs 3
muon tests — partition + NS singular values (0.5–1.3) + toy regression
(0.07→0.0005) + full-model 12 synthetic muon steps — and prints
`muon=65,874,944 adamw=19,352,200`.

Head-to-head on the laptop CPU (FineWeb-Edu local shard, 50 micro-steps,
batch 4 × seq 64, `accum 2`, `lr 3e-4`, same init/data order):
`muon 10.16` vs `adamw 8.82` last5 (adam wins) and `5.7 s/step` vs `1.6 s/step`
(+256 % NS overhead on CPU). This short/synthetic horizon is not the
regime Muon was designed for; literature (Modded-NanoGPT, Kimi K2/Moonshot)
reports Muon overtaking AdamW only over **thousands of updates / 100M–1B tokens**
and with TPU bf16. **Verdict on the laptop: INCONCLUSIVE / not useful for
wall-clock on CPU.** Use the bench for a quick sanity check and the full
TPU run for the real comparison:

```bash
python3.11 -m jax_mokre.bench_muon --steps 50 --real_data           # laptop, uses ~/Downloads/*.parquet if present, else synthetic
python3.11 -m jax_mokre.bench_muon --steps 50 --real_data --muon_lr_scale 3.0
```

## Local CPU smoke (dev box)
```bash
pip install "jax[cpu]" optax transformers huggingface_hub pyarrow
python3.11 -m jax_mokre.smoke_test     # ALL SMOKE TESTS PASSED (now 9 tests incl. 3 muon)
python3.11 -m jax_mokre.train --optim muon --config tinystories --seq_len 64 \
    --batch_size 2 --total_steps 5 --synthetic --out_dir /tmp/run
# laptop dev tip: adamw is 3.5× faster per step on CPU
python3.11 -m jax_mokre.train --optim adamw --config tinystories --seq_len 64 \
    --batch_size 4 --total_steps 50 --synthetic --out_dir /tmp/run_adam
```

## Real TinyStories run on a TPU v5e-1
The TPU host (or Colab/Kaggle with `TPU v5e-1`) needs the TPU jax wheel:

```bash
pip install -q --upgrade "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/lts/jax_releases.html
pip install -q optax transformers huggingface_hub pyarrow
bash jax_mokre/run_tpu.sh                 # or run the python command inside
```

Or, in a Colab cell:
```python
!git clone https://github.com/you/COMBO.git && cd COMBO   # your repo
!bash jax_mokre/run_tpu.sh
!tail -f jax_run/train.log
```

Notes:
- First run downloads 4 train parquet shards (~1 GB) and tokenizes them
  (~10–20 min); the token array is cached to `tinystories_data/`.
- Batch 64 x seq 512 = 32k tok/step; accum 8 => 262k tok/update.
  ~1B tokens ≈ 30k micro-steps, comfortably < 30 h on one v5e-1.
- Logs: `jax_run/train.log` (`step=N loss=<finite> aux=.. lr=.. tok/s=..`),
  greedy samples every `--gen_every`, checkpoints as `jax_run/ckpt_*.pkl`.

## Resume
```bash
python3 -m jax_mokre.train --resume jax_run/ckpt_2000.pkl [same flags]
```

## Success criteria
- `smoke_test.py` prints `ALL SMOKE TESTS PASSED` (CPU).
- `train.log` shows decreasing finite loss; loss should drop from ~10.8
  (ln 50257) toward ~4–5 within the first few thousand steps, and generation
  should produce coherent TinyStories-style text.