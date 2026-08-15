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
- `train.py`    — jitted train step, optax adamw + cosine + grad-accum, ckpt
- `smoke_test.py` — CPU verification suite
- `run_tpu.sh`  — TPU launcher

## Local CPU smoke (dev box)
```bash
pip install "jax[cpu]" optax transformers huggingface_hub pyarrow
python3 -m jax_mokre.smoke_test     # ALL SMOKE TESTS PASSED
python3 -m jax_mokre.train --config tinystories --seq_len 64 \
    --batch_size 2 --total_steps 5 --synthetic --out_dir /tmp/run
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