#!/usr/bin/env bash
# Launch MoRE TinyStories training on a TPU v5e-1 (16GB HBM) host.
# Run from the repo root:  bash jax_mokre/run_tpu.sh [extra args ...]

set -euo pipefail

# --- JAX for TPU (Colab / Kaggle / TPU VM all use the same wheel) ----------
pip install -q --upgrade "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/lts/jax_releases.html || true
pip install -q optax transformers huggingface_hub pyarrow

python3 -c "import jax; print('devices:', jax.devices())"

# --- Training ----------------------------------------------------------------
# L5 fix: CLI pass-through for dataset + mesh + dtype; no hardcoded TinyStories.
# batch 64 x seq 512 = 32768 tok/step; accum 8 => 262144 tok/update.
# 1B tokens ~= 30517 micro-steps (3815 updates). Defaults fit 16GB HBM in bf16.
mkdir -p jax_run
# Default dtype auto-switches to bfloat16 on TPU (see train.py build_config); override via --param_dtype/--compute_dtype
# Default mesh: single-device; for 8-way v5e set --mesh 2,4
# Dataset: default TinyStories; for FineWeb use --fineweb --fineweb_source /path/to/parquet
python3 -m jax_mokre.train \
    --config tinystories \
    --seq_len 512 \
    --batch_size 64 \
    --accum 8 \
    --total_steps "${TOTAL_STEPS:-30517}" \
    --lr 3e-4 \
    --warmup_steps 2000 \
    --data_dir ./tinystories_data \
    --out_dir ./jax_run \
    --ckpt_every 2000 \
    --gen_every 500 \
    --gen_prompt "Once upon a time," \
    "$@"