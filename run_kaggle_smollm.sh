#!/usr/bin/env bash
# Kaggle 2xT4 launcher for HuggingFaceTB/smollm-corpus (cosmopedia-v2 + fineweb-edu-dedup)
# Usage in Kaggle notebook:
#   !bash jax_mokre/run_kaggle_smollm.sh
# Or with overrides:
#   TOTAL_STEPS=30000 BATCH_SIZE=32 bash jax_mokre/run_kaggle_smollm.sh --smollm_weights 0.5,0.5
set -euo pipefail

pip install -q "jax[cuda12]" optax transformers huggingface_hub pyarrow \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || true
pip install -q optax transformers huggingface_hub pyarrow

python3 -c "import jax; print('devices:', jax.devices())"

# Tuned for Kaggle 2xT4 (16GB each, JAX pmap data-parallel)
# seq 1024 x batch 32 x accum 4 = 131k tok/update; 30k steps ~ 1B tokens
# For smoke test set SMOLLM_MAX_FILES=10 (fast, ~10 shards)
mkdir -p /kaggle/working/jax_run 2>/dev/null || mkdir -p ./jax_run
DATA_DIR="${DATA_DIR:-./smollm_data}"
OUT_DIR="${OUT_DIR:-./jax_run}"

python3 -m jax_mokre.train \
  --smollm \
  --smollm_subsets "${SMOLLM_SUBSETS:-cosmopedia-v2,fineweb-edu-dedup}" \
  ${SMOLLM_WEIGHTS:+--smollm_weights $SMOLLM_WEIGHTS} \
  ${SMOLLM_MAX_FILES:+--smollm_max_files $SMOLLM_MAX_FILES} \
  --config tinystories \
  --seq_len 1024 \
  --batch_size "${BATCH_SIZE:-32}" \
  --accum 4 \
  --total_steps "${TOTAL_STEPS:-30000}" \
  --lr 3e-4 \
  --warmup_steps 2000 \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --ckpt_every 2000 \
  --gen_every 500 \
  --gen_prompt "Once upon a time," \
  "$@"

echo "Done. Logs: $OUT_DIR/train.log  Checkpoints: $OUT_DIR/ckpt_*.pkl"
