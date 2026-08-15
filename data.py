"""TinyStories loader for JAX training.

Reads the roneneldan/TinyStories parquet shards via huggingface_hub (the
`datasets` package is unusable on some hosts because of the missing `_lzma`
extension module), tokenizes with the GPT-2 tokenizer, and exposes a numpy
batch iterator. Tokens are cached to disk as uint16 so the TPU run reuses them.
"""

from __future__ import annotations

import os
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "roneneldan/TinyStories"
CACHE_NAME = "tinystories_{split}.npy"


def list_parquet(split: str) -> list[str]:
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.startswith(f"data/{split}") and f.endswith(".parquet")]
    return sorted(files)


def _download(split: str, data_dir: str, max_files: int | None) -> list[str]:
    files = list_parquet(split)
    if max_files:
        files = files[:max_files]
    paths = [hf_hub_download(REPO, f, repo_type="dataset", cache_dir=data_dir) for f in files]
    return paths


def _tokenize_shard(path: str, tokenizer, target: int | None, buf: list[int]) -> int:
    """Append GPT-2 tokens of one parquet shard to buf. Returns token count."""
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]
    table = pq.read_table(path, columns=["text"])
    texts = table["text"].to_pylist()
    total = 0
    for text in texts:
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        buf.append(nl)
        total += 1
        if target is not None and total >= target:
            break
    return total


def ensure_tokens(split: str, tokenizer, data_dir: str, *, max_files: int | None = None,
                  max_stories: int | None = None, max_tokens: int | None = None,
                  force: bool = False) -> np.ndarray:
    """Return uint16 token array for a split; download+tokenize if not cached."""
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, CACHE_NAME.format(split=split))
    if os.path.exists(cache) and not force:
        arr = np.load(cache, mmap_mode="r")
        print(f"Loaded cached {split}: {arr.shape[0]:,} tokens ({cache})")
        return arr

    paths = _download(split, data_dir, max_files)
    print(f"Tokenizing {split} ({len(paths)} shard(s)) with GPT-2...")
    buf: list[int] = []
    stories = 0
    for p in paths:
        n = _tokenize_shard(p, tokenizer, max_stories, buf)
        stories += n
        print(f"  {os.path.basename(p)}: {n} stories, {len(buf):,} tokens so far")

    tokens = np.asarray(buf, dtype=np.uint16)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    np.save(cache, tokens)
    print(f"Saved cache: {tokens.shape[0]:,} tokens, {stories} stories ({cache})")
    return tokens


def make_iter(tokens: np.ndarray, batch_size: int, seq_len: int, rng=None):
    """Yield (x, y) numpy int32 pairs of shape (batch, seq).
    x[i, j+1] == y[i, j] (shifted next-token prediction) across the stream."""
    total = tokens.shape[0]
    max_start = total - batch_size * seq_len - 1
    if rng is None:
        rng = np.random.default_rng(0)

    while True:
        start = rng.integers(0, max_start)
        seg = tokens[start:start + batch_size * seq_len + 1]
        x = seg[:-1].reshape(batch_size, seq_len).astype(np.int32)
        y = seg[1:].reshape(batch_size, seq_len).astype(np.int32)
        yield x, y