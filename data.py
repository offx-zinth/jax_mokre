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


def _tokenize_shard(path: str, tokenizer, target: int | None,
                    arr: np.ndarray, pos: int) -> tuple[int, int, bool]:
    """Batch-encode one parquet shard into arr[pos:]; returns (new pos, stories, truncated)."""
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]
    table = pq.read_table(path, columns=["text"])
    texts = table["text"].to_pylist()
    bs = 512
    stories = 0
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], add_special_tokens=False)["input_ids"]
        for ids in enc:
            n = len(ids)
            if pos + n + 1 > arr.shape[0]:
                print(f"  WARNING: buffer full mid-shard {os.path.basename(path)}; "
                      f"rest of corpus dropped (raise the token estimate)")
                return pos, stories, True
            arr[pos:pos + n] = np.asarray(ids, dtype=np.uint16)
            pos += n
            arr[pos] = nl
            pos += 1
            stories += 1
            if target is not None and stories >= target:
                return pos, stories, False
    return pos, stories, False


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
    total_rows = sum(pq.read_metadata(p).num_rows for p in paths)

    cap = max_tokens if max_tokens else total_rows * 400 + 1_000_000
    arr = np.empty(cap, dtype=np.uint16)
    pos = 0
    stories = 0
    for p in paths:
        before = pos
        pos, n_stories, _ = _tokenize_shard(p, tokenizer, max_stories, arr, pos)
        stories += n_stories
        print(f"  {os.path.basename(p)}: +{pos - before:,} tokens "
              f"({pos:,} total so far)")

    tokens = arr[:pos]
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    np.save(cache, tokens)
    print(f"Saved cache: {tokens.shape[0]:,} tokens, {stories} stories ({cache})")
    return tokens


def make_iter(tokens: np.ndarray, batch_size: int, seq_len: int, rng=None):
    """Yield (x, y) numpy int32 pairs of shape (batch, seq).
    x[i, j+1] == y[i, j] (shifted next-token prediction) across the stream."""
    total = tokens.shape[0]
    need = batch_size * seq_len + 1
    if total < need:
        raise ValueError(
            f"token stream too small ({total:,}) for batch_size={batch_size} x "
            f"seq_len={seq_len} (+1); need at least {need:,} tokens")
    max_start = total - need
    if rng is None:
        rng = np.random.default_rng(0)

    while True:
        start = rng.integers(0, max_start)
        seg = tokens[start:start + need]
        x = seg[:-1].reshape(batch_size, seq_len).astype(np.int32)
        y = seg[1:].reshape(batch_size, seq_len).astype(np.int32)
        yield x, y


def ensure_shards(split: str, tokenizer, data_dir: str, *, max_files: int | None = None,
                  max_stories: int | None = None, force: bool = False) -> list[np.ndarray]:
    """Tokenize TinyStories **one parquet shard at a time**, returning per-shard
    uint16 arrays (memory-mapped). Peak host memory ~= one shard, not the corpus."""
    os.makedirs(data_dir, exist_ok=True)
    paths = _download(split, data_dir, max_files)
    print(f"Tokenizing {split} shard-by-shard ({len(paths)} shard(s))...")
    shards: list[np.ndarray] = []
    for i, p in enumerate(paths):
        cache = os.path.join(data_dir, f"{split}_shard{i}.npy")
        if os.path.exists(cache) and not force:
            arr = np.load(cache, mmap_mode="r")
            print(f"  {os.path.basename(p)}: cached {arr.shape[0]:,} tokens")
        else:
            nrows = pq.read_metadata(p).num_rows
            cap = nrows * 400 + 1_000_000
            arr = np.empty(cap, dtype=np.uint16)
            pos, stories, _ = _tokenize_shard(p, tokenizer, max_stories, arr, 0)
            arr = arr[:pos]
            np.save(cache, arr)
            print(f"  {os.path.basename(p)}: tokenized {pos:,} tokens, "
                  f"{stories} stories")
        shards.append(arr)
    return shards


def stream_iter(shards: list[np.ndarray], batch_size: int, seq_len: int,
                steps_per_shard: int, rng=None):
    """Infinite generator: cycles shards (shuffled order each pass), drawing
    random-offset batches from ONE shard at a time for `steps_per_shard` steps."""
    if rng is None:
        rng = np.random.default_rng(0)
    order = rng.permutation(len(shards))
    while True:
        for idx in order:
            it = make_iter(shards[idx], batch_size, seq_len, rng)
            for _ in range(steps_per_shard):
                yield next(it)
        order = rng.permutation(len(shards))

# ---------------------------------------------------------------------------
# FineWeb-Edu loader (EleutherAI/fineweb-edu-dedup-10b style).
# Works with the local subset parquet file(s) already downloaded; each shard is
# tokenized once to a uint16 npy cache and streamed shard-by-shard exactly like
# the TinyStories path, so peak host RAM stays ~one shard.

REPO_FINEWEB = "EleutherAI/fineweb-edu-dedup-10b"


def fineweb_parquets(path: str | os.PathLike) -> list[str]:
    """Return local *.parquet files with a usable `text` column (recursively), sorted.

    Filters out non-corpus parquets (e.g. reward/preference datasets that lack
    the `text` column), which would otherwise crash tokenization."""
    import glob as _glob
    out = []
    for p in sorted(_glob.glob(os.path.join(str(path), "**", "*.parquet"), recursive=True)):
        try:
            cols = pq.read_schema(p).names
        except Exception:
            continue
        if "text" in cols:
            out.append(p)
    return out


def _tokenize_texts(texts, tokenizer, arr, pos, nl):
    bs = 512
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], add_special_tokens=False)["input_ids"]
        for ids in enc:
            n = len(ids)
            if pos + n + 1 > arr.shape[0]:
                return pos
            arr[pos:pos + n] = np.asarray(ids, dtype=np.uint16)
            pos += n
            arr[pos] = nl
            pos += 1
    return pos


def ensure_fineweb_shards(tokenizer, data_dir: str,
                          source: str | os.PathLike,
                          *, max_shards: int | None = None,
                          force: bool = False) -> list[np.ndarray]:
    """Tokenize local FineWeb-Edu parquet shards one at a time -> npy caches.
    `source` is the local folder containing train-*.parquet."""
    os.makedirs(data_dir, exist_ok=True)
    paths = fineweb_parquets(source)
    if max_shards:
        paths = paths[:max_shards]
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]
    print(f"FineWeb-Edu: {len(paths)} shard(s) under {source}")
    shards: list[np.ndarray] = []
    for i, p in enumerate(paths):
        cache = os.path.join(data_dir, f"fineweb_shard{i}.npy")
        if os.path.exists(cache) and not force:
            arr = np.load(cache, mmap_mode="r")
            print(f"  {os.path.basename(p)}: cached {arr.shape[0]:,} tokens")
        else:
            nrows = pq.read_metadata(p).num_rows
            cap = nrows * 400 + 1_000_000
            arr = np.empty(cap, dtype=np.uint16)
            pos = 0
            truncated = False
            table = pq.read_table(p, columns=["text"])
            texts = table["text"].to_pylist()
            bs = 512
            for i2 in range(0, len(texts), bs):
                enc = tokenizer(texts[i2:i2 + bs], add_special_tokens=False)["input_ids"]
                for ids in enc:
                    n = len(ids)
                    if pos + n + 1 > arr.shape[0]:
                        truncated = True
                        break
                    arr[pos:pos + n] = np.asarray(ids, dtype=np.uint16)
                    pos += n
                    arr[pos] = nl
                    pos += 1
                else:
                    continue
                break
            if truncated:
                print(f"  WARNING: buffer full mid-shard {os.path.basename(p)}; "
                      f"rest of corpus dropped (raise the token estimate)")
            arr = arr[:pos]
            np.save(cache, arr)
            print(f"  {os.path.basename(p)}: tokenized {pos:,} tokens")
        shards.append(arr)
    return shards
