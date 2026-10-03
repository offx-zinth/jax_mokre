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


def _ensure_capacity(arr: np.ndarray, needed: int) -> np.ndarray:
    """Grow arr (uint16) if needed > len(arr); doubles until fit."""
    if needed <= arr.shape[0]:
        return arr
    new_cap = max(needed, arr.shape[0] * 2)
    # also grow at least 1M to avoid many resizes
    if new_cap < arr.shape[0] + 1_000_000:
        new_cap = arr.shape[0] + 1_000_000
    print(f"  [data] growing token buffer {arr.shape[0]:,} -> {new_cap:,} (needed {needed:,})")
    grown = np.empty(new_cap, dtype=arr.dtype)
    grown[: arr.shape[0]] = arr[: arr.shape[0]]
    return grown


def _tokenize_shard(path: str, tokenizer, target: int | None,
                    arr: np.ndarray, pos: int) -> tuple[np.ndarray, int, int, bool]:
    """Batch-encode one parquet shard into arr[pos:]; auto-grows buffer.

    Returns (arr_maybe_grown, new pos, stories, truncated_due_to_target).
    Never truncates due to cap - buffer grows as needed. `truncated` now
    only reflects `target` story limit, not cap overflow.
    """
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]
    table = pq.read_table(path, columns=["text"])
    texts = table["text"].to_pylist()
    bs = 512
    stories = 0
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], add_special_tokens=False)["input_ids"]
        for ids in enc:
            n = len(ids)
            need = pos + n + 1
            if need > arr.shape[0]:
                arr = _ensure_capacity(arr, need)
            arr[pos:pos + n] = np.asarray(ids, dtype=np.uint16)
            pos += n
            arr[pos] = nl
            pos += 1
            stories += 1
            if target is not None and stories >= target:
                return arr, pos, stories, False
    return arr, pos, stories, False


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
        arr, pos, n_stories, _trunc = _tokenize_shard(p, tokenizer, max_stories, arr, pos)
        stories += n_stories
        cap_now = arr.shape[0]
        util = pos / cap_now * 100 if cap_now else 0
        print(f"  {os.path.basename(p)}: +{pos - before:,} tokens "
              f"({pos:,} total so far, {util:.1f}% of cap {cap_now:,})")
        if arr.shape[0] != cap and cap != cap_now:
            print(f"  [data] buffer auto-grown {cap:,} -> {cap_now:,}")

    tokens = arr[:pos]
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    np.save(cache, tokens)
    util = tokens.shape[0] / max(arr.shape[0], 1) * 100
    print(f"Saved cache: {tokens.shape[0]:,} tokens, {stories} stories ({cache}) cap {arr.shape[0]:,} util {util:.1f}%")
    if max_tokens is not None and tokens.shape[0] == max_tokens:
        print(f"  [data] note: truncated to max_tokens={max_tokens:,}")
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
            arr, pos, stories, _trunc = _tokenize_shard(p, tokenizer, max_stories, arr, 0)
            cap_now = arr.shape[0]
            util = pos / cap_now * 100 if cap_now else 0
            arr = arr[:pos]
            np.save(cache, arr)
            print(f"  {os.path.basename(p)}: tokenized {pos:,} tokens, "
                  f"{stories} stories cap {cap_now:,} util {util:.1f}%")
            if cap_now != cap:
                print(f"  [data] shard buffer grown {cap:,} -> {cap_now:,}")
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
            table = pq.read_table(p, columns=["text"])
            texts = table["text"].to_pylist()
            bs = 512
            for i2 in range(0, len(texts), bs):
                enc = tokenizer(texts[i2:i2 + bs], add_special_tokens=False)["input_ids"]
                for ids in enc:
                    n = len(ids)
                    need = pos + n + 1
                    if need > arr.shape[0]:
                        arr = _ensure_capacity(arr, need)
                    arr[pos:pos + n] = np.asarray(ids, dtype=np.uint16)
                    pos += n
                    arr[pos] = nl
                    pos += 1
            cap_now = arr.shape[0]
            arr = arr[:pos]
            np.save(cache, arr)
            print(f"  {os.path.basename(p)}: tokenized {pos:,} tokens cap {cap_now:,} util {pos/cap_now*100:.1f}%")
            if cap_now != cap:
                print(f"  [data] FineWeb shard buffer grown {cap:,} -> {cap_now:,}")
        shards.append(arr)
    return shards


# ---------------------------------------------------------------------------
# HuggingFaceTB/smollm-corpus — cosmopedia-v2 + fineweb-edu-dedup
# Two configs living in one HF dataset repo:
#   HuggingFaceTB/smollm-corpus / cosmopedia-v2       -> 104 shards
#   HuggingFaceTB/smollm-corpus / fineweb-edu-dedup   -> 234 shards
# Each parquet has `text` (string) column. Tokenized exactly like TinyStories:
#   one shard = one uint16 npy cache, streamed shard-by-shard so peak RAM ~1 shard.
# Use via --smollm on train.py (mixes both subsets).

REPO_SMOLLM = "HuggingFaceTB/smollm-corpus"
SMOLLM_SUBSETS = ("cosmopedia-v2", "fineweb-edu-dedup")


def list_smollm(subset: str) -> list[str]:
    assert subset in SMOLLM_SUBSETS, f"unknown smollm subset {subset!r}"
    files = [f for f in list_repo_files(REPO_SMOLLM, repo_type="dataset")
             if f.startswith(f"{subset}/") and f.endswith(".parquet")]
    return sorted(files)


def _download_smollm(subset: str, data_dir: str, max_files: int | None) -> list[str]:
    files = list_smollm(subset)
    if max_files:
        files = files[:max_files]
    return [hf_hub_download(REPO_SMOLLM, f, repo_type="dataset", cache_dir=data_dir) for f in files]


def ensure_smollm_shards(tokenizer, data_dir: str, *,
                         subsets: list[str] | None = None,
                         max_files_per_subset: int | None = None,
                         max_files_total: int | None = None,
                         force: bool = False) -> list[np.ndarray]:
    """Tokenize smollm-corpus subsets shard-by-shard -> per-shard uint16 npy caches.

    Returns a flat list of shards mixing both subsets (cosmopedia first, then
    fineweb). Call stream_iter or mixture_stream_iter over the result.

    Caches are named {subset}_shard{i}.npy under data_dir.
    Disk-efficient: downloads one parquet at a time, tokenizes to npy, then
    deletes the parquet blob immediately to keep peak disk ~1 shard (fixes
    OOM on Kaggle TPU where 340*1.1GB would otherwise fill disk).
    """
    if subsets is None:
        subsets = list(SMOLLM_SUBSETS)
    for s in subsets:
        assert s in SMOLLM_SUBSETS, s
    os.makedirs(data_dir, exist_ok=True)
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]
    shards: list[np.ndarray] = []
    total_cap = 0
    for subset in subsets:
        files = list_smollm(subset)
        if max_files_per_subset:
            files = files[:max_files_per_subset]
        # global total cap across subsets
        if max_files_total and len(shards) >= max_files_total:
            break
        if max_files_total:
            files = files[: max(0, max_files_total - len(shards))]
        print(f"SmollM [{subset}]: {len(files)} shard(s)")
        for i, f in enumerate(files):
            # global index for cache name to avoid collision across subsets
            gi = len(shards)
            cache = os.path.join(data_dir, f"smollm_{subset}_shard{i}.npy")
            # also support legacy flat cache name for backward compat check
            if os.path.exists(cache) and not force:
                arr = np.load(cache, mmap_mode="r")
                print(f"  {subset}/{os.path.basename(f)}: cached {arr.shape[0]:,} tokens")
            else:
                # streaming download: one parquet at a time
                p = hf_hub_download(REPO_SMOLLM, f, repo_type="dataset", cache_dir=data_dir)
                try:
                    nrows = pq.read_metadata(p).num_rows
                    cap = nrows * 800 + 1_000_000  # ~800 tok/doc is safer for edu web
                    arr = np.empty(cap, dtype=np.uint16)
                    pos = 0
                    table = pq.read_table(p, columns=["text"])
                    texts = table["text"].to_pylist()
                    bs = 512
                    for b in range(0, len(texts), bs):
                        enc = tokenizer(texts[b:b+bs], add_special_tokens=False)["input_ids"]
                        for ids in enc:
                            n = len(ids)
                            need = pos + n + 1
                            if need > arr.shape[0]:
                                arr = _ensure_capacity(arr, need)
                            arr[pos:pos+n] = np.asarray(ids, dtype=np.uint16)
                            pos += n
                            arr[pos] = nl
                            pos += 1
                    cap_now = arr.shape[0]
                    arr = arr[:pos]
                    np.save(cache, arr)
                    print(f"  {subset}/{os.path.basename(f)}: tokenized {pos:,} tokens cap {cap_now:,} util {pos/cap_now*100:.1f}%")
                    if cap_now != cap:
                        print(f"  [data] smollm {subset} shard buffer grown {cap:,} -> {cap_now:,}")
                    total_cap += pos
                    arr = np.load(cache, mmap_mode="r")
                finally:
                    # free disk: delete parquet blob immediately (peak disk ~1 shard)
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                        # also try to remove empty snapshot dir symlink target if present
                        # HF cache keeps blobs under datasets--HuggingFaceTB--smollm-corpus/blobs/
                        # removing the file above is enough to free 1.1GB per shard
                    except Exception as e:
                        print(f"  [data] warning: could not remove {p}: {e}")
            shards.append(arr)
            if max_files_total and len(shards) >= max_files_total:
                break
    print(f"SmollM total: {len(shards)} shards, ~{total_cap:,} tokens tokenized (cached on disk)")
    return shards


# ---------------------------------------------------------------------------
# EleutherAI/SmolLM2-135M-10B — 10B sample of SmolLM2 2T mixture (85 parquet shards)
# HF dataset: data/train-*.parquet (85 files, ~25GB download, 42GB uncompressed)
# Features: text (string), source (string). Tokenized like smollm: one shard
# = one uint16 npy cache, streamed. Use via --smollm2 on train.py.
REPO_SMOLM2 = "EleutherAI/SmolLM2-135M-10B"

def list_smolm2() -> list[str]:
    files = [f for f in list_repo_files(REPO_SMOLM2, repo_type="dataset") if f.endswith(".parquet")]
    return sorted(files)

def ensure_smolm2_shards(tokenizer, data_dir: str, *, max_files: int | None = None, offset: int = 0, force: bool = False, reuse_cache_slot: bool = False) -> list[np.ndarray]:
    """Tokenize EleutherAI/SmolLM2-135M-10B shard-by-shard -> per-shard uint16 npy caches.
    offset: start index in full 85 list (for disk-friendly phased training 3-4 shards at a time)
    reuse_cache_slot: if True, cache files are named smollm2_shard0..3 reused each phase (keeps disk ~4*0.6GB)
    """
    os.makedirs(data_dir, exist_ok=True)
    files = list_smolm2()
    # apply offset + max_files slicing
    if offset:
        files = files[offset: offset + max_files] if max_files else files[offset:]
    elif max_files:
        files = files[:max_files]
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]
    print(f"SmolLM2-10B: {len(files)} shard(s) from {REPO_SMOLM2}")
    shards: list[np.ndarray] = []
    total = 0
    for i, f in enumerate(files):
        # cache slot reuse keeps disk at ~4 shards (~2.4GB) instead of 85*0.6GB
        slot = i if not reuse_cache_slot else i % 4
        # when offset is used, include global index in cache name unless reuse
        if reuse_cache_slot:
            cache = os.path.join(data_dir, f"smollm2_shard{slot}.npy")
        else:
            global_i = offset + i
            cache = os.path.join(data_dir, f"smollm2_shard{global_i}.npy")
        if os.path.exists(cache) and not force:
            arr = np.load(cache, mmap_mode="r")
            print(f"  {os.path.basename(f)}: cached {arr.shape[0]:,} tokens")
        else:
            p = hf_hub_download(REPO_SMOLM2, f, repo_type="dataset", cache_dir=data_dir)
            try:
                nrows = pq.read_metadata(p).num_rows
                cap = nrows * 800 + 1_000_000
                arr = np.empty(cap, dtype=np.uint16)
                pos = 0
                table = pq.read_table(p, columns=["text"])
                texts = table["text"].to_pylist()
                bs = 512
                for b in range(0, len(texts), bs):
                    enc = tokenizer(texts[b:b+bs], add_special_tokens=False)["input_ids"]
                    for ids in enc:
                        n = len(ids); need = pos + n + 1
                        if need > arr.shape[0]:
                            arr = _ensure_capacity(arr, need)
                        arr[pos:pos+n] = np.asarray(ids, dtype=np.uint16); pos += n; arr[pos]=nl; pos+=1
                cap_now = arr.shape[0]; arr = arr[:pos]; np.save(cache, arr)
                print(f"  {os.path.basename(f)}: tokenized {pos:,} tokens cap {cap_now:,} util {pos/cap_now*100:.1f}%")
                total += pos
                arr = np.load(cache, mmap_mode="r")
            finally:
                try:
                    if os.path.exists(p): os.remove(p)
                except Exception as e:
                    print(f"  [data] warning: could not remove {p}: {e}")
        shards.append(arr)
    print(f"SmolLM2 total: {len(shards)} shards, ~{total:,} tokens tokenized (cached)")
    return shards


def _ensure_smolm2_one(tokenizer, data_dir: str, fname: str, global_idx: int, force: bool = False) -> np.ndarray:
    """Download+tokenize single SmolLM2 shard -> smollm2_shard{global_idx}.npy (mmap)."""
    cache = os.path.join(data_dir, f"smollm2_shard{global_idx}.npy")
    if os.path.exists(cache) and not force:
        arr = np.load(cache, mmap_mode="r")
        print(f"  [sliding] {os.path.basename(fname)}: cached {arr.shape[0]:,} tokens -> {cache}")
        return arr
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]
    p = hf_hub_download(REPO_SMOLM2, fname, repo_type="dataset", cache_dir=data_dir)
    try:
        nrows = pq.read_metadata(p).num_rows
        cap = nrows * 800 + 1_000_000
        arr = np.empty(cap, dtype=np.uint16)
        pos = 0
        table = pq.read_table(p, columns=["text"])
        texts = table["text"].to_pylist()
        bs = 512
        for b in range(0, len(texts), bs):
            enc = tokenizer(texts[b:b+bs], add_special_tokens=False)["input_ids"]
            for ids in enc:
                n = len(ids); need = pos + n + 1
                if need > arr.shape[0]:
                    arr = _ensure_capacity(arr, need)
                arr[pos:pos+n] = np.asarray(ids, dtype=np.uint16); pos += n; arr[pos]=nl; pos+=1
        cap_now = arr.shape[0]; arr = arr[:pos]; np.save(cache, arr)
        print(f"  [sliding] {os.path.basename(fname)}: tokenized {pos:,} tokens cap {cap_now:,} util {pos/cap_now*100:.1f}%")
        arr = np.load(cache, mmap_mode="r")
    finally:
        try:
            if os.path.exists(p): os.remove(p)
        except Exception:
            pass
        # also clean HF blob cache under datasets--EleutherAI--SmolLM2-135M-10B
        try:
            import glob as _g, shutil as _sh
            for d in _g.glob(os.path.join(data_dir, "datasets--EleutherAI--SmolLM2-135M-10B")):
                # don't delete npy caches, only blobs
                pass
        except Exception:
            pass
    return arr


def sliding_smolm2_iter(tokenizer, data_dir: str, batch_size: int, seq_len: int, total_steps: int,
                        window: int = 4, overlap: int = 1, shuffle: bool = True, rng=None, skip_steps: int = 0):
    """Sliding overlapped-window iterator for 85-shard SmolLM2 — resume-aware.

    Keeps optimizer/params resident in one process. Disk holds at most `window` .npy
    files (std smollm2_shard{i}.npy); oldest is deleted as window slides — so peak
    disk ~ window*0.6GB. Overlap keeps `overlap` shards from previous window in the
    mixture for continuity (loss plateau break). Yields indefinitely until caller
    stops at total_steps.
    skip_steps: resume offset (= start_step from ckpt). Fast-forwards perm window
    to the slide containing skip_steps (floored to slide boundary) so training
    restarts from the shard it left, not from shard 0.
    """
    import collections
    os.makedirs(data_dir, exist_ok=True)
    files = list_smolm2()
    n = len(files)
    if rng is None:
        rng = np.random.default_rng(0)
    perm = rng.permutation(n) if shuffle else np.arange(n)
    steps_per_shard = max(total_steps // max(n, 1), 1)
    drop_n = max((window - overlap), 1)
    advance_every = drop_n * steps_per_shard
    print(f"[sliding] window={window} overlap={overlap} n={n} steps_per_shard={steps_per_shard} total_steps={total_steps} shuffle={shuffle} skip_steps={skip_steps}")
    # deque of (global_idx, mmap array)
    dq: collections.deque = collections.deque()
    # resume: fast-forward to slide containing skip_steps (floored)
    _skip_floor = 0
    _start_pos = 0
    if skip_steps and skip_steps > 0:
        num_slides = skip_steps // max(advance_every, 1)
        _start_pos = int(num_slides * drop_n)
        # clamp so window fits in 85
        if _start_pos + window > n:
            _start_pos = max(0, n - window)
            num_slides = _start_pos // max(drop_n, 1)
        _skip_floor = int(num_slides * advance_every)
        print(f"  [sliding-resume] skip_steps={skip_steps} -> start_pos={_start_pos} next_ptr={_start_pos+window}/{n} emitted_floor={_skip_floor} (repeat {skip_steps-_skip_floor} steps in current window)")
    # sequential sliding: for each position pos in perm order, ensure that shard,
    # push to dq, and when dq exceeds window, drop oldest file from disk.
    # Within a window we interleave batches round-robin across shards in dq.
    next_ptr = _start_pos
    # preload window at resume position (or first window on fresh start)
    for _ in range(min(window, n - _start_pos)):
        gi = int(perm[next_ptr]); fname = files[gi]
        arr = _ensure_smolm2_one(tokenizer, data_dir, fname, gi)
        dq.append((gi, arr))
        next_ptr += 1
    print(f"  [sliding] preloaded window {[g for g,_ in dq]} (start_pos={_start_pos})")
    emitted = _skip_floor
    # round-robin iterators per shard in window
    iters = [make_iter(arr, batch_size, seq_len, rng) for _, arr in dq]
    while emitted < total_steps:
        # yield one batch round-robin from current window
        for wi in range(len(dq)):
            if emitted >= total_steps:
                break
            it = iters[wi]
            # steps_per_shard controls how many batches we draw before sliding;
            # we slide every (window-overlap)*steps_per_shard batches
            yield next(it)
            emitted += 1
            # check if window should slide
            advance_every = max((window - overlap), 1) * steps_per_shard
            if emitted % advance_every == 0 and next_ptr < n:
                # slide by (window-overlap) shards: drop oldest `window-overlap`, keep `overlap`
                drop_n = window - overlap
                for _ in range(drop_n):
                    if dq:
                        old_gi, old_arr = dq.popleft()
                        iters.pop(0)
                        # close mmap before delete (numpy memmap)
                        try:
                            del old_arr
                        except Exception:
                            pass
                        old_cache = os.path.join(data_dir, f"smollm2_shard{old_gi}.npy")
                        try:
                            if os.path.exists(old_cache):
                                os.remove(old_cache)
                                print(f"  [sliding] cleaned shard {old_gi} -> {old_cache} ({emitted}/{total_steps} steps)")
                        except Exception as e:
                            print(f"  [sliding] warning clean {old_gi}: {e}")
                # pull next drop_n shards
                for _ in range(drop_n):
                    if next_ptr >= n:
                        break
                    gi = int(perm[next_ptr]); fname = files[gi]
                    arr = _ensure_smolm2_one(tokenizer, data_dir, fname, gi)
                    dq.append((gi, arr))
                    iters.append(make_iter(arr, batch_size, seq_len, rng))
                    next_ptr += 1
                print(f"  [sliding] window slid -> {[g for g,_ in dq]} next_ptr={next_ptr}/{n}")
                # clean HF blob cache leftovers
                try:
                    import glob as _g, shutil as _sh
                    for d in _g.glob(os.path.join(data_dir, "datasets--EleutherAI--SmolLM2-135M-10B")):
                        # keep dir but blobs already removed per shard
                        pass
                    for d in _g.glob(os.path.join(data_dir, ".cache")):
                        _sh.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
                break  # restart round-robin after slide
        # if we have exhausted all shards but still need steps, reshuffle and continue (second epoch)
        if next_ptr >= n and emitted < total_steps and len(dq) == 0:
            perm = rng.permutation(n) if shuffle else np.arange(n)
            next_ptr = 0
            for _ in range(min(window, n)):
                gi = int(perm[next_ptr]); fname = files[gi]
                arr = _ensure_smolm2_one(tokenizer, data_dir, fname, gi)
                dq.append((gi, arr))
                next_ptr += 1
            iters = [make_iter(arr, batch_size, seq_len, rng) for _, arr in dq]
            print(f"  [sliding] epoch wrap -> new window {[g for g,_ in dq]}")


def _ensure_smollm_one(tokenizer, data_dir: str, subset: str, fname: str, global_idx: int, force: bool = False) -> np.ndarray:
    """Download+tokenize single smollm-corpus shard -> smollm_{subset}_shard{global_idx}.npy (mmap)."""
    assert subset in SMOLLM_SUBSETS, subset
    cache = os.path.join(data_dir, f"smollm_{subset}_shard{global_idx}.npy")
    if os.path.exists(cache) and not force:
        arr = np.load(cache, mmap_mode="r")
        print(f"  [sliding-smollm] {subset}/{os.path.basename(fname)}: cached {arr.shape[0]:,} tokens -> {cache}")
        return arr
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]
    p = hf_hub_download(REPO_SMOLLM, fname, repo_type="dataset", cache_dir=data_dir)
    try:
        nrows = pq.read_metadata(p).num_rows
        cap = nrows * 800 + 1_000_000  # ~800 tok/doc for edu web/textbook
        arr = np.empty(cap, dtype=np.uint16)
        pos = 0
        table = pq.read_table(p, columns=["text"])
        texts = table["text"].to_pylist()
        bs = 512
        for b in range(0, len(texts), bs):
            enc = tokenizer(texts[b:b+bs], add_special_tokens=False)["input_ids"]
            for ids in enc:
                n = len(ids); need = pos + n + 1
                if need > arr.shape[0]:
                    arr = _ensure_capacity(arr, need)
                arr[pos:pos+n] = np.asarray(ids, dtype=np.uint16); pos += n; arr[pos] = nl; pos += 1
        cap_now = arr.shape[0]; arr = arr[:pos]; np.save(cache, arr)
        print(f"  [sliding-smollm] {subset}/{os.path.basename(fname)}: tokenized {pos:,} tokens cap {cap_now:,} util {pos/cap_now*100:.1f}%")
        arr = np.load(cache, mmap_mode="r")
    finally:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    return arr


def sliding_smollm_iter(tokenizer, data_dir: str, subset: str, batch_size: int, seq_len: int, total_steps: int,
                        window: int = 4, overlap: int = 1, shuffle: bool = True, rng=None, skip_steps: int = 0):
    """Sliding overlapped-window iterator for smollm-corpus single subset — resume-aware.

    Same protocol as sliding_smolm2_iter but over list_smollm(subset) (e.g.
    cosmopedia-v2 104 shards, ~28B GPT-2 tokens, ~1.1GB npy/shard). Disk holds at
    most `window` .npy files (default 4 -> ~4.4GB); oldest are deleted as the
    window slides. Overlap keeps `overlap` shards for continuity.
    skip_steps (= start_step from ckpt) fast-forwards to the slide containing it.
    """
    import collections
    assert subset in SMOLLM_SUBSETS, subset
    os.makedirs(data_dir, exist_ok=True)
    files = list_smollm(subset)
    n = len(files)
    if rng is None:
        rng = np.random.default_rng(0)
    perm = rng.permutation(n) if shuffle else np.arange(n)
    steps_per_shard = max(total_steps // max(n, 1), 1)
    drop_n = max((window - overlap), 1)
    advance_every = drop_n * steps_per_shard
    print(f"[sliding-smollm] subset={subset} window={window} overlap={overlap} n={n} steps_per_shard={steps_per_shard} total_steps={total_steps} shuffle={shuffle} skip_steps={skip_steps}")
    dq: collections.deque = collections.deque()
    _skip_floor = 0
    _start_pos = 0
    if skip_steps and skip_steps > 0:
        num_slides = skip_steps // max(advance_every, 1)
        _start_pos = int(num_slides * drop_n)
        if _start_pos + window > n:
            _start_pos = max(0, n - window)
            num_slides = _start_pos // max(drop_n, 1)
        _skip_floor = int(num_slides * advance_every)
        print(f"  [sliding-smollm-resume] skip_steps={skip_steps} -> start_pos={_start_pos} next_ptr={_start_pos+window}/{n} emitted_floor={_skip_floor}")
    next_ptr = _start_pos
    for _ in range(min(window, n - _start_pos)):
        gi = int(perm[next_ptr]); fname = files[gi]
        arr = _ensure_smollm_one(tokenizer, data_dir, subset, fname, gi)
        dq.append((gi, arr))
        next_ptr += 1
    print(f"  [sliding-smollm] preloaded window {[g for g, _ in dq]} (start_pos={_start_pos})")
    emitted = _skip_floor
    iters = [make_iter(arr, batch_size, seq_len, rng) for _, arr in dq]
    while emitted < total_steps:
        for wi in range(len(dq)):
            if emitted >= total_steps:
                break
            it = iters[wi]
            yield next(it)
            emitted += 1
            advance_every = max((window - overlap), 1) * steps_per_shard
            if emitted % advance_every == 0 and next_ptr < n:
                drop_n = window - overlap
                for _ in range(drop_n):
                    if dq:
                        old_gi, old_arr = dq.popleft()
                        iters.pop(0)
                        try:
                            del old_arr
                        except Exception:
                            pass
                        old_cache = os.path.join(data_dir, f"smollm_{subset}_shard{old_gi}.npy")
                        try:
                            if os.path.exists(old_cache):
                                os.remove(old_cache)
                                print(f"  [sliding-smollm] cleaned shard {old_gi} -> {old_cache} ({emitted}/{total_steps} steps)")
                        except Exception as e:
                            print(f"  [sliding-smollm] warning clean {old_gi}: {e}")
                for _ in range(drop_n):
                    if next_ptr >= n:
                        break
                    gi = int(perm[next_ptr]); fname = files[gi]
                    arr = _ensure_smollm_one(tokenizer, data_dir, subset, fname, gi)
                    dq.append((gi, arr))
                    iters.append(make_iter(arr, batch_size, seq_len, rng))
                    next_ptr += 1
                print(f"  [sliding-smollm] window slid -> {[g for g, _ in dq]} next_ptr={next_ptr}/{n}")
                try:
                    import glob as _g, shutil as _sh
                    for d in _g.glob(os.path.join(data_dir, ".cache")):
                        _sh.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
                break
        if next_ptr >= n and emitted < total_steps and len(dq) == 0:
            perm = rng.permutation(n) if shuffle else np.arange(n)
            next_ptr = 0
            for _ in range(min(window, n)):
                gi = int(perm[next_ptr]); fname = files[gi]
                arr = _ensure_smollm_one(tokenizer, data_dir, subset, fname, gi)
                dq.append((gi, arr))
                next_ptr += 1
            iters = [make_iter(arr, batch_size, seq_len, rng) for _, arr in dq]
            print(f"  [sliding-smollm] epoch wrap -> new window {[g for g, _ in dq]}")


def mixture_stream_iter(shards_a: list[np.ndarray], shards_b: list[np.ndarray],
                        batch_size: int, seq_len: int,
                        steps_per_shard: int,
                        weight_a: float = 0.5,
                        rng=None):
    """Interleaved mixture of two shard pools (e.g. cosmopedia vs fineweb).

    At each shard pick, choose pool A with prob weight_a else pool B.
    Falls back to whichever pool is non-empty. Steps_per_shard batches are
    drawn from the chosen shard before picking the next shard.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if not shards_a and not shards_b:
        raise ValueError("both shard pools empty")
    if not shards_a:
        yield from stream_iter(shards_b, batch_size, seq_len, steps_per_shard, rng)
        return
    if not shards_b:
        yield from stream_iter(shards_a, batch_size, seq_len, steps_per_shard, rng)
        return
    # shuffle orders within each pool independently
    order_a = rng.permutation(len(shards_a))
    order_b = rng.permutation(len(shards_b))
    ia = ib = 0
    while True:
        pick_a = rng.random() < weight_a
        if pick_a:
            idx = int(order_a[ia % len(order_a)])
            ia += 1
            if ia % len(order_a) == 0:
                order_a = rng.permutation(len(shards_a))
            it = make_iter(shards_a[idx], batch_size, seq_len, rng)
        else:
            idx = int(order_b[ib % len(order_b)])
            ib += 1
            if ib % len(order_b) == 0:
                order_b = rng.permutation(len(shards_b))
            it = make_iter(shards_b[idx], batch_size, seq_len, rng)
        for _ in range(steps_per_shard):
            yield next(it)
