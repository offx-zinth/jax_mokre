# jax_mokre — Review Problems

**Date:** 2026-08-27
**Scope:** `jax_mokre/model.py`, `jax_mokre/config.py`, `jax_mokre/train.py`, `jax_mokre/data.py`, `jax_mokre/smoke_test.py` vs `mokre/` (PyTorch reference)
**Config inspected:** `tinystories` (`hidden=384, heads=6, kv=2, head_dim=64, E=8, top_k=1, depth=4, layer_types=[kda,kda,msa,kda]`) and `default` (`hidden=1024, heads=16, kv=8, top_k=2`)

---

## 1. CRITICAL / HIGH — Fix before scaling on TPU

### H1 — Dead learning-rate schedule (`jax_mokre/train.py:57`, `jax_mokre/train.py:371`)
**What:** `schedule()` computes `lr` with warmup 2000 + cosine decay `lr_min=3e-5` (`train.py:148`), and `lr = jnp.asarray(lr)` at `train.py:371`. But `make_train_step` at `train.py:57-63` does:
```python
def step(params, opt_state, x, y, lr):
    (loss,(ce,aux)), grads = value_and_grad(...)(params,x,y)
    updates, opt_state = opt.update(grads, opt_state, params)  # lr is ignored
```
`opt` is `optax.chain(clip, adamw(3e-4))` with a **constant** `3e-4` (`train.py:278`). The `lr` argument is never read and never injected via `optax.inject_hyperparams` or `optax.scale_by_schedule`. Warmup/cosine has no effect.

**Impact:** Training runs at flat 3e-4 for 40k steps. Warmup (stability at init) and decay (convergence) are silently disabled. Loss curves will look unexpectedly noisy/plateaued.

**Fix:** Wrap base optimizer with hyperparams injection:
```python
base_opt = optax.chain(
    optax.clip_by_global_norm(args.grad_clip),
    optax.inject_hyperparams(optax.adamw)(learning_rate=0.0, b1=0.9, b2=0.95, eps=1e-8, weight_decay=args.weight_decay),
)
# then at step:
updates, opt_state = opt.update(grads, opt_state, params, learning_rate=lr)
```
Or use `optax.scale_by_schedule` + `optax.scale(-lr)` pattern. Update both `make_train_step` and `make_train_step_pjit` (`train.py:100`).

---

### H2 — Recursion auxiliary loss dominates cross-entropy (`jax_mokre/model.py:787`, `jax_mokre/model.py:937`, `jax_mokre/config.py:39`)
**What:** `router_forward` pushes depth via `aux_push = rec_aux_coef * max(Nr - exp_depth, 0)` with default `rec_aux_coef=0.1` and `Nr=4`. If the router collapses to depth 1 (`exp_depth≈1`), `aux_push≈0.3`. `aux_lb = coef * sum(f*P)` is ~0.01-0.02. These are summed into `aux` and **added directly to CE** (`train.py:49` `ce+aux`). `forward` then does `aux = aux + block_aux/Nr` per recursion step (`model.py:937`). `block_aux` already sums 4 layers of `moe_aux + msa_kl` per step, so the final `aux` is a mean over steps plus `a_router + a_first + a_last`.

**Impact:** At initialization the total aux can be 0.3-0.5 vs CE ~10.8 (tiny), but after ~2k steps when CE ~4-5, aux is ~6-10% of loss and its gradient competes with LM signal. With `rec_aux_coef=0.1` the model is forced deeper even when shallower is optimal for easy tokens. No ablation in repo.

**Fix:** Reduce `rec_aux_coef` to `0.01-0.03` or make it scheduled (warmup 0 → 0.05). Log `aux` decomposed (`aux_lb`, `aux_push`, `moe_aux`, `msa_kl`) separately in `train.log` to monitor dominance. Consider normalizing `aux_push` by `Nr`.

---

### H3 — KDA chunked path dead code and misleading naming (`jax_mokre/model.py:108-147`)
**What:** For `S <= chunk`, the function computes `s_t` then `out = s_t * q` and falls through to `return lin(out, o_w)` at `model.py:147`. For `S > chunk` it builds `out_parts` as `s_chunk * q_chunk` and at `model.py:140-144` does:
```python
s_t = jnp.concatenate(out_parts, axis=1)  # actually already s*q
out = s_t.reshape(bsz, S, NH*D)
return lin(out, p["o_w"])  # early return, lines 145-147 unreachable
```
`s_t` is misnamed (it holds the projected-ready tensor, not the state `s_t`). Lines 145-146 are dead on the large-S path, which is confusing for future edits and hides that the two branches return different intermediate names.

**Impact:** No numerical bug currently — both paths return `lin(s*q)`. But a future edit adding logic after line 143 (e.g., dropout) would silently not execute for large S, creating a divergence between short/long sequence behavior. Also allocates `s_t` as `(B,S,NH,D)` twice in the small-S branch (memory waste).

**Fix:** Rename `out_parts` concatenation to `out` and unify return:
```python
else:
    ...
    out = jnp.concatenate(out_parts, axis=1).reshape(bsz, S, NH*D)
    return lin(out, p["o_w"])
out = (s_t * q).reshape(bsz, S, NH*D)
return lin(out, p["o_w"])
```

---

### H4 — Inference recursion has no true gather-compute-scatter; `lax.cond` is cosmetic (`jax_mokre/model.py:898-928`)
**What:** The forward recursion advertises conditional execution to save FLOPs. For `training=False` it computes `is_empty = jnp.all(m_hard==0)` and wraps `do_block` in `jax.lax.cond` (`model.py:925`). But `do_block` still calls `mor_layer` with `token_mask=m_hard` on **all** `B*S` tokens; `mor_layer` only masks the residual (`m*x + (1-m)*orig` at `model.py:816-817`), while KDA/MSA/MoE still run densely on frozen tokens. The comment at `model.py:917` admits "we would gather indices ... but keep masked path". No gathering happens.

**Impact:** Zero FLOP saving except the trivial `is_empty` full-skip (rare when depth varies per token). At S=512/B=16, ~40-60% of tokens typically freeze at depth 1-2, so 1-2 of the 4 recursion steps waste ~50% of their attention+MoE FLOPs on frozen tokens. On TPU this costs ~20-30% extra step time with no benefit.

**Fix:** Either remove the `lax.cond` indirection (keep dense, document accurately) or implement real gather:
```python
active = jnp.where(m_hard.reshape(-1))[0]
h_flat = h.reshape(-1, H)[active]  # gather
# run block on h_flat only, scatter back
```
Needs careful handling of KDA state (causal recurrence cannot be arbitrarily gathered without breaking positions) — so document as known limitation if not fixed.

---

## 2. MEDIUM — Correctness / performance traps

### M1 — `init_state` broadcast assumes divisibility (`jax_mokre/model.py:115`, `jax_mokre/model.py:120`)
```python
s0 = jnp.repeat(p["init_state"], G, axis=1)  # (1,NH,D)
s0 = jnp.broadcast_to(s0[:, None, :, :], (bsz, S, NH, D))
```
`p["init_state"]` is `(1, NG, D)`. `repeat(G)` with `G=NH//NG` works only when `NH % NG ==0` (asserted in config). But `init_state` is never updated (learned initial state stays zero) and is broadcast to `(B,S,NH,D)` which allocates `B*S*NH*D` floats (e.g., 16*512*16*64 ≈ 8M) — same size as `A`. Acceptable at 384H but OOM risk at 1024H/S=8192.

**Fix:** Keep carry as `(B,NH,D)` (as done in chunked path) even for small S; don't expand to `(B,S,NH,D)` until needed.

---

### M2 — Data buffer slicing hides truncation (`jax_mokre/data.py:75`, `jax_mokre/data.py:87`, `jax_mokre/data.py:216`)
`arr = np.empty(cap)` where `cap = nrows*400 + 1_000_000`. `_tokenize_shard` writes `arr[pos:pos+n]` and returns `arr[:pos]` after. If `cap` is underestimated (long TinyStories docs, >400 tok avg), the warning at `data.py:46` fires and `rest of corpus dropped`. No hard error, just truncated corpus without surfacing `cap` vs `pos` to the caller. The `np.empty` tail is uninitialized but never read after slicing, so not a bug — but `np.save(cache, tokens)` saves only `arr[:pos]`, so subsequent `np.load(mmap)` returns the truncated size silently.

**Fix:** After each shard, log `pos/cap` utilization and raise if `truncated` is True, or grow `cap` via `np.resize` / double allocation.

---

### M3 — Engram `total_N` double compute with tracer (`jax_mokre/model.py:551-553`)
```python
total_N = int(jnp.sum(static["primes"]).item()) if hasattr(...) else sum(...)
total_N = sum(static["primes_list"])  # immediately overwrites
```
First line tries to use a JAX tracer in Python `int()` context (would fail under jit), then is overwritten. Dead code that confuses readers into thinking `total_N` is JAX-computed.

**Fix:** Keep only `total_N = sum(static["primes_list"])`.

---

### M4 — `JAX_MESH` / `make_train_step_pjit` is stubbed (`jax_mokre/train.py:74-114`)
The mesh block at `train.py:77-83` is `pass`, and `make_train_step_pjit` at `train.py:90-114` just returns `jax.jit(step)` regardless of `mesh`, with comment "caller should wrap with pjit with explicit in_shardings". So `--mesh` / H=1024/E=16 scale claims in README are not actually sharded — params stay replicated per device, doubling HBM usage.

**Impact:** At `hidden=1024, E=8, vocab=50257` the model is ~250M params (~1GB bf16). On 8-way TPU v5e (16GB/HBM), replication works, but at H=2048/E=32 it will OOM. No test covers multi-device.

**Fix:** Either remove the stub and document "data-parallel via pmap only", or implement real `NamedSharding` for `embed_tokens (vocab, hidden)` sharded on `model` axis and MoE `expert_* (E,M,H)` sharded on `E`.

---

### M5 — No bf16 / dtype control (`jax_mokre/model.py:105`, `jax_mokre/train.py:277`)
All params initialized `float32` (`jax_mokre/model.py:17-72`), no `jnp.bfloat16` cast for matmuls. On TPU v5e, float32 halves throughput and doubles HBM. `kda_forward` norms use `1e-6` eps which underflows in bf16. Needs dtype policy.

**Fix:** Add `cfg.param_dtype / compute_dtype` and cast `x` to `bf16` inside `kda_forward`/`msa_forward`/`moe_forward`, keep `rmsnorm` in `float32`.

---

## 3. LOW — Nits and documentation debt

### L1 — Param-count assertion drift (`jax_mokre/smoke_test.py:41`, `jax_mokre/README.md:4`)
README says `85,227,144 params` (MLA config), smoke test asserts `80M < n < 90M` with note `MSA ~85,255,816`. The two counts differ by ~28k (index branch `q_idx_w + k_idx_w`). Not asserted exactly, so a regression adding/removing params won't be caught.

### L2 — `chunk` clamping without warning (`jax_mokre/model.py:111-112`)
```python
if chunk < 16: chunk = 16
```
Silently overrides user-specified `kda_chunk_size <16`. Should warn or raise.

### L3 — `attention_mask` vs `token_mask` dual semantics (`jax_mokre/model.py:178-183`)
`attention_mask` (padding) and `token_mask` (recursion freezing) are both applied as `key_valid` masks to `S_idx` and `selected_mask` identically. No distinction in KL path, so padding tokens still contribute to `has_valid` if `attention_mask` is None (default training path). Training without padding mask will include BOS/pad confusion at S=512.

### L4 — Checkpoint pickle without version (`jax_mokre/train.py:124-135`)
`pickle.dump({"config": cfg.__dict__, ...})` stores raw `__dict__`. Adding a new config field (e.g., `msa_topk`) breaks old `load_state` silently. Use explicit `cfg` dataclass `asdict` + version tag.

### L5 — `run_tpu.sh` hardcodes TinyStories; FineWeb path needs manual edit (no CLI pass-through).

---

## 4. Impact matrix

| ID | Severity | Affects training correctness? | Affects performance? | Effort |
|---|---|---|---|---|
| H1 | Critical | Yes — LR schedule dead | — | 2 lines |
| H2 | High | Yes — aux tuning | Indirect (forces depth) | Config + logging |
| H3 | High (maintainability) | Latent | — | 5 lines |
| H4 | High | No | Yes ~20-30% waste | Medium (gather impl) |
| M1 | Medium | No | Memory | 3 lines |
| M2 | Medium | Yes — silent truncation | — | 10 lines |
| M3 | Low | No | — | 1 line |
| M4 | Medium | No | Scale OOM | Larger |
| M5 | Medium | No | 2x throughput | Config |
| L1-L5 | Low | No | No | Docs |

---

## 5. Recommended fix order (if you want a follow-up PR)

1. H1 (LR) — 5 min, highest ROI.
2. H3 (KDA rename) + M3 (Engram) — cleanup.
3. H2 (aux logging + reduce `rec_aux_coef` to 0.03, sweep).
4. M2 (buffer growth warning).
5. M5 (bf16 policy) + M1 (carry shape) before TPU scale.
6. M4 (real pjit) or remove claim.
7. H4 (true gather) — only after correctness fixes.

## 6. How to verify after fixes

```bash
pip install "jax[cpu]" optax transformers huggingface_hub pyarrow
python3 -m jax_mokre.smoke_test
# expect: ALL SMOKE TESTS PASSED (MSA)
python3 -m jax_mokre.train --config tinystories --seq_len 64 --batch_size 2 --total_steps 5 --synthetic --out_dir /tmp/run
# check train.log: lr varies, aux decomposed, loss finite and decreasing
```

---

*Generated from read-only inspection; no runtime execution performed beyond static analysis. Re-run smoke_test after any fix to confirm bit-close KDA (`err <1e-5`) and MSA KL grad (`mean|g| >0`).*
