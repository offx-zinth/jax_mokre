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

# Helper to inject dynamic LR into opt state (H1 fix). Works for:
# - MultiSteps( chain(clip, inject_adamw) )  -> MultiStepsState
# - chain(clip, inject_adamw)                -> tuple
def _set_lr_in_state(opt_state, lr):
    """Return new opt_state with hyperparams['learning_rate'] = lr (jittable)."""
    # MultiSteps wrapper
    if hasattr(opt_state, "inner_opt_state"):
        inner = opt_state.inner_opt_state  # tuple (clip_state, inject_state)
        # inject is second element of chain
        if isinstance(inner, tuple) and len(inner) == 2 and hasattr(inner[1], "hyperparams"):
            inject = inner[1]
            new_inject = inject._replace(hyperparams={**inject.hyperparams, "learning_rate": lr})
            new_inner = (inner[0], new_inject)
            return opt_state._replace(inner_opt_state=new_inner)
        # fallback: try to walk generic
        return opt_state
    # plain chain tuple
    if isinstance(opt_state, tuple) and len(opt_state) == 2 and hasattr(opt_state[1], "hyperparams"):
        inject = opt_state[1]
        new_inject = inject._replace(hyperparams={**inject.hyperparams, "learning_rate": lr})
        return (opt_state[0], new_inject)
    # unknown structure: try generic tree walk (best-effort)
    return opt_state

from .config import MoREConfig
from . import model as M
from . import data as D


def chunked_ce(cfg, Wt, hidden, labels, chunk):
    """Cross-entropy over the vocab head, computed in seq-chunks to bound memory."""
    B, S, H = hidden.shape
    total = 0.0
    denom = 0
    for i in range(0, S, chunk):
        h = hidden[:, i:i + chunk].reshape(-1, H)
        y = labels[:, i:i + chunk].reshape(-1)
        lg = jnp.einsum("nh,vh->nv", h, Wt)
        logp = jax.nn.log_softmax(lg, axis=-1)
        total = total + (-jnp.sum(logp[jnp.arange(y.shape[0]), y]))
        denom = denom + y.shape[0]
    return total / denom


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
        opt_state = _set_lr_in_state(opt_state, lr)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, ce, aux

    if dist:
        return jax.pmap(step, axis_name="batch",
                        in_axes=(None, None, 0, 0, None),
                        out_axes=(None, None, 0, 0, 0))
    # Data-parallel only via pmap above; model-parallel (FSDP) is opt-in via
    # make_train_step_pjit with an explicit Mesh. JAX_MESH env is not auto-wired
    # here to avoid silent replication at H>=1024/E>=16 — use pjit path instead.
    # See make_train_step_pjit docstring for sharding example.
    return jax.jit(step)


def _make_param_shardings(params, mesh):
    """Build sharding pytree for params: embed_tokens on vocab/model, experts on E."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    def spec_for_path(path, val):
        # path is tuple of keys like ('embed_tokens',) or ('block', 0, 'moe', 'expert_gate_w')
        p = "/".join(str(k) for k in path)
        if isinstance(val, jnp.ndarray):
            if val.ndim == 2 and "embed_tokens" in p:
                # (vocab, hidden) -> shard vocab on model
                return NamedSharding(mesh, P("model", None) if "model" in mesh.axis_names else P(None))
            if val.ndim == 3 and "expert" in p:
                # (E, M, H) or (E, H, M) -> shard E on model
                return NamedSharding(mesh, P("model", None, None) if "model" in mesh.axis_names else P(None))
            if val.ndim == 2 and val.shape[0] == 50257:
                return NamedSharding(mesh, P("model", None) if "model" in mesh.axis_names else P(None))
        # default: replicated (or data-sharded if needed)
        return NamedSharding(mesh, P())
    # jax.tree_util.tree_map_with_path available in newer JAX
    try:
        from jax.tree_util import tree_map_with_path
        return tree_map_with_path(lambda path, v: spec_for_path(path, v) if isinstance(v, jnp.ndarray) else v, params)
    except ImportError:
        # fallback: replicate all
        return jax.tree_util.tree_map(lambda v: NamedSharding(mesh, P()) if isinstance(v, jnp.ndarray) else v, params)


def make_train_step_pjit(cfg, opt, ce_chunk, mesh=None, remat=True):
    """pjit/FSDP variant for large-scale: shards params across mesh (data, model).

    Data-parallel via pmap is used by make_train_step when len(devices)>1.
    For H>=1024 or E>=16 where replication would OOM (250M params ~1GB bf16
    replicated ×8 = 8GB), use this pjit path with explicit sharding:

        from jax.sharding import Mesh
        mesh = Mesh(jax.devices().reshape(2,4), ("data","model"))
        step_fn = make_train_step_pjit(cfg, opt, 32, mesh=mesh)
        # Params will be auto-sharded: embed_tokens on vocab/model, experts on E.
        # Batch is sharded on "data".

    If mesh is None this falls back to jit (same as make_train_step single-host).
    """
    loss_fn = make_loss_fn(cfg, ce_chunk, remat)

    def step(params, opt_state, x, y, lr):
        (loss, (ce, aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y)
        # NOTE: for pjit/shard_map, grads are already sharded on data axis;
        # do NOT pmean here unless inside a named axis context (pmap/shard_map).
        # Keeping this as no-op for single-host mesh avoids unbound axis error.
        # For multi-host data-parallel with shard_map, caller should wrap step
        # with shard_map and handle reduction externally.
        opt_state = _set_lr_in_state(opt_state, lr)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, ce, aux

    if mesh is None:
        return jax.jit(step)
    from jax.sharding import NamedSharding, PartitionSpec as P
    # M4 fix: real sharding instead of stub. Build in/out shardings.
    # Data sharding for batch (B,S) -> data axis
    data_sharding = NamedSharding(mesh, P("data", None) if "data" in mesh.axis_names else P())
    # We create a dummy params pytree to derive spec shapes; but we can lazily use jit with in_shardings as pytrees
    # Here we return a jit with explicit in_shardings that will be resolved at call time via device_put.
    # To allow caller to pass already-sharded arrays, we use jit with shardings.
    # Note: opt_state sharding mirrors params (replicated or model-sharded for Adam moments).
    # Use jit with static mesh: caller should device_put params with _make_param_shardings before first step.
    try:
        # Attempt to create a pjit with explicit shardings; if this fails we fall back to jit
        return jax.jit(
            step,
            in_shardings=(None, None, data_sharding, data_sharding, None),
            out_shardings=(None, None, None, None, None),
        )
    except Exception as e:
        print(f"  [pjit] warning: could not create sharded jit ({e}), falling back to jit")
        return jax.jit(step)


def make_gen_step(cfg):
    @jax.jit
    def step(params, ids):
        logits, _, _ = M.forward(cfg, params, ids, training=False)
        return jnp.argmax(logits[:, -1, :], axis=-1)
    return step


CKPT_VERSION = 2

def save_state(path, cfg, params, opt_state, step, key):
    import dataclasses
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # L4 fix: explicit asdict + version tag, not raw __dict__
    cfg_dict = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg.__dict__)
    with open(path, "wb") as f:
        pickle.dump({
            "version": CKPT_VERSION,
            "config": cfg_dict,
            "params": params,
            "opt_state": opt_state,
            "step": step,
            "key": key,
        }, f)
    print(f"  checkpoint v{CKPT_VERSION} -> {path}")


def load_state(path):
    with open(path, "rb") as f:
        st = pickle.load(f)
    # backward compat: v1 had raw __dict__ without version
    if "version" not in st:
        print(f"  [ckpt] legacy v1 checkpoint (no version), migrating")
        st["version"] = 1
    if st["version"] < CKPT_VERSION:
        print(f"  [ckpt] checkpoint v{st['version']} < current v{CKPT_VERSION}: fields may be missing, using defaults")
    # Re-hydrate config dataclass with defaults for new fields
    cfg_dict = st["config"]
    try:
        cfg = MoREConfig(**{k: v for k, v in cfg_dict.items() if k in MoREConfig.__dataclass_fields__})
        # fill any missing fields with defaults by re-instantiating
        for k, field in MoREConfig.__dataclass_fields__.items():
            if k not in cfg_dict:
                print(f"  [ckpt] missing field {k!r}, using default {field.default!r}")
        st["config_obj"] = cfg
    except Exception as e:
        print(f"  [ckpt] warning: could not rehydrate MoREConfig: {e}")
        st["config_obj"] = None
    return st


def _first(p):
    """pmap leaves with out_axes=(None, ...) are already single-device (replicated)."""
    return p


def schedule(step, lr, lr_min, warmup, total):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= total:
        return lr_min
    t = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * t))


def build_config(args) -> MoREConfig:
    # 12B scale presets bypass engram_kw and return factory directly (no local training needed for param inspection)
    if getattr(args, "config", None) in ("12b-3840", "12b-4096", "12b"):
        # hidden via factory + allow overrides from CLI
        hidden = 3840 if args.config in ("12b-3840", "12b") else 4096
        # optional override: --hidden 4096 or --intermediate 1024
        if getattr(args, "hidden_size", None):
            hidden = args.hidden_size
        inter = getattr(args, "intermediate_size", 1024) or 1024
        from .config import get_12b_config
        return get_12b_config(
            hidden_size=hidden,
            intermediate_size=inter,
            param_dtype=getattr(args, "param_dtype", "bfloat16"),
            compute_dtype=getattr(args, "compute_dtype", "bfloat16"),
            max_seq_len=args.seq_len,
        )
    engram_kw = dict(
        engram_enabled=args.engram,
        engram_vocab_size=args.engram_vocab_size,
        engram_max_ngram=3,
        engram_n_embed=args.engram_n_embed,
        engram_n_head=args.engram_n_head,
        engram_kernel_size=args.engram_kernel_size,
        engram_seed=args.seed,
        engram_pad_id=0,
    )
    # M5 fix: dtype policy - use CLI if provided, else auto bfloat16 on TPU
    param_dtype = getattr(args, "param_dtype", "float32")
    compute_dtype = getattr(args, "compute_dtype", "float32")
    # auto-detect TPU for sensible default if user left defaults
    if param_dtype == "float32" and compute_dtype == "float32":
        try:
            devs = jax.devices()
            if any("tpu" in d.platform.lower() or "TPU" in d.device_kind for d in devs):
                # Keep params float32 for determinism but compute in bf16 for throughput
                # User can override with --param_dtype bfloat16 --compute_dtype bfloat16 for full bf16
                compute_dtype = "bfloat16"
                print(f"  [dtype] TPU detected, auto compute_dtype=bfloat16 (override with --compute_dtype)")
        except Exception:
            pass
    if args.config == "tinystories":
        cfg = MoREConfig(
            vocab_size=50257, hidden_size=384, intermediate_size=1024,
            num_attention_heads=6, num_key_value_heads=2, head_dim=64,
            max_seq_len=args.seq_len, max_recursion_depth=4,
            num_experts=8, num_local_experts=8, num_shared_experts=1, top_k=1,
            router_hidden_size=64, kda_state_size=64, kda_chunk_size=128,
            layer_types=["kda", "kda", "msa", "kda"],
            msa_block_size=64, msa_topk=4, msa_index_dim=32, msa_kl_coef=0.01,
            load_balancing_loss_coef=args.aux_coef, recursion_aux_coef=args.rec_aux_coef,
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            **engram_kw,
        )
    else:
        cfg = MoREConfig(
            vocab_size=50257, hidden_size=1024, intermediate_size=1024,
            num_attention_heads=16, num_key_value_heads=8, head_dim=64,
            max_seq_len=args.seq_len, max_recursion_depth=4,
            num_experts=8, num_local_experts=8, num_shared_experts=1, top_k=2,
            router_hidden_size=128, kda_state_size=64, kda_chunk_size=128,
            layer_types=["kda", "kda", "msa", "kda"],
            msa_block_size=64, msa_topk=4, msa_index_dim=32, msa_kl_coef=0.01,
            load_balancing_loss_coef=args.aux_coef, recursion_aux_coef=args.rec_aux_coef,
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            **engram_kw,
        )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tinystories", choices=["tinystories", "default", "12b-3840", "12b-4096", "12b"])
    ap.add_argument("--hidden_size", type=int, default=None, help="override hidden for 12b (3840 or 4096)")
    ap.add_argument("--intermediate_size", type=int, default=None, help="override per-expert intermediate for 12b (default 1024)")
    ap.add_argument("--dry_run", action="store_true", help="for 12b: only build config + count params, no training (no local training)")
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
    ap.add_argument("--rec_aux_coef", type=float, default=0.03,
                    help="recursion depth-push aux: pushes router toward deeper loops (tuned 0.03; was 0.1)")
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
    ap.add_argument("--fineweb", action="store_true",
                    help="train on local FineWeb-Edu parquet (--fineweb_source)")
    ap.add_argument("--fineweb_source", type=str, default=None,
                    help="folder containing fineweb-edu-dedup-10b train-*.parquet")
    # --- SmollM corpus (HuggingFaceTB/smollm-corpus) ---
    ap.add_argument("--smollm", action="store_true",
                    help="train on HuggingFaceTB/smollm-corpus (cosmopedia-v2 + fineweb-edu-dedup via HF Hub)")
    ap.add_argument("--smollm_subsets", type=str, default="cosmopedia-v2,fineweb-edu-dedup",
                    help="comma-separated subsets of smollm-corpus (default: both)")
    ap.add_argument("--smollm_weights", type=str, default=None,
                    help="comma-separated sampling weights for subsets, e.g. '0.5,0.5' (default: equal, interleaved)")
    ap.add_argument("--smollm_max_files", type=int, default=None,
                    help="max files TOTAL across all smollm subsets (e.g. 10 for smoke)")
    ap.add_argument("--smollm_max_per_subset", type=int, default=None,
                    help="max files per subset (overrides total if set)")
    # --- Dtype (M5) ---
    ap.add_argument("--param_dtype", type=str, default="float32", choices=["float32","bfloat16","float16"],
                    help="param storage dtype (float32 default; bfloat16 on TPU for HBM/throughput)")
    ap.add_argument("--compute_dtype", type=str, default="float32", choices=["float32","bfloat16","float16"],
                    help="matmul compute dtype (rmsnorm always float32); auto bf16 on TPU if left default")
    # --- Mesh for pjit (M4) ---
    ap.add_argument("--mesh", type=str, default=None,
                    help="pjit mesh shape like '2' (data) or '2,4' (data,model); enables FSDP sharding for H>=1024")
    # --- Engram (DeepSeek 2601.07372) ---
    ap.add_argument("--engram", action="store_true",
                    help="enable Engram 2,3-gram conditional memory (residual after embedding)")
    ap.add_argument("--engram_vocab_size", type=int, default=8192,
                    help="Engram base vocab per n-gram before prime inflation (tiny 8192, paper 646k)")
    ap.add_argument("--engram_n_embed", type=int, default=64,
                    help="Engram embedding dim per n-gram (E=(max_n-1)*n_embed)")
    ap.add_argument("--engram_n_head", type=int, default=4,
                    help="Engram heads per n-gram (D_head=n_embed//n_head)")
    ap.add_argument("--engram_kernel_size", type=int, default=4,
                    help="Engram short-conv kernel size")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")

    devices = jax.devices()
    dist = len(devices) > 1
    print(f"JAX devices: {devices}  dist={dist}")

    # Optional mesh for pjit path (M4): --mesh 2,4 -> (data=2, model=4)
    mesh = None
    if getattr(args, "mesh", None):
        try:
            from jax.sharding import Mesh
            dims = [int(x) for x in args.mesh.split(",")]
            if len(dims) == 1:
                mesh = Mesh(jax.devices().reshape(dims[0]), ("data",))
            elif len(dims) == 2:
                mesh = Mesh(jax.devices().reshape(dims[0], dims[1]), ("data", "model"))
            else:
                raise ValueError("mesh must be like '2' or '2,4'")
            print(f"  [mesh] {mesh.axis_names} shape {mesh.shape}")
        except Exception as e:
            print(f"  [mesh] failed to create mesh {args.mesh}: {e}")
            mesh = None

    cfg = build_config(args)
    print(f"Config: hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
          f"kv={cfg.num_key_value_heads} experts={cfg.num_local_experts} "
          f"topk={cfg.top_k} layers={len(cfg.layer_types)} depth={cfg.max_recursion_depth} "
          f"blocks={cfg.num_recursion_blocks} dtype={cfg.param_dtype}/{cfg.compute_dtype} "
          f"engram={cfg.engram_enabled} engram_layers={getattr(cfg,'engram_layers',None)} ngrams={getattr(cfg,'engram_ngrams',None)} "
          f"msa_rope={getattr(cfg,'msa_use_rope',False)} rope_dim={getattr(cfg,'msa_rope_dim',0)}")
    if cfg.engram_enabled:
        print(f"  Engram: max_ngram={cfg.engram_max_ngram} ngrams={cfg.engram_ngrams} layers={cfg.engram_layers} (only layer 2)")
    if getattr(cfg,'msa_use_rope',False):
        print(f"  MSA RoPE: theta={cfg.msa_rope_theta} dim={cfg.msa_rope_dim} (KDA NoPE, MSA RoPE -> MSA compute + MLA retrieval)")
    # dry_run for 12B: count params analytically, no allocation / no training (respects 'no local training')
    if getattr(args, "dry_run", False):
        # analytic count without materializing 12B params on CPU (would OOM 40GB)
        try:
            from .config import get_12b_config as _g
            # use analytic helper
            H, NH, NG, Hd, Inter, E = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size, cfg.num_local_experts
            per_kda = NH*Hd*H + NG*Hd*H + NG*Hd*H + H*NH*Hd + NG*(Hd+1)*H + NG*(Hd+1) + H + NG*Hd
            per_msa = NH*Hd*H + NG*Hd*H + NG*Hd*H + H*NH*Hd + NG*cfg.msa_index_dim*H + cfg.msa_index_dim*H + H
            per_moe = E*H + E*Inter*H + E*Inter*H + E*H*Inter + Inter*H + Inter*H + H*Inter + H
            per_kda_l = per_kda + per_moe + 2*H
            per_msa_l = per_msa + per_moe + 2*H
            n_kda = cfg.layer_types.count("kda"); n_msa = cfg.layer_types.count("msa")
            total_block = n_kda*per_kda_l + n_msa*per_msa_l
            embed = cfg.vocab_size*H + H
            first = per_kda + per_moe + 2*H
            last = per_moe + H
            router = H*cfg.router_hidden_size + cfg.max_recursion_depth*cfg.router_hidden_size + H
            # engram analytic via static
            from .model import _engram_static
            static = _engram_static(cfg)
            total_N = sum(static["primes_list"]); D_head = cfg.engram_n_embed // cfg.engram_n_head
            E_emb = len(cfg.engram_ngrams)*cfg.engram_n_embed if cfg.engram_ngrams else 0
            engram_p = total_N*D_head + H*E_emb*2 + H*cfg.engram_kernel_size + 3*H if cfg.engram_enabled else 0
            total = total_block + embed + first + last + router + engram_p
            per_moe_a = E*H + cfg.top_k*Inter*H + cfg.top_k*Inter*H + cfg.top_k*H*Inter + Inter*H + Inter*H + H*Inter + H
            per_kda_a = per_kda + per_moe_a + 2*H
            per_msa_a = per_msa + per_moe_a + 2*H
            active_block = n_kda*per_kda_a + n_msa*per_msa_a
            active = active_block + embed + (per_kda + per_moe_a + 2*H) + (per_moe_a + H) + router + engram_p
            print(f"  [dry_run] 12B analytic: total={total/1e9:.2f}B  active={active/1e9:.2f}B  engram={engram_p/1e6:.2f}M  (no params materialized, no local training)")
            print(f"  Layers: 48 (4*12) = 36 KDA +12 MSA, each attn+MoE (16 experts top2), MoR depth 4 in middle (22+4x4+22)")
            print(f"  Hidden {cfg.hidden_size} intermediate {cfg.intermediate_size} experts {cfg.num_experts}x{cfg.top_k} -> active ratio {active/total:.1%}")
        except Exception as e:
            print(f"  [dry_run] analytic failed: {e}")
        print("  dry_run complete — no training launched (as requested 'no local training').")
        return

    rng = jax.random.PRNGKey(args.seed)

    if args.resume and os.path.exists(args.resume):
        st = load_state(args.resume)
        params = st["params"]
        opt_state = st["opt_state"]
        start_step = st["step"] + 1
        rng = st["key"]
        # L4: if checkpoint has config_obj, optionally verify compatibility
        if st.get("config_obj") is not None:
            ckpt_cfg = st["config_obj"]
            cur_cfg = cfg
            if ckpt_cfg.hidden_size != cur_cfg.hidden_size or ckpt_cfg.layer_types != cur_cfg.layer_types:
                print(f"  [ckpt] WARNING: checkpoint hidden {ckpt_cfg.hidden_size} layers {ckpt_cfg.layer_types} != current {cur_cfg.hidden_size} {cur_cfg.layer_types}")
        print(f"Resumed from {args.resume} at step {start_step} (v{st.get('version',1)})")
    else:
        params = M.init_model(cfg, rng)
        rng, _ = jax.random.split(rng)
        start_step = 0
        opt_state = None
        print(f"Initialized model: {M.count_params(params):,} params")

    base_opt = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.inject_hyperparams(optax.adamw)(learning_rate=0.0, b1=0.9, b2=0.95, eps=1e-8, weight_decay=args.weight_decay),
    )
    opt = optax.MultiSteps(base_opt, every_k_schedule=args.accum)

    if opt_state is None:
        opt_state = opt.init(params)

    # Choose pjit vs pmap path
    if mesh is not None:
        step_fn = make_train_step_pjit(cfg, opt, args.ce_chunk, mesh=mesh, remat=not args.no_remat)
        # Optionally shard params for pjit (best-effort)
        if len(mesh.devices) > 1:
            try:
                from jax.sharding import PartitionSpec as P, NamedSharding
                # Shard params eagerly for pjit
                param_shardings = _make_param_shardings(params, mesh)
                params = jax.device_put(params, param_shardings)
                opt_state = jax.device_put(opt_state, jax.tree_util.tree_map(lambda _: NamedSharding(mesh, P()), opt_state) if hasattr(opt_state, "_fields") else opt_state)
                print(f"  [pjit] params sharded on mesh {mesh.axis_names}")
            except Exception as e:
                print(f"  [pjit] shard put failed ({e}), using replicated")
    else:
        step_fn = make_train_step(cfg, opt, args.ce_chunk, dist, remat=not args.no_remat)
    gen_fn = make_gen_step(cfg)

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    if args.smollm:
        subsets = [s.strip() for s in args.smollm_subsets.split(",") if s.strip()]
        weights = None
        if args.smollm_weights:
            weights = [float(w) for w in args.smollm_weights.split(",")]
            assert len(weights) == len(subsets), "--smollm_weights must match --smollm_subsets length"
            s = sum(weights)
            weights = [w/s for w in weights]
        print(f"Dataset: HuggingFaceTB/smollm-corpus subsets={subsets} weights={weights}")
        if weights is not None and len(subsets) == 2:
            # weighted mixture: download each subset's shards separately so weights matter
            shards_a = D.ensure_smollm_shards(tokenizer, args.data_dir,
                                              subsets=[subsets[0]],
                                              max_files_per_subset=args.smollm_max_per_subset,
                                              max_files_total=args.smollm_max_files)
            shards_b = D.ensure_smollm_shards(tokenizer, args.data_dir,
                                              subsets=[subsets[1]],
                                              max_files_per_subset=args.smollm_max_per_subset,
                                              max_files_total=args.smollm_max_files)
            n_shards = len(shards_a) + len(shards_b)
            steps_per_shard = max(int(np.ceil(args.total_steps / max(n_shards, 1))), 1)
            data_iter = D.mixture_stream_iter(shards_a, shards_b,
                                              args.batch_size, args.seq_len,
                                              steps_per_shard,
                                              weight_a=weights[0])
            print(f"  mixture_stream: {len(shards_a)} + {len(shards_b)} shards, "
                  f"weight_a={weights[0]:.2f} steps_per_shard={steps_per_shard}")
        else:
            shards = D.ensure_smollm_shards(tokenizer, args.data_dir,
                                            subsets=subsets,
                                            max_files_per_subset=args.smollm_max_per_subset,
                                            max_files_total=args.smollm_max_files)
            n_shards = len(shards)
            steps_per_shard = max(int(np.ceil(args.total_steps / n_shards)), 1)
            data_iter = D.stream_iter(shards, args.batch_size, args.seq_len,
                                      steps_per_shard)
            print(f"Dataset: smollm-corpus streamed across {n_shards} shard(s), "
                  f"{steps_per_shard} steps/shard (uniform interleave)")
    elif args.fineweb:
        if not args.fineweb_source:
            raise SystemExit("--fineweb requires --fineweb_source=<folder of train-*.parquet>")
        shards = D.ensure_fineweb_shards(tokenizer, args.data_dir,
                                         args.fineweb_source,
                                         max_shards=args.max_files)
        n_shards = len(shards)
        steps_per_shard = max(int(np.ceil(args.total_steps / n_shards)), 1)
        data_iter = D.stream_iter(shards, args.batch_size, args.seq_len,
                                  steps_per_shard)
        print(f"Dataset: FineWeb-Edu streamed across {n_shards} shard(s), "
              f"{steps_per_shard} steps/shard")
    elif args.synthetic:
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
    if dist and args.batch_size % len(devices) != 0:
        raise SystemExit(
            f"--batch_size {args.batch_size} must be divisible by "
            f"{len(devices)} device(s); got per-device batch {per}")
    step_tokens = per * args.seq_len * len(devices)

    step = start_step
    losses = []
    last_good = (params, opt_state)
    t0 = time.time()

    def log(step_, loss, aux, lr_, tok_per_s, breakdown=None):
        line = f"step={step_} loss={float(loss):.4f} aux={float(aux):.5f} lr={lr_:.2e} tok/s={tok_per_s:.0f}"
        if breakdown is not None:
            # decomposed aux (H2): first / router_lb / router_push / block / last
            bd = {k: float(v) for k, v in breakdown.items()}
            line += f" aux_bd=[first={bd['first']:.4f} lb={bd['router_lb']:.4f} push={bd['router_push']:.4f} block={bd['block']:.4f} last={bd['last']:.4f}]"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    def sample(step_):
        ids = tokenizer.encode(args.gen_prompt, add_special_tokens=False)
        ids = np.asarray(ids[:16], dtype=np.int32)[None, :]
        gen_params = params
        out = list(ids[0])
        for _ in range(args.gen_len):
            nxt = np.asarray(gen_fn(gen_params, ids))
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
            breakdown = None
            # H2: decomposed aux logging (only single-device to avoid pmap overhead)
            if not dist:
                try:
                    _, _, _, breakdown = M.forward(cfg, params, x, training=True, return_aux_breakdown=True)
                except Exception:
                    breakdown = None
            log(step, loss_v, float(np.asarray(aux).mean()),
                lr, tok_per_s, breakdown)
            t0 = time.time()

        if step % args.ckpt_every == 0:
            save_state(os.path.join(args.out_dir, f"ckpt_{step}.pkl"), cfg,
                       params,
                       opt_state, step, rng)

        if step % args.gen_every == 0:
            sample(step)

        if step >= args.total_steps:
            break

    save_state(os.path.join(args.out_dir, "final.pkl"), cfg,
               params,
               opt_state, step, rng)
    print(f"Done. {step} steps. final avg loss (last 50): {np.mean(losses[-50:]):.4f}")


if __name__ == "__main__":
    main()
