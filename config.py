from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MoREConfig:
    """MoRE (Mixture-of-Recursions) config, pure-JAX port.

    Tinystories-scale defaults matching mokre/train.py get_tinystories_config().
    Architecture: 4-layer shared recursion block, 3:1 KDA:MSA, MoE = FFN,
    token-choice router, recursion depth up to max_recursion_depth.

    MSA (MiniMax Sparse Attention) replaces MLA:
        - Index Branch: per-GQA-group block selection via max-pool over
          Q_idx * K_idx^T scores (block size = msa_block_size, top-k = msa_topk,
          always includes local block).
        - Main Branch: GQA sparse attention over only selected blocks.
        - Training: KL alignment loss between index distribution and
          group-averaged main distribution (stop-gradient on teacher & input),
          weighted by msa_kl_coef. Two-stage warmup is optional via
          msa_warmup_steps.
    """

    vocab_size: int = 50257        # GPT-2
    max_seq_len: int = 1024

    hidden_size: int = 384
    intermediate_size: int = 1024  # per-expert SwiGLU hidden
    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    head_dim: int = 64             # hidden = heads * head_dim

    num_recursion_blocks: int = 1
    max_recursion_depth: int = 4
    layer_types: list = field(default_factory=lambda: ["kda", "kda", "msa", "kda"])

    router_hidden_size: int = 64
    load_balancing_loss_coef: float = 0.02  # doubled from 0.01 to fix MoE load imbalance (mi*P ~1/E, needs ~0.02 to be non-negligible vs CE ~3)
    recursion_aux_coef: float = 0.07  # restored from 0.03 (was 0.1) to fix MoR collapse to depth 1; 0.07 balances depth push vs CE
    router_z_loss_coef: float = 0.001  # z-loss on router logits to prevent logit blowup and collapse
    capacity_factor: float = 1.5  # increased from 1.25 to reduce expert overflow drops (G = ceil(Nk/E*1.5))

    num_experts: int = 8           # routed experts
    num_shared_experts: int = 1
    num_local_experts: int = 8     # effective routed experts (mirrors torch default)
    top_k: int = 1
    expert_capacity: int = 64

    kda_state_size: Optional[int] = None
    kda_chunk_size: int = 128      # chunked associative_scan with carry (128-256 for S=2048/8192)
    kda_use_nope: bool = True

    # --- MSA (MiniMax Sparse Attention) ---
    # Replaces MLA.  GQA-based block-sparse attention:
    #   * Index branch: Q_idx (per-GQA-group, dim=msa_index_dim) + K_idx (shared)
    #     scores pooled to block level (max) and top-k selected per group.
    #   * Main branch: exact sparse attention over selected blocks only.
    #   * KL alignment loss (see model.msa_forward) weighted by msa_kl_coef.
    msa_block_size: int = 128      # Bk, tokens per block (attention budget 2048 => topk=16)
    msa_topk: int = 16             # k, blocks selected per query per group (budget = k*Bk = 2048 tokens)
    msa_index_dim: int = 32        # d_idx, index head dim (paper: 64-128)
    msa_kl_coef: float = 0.01      # lambda for KL(teacher || index) aux loss
    msa_warmup_steps: int = 0      # 0 = sparse from step 0; >0 = full attn warmup
    # --- MSA RoPE (Qwen/MLA retrieval + MSA sparse compute) ---
    # True = apply RoPE to main branch Q/K (head_dim), keeping block-sparse top-k.
    # Index branch can stay NoPE (pure learned block scoring) to keep compute cheap.
    msa_use_rope: bool = True      # enable RoPE on MSA main Q/K for retrieval (MLA-like)
    msa_rope_theta: float = 10000.0
    msa_rope_dim: int = 64         # rotary dim <= head_dim (64 as in template), rest NoPE

    # Legacy MLA fields kept for checkpoint compat (unused when layer_types uses msa)
    mla_qk_latent_dim: int = 64
    mla_v_latent_dim: int = 64

    mlp_gate_type: str = "swiglu"
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    initializer_range: float = 0.02

    # --- DType policy (TPU throughput) ---
    # param_dtype: storage dtype for params (float32 default for determinism; use "bfloat16" on TPU)
    # compute_dtype: matmul dtype; rmsnorm always stays in float32 for stability
    param_dtype: str = "float32"
    compute_dtype: str = "float32"

    # --- Inference gather (H4/SCALE3) ---
    # If True, uses experimental ordered gather for active tokens at inference
    # to save MoE FLOPs (KDA identity means gather matches dense). Default False
    # keeps dense masked residual (honest, no per-token saving except full-empty).
    enable_gather: bool = False

    # --- Engram: Conditional Memory via Scalable N-gram Lookup ---
    # DeepSeek Engram (arXiv:2601.07372): O(1) n-gram embedding with
    # hash mapping + multi-head embedding + signed-sqrt gating + short conv.
    # Supports 2,3,5-gram (max_ngram up to 5). Can be injected globally after
    # embedding (early) or per-layer via engram_layers (e.g. only layer 2).
    engram_enabled: bool = False
    # Base vocab size per n-gram before prime inflation (small for TinyStories).
    # Original DeepSeek: 129280*5 per n-gram; tiny default 8192 keeps <1M params.
    engram_vocab_size: int = 8192
    engram_max_ngram: int = 3          # 3 -> 2,3-gram; 5 -> 2,3,5-gram (skips 4)
    engram_n_embed: int = 64           # per n-gram embedding dim (flattened -> ngram-specific)
    engram_n_head: int = 4             # heads per n-gram (D_head = n_embed // n_head)
    engram_kernel_size: int = 4
    engram_seed: int = 0
    engram_pad_id: int = 0
    # Layer-specific Engram: list of layer indices (0-based) where engram delta is
    # added residually. Empty or None -> global after embedding (legacy). For
    # 12B scale: engram_layers=[1] means only layer 2 (0-indexed 1).
    engram_layers: Optional[list] = None
    # Which n-grams to use when max_ngram >=5. None -> all 2..max_ngram.
    # For spec "2,3,5 only": set to [2,3,5].
    engram_ngrams: Optional[list] = None

    def __post_init__(self):
        if self.kda_state_size is None:
            self.kda_state_size = self.head_dim
        assert self.param_dtype in ("float32", "bfloat16", "float16")
        assert self.compute_dtype in ("float32", "bfloat16", "float16")
        # --- Layer pattern: support 4-layer recursion, 48-layer (4*12) and 192-layer (4*48) ---
        # Legacy: 4 layers ["kda","kda","msa","kda"]; Scale: 48 layers (4*12) or 192 layers (4*48) flat unrolled
        valid_lens = {4, 48, 93, 192}
        assert len(self.layer_types) in valid_lens, f"layer_types must have 4, 48, 93 or 192 layers, got {len(self.layer_types)}"
        # Support both 'msa' (new) and 'mla' (legacy checkpoint) as sparse layers
        n_kda = self.layer_types.count("kda")
        n_sparse = self.layer_types.count("msa") + self.layer_types.count("mla")
        # For 4 layers: allow 0-3 KDA + 1-4 MSA (3:1 default, 4 MSA for recall-precise)
        # For 48 layers: allow flexible 12-36 KDA + 12-36 MSA (3:1 default 36+12, recall-precise 12+36 or 0+48)
        if len(self.layer_types) == 4:
            assert n_kda + n_sparse == 4 and n_sparse >= 1, f"4-layer must have 4 total with >=1 sparse, got {n_kda} KDA + {n_sparse} sparse"
        elif len(self.layer_types) == 48:
            assert n_kda + n_sparse == 48 and n_sparse >= 12, f"48-layer must have 48 total with >=12 sparse, got {n_kda} KDA + {n_sparse} sparse"
        elif len(self.layer_types) == 93:
            assert n_kda + n_sparse == 93 and n_sparse >= 23, f"93-layer must have 93 total with >=23 sparse, got {n_kda} KDA + {n_sparse} sparse"
        else:  # 192
            assert n_kda + n_sparse == 192 and n_sparse >= 48, f"192-layer must have 192 total with >=48 sparse, got {n_kda} KDA + {n_sparse} sparse"
        # Normalize legacy 'mla' -> 'msa' for forward compat
        self.layer_types = ["msa" if t == "mla" else t for t in self.layer_types]
        assert self.hidden_size == self.num_attention_heads * self.head_dim
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.num_local_experts = self.num_local_experts or self.num_experts
        # MSA sanity
        assert self.msa_block_size >= 8 and self.msa_block_size <= 512
        assert self.msa_topk >= 1
        assert self.msa_index_dim >= 8
        # Engram sanity
        if self.engram_enabled:
            assert self.engram_max_ngram in (2, 3, 4, 5), "Engram supports 2..5"
            # normalize engram_ngrams
            if self.engram_ngrams is None:
                self.engram_ngrams = list(range(2, self.engram_max_ngram + 1))
            else:
                for n in self.engram_ngrams:
                    assert 2 <= n <= self.engram_max_ngram, f"engram n-gram {n} out of range 2..{self.engram_max_ngram}"
            # For 2,3,5 skip-4 pattern keep as is; ensure sorted unique
            self.engram_ngrams = sorted(set(self.engram_ngrams))
            assert self.engram_n_embed % self.engram_n_head == 0, "n_embed must be divisible by n_head"
            assert self.engram_n_embed >= 8
            assert self.engram_kernel_size >= 2
            assert 0 <= self.engram_pad_id < self.vocab_size
            if self.engram_layers is not None:
                assert all(0 <= i < len(self.layer_types) for i in self.engram_layers), \
                    f"engram_layers {self.engram_layers} out of range 0..{len(self.layer_types)-1}"

    @property
    def num_layers(self) -> int:
        # For flat 48/93/192-layer model, num_layers is len(layer_types).
        # For recursion model, num_layers = num_recursion_blocks * max_recursion_depth
        # where recursion depth is steps over the shared block.
        if len(self.layer_types) in (48, 93, 192):
            return len(self.layer_types)
        return self.num_recursion_blocks * self.max_recursion_depth

    @property
    def total_forward_layers(self) -> int:
        """Total transformer layer forward passes (attention+MoE) per token."""
        if len(self.layer_types) in (48, 93, 192):
            # flat unique layers, no recursion sharing
            return len(self.layer_types)
        # recursion: first + block*depth + last is accounted separately, but num_layers is depth*blocks
        return self.num_recursion_blocks * self.max_recursion_depth * len(self.layer_types) // 4 + 1  # +1 for first/last nuance


# ----------------------------------------------------------------
# 12B scale presets  (hidden 3840 or 4096, 48 layers = 4*12, 16 experts top2, 2/3/5-gram on layer 2)
# ----------------------------------------------------------------
def _layer_pattern_48():
    """Return 48-layer pattern: 4-layer block repeated 12 times (3 KDA +1 MSA per block)."""
    base = ["kda", "kda", "msa", "kda"]
    return base * 12  # 36 KDA +12 MSA


def _layer_pattern_48_middle6():
    """12 blocks = 3 prefix + 6 middle (recursive) + 3 suffix. Block0 is 2:2 msa,msa,kda,kda per request."""
    blocks = []
    for i in range(12):
        if i == 0:
            blocks.extend(["msa", "msa", "kda", "kda"])  # 2:2 first block
        else:
            blocks.extend(["kda", "kda", "msa", "kda"])  # 3:1 rest
    # 12*4=48: 2*MSA+2*KDA in block0 + 11* (3KDA+1MSA) = 35 KDA +13 MSA (was 36+12)
    return blocks


def _layer_pattern_192():
    """48 blocks = 12 prefix + 24 middle + 12 suffix = 192 layers. Block0 2:2, rest 3:1."""
    blocks = []
    for i in range(48):
        if i == 0:
            blocks.extend(["msa", "msa", "kda", "kda"])
        else:
            blocks.extend(["kda", "kda", "msa", "kda"])
    # 48*4=192: 140 KDA +52 MSA (block0 2:2 + 47*3:1)
    return blocks


def _layer_pattern_93():
    """93 layers baked: block0 unique 2:2, blocks1-6 =2 distinct x3 forced R, blocks7-17 =11 MoR, blocks18-23 =3 distinct x2 forced R =>96 fw truncated to 93."""
    # distinct patterns (will be expanded as forwards for now, weight tying later)
    b0 = ["msa", "msa", "kda", "kda"]  # block0 1x
    bA = ["kda", "kda", "msa", "kda"]  # for 1-6: 2 distinct A,B
    bB = ["kda", "kda", "msa", "kda"]
    # expand forced R: 2 distinct x3 =6 blocks (24 fw)
    prefix_mid = []
    for _ in range(3):
        prefix_mid.extend(bA)
    for _ in range(3):
        prefix_mid.extend(bB)
    # middle 11 blocks MoR distinct 44 fw
    middle = []
    for _ in range(11):
        middle.extend(["kda", "kda", "msa", "kda"])
    # suffix 3 distinct x2 =6 blocks (24 fw) C,D,E
    bC = ["kda", "kda", "msa", "kda"]
    bD = ["kda", "kda", "msa", "kda"]
    bE = ["kda", "kda", "msa", "kda"]
    suffix = []
    for _ in range(2):
        suffix.extend(bC)
    for _ in range(2):
        suffix.extend(bD)
    for _ in range(2):
        suffix.extend(bE)
    fw = b0 + prefix_mid + middle + suffix  # 4+24+44+24=96
    # truncate to 93 as requested (drop last 3 of suffix)
    fw = fw[:93]
    # 69 KDA+24 MSA =93, 93*8=744 vs 192*8=1536
    return fw


def _split_93():
    # baked 93 fw: 0: block0 4, 1-6: 2x3=24 (4-27), 7-17: 11=44 (28-71), 18-23: 6=24 truncated to 21 (72-92)
    return {"prefix_block0": list(range(0, 4)), "prefix_R_1_6": list(range(4, 28)), "middle_MoR": list(range(28, 72)), "suffix_R_18_23": list(range(72, 93))}


def get_12b_config(
    hidden_size: int = 3840,
    intermediate_size: int = 1024,
    num_experts: int = 16,
    top_k: int = 2,
    vocab_size: int = 262144,          # SOTA Google-like (Gemma 256K, power-of-two 262144)
    engram_vocab_size: int = 1_000_000,  # ~3B Engram: see doc below
    engram_n_embed: int = 1024,        # 1024*3=3072 E_emb -> 3B with 1M vocab
    engram_n_head: int = 8,            # D_head=128
    param_dtype: str = "bfloat16",
    compute_dtype: str = "bfloat16",
    max_seq_len: int = 4096,
) -> "MoREConfig":
    """Factory for the 12B/3B-active MoE scale requested.

    Spec: hidden 3840 or 4096, 48 layers (4*12), 16 total experts top-2,
    2,3,5-gram Engram only on layer 2 (index 1). Active ~3B via 2/16 routed
    +1 shared SwiGLU per layer. BFloat16 on TPU for HBM/throughput.

    SOTA vocab: 262144 (Gemma/Gemini SentencePiece 256K) vs 50257 GPT-2.
    Embed 262K*3840=1.01B (vs 0.19B) — Google-like 2024-25 frontier.
    Engram 3B: vocab 1M * 24 heads (3 ngrams*8 heads) * D_head 128 = 3.07B
    + proj 2*H*3072 ~23M. Total non-Engram ~12.1B -> 15.2B total. Keep
    param_dtype bfloat16 + FSDP sharded.

    For exact 12B+3B split, use engram_vocab_size=500K, n_embed=2048 -> also 3.12B.
    """
    assert hidden_size in (3840, 4096), f"hidden_size must be 3840 or 4096, got {hidden_size}"
    if hidden_size == 3840:
        num_heads, num_kv, head_dim = 30, 6, 128  # 30*128=3840, GQA 5:1
    else:
        num_heads, num_kv, head_dim = 32, 8, 128  # 32*128=4096, GQA 4:1
    return MoREConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv,
        head_dim=head_dim,
        # 48 layers (4*12): 3 prefix + 6 middle recursive + 3 suffix = 12 blocks.
        # Block0 is 2:2 msa,msa,kda,kda per request; rest 3:1. Middle 6 blocks (layers 12-35) are MoR depth-4.
        num_recursion_blocks=6,
        max_recursion_depth=4,
        layer_types=_layer_pattern_48_middle6(),
        router_hidden_size=256,
        load_balancing_loss_coef=0.02,
        recursion_aux_coef=0.07,  # push router toward deeper recursion (middle block)
        router_z_loss_coef=0.001,
        capacity_factor=1.5,
        num_experts=num_experts,
        num_shared_experts=1,
        num_local_experts=num_experts,
        top_k=top_k,
        expert_capacity=64,
        kda_state_size=head_dim,
        kda_chunk_size=128,
        msa_block_size=128,
        msa_topk=16,
        msa_index_dim=128,
        msa_kl_coef=0.01,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        # Engram: 2,3,5-gram only on layer 2 (index 1, prefix)
        engram_enabled=True,
        engram_vocab_size=engram_vocab_size,
        engram_max_ngram=5,
        engram_n_embed=engram_n_embed,
        engram_n_head=engram_n_head,
        engram_kernel_size=4,
        engram_seed=0,
        engram_pad_id=0,
        engram_layers=[1],
        engram_ngrams=[2, 3, 5],
    )


def get_12b_3840_config(**kw) -> "MoREConfig":
    return get_12b_config(hidden_size=3840, **kw)


def get_12b_4096_config(**kw) -> "MoREConfig":
    return get_12b_config(hidden_size=4096, **kw)


def get_93l_config(hidden_size: int = 3840, num_experts: int = 8, **kw) -> "MoREConfig":
    """93 layers 69KDA+24MSA, 8 experts top2 => 744 experts (vs 192*8=1536). 6 prefix(24)+11 middle(44)+6 suffix(24)+1."""
    cfg = get_12b_config(hidden_size=hidden_size, num_experts=num_experts, **kw)
    cfg.layer_types = _layer_pattern_93()
    cfg.num_recursion_blocks = 11  # middle 11 blocks (44 layers) recursive depth4
    cfg.__post_init__()
    return cfg