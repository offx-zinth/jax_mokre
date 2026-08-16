from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MoREConfig:
    """MoRE (Mixture-of-Recursions) config, pure-JAX port.

    Tinystories-scale defaults matching mokre/train.py get_tinystories_config().
    Architecture: 4-layer shared recursion block, 3:1 KDA:MLA, MoE = FFN,
    token-choice router, recursion depth up to max_recursion_depth.
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
    layer_types: list = field(default_factory=lambda: ["kda", "kda", "mla", "kda"])

    router_hidden_size: int = 64
    load_balancing_loss_coef: float = 0.01
    recursion_aux_coef: float = 0.1   # pushes router toward deeper recursion
    capacity_factor: float = 1.25

    num_experts: int = 8           # routed experts
    num_shared_experts: int = 1
    num_local_experts: int = 8     # effective routed experts (mirrors torch default)
    top_k: int = 1
    expert_capacity: int = 64

    kda_state_size: Optional[int] = None
    kda_chunk_size: int = 16       # kept for config parity; unused (assoc. scan)
    kda_use_nope: bool = True

    mla_qk_latent_dim: int = 64
    mla_v_latent_dim: int = 64

    mlp_gate_type: str = "swiglu"
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    initializer_range: float = 0.02

    def __post_init__(self):
        if self.kda_state_size is None:
            self.kda_state_size = self.head_dim
        assert len(self.layer_types) == 4, "recursion block must have 4 layers"
        assert self.layer_types.count("kda") == 3
        assert self.layer_types.count("mla") == 1
        assert self.hidden_size == self.num_attention_heads * self.head_dim
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.num_local_experts = self.num_local_experts or self.num_experts

    @property
    def num_layers(self) -> int:
        return self.num_recursion_blocks * self.max_recursion_depth