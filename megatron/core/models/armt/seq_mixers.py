"""Seq-level mixers for ARMT recurrent-memory write path.

These modules sit between ``ARMTLayer._resolve_write_source`` and
``recurrent_memory_layer.update_mem``. Their sole purpose is to let tokens
inside a single chunk exchange information *before* the recurrent memory
backend integrates the write tensor into its state.

Shape contract: input and output are both ``[S, B, H]`` (SBH layout) and have
identical shape; no compression is performed.

Initialization strategy is selected by ``init_strategy``:

- ``identity``: at step 0 the mixer is (approximately) a no-op so that the
  initial forward pass matches the baseline without the mixer. The strictness
  of this guarantee depends on the variant — see each class for details.
- ``megatron``: use the project's standard ``config.init_method`` /
  ``config.output_layer_init_method`` so the mixer behaves like any other
  fresh Megatron submodule.
"""

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from .init_utils import init_linear_weight_and_bias
from .norm_utils import build_recurrent_norm


_SUPPORTED_SEQ_MIXER_TYPES = ("none", "mlp1", "mlp2", "attn")
_SUPPORTED_SEQ_MIXER_INITS = ("identity", "megatron")
_SUPPORTED_SEQ_MIXER_ATTN_BACKENDS = ("flash", "sdpa")


class SeqMixerMLP1(MegatronModule):
    """Single Linear(S -> S) along the sequence axis, weights shared across H.

    Treats the chunk as a length-``seq_len`` vector per (batch, hidden) cell and
    applies one shared S->S linear projection. Bias is omitted to keep the
    "pure mix" interpretation: the output is always a linear combination of
    the input positions.

    Identity init: ``W = I`` so the layer's initial output is exactly the
    input.
    """

    def __init__(
        self,
        seq_len: int,
        *,
        config: TransformerConfig,
        init_strategy: str,
        dtype: torch.dtype,
    ):
        super().__init__(config=config)
        # nn.Linear stores weight as [out, in]; here both are seq_len.
        self.linear = nn.Linear(seq_len, seq_len, bias=False, dtype=dtype)
        self._reset_parameters(init_strategy=init_strategy)

    def _reset_parameters(self, *, init_strategy: str) -> None:
        if not self.config.perform_initialization:
            return
        if init_strategy == "identity":
            with torch.no_grad():
                nn.init.eye_(self.linear.weight)
        else:  # "megatron"
            init_linear_weight_and_bias(
                self.linear,
                self.config.init_method,
                perform_initialization=True,
            )

    def forward(self, x: torch.Tensor, *, input_is_sbh: bool) -> torch.Tensor:
        # Project convention: ARMT mixers always receive SBH layout.
        assert input_is_sbh, "SeqMixerMLP1 only supports SBH layout"
        # [S, B, H] -> [B, H, S] so Linear acts along the trailing S axis.
        h = x.permute(1, 2, 0)
        h = self.linear(h)
        return h.permute(2, 0, 1).contiguous()


class SeqMixerMLP2(MegatronModule):
    """Two-layer Linear(S -> S*e -> S) with SiLU in between, no residual.

    Identity init for this variant is only **approximate** — without a
    residual connection or linear shortcut a non-linear MLP cannot reduce to
    the identity. We pick ``fc2.weight ~ 0`` so the initial output magnitude
    is small (mixer ≈ 0). This means the very first ``update_mem`` call
    receives a near-zero tensor; gradient still flows so the mixer learns
    quickly from step 1. Use ``mlp1`` or ``attn`` (residual) if a strict
    identity start is required.
    """

    def __init__(
        self,
        seq_len: int,
        expansion: int,
        *,
        config: TransformerConfig,
        init_strategy: str,
        dtype: torch.dtype,
    ):
        super().__init__(config=config)
        mid = seq_len * expansion
        self.fc1 = nn.Linear(seq_len, mid, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(mid, seq_len, bias=False, dtype=dtype)
        self._reset_parameters(init_strategy=init_strategy)

    def _reset_parameters(self, *, init_strategy: str) -> None:
        if not self.config.perform_initialization:
            return
        if init_strategy == "identity":
            # fc1: standard Xavier so SiLU(fc1(x)) is well-scaled.
            # fc2: near-zero so the overall mixer output magnitude ~ 0.
            with torch.no_grad():
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-2)
        else:  # "megatron"
            init_linear_weight_and_bias(
                self.fc1,
                self.config.init_method,
                perform_initialization=True,
            )
            init_linear_weight_and_bias(
                self.fc2,
                self.config.output_layer_init_method,
                perform_initialization=True,
            )

    def forward(self, x: torch.Tensor, *, input_is_sbh: bool) -> torch.Tensor:
        assert input_is_sbh, "SeqMixerMLP2 only supports SBH layout"
        # [S, B, H] -> [B, H, S] -> fc1 -> SiLU -> fc2 -> [S, B, H]
        h = x.permute(1, 2, 0)
        h = self.fc2(F.silu(self.fc1(h)))
        return h.permute(2, 0, 1).contiguous()


class SeqMixerAttn(MegatronModule):
    """Bidirectional self-attention along the chunk axis.

    Mirrors the lightweight SDPA pattern used by
    ``CrossAttentionSlotMemory._read_attention``: ``F.scaled_dot_product_attention``
    with ``is_causal=False`` and no mask, optionally wrapped in the flash
    backend kernel.

    Q/K/V are produced from a single fused ``linear_qkv`` (output dim
    ``3 * num_heads * head_dim``), matching Megatron's ``SelfAttention`` so we
    get a single GEMM instead of three smaller ones. Q/K/V are then sliced
    along the trailing dim.

    Identity init:
      - ``residual=True``: ``linear_proj.weight = 0`` makes the mixer output
        the zero tensor, and ``x + 0 = x``, i.e. a strict identity.
      - ``residual=False``: cannot be a strict identity. We initialize
        ``linear_proj.weight`` with a very small std so the initial mixer
        output is near zero and the mixer learns from a controlled start.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int],
        residual: bool,
        prenorm: bool,
        backend: str,
        *,
        config: TransformerConfig,
        init_strategy: str,
        dtype: torch.dtype,
    ):
        super().__init__(config=config)
        if backend not in _SUPPORTED_SEQ_MIXER_ATTN_BACKENDS:
            raise ValueError(
                f"attn backend must be one of {_SUPPORTED_SEQ_MIXER_ATTN_BACKENDS}, "
                f"got {backend!r}"
            )
        if num_heads <= 0:
            raise ValueError("attn_num_heads must be > 0")
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by attn_num_heads when "
                    "head_dim is omitted"
                )
            head_dim = hidden_size // num_heads
        if head_dim <= 0:
            raise ValueError("attn_head_dim must be > 0")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.residual = bool(residual)
        self.backend = backend
        proj = num_heads * head_dim

        # Optional pre-norm before QKV projection; mirrors the modern
        # "pre-LN" attention pattern.
        self.prenorm = None
        if prenorm:
            self.prenorm = build_recurrent_norm(
                hidden_size,
                normalization=getattr(config, "normalization", "LayerNorm"),
                eps=getattr(config, "layernorm_epsilon", 1e-5),
                dtype=dtype,
            )

        # Fused Q/K/V projection: one GEMM produces all three streams. They
        # are split along the trailing dim in forward(). Output layout is
        # [..., Q | K | V] with each chunk of size ``proj``.
        self.linear_qkv = nn.Linear(hidden_size, 3 * proj, bias=False, dtype=dtype)
        self.linear_proj = nn.Linear(proj, hidden_size, bias=False, dtype=dtype)
        self._reset_parameters(init_strategy=init_strategy)

    def _reset_parameters(self, *, init_strategy: str) -> None:
        if not self.config.perform_initialization:
            return
        if init_strategy == "identity":
            # Use the standard Megatron init for Q/K/V (now a single fused
            # linear) — the output norm is solely controlled by linear_proj,
            # which we zero (or near-zero for the no-residual variant).
            init_linear_weight_and_bias(
                self.linear_qkv,
                self.config.init_method,
                perform_initialization=True,
            )
            with torch.no_grad():
                if self.residual:
                    # Strict identity: x + linear_proj(SDPA(...)) = x when
                    # linear_proj.weight = 0.
                    self.linear_proj.weight.zero_()
                else:
                    # No residual: only an approximate identity is possible.
                    # Keep the initial output magnitude tiny so the first
                    # step's write_part is approximately zero.
                    nn.init.normal_(self.linear_proj.weight, mean=0.0, std=1e-2)
        else:  # "megatron"
            init_linear_weight_and_bias(
                self.linear_qkv,
                self.config.init_method,
                perform_initialization=True,
            )
            init_linear_weight_and_bias(
                self.linear_proj,
                self.config.output_layer_init_method,
                perform_initialization=True,
            )

    def _split_to_heads(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # qkv: (B, S, 3 * num_heads * head_dim)
        # Reshape so the QKV split lives on a contiguous trailing dim, then
        # slice. Output shape per stream: (B, num_heads, S, head_dim).
        b, s, _ = qkv.shape
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)
        # Permute -> (3, B, num_heads, S, head_dim) then unbind on dim 0; this
        # avoids three independent reshapes and keeps memory contiguous in the
        # SDPA-friendly layout.
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        return qkv[0], qkv[1], qkv[2]

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_heads, S, head_dim) -> (B, S, num_heads * head_dim)
        b, h, s, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, s, h * d)

    def _attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # Flash kernel needs CUDA + fp16/bf16. Outside that, fall back to the
        # generic SDPA dispatcher; the caller can also force "sdpa" backend.
        ctx = nullcontext()
        if (
            self.backend == "flash"
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        with ctx:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )

    def forward(self, x: torch.Tensor, *, input_is_sbh: bool) -> torch.Tensor:
        assert input_is_sbh, "SeqMixerAttn only supports SBH layout"
        # [S, B, H] -> [B, S, H] so SDPA sees a batch-first 4D tensor.
        bs = x.permute(1, 0, 2).contiguous()
        h = self.prenorm(bs) if self.prenorm is not None else bs
        q, k, v = self._split_to_heads(self.linear_qkv(h))
        out = self.linear_proj(self._from_heads(self._attn(q, k, v)))
        if self.residual:
            out = bs + out
        return out.permute(1, 0, 2).contiguous()


def build_seq_mixer(
    mixer_type: str,
    *,
    config: TransformerConfig,
    hidden_size: int,
    seq_len: Optional[int],
    init_strategy: str = "identity",
    mlp_expansion: int = 2,
    attn_num_heads: int = 1,
    attn_head_dim: Optional[int] = None,
    attn_residual: bool = True,
    attn_prenorm: bool = True,
    attn_backend: str = "flash",
    dtype: Optional[torch.dtype] = None,
) -> Optional[MegatronModule]:
    """Return a configured seq mixer, or ``None`` when disabled.

    ``seq_len`` is only consulted by the MLP variants (their weight shape is
    ``[seq_len, seq_len]``). The attention variant operates on a runtime-known
    chunk length, so ``seq_len`` may be ``None`` for ``mixer_type='attn'``.
    """
    if mixer_type not in _SUPPORTED_SEQ_MIXER_TYPES:
        raise ValueError(
            f"armt_seq_mixer_type must be one of {_SUPPORTED_SEQ_MIXER_TYPES}, "
            f"got {mixer_type!r}"
        )
    if mixer_type == "none":
        return None

    if init_strategy not in _SUPPORTED_SEQ_MIXER_INITS:
        raise ValueError(
            f"armt_seq_mixer_init must be one of {_SUPPORTED_SEQ_MIXER_INITS}, "
            f"got {init_strategy!r}"
        )

    if dtype is None:
        dtype = getattr(config, "params_dtype", torch.bfloat16)

    if mixer_type == "mlp1":
        if seq_len is None or seq_len <= 0:
            raise ValueError(
                "mlp1 requires a positive seq_len (got "
                f"{seq_len!r}); for mem_tokens write source this means "
                "num_mem_tokens must be > 0."
            )
        return SeqMixerMLP1(
            seq_len,
            config=config,
            init_strategy=init_strategy,
            dtype=dtype,
        )

    if mixer_type == "mlp2":
        if seq_len is None or seq_len <= 0:
            raise ValueError(
                "mlp2 requires a positive seq_len (got "
                f"{seq_len!r}); for mem_tokens write source this means "
                "num_mem_tokens must be > 0."
            )
        if mlp_expansion <= 0:
            raise ValueError("armt_seq_mixer_mlp_expansion must be > 0")
        return SeqMixerMLP2(
            seq_len,
            mlp_expansion,
            config=config,
            init_strategy=init_strategy,
            dtype=dtype,
        )

    # mixer_type == "attn"
    return SeqMixerAttn(
        hidden_size=hidden_size,
        num_heads=attn_num_heads,
        head_dim=attn_head_dim,
        residual=attn_residual,
        prenorm=attn_prenorm,
        backend=attn_backend,
        config=config,
        init_strategy=init_strategy,
        dtype=dtype,
    )
