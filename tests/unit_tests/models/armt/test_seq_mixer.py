"""Unit tests for ARMT seq mixers (chunk-internal token mixing modules)."""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.models.armt.seq_mixers import (
    SeqMixerAttn,
    SeqMixerMLP1,
    SeqMixerMLP2,
    build_seq_mixer,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _build_config(hidden_size: int, *, dtype: torch.dtype = torch.float32, num_layers: int = 2):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=4 if hidden_size % 4 == 0 else 1,
        ffn_hidden_size=hidden_size * 4,
        params_dtype=dtype,
    )


def _make_input(seq_len: int, batch: int, hidden: int, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(seq_len, batch, hidden, dtype=dtype)


class TestBuildSeqMixer:
    def test_none_returns_none(self):
        config = _build_config(hidden_size=64)
        assert (
            build_seq_mixer(
                "none",
                config=config,
                hidden_size=64,
                seq_len=8,
            )
            is None
        )

    def test_unknown_type_raises(self):
        config = _build_config(hidden_size=64)
        with pytest.raises(ValueError, match="armt_seq_mixer_type"):
            build_seq_mixer(
                "no_such_mixer",
                config=config,
                hidden_size=64,
                seq_len=8,
            )

    def test_unknown_init_raises(self):
        config = _build_config(hidden_size=64)
        with pytest.raises(ValueError, match="armt_seq_mixer_init"):
            build_seq_mixer(
                "mlp1",
                config=config,
                hidden_size=64,
                seq_len=8,
                init_strategy="no_such_init",
            )

    def test_mlp_requires_positive_seq_len(self):
        config = _build_config(hidden_size=64)
        for mixer_type in ("mlp1", "mlp2"):
            with pytest.raises(ValueError, match="positive seq_len"):
                build_seq_mixer(
                    mixer_type,
                    config=config,
                    hidden_size=64,
                    seq_len=0,
                )
            with pytest.raises(ValueError, match="positive seq_len"):
                build_seq_mixer(
                    mixer_type,
                    config=config,
                    hidden_size=64,
                    seq_len=None,
                )

    def test_returns_correct_class(self):
        config = _build_config(hidden_size=64)
        assert isinstance(
            build_seq_mixer("mlp1", config=config, hidden_size=64, seq_len=8),
            SeqMixerMLP1,
        )
        assert isinstance(
            build_seq_mixer("mlp2", config=config, hidden_size=64, seq_len=8),
            SeqMixerMLP2,
        )
        assert isinstance(
            build_seq_mixer(
                "attn",
                config=config,
                hidden_size=64,
                seq_len=None,
                attn_num_heads=4,
            ),
            SeqMixerAttn,
        )


class TestShapeInvariance:
    """All mixers must preserve [S, B, H] shape regardless of variant or init."""

    @pytest.mark.parametrize("init_strategy", ["identity", "megatron"])
    @pytest.mark.parametrize(
        "mixer_type,extra_kwargs",
        [
            ("mlp1", {}),
            ("mlp2", {"mlp_expansion": 2}),
            ("attn", {"attn_num_heads": 4, "attn_residual": True}),
            ("attn", {"attn_num_heads": 4, "attn_residual": False}),
        ],
    )
    def test_output_shape_matches_input(self, mixer_type, extra_kwargs, init_strategy):
        S, B, H = 16, 2, 64
        config = _build_config(hidden_size=H, dtype=torch.float32)
        mixer = build_seq_mixer(
            mixer_type,
            config=config,
            hidden_size=H,
            seq_len=S,
            init_strategy=init_strategy,
            attn_backend="sdpa",
            **extra_kwargs,
        )
        x = _make_input(S, B, H, dtype=torch.float32)
        y = mixer(x, input_is_sbh=True)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


class TestIdentityInitMLP1:
    """mlp1 + identity gives a strict identity: output == input."""

    def test_strict_identity(self):
        S, B, H = 12, 3, 32
        config = _build_config(hidden_size=H, dtype=torch.float32)
        mixer = build_seq_mixer(
            "mlp1",
            config=config,
            hidden_size=H,
            seq_len=S,
            init_strategy="identity",
        )
        x = _make_input(S, B, H, dtype=torch.float32)
        y = mixer(x, input_is_sbh=True)
        torch.testing.assert_close(y, x, rtol=1e-6, atol=1e-6)


class TestIdentityInitAttnResidual:
    """attn + residual + identity: W_o is zero, residual passes input through."""

    def test_strict_identity(self):
        S, B, H = 16, 2, 64
        config = _build_config(hidden_size=H, dtype=torch.float32)
        mixer = build_seq_mixer(
            "attn",
            config=config,
            hidden_size=H,
            seq_len=None,
            init_strategy="identity",
            attn_num_heads=4,
            attn_residual=True,
            attn_prenorm=True,
            attn_backend="sdpa",
        )
        # linear_proj initialized to zero so the attention branch contributes nothing.
        assert torch.all(mixer.linear_proj.weight == 0)
        x = _make_input(S, B, H, dtype=torch.float32)
        y = mixer(x, input_is_sbh=True)
        torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-5)


class TestIdentityInitApproxZero:
    """mlp2 / attn-no-residual: identity init only guarantees a small-magnitude output."""

    @pytest.mark.parametrize(
        "mixer_type,extra_kwargs",
        [
            ("mlp2", {"mlp_expansion": 2}),
            ("attn", {"attn_num_heads": 4, "attn_residual": False, "attn_backend": "sdpa"}),
        ],
    )
    def test_output_norm_much_smaller_than_input(self, mixer_type, extra_kwargs):
        S, B, H = 16, 2, 64
        config = _build_config(hidden_size=H, dtype=torch.float32)
        mixer = build_seq_mixer(
            mixer_type,
            config=config,
            hidden_size=H,
            seq_len=S,
            init_strategy="identity",
            **extra_kwargs,
        )
        x = _make_input(S, B, H, dtype=torch.float32)
        y = mixer(x, input_is_sbh=True)
        input_norm = x.float().norm()
        output_norm = y.float().norm()
        # near-zero output: at least 100x smaller than the input norm.
        assert output_norm < 0.01 * input_norm, (
            f"identity init expected near-zero output for {mixer_type}, "
            f"but got output_norm={output_norm} vs input_norm={input_norm}"
        )


class TestMegatronInitProducesNonTrivialOutput:
    """megatron init should yield a non-identity, finite output."""

    @pytest.mark.parametrize(
        "mixer_type,extra_kwargs",
        [
            ("mlp1", {}),
            ("mlp2", {"mlp_expansion": 2}),
            ("attn", {"attn_num_heads": 4, "attn_residual": True, "attn_backend": "sdpa"}),
            ("attn", {"attn_num_heads": 4, "attn_residual": False, "attn_backend": "sdpa"}),
        ],
    )
    def test_output_differs_from_input(self, mixer_type, extra_kwargs):
        S, B, H = 16, 2, 64
        config = _build_config(hidden_size=H, dtype=torch.float32)
        mixer = build_seq_mixer(
            mixer_type,
            config=config,
            hidden_size=H,
            seq_len=S,
            init_strategy="megatron",
            **extra_kwargs,
        )
        x = _make_input(S, B, H, dtype=torch.float32)
        y = mixer(x, input_is_sbh=True)
        assert torch.isfinite(y).all()
        # The output should differ from the input (mixer is not identity here).
        assert (y - x).abs().mean() > 1e-4


class TestAttnInvalidArgs:
    def test_indivisible_hidden_raises_without_head_dim(self):
        config = _build_config(hidden_size=64)
        with pytest.raises(ValueError, match="divisible"):
            build_seq_mixer(
                "attn",
                config=config,
                hidden_size=64,
                seq_len=None,
                attn_num_heads=6,  # 64 % 6 != 0
                attn_head_dim=None,
            )

    def test_unknown_attn_backend_raises(self):
        config = _build_config(hidden_size=64)
        with pytest.raises(ValueError, match="attn backend"):
            build_seq_mixer(
                "attn",
                config=config,
                hidden_size=64,
                seq_len=None,
                attn_num_heads=4,
                attn_backend="no_such_backend",
            )


class TestAttnFusedQKV:
    """linear_qkv is a fused [Q | K | V] projection. Sanity-check that the split
    produces tensors equivalent to applying three independent linears with the
    matching weight slices.
    """

    def test_fused_qkv_matches_three_independent_linears(self):
        S, B, H = 8, 2, 32
        num_heads, head_dim = 4, 8
        config = _build_config(hidden_size=H, dtype=torch.float32)
        mixer = build_seq_mixer(
            "attn",
            config=config,
            hidden_size=H,
            seq_len=None,
            init_strategy="megatron",
            attn_num_heads=num_heads,
            attn_residual=False,
            attn_prenorm=False,
            attn_backend="sdpa",
        )
        # Reconstruct what Q/K/V would have been with three separate linears
        # by slicing the fused weight along the [Q | K | V] axis.
        proj = num_heads * head_dim
        w = mixer.linear_qkv.weight  # [3*proj, H]
        wq = w[0 * proj : 1 * proj]
        wk = w[1 * proj : 2 * proj]
        wv = w[2 * proj : 3 * proj]

        x = _make_input(S, B, H, dtype=torch.float32)
        # x: [S, B, H] -> [B, S, H] (matches SeqMixerAttn.forward)
        bs = x.permute(1, 0, 2).contiguous()
        qkv_fused = mixer.linear_qkv(bs)
        # Manual split mirrors _split_to_heads: reshape into (B, S, 3, h, d).
        b, s, _ = qkv_fused.shape
        qkv_split = qkv_fused.view(b, s, 3, num_heads, head_dim)
        q_fused = qkv_split[:, :, 0]
        k_fused = qkv_split[:, :, 1]
        v_fused = qkv_split[:, :, 2]

        # Compute the equivalent independent-linear outputs.
        q_indep = F.linear(bs, wq).view(b, s, num_heads, head_dim)
        k_indep = F.linear(bs, wk).view(b, s, num_heads, head_dim)
        v_indep = F.linear(bs, wv).view(b, s, num_heads, head_dim)

        torch.testing.assert_close(q_fused, q_indep, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k_fused, k_indep, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(v_fused, v_indep, rtol=1e-5, atol=1e-5)


class TestMegatronModuleInheritance:
    """Mixers should be MegatronModule subclasses so they get config + sharded
    state-dict integration for free.
    """

    @pytest.mark.parametrize(
        "mixer_type,extra_kwargs",
        [
            ("mlp1", {}),
            ("mlp2", {"mlp_expansion": 2}),
            ("attn", {"attn_num_heads": 4, "attn_backend": "sdpa"}),
        ],
    )
    def test_inherits_megatron_module(self, mixer_type, extra_kwargs):
        from megatron.core.transformer.module import MegatronModule

        config = _build_config(hidden_size=64)
        mixer = build_seq_mixer(
            mixer_type,
            config=config,
            hidden_size=64,
            seq_len=8,
            init_strategy="megatron",
            **extra_kwargs,
        )
        assert isinstance(mixer, MegatronModule)
        # config should be exposed via the MegatronModule contract.
        assert mixer.config is config


class TestGradientFlow:
    """Gradients should flow through every mixer variant in both init modes."""

    @pytest.mark.parametrize("init_strategy", ["identity", "megatron"])
    @pytest.mark.parametrize(
        "mixer_type,extra_kwargs",
        [
            ("mlp1", {}),
            ("mlp2", {"mlp_expansion": 2}),
            ("attn", {"attn_num_heads": 4, "attn_residual": True, "attn_backend": "sdpa"}),
            ("attn", {"attn_num_heads": 4, "attn_residual": False, "attn_backend": "sdpa"}),
        ],
    )
    def test_backward(self, mixer_type, extra_kwargs, init_strategy):
        S, B, H = 16, 2, 64
        config = _build_config(hidden_size=H, dtype=torch.float32)
        mixer = build_seq_mixer(
            mixer_type,
            config=config,
            hidden_size=H,
            seq_len=S,
            init_strategy=init_strategy,
            **extra_kwargs,
        )
        x = _make_input(S, B, H, dtype=torch.float32).requires_grad_(True)
        y = mixer(x, input_is_sbh=True)
        loss = y.float().square().mean()
        loss.backward()
        assert x.grad is not None
        # At least one parameter receives a gradient with non-trivial magnitude.
        param_grads = [p.grad for p in mixer.parameters() if p.grad is not None]
        assert any(g.abs().sum() > 0 for g in param_grads)
