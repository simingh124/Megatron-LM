import torch
import pytest
from unittest.mock import MagicMock, patch

from megatron.core.models.armt.armt_model import ARMTModel
from megatron.core.models.armt.armt_layer import ARMTLayer
from megatron.core.models.armt.associative_layer import AssociativeLayer
from megatron.core.models.armt.cross_attention_slot_memory import CrossAttentionSlotMemory
from megatron.core.models.armt.gated_deltanet_memory import GatedDeltaNetMemory
from megatron.core.models.armt.monitoring import build_mean_metric, build_ratio_of_means_metric


def _minimal_gpt_init(self, config, transformer_layer_spec, vocab_size, max_sequence_length, **kwargs):
    torch.nn.Module.__init__(self)
    self.config = config
    self.position_embedding_type = getattr(config, "position_embedding_type", "rope")
    self.mtp_process = False
    self.decoder = MagicMock()


def _minimal_armt_layer_init(self, config, submodules, layer_number=1, **kwargs):
    torch.nn.Module.__init__(self)
    self.config = config
    self.layer_number = layer_number
    self.associative_layer = None
    self.recurrent_memory_layer = None
    self._skip_read_memory_for_current_chunk = False
    self._current_chunk_is_first = False
    self._current_chunk_start_position = 0


class TestARMTModel:
    @pytest.mark.parametrize("backend_attr", ["associative_layer", "recurrent_memory_layer"])
    def test_armt_model_memory_parameter_breakdown(self, backend_attr):
        config = MagicMock()
        config.hidden_size = 8
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=4,
            )

        with patch(
            "megatron.core.models.armt.armt_model.ARMTLayer.__init__",
            new=_minimal_armt_layer_init,
        ):
            layer = ARMTLayer(config=config, submodules=MagicMock(), layer_number=1)

        memory_module = torch.nn.Sequential(
            torch.nn.Linear(8, 4, bias=False),
            torch.nn.Linear(4, 2, bias=True),
        )
        setattr(layer, backend_attr, memory_module)
        model.add_module("armt_layer", layer)

        breakdown = model.get_memory_parameter_breakdown()

        assert breakdown == [
            ("memory_embeddings", 32),
            (f"armt_layer.{backend_attr}", sum(p.numel() for p in memory_module.parameters())),
        ]

    def test_armt_model_memory_parameter_breakdown_is_empty_when_num_mem_tokens_is_zero(self):
        config = MagicMock()
        config.hidden_size = 8
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=0,
            )

        with patch(
            "megatron.core.models.armt.armt_model.ARMTLayer.__init__",
            new=_minimal_armt_layer_init,
        ):
            layer = ARMTLayer(config=config, submodules=MagicMock(), layer_number=1)

        layer.recurrent_memory_layer = torch.nn.Linear(8, 4, bias=False)
        model.add_module("armt_layer", layer)

        assert model.get_memory_parameter_breakdown() == []

    def test_armt_model_memory_parameter_breakdown_counts_cross_attention_slot_backend(self):
        config = MagicMock()
        config.hidden_size = 16
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02
        config.params_dtype = torch.float32

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=4,
            )

        with patch(
            "megatron.core.models.armt.armt_model.ARMTLayer.__init__",
            new=_minimal_armt_layer_init,
        ):
            layer = ARMTLayer(config=config, submodules=MagicMock(), layer_number=1)

        layer.recurrent_memory_layer = CrossAttentionSlotMemory(
            d_model=16,
            num_mem_tokens=4,
            num_slots=6,
            num_heads=4,
            head_dim=4,
            dtype=torch.float32,
        )
        model.add_module("armt_layer", layer)

        breakdown = model.get_memory_parameter_breakdown()
        backend_params = sum(p.numel() for p in layer.recurrent_memory_layer.parameters())

        assert breakdown == [
            ("memory_embeddings", 64),
            ("armt_layer.recurrent_memory_layer", backend_params),
        ]

    def test_armt_model_memory_state_breakdown_aggregates_single_sample_state_sizes(self):
        config = MagicMock()
        config.hidden_size = 16
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02
        config.params_dtype = torch.float32

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=4,
            )

        with patch(
            "megatron.core.models.armt.armt_model.ARMTLayer.__init__",
            new=_minimal_armt_layer_init,
        ):
            slot_layer = ARMTLayer(config=config, submodules=MagicMock(), layer_number=1)
            assoc_layer = ARMTLayer(config=config, submodules=MagicMock(), layer_number=2)
            gdn_layer = ARMTLayer(config=config, submodules=MagicMock(), layer_number=3)

        slot_layer.recurrent_memory_layer = CrossAttentionSlotMemory(
            d_model=16,
            num_mem_tokens=4,
            num_slots=6,
            num_heads=4,
            head_dim=4,
            dtype=torch.float32,
        )
        assoc_layer.associative_layer = AssociativeLayer(
            d_model=16,
            num_mem_tokens=4,
            d_mem=16,
            n_heads=4,
            head_dim=4,
            dtype=torch.float32,
        )
        gdn_layer.recurrent_memory_layer = GatedDeltaNetMemory(
            d_model=16,
            num_mem_tokens=4,
            key_head_dim=4,
            value_head_dim=8,
            num_key_heads=2,
            num_value_heads=4,
            use_fla_kernel=False,
            use_causal_conv1d=False,
            dtype=torch.float32,
        )
        model.add_module("slot_layer", slot_layer)
        model.add_module("assoc_layer", assoc_layer)
        model.add_module("gdn_layer", gdn_layer)

        assert model.get_memory_state_breakdown(batch_size=1) == [
            ("initial_slots", slot_layer.recurrent_memory_layer.initial_slots.numel()),
            (
                "W_mem",
                assoc_layer.associative_layer.n_heads
                * (assoc_layer.associative_layer.d_mem // assoc_layer.associative_layer.n_heads)
                * assoc_layer.associative_layer.head_dim
                + gdn_layer.recurrent_memory_layer.num_value_heads
                * gdn_layer.recurrent_memory_layer.key_head_dim
                * gdn_layer.recurrent_memory_layer.value_head_dim,
            ),
        ]

    def test_armt_model_memory_concat_strip(self):
        """验证 ARMTModel 的 memory embedding concat/strip 形状逻辑（S -> S+M -> S）。"""
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=16,
            )

        B, S, H = 2, 512, 256
        decoder_input = torch.randn(S, B, H)
        concat_hidden = model._concat_memory_embeddings(decoder_input)
        assert concat_hidden.shape == (S + model.num_mem_tokens, B, H)

        stripped = model._strip_memory_tokens(concat_hidden, S)
        assert stripped.shape == (S, B, H)

    def test_armt_model_reset_all_memory(self):
        """验证 reset_all_memory 会遍历并调用每个 ARMTLayer.reset_memory。"""
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=16,
            )

        with patch(
            "megatron.core.models.armt.armt_model.ARMTLayer.__init__",
            new=_minimal_armt_layer_init,
        ):
            layer = ARMTLayer(config=config, submodules=MagicMock(), layer_number=1)
            layer.reset_memory = MagicMock()
            model.add_module("armt_layer", layer)

        model.reset_all_memory()
        layer.reset_memory.assert_called_once()

    def test_armt_model_propagates_chunk_state_to_layers(self):
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=16,
            )

        with patch(
            "megatron.core.models.armt.armt_model.ARMTLayer.__init__",
            new=_minimal_armt_layer_init,
        ):
            layer = ARMTLayer(config=config, submodules=MagicMock(), layer_number=1)

        layer.associative_layer = MagicMock()
        model.add_module("armt_layer", layer)

        model.set_current_chunk_is_first(True)
        model.set_skip_read_memory_for_current_chunk(True)
        model.set_current_chunk_start_position(128)

        assert model._current_chunk_is_first is True
        assert model._skip_read_memory_for_current_chunk is True
        assert model._current_chunk_start_position == 128
        assert layer._current_chunk_is_first is True
        assert layer._skip_read_memory_for_current_chunk is True
        assert layer._current_chunk_start_position == 128

        model.reset_all_memory()

        assert model._current_chunk_is_first is False
        assert model._skip_read_memory_for_current_chunk is False
        assert model._current_chunk_start_position == 0
        assert layer._current_chunk_is_first is False
        assert layer._skip_read_memory_for_current_chunk is False
        assert layer._current_chunk_start_position == 0
        layer.associative_layer.reset_memory.assert_called_once()

    def test_armt_model_monitoring_helpers(self):
        """验证默认只产出聚合 monitoring 指标。"""
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=16,
            )

        with patch(
            "megatron.core.models.armt.armt_model.ARMTLayer.__init__",
            new=_minimal_armt_layer_init,
        ):
            layer_one = ARMTLayer(config=config, submodules=MagicMock(), layer_number=1)
            layer_two = ARMTLayer(config=config, submodules=MagicMock(), layer_number=2)

        layer_one.reset_monitoring_stats = MagicMock()
        layer_two.reset_monitoring_stats = MagicMock()
        layer_one.consume_monitoring_primitives = MagicMock(
            return_value={
                "armt/read/retrieved_norm_mean": build_mean_metric(
                    torch.tensor(4.0),
                    torch.tensor(2.0),
                ),
                "armt/token/mem_ctx_norm_ratio": build_ratio_of_means_metric(
                    torch.tensor(6.0),
                    torch.tensor(3.0),
                    torch.tensor(18.0),
                    torch.tensor(2.0),
                ),
            }
        )
        layer_two.consume_monitoring_primitives = MagicMock(
            return_value={
                "armt/read/retrieved_norm_mean": build_mean_metric(
                    torch.tensor(9.0),
                    torch.tensor(3.0),
                ),
                "armt/token/mem_ctx_norm_ratio": build_ratio_of_means_metric(
                    torch.tensor(9.0),
                    torch.tensor(1.0),
                    torch.tensor(8.0),
                    torch.tensor(4.0),
                ),
            }
        )
        model.add_module("armt_layer_one", layer_one)
        model.add_module("armt_layer_two", layer_two)

        model.reset_all_monitoring_stats()
        layer_one.reset_monitoring_stats.assert_called_once()
        layer_two.reset_monitoring_stats.assert_called_once()

        metrics = model.consume_all_monitoring_metrics()

        assert float(metrics["armt/read/retrieved_norm_mean"]) == pytest.approx(13.0 / 5.0)
        assert float(metrics["armt/token/mem_ctx_norm_ratio"]) == pytest.approx(45.0 / 52.0)
        assert "armt/read/retrieved_norm_mean/layer_01" not in metrics
        assert "armt/read/retrieved_norm_mean/layer_02" not in metrics

    def test_armt_model_monitoring_helpers_with_layer_metrics_enabled(self):
        """验证显式开关打开后会同时产出逐层 monitoring 指标。"""
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=16,
                log_layer_metrics_to_tensorboard=True,
            )

        with patch(
            "megatron.core.models.armt.armt_model.ARMTLayer.__init__",
            new=_minimal_armt_layer_init,
        ):
            layer_one = ARMTLayer(config=config, submodules=MagicMock(), layer_number=1)
            layer_two = ARMTLayer(config=config, submodules=MagicMock(), layer_number=2)

        layer_one.consume_monitoring_primitives = MagicMock(
            return_value={
                "armt/read/retrieved_norm_mean": build_mean_metric(
                    torch.tensor(4.0),
                    torch.tensor(2.0),
                ),
                "armt/token/mem_ctx_norm_ratio": build_ratio_of_means_metric(
                    torch.tensor(6.0),
                    torch.tensor(3.0),
                    torch.tensor(18.0),
                    torch.tensor(2.0),
                ),
            }
        )
        layer_two.consume_monitoring_primitives = MagicMock(
            return_value={
                "armt/read/retrieved_norm_mean": build_mean_metric(
                    torch.tensor(9.0),
                    torch.tensor(3.0),
                ),
                "armt/token/mem_ctx_norm_ratio": build_ratio_of_means_metric(
                    torch.tensor(9.0),
                    torch.tensor(1.0),
                    torch.tensor(8.0),
                    torch.tensor(4.0),
                ),
            }
        )
        model.add_module("armt_layer_one", layer_one)
        model.add_module("armt_layer_two", layer_two)

        metrics = model.consume_all_monitoring_metrics()

        assert float(metrics["armt/read/retrieved_norm_mean"]) == pytest.approx(13.0 / 5.0)
        assert float(metrics["armt/token/mem_ctx_norm_ratio"]) == pytest.approx(45.0 / 52.0)
        assert float(metrics["armt/read/retrieved_norm_mean/layer_01"]) == pytest.approx(2.0)
        assert float(metrics["armt/read/retrieved_norm_mean/layer_02"]) == pytest.approx(3.0)
        assert float(metrics["armt/token/mem_ctx_norm_ratio/layer_01"]) == pytest.approx(
            2.0 / 9.0
        )
        assert float(metrics["armt/token/mem_ctx_norm_ratio/layer_02"]) == pytest.approx(4.5)

    def test_armt_model_rope_extension(self):
        """验证 _preprocess 在拼接 memory tokens 后会重新生成/扩展 RoPE 到 S+M。"""
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=16,
            )

        S, B, H = 512, 2, 256
        decoder_input = torch.randn(S, B, H)
        input_ids = torch.randint(0, 32000, (B, S))
        position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

        model.rotary_pos_emb = MagicMock(return_value="rotary")

        preproc_output = (
            decoder_input,
            "old_rotary",
            None,
            None,
            None,
            None,
        )
        with patch(
            "megatron.core.models.armt.armt_model.GPTModel._preprocess",
            return_value=preproc_output,
        ):
            output = model._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=None,
                packed_seq_params=None,
                padding_mask=None,
            )

        model.rotary_pos_emb.assert_called_once()
        assert output[1] == "rotary"

    def test_armt_model_rope_extension_uses_chunk_start_offset_for_windowed_attention(self):
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch(
            "megatron.core.models.armt.armt_model.GPTModel.__init__",
            new=_minimal_gpt_init,
        ):
            model = ARMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=16,
                recurrent_chunk_size=64,
                full_attn_window_size=512,
            )

        S, B, H = 64, 2, 256
        decoder_input = torch.randn(S, B, H)
        input_ids = torch.randint(0, 32000, (B, S))
        position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

        model.set_current_chunk_start_position(128)
        model.rotary_pos_emb = MagicMock(return_value="rotary")

        preproc_output = (
            decoder_input,
            "old_rotary",
            None,
            None,
            None,
            None,
        )
        with patch(
            "megatron.core.models.armt.armt_model.GPTModel._preprocess",
            return_value=preproc_output,
        ):
            output = model._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=None,
                packed_seq_params=None,
                padding_mask=None,
            )

        model.rotary_pos_emb.assert_called_once_with(
            S + model.num_mem_tokens,
            offset=128,
            packed_seq=False,
            cp_group=None,
        )
        assert output[1] == "rotary"
