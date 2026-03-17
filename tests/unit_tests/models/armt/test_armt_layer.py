import torch
import pytest
from unittest.mock import MagicMock, patch

from megatron.core.models.armt.armt_layer import ARMTLayer
from megatron.core.models.armt.monitoring import finalize_metric_primitives


class _DummyAssociativeLayer(torch.nn.Module):
    def __init__(self, associate_return):
        super().__init__()
        self.associate = MagicMock(return_value=associate_return)
        self.update_mem = MagicMock()
        self.reset_monitoring_stats = MagicMock()
        self.consume_monitoring_primitives = MagicMock(return_value={})


class TestARMTLayer:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.params_dtype = torch.bfloat16
        return config

    def test_armt_layer_forward_shape(self, mock_config):
        """验证 ARMTLayer.forward 的输出 shape 与输入一致（只做插入，不改主干维度）。"""
        S, B, H = 128 + 16, 2, 256
        hidden_states = torch.randn(S, B, H)

        def _minimal_init(self, config, submodules, layer_number=1, **kwargs):
            torch.nn.Module.__init__(self)
            self.config = config
            self.submodules_config = submodules

        with patch(
            "megatron.core.models.armt.armt_layer.TransformerLayer.__init__",
            new=_minimal_init,
        ):
            layer = ARMTLayer(config=mock_config, submodules=MagicMock(), layer_number=1)
            layer.associative_layer = _DummyAssociativeLayer(torch.zeros_like(hidden_states))

        with patch(
            "megatron.core.models.armt.armt_layer.TransformerLayer.forward",
            return_value=(hidden_states, None),
        ):
            out, _ = layer.forward(hidden_states, attention_mask=None)

        assert out.shape == hidden_states.shape

    def test_armt_layer_memory_update(self, mock_config):
        """验证 ARMTLayer 的调用顺序：associate 被调用，且 update_mem 使用尾部 M 个 memory tokens。"""
        S, B, H = 128 + 16, 2, 256
        num_mem_tokens = 16
        hidden_states = torch.randn(S, B, H)

        def _minimal_init(self, config, submodules, layer_number=1, **kwargs):
            torch.nn.Module.__init__(self)
            self.config = config
            self.submodules_config = submodules

        with patch(
            "megatron.core.models.armt.armt_layer.TransformerLayer.__init__",
            new=_minimal_init,
        ):
            layer = ARMTLayer(config=mock_config, submodules=MagicMock(), layer_number=1)
            layer.associative_layer = _DummyAssociativeLayer(torch.zeros_like(hidden_states))
            layer.num_mem_tokens = num_mem_tokens

        with patch(
            "megatron.core.models.armt.armt_layer.TransformerLayer.forward",
            return_value=(hidden_states, None),
        ):
            layer.forward(hidden_states, attention_mask=None)

        layer.associative_layer.associate.assert_called_once()
        layer.associative_layer.update_mem.assert_called_once()
        mem_hidden = layer.associative_layer.update_mem.call_args[0][0]
        assert mem_hidden.shape == (num_mem_tokens, B, H)

    def test_armt_layer_token_monitoring_metrics(self, mock_config):
        """验证 token 监控指标按最终层输出计算。"""
        num_mem_tokens = 4
        hidden_states = torch.zeros(6, 1, 2)
        monitored_hidden = torch.tensor(
            [
                [[3.0, 4.0]],
                [[0.0, 5.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[-1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        )

        def _minimal_init(self, config, submodules, layer_number=1, **kwargs):
            torch.nn.Module.__init__(self)
            self.config = config
            self.submodules_config = submodules

        with patch(
            "megatron.core.models.armt.armt_layer.TransformerLayer.__init__",
            new=_minimal_init,
        ):
            layer = ARMTLayer(
                config=mock_config,
                submodules=MagicMock(),
                layer_number=1,
                num_mem_tokens=num_mem_tokens,
            )
            layer.associative_layer = _DummyAssociativeLayer(torch.zeros_like(hidden_states))

        with patch(
            "megatron.core.models.armt.armt_layer.TransformerLayer.forward",
            return_value=(monitored_hidden, None),
        ):
            layer.forward(hidden_states, attention_mask=None)

        metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())

        assert float(metrics["armt/token/context_token_norm_mean"]) == pytest.approx(5.0)
        assert float(metrics["armt/token/mem_token_norm_mean"]) == pytest.approx(1.0)
        assert float(metrics["armt/token/mem_ctx_norm_ratio"]) == pytest.approx(0.4)
        assert float(metrics["armt/token/mem_token_cosine_mean"]) == pytest.approx(-1.0 / 6.0)
        assert float(metrics["armt/token/mem_token_cosine_max_mean"]) == pytest.approx(1.0)
        assert float(metrics["armt/token/mem_token_cosine_min_mean"]) == pytest.approx(-1.0)
        assert float(metrics["armt/token/mem_token_cosine_gt_0p8_ratio_mean"]) == pytest.approx(
            1.0 / 6.0
        )
        mem_hidden = layer.associative_layer.update_mem.call_args[0][0]
        assert torch.equal(mem_hidden, monitored_hidden[-num_mem_tokens:])

    def test_armt_layer_monitoring_skips_cosine_for_single_mem_token(self, mock_config):
        """验证 num_mem_tokens=1 时不记录 cosine 指标。"""
        monitored_hidden = torch.tensor(
            [
                [[3.0, 4.0]],
                [[0.0, 5.0]],
                [[1.0, 0.0]],
            ]
        )

        def _minimal_init(self, config, submodules, layer_number=1, **kwargs):
            torch.nn.Module.__init__(self)
            self.config = config
            self.submodules_config = submodules

        with patch(
            "megatron.core.models.armt.armt_layer.TransformerLayer.__init__",
            new=_minimal_init,
        ):
            layer = ARMTLayer(
                config=mock_config,
                submodules=MagicMock(),
                layer_number=1,
                num_mem_tokens=1,
            )
            layer.associative_layer = _DummyAssociativeLayer(torch.zeros_like(monitored_hidden))

        with patch(
            "megatron.core.models.armt.armt_layer.TransformerLayer.forward",
            return_value=(monitored_hidden, None),
        ):
            layer.forward(monitored_hidden, attention_mask=None)

        metrics = finalize_metric_primitives(layer.consume_monitoring_primitives())

        assert "armt/token/mem_token_cosine_mean" not in metrics
        assert "armt/token/mem_token_cosine_max_mean" not in metrics
        assert "armt/token/mem_token_cosine_min_mean" not in metrics
        assert "armt/token/mem_token_cosine_gt_0p8_ratio_mean" not in metrics

    def test_armt_layer_sequence_parallel_monitoring_gathers_full_hidden_states(
        self, mock_config
    ):
        """验证 sequence parallel 场景会先 gather 完整 hidden 再做 token 监控。"""
        mock_config.sequence_parallel = True
        sharded_hidden = torch.randn(4, 1, 1)
        full_hidden = torch.tensor(
            [
                [[3.0, 4.0]],
                [[0.0, 5.0]],
                [[1.0, 0.0]],
                [[0.0, 2.0]],
            ]
        )

        def _minimal_init(self, config, submodules, layer_number=1, **kwargs):
            torch.nn.Module.__init__(self)
            self.config = config
            self.submodules_config = submodules

        with patch(
            "megatron.core.models.armt.armt_layer.TransformerLayer.__init__",
            new=_minimal_init,
        ):
            layer = ARMTLayer(
                config=mock_config,
                submodules=MagicMock(),
                layer_number=1,
                num_mem_tokens=2,
            )
            layer.associative_layer = _DummyAssociativeLayer(torch.zeros_like(sharded_hidden))

        with (
            patch(
                "megatron.core.models.armt.armt_layer.TransformerLayer.forward",
                return_value=(sharded_hidden, None),
            ),
            patch(
                "megatron.core.models.armt.armt_layer.parallel_state.get_tensor_model_parallel_group",
                return_value=MagicMock(),
            ),
            patch(
                "megatron.core.models.armt.armt_layer.parallel_state.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "megatron.core.models.armt.armt_layer.parallel_state.get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            patch(
                "megatron.core.models.armt.armt_layer.tensor_parallel.gather_from_sequence_parallel_region",
                return_value=sharded_hidden,
            ),
            patch(
                "megatron.core.models.armt.armt_layer.tensor_parallel.gather_from_tensor_model_parallel_region",
                return_value=full_hidden,
            ),
        ):
            layer.forward(sharded_hidden, attention_mask=None)

        mem_hidden = layer.associative_layer.update_mem.call_args[0][0]
        assert torch.equal(mem_hidden, full_hidden[-2:])
