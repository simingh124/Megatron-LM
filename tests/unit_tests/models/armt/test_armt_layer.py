import torch
import pytest
from unittest.mock import MagicMock, patch

from megatron.core.models.armt.armt_layer import ARMTLayer


class _DummyAssociativeLayer(torch.nn.Module):
    def __init__(self, associate_return):
        super().__init__()
        self.associate = MagicMock(return_value=associate_return)
        self.update_mem = MagicMock()


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
