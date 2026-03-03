import torch
from unittest.mock import MagicMock, patch

from megatron.core.models.armt.armt_model import ARMTModel
from megatron.core.models.armt.armt_layer import ARMTLayer


def _minimal_gpt_init(self, config, transformer_layer_spec, vocab_size, max_sequence_length, **kwargs):
    torch.nn.Module.__init__(self)
    self.config = config
    self.position_embedding_type = getattr(config, "position_embedding_type", "rope")
    self.mtp_process = False
    self.decoder = MagicMock()


def _minimal_armt_layer_init(self, config, submodules, layer_number=1, **kwargs):
    torch.nn.Module.__init__(self)
    self.config = config


class TestARMTModel:
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
