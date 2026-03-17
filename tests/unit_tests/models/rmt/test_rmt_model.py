from unittest.mock import MagicMock, patch

import torch

from megatron.core.models.rmt.rmt_model import RMTModel


def _minimal_gpt_init(self, config, transformer_layer_spec, vocab_size, max_sequence_length, **kwargs):
    del transformer_layer_spec, vocab_size, max_sequence_length, kwargs
    torch.nn.Module.__init__(self)
    self.config = config
    self.position_embedding_type = getattr(config, "position_embedding_type", "rope")
    self.mtp_process = False
    self.decoder = MagicMock()


class TestRMTModel:
    def test_rmt_model_memory_concat_strip(self):
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=16,
            )

        batch_size, seq_length, hidden_size = 2, 512, 256
        decoder_input = torch.randn(seq_length, batch_size, hidden_size)
        concat_hidden = model._concat_memory_embeddings(decoder_input)
        assert concat_hidden.shape == (seq_length + 2 * model.num_mem_tokens, batch_size, hidden_size)

        stripped = model._strip_memory_tokens(concat_hidden, seq_length)
        assert stripped.shape == (seq_length, batch_size, hidden_size)
        assert torch.equal(
            stripped,
            concat_hidden[model.num_mem_tokens : model.num_mem_tokens + seq_length],
        )

    def test_rmt_model_reset_all_memory(self):
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=16,
            )

        model.memory_state = torch.randn(16, 2, 256)
        model.reset_all_memory()
        assert model.memory_state is None

    def test_rmt_model_updates_memory_state_and_detaches(self):
        config = MagicMock()
        config.hidden_size = 128
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=8,
                tbptt_mode=True,
            )

        hidden_states = torch.randn(48, 2, 128, requires_grad=True)
        model._update_memory_state_from_hidden_states(hidden_states)
        assert model.memory_state.shape == (8, 2, 128)
        assert torch.equal(model.memory_state, hidden_states[-8:].detach())
        assert model.memory_state.requires_grad is False

    def test_rmt_model_attention_mask_inserts_token_block(self):
        config = MagicMock()
        config.hidden_size = 128
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=4,
            )

        seq_length = 6
        attention_mask = torch.triu(torch.ones(1, 1, seq_length, seq_length), diagonal=1).bool()
        new_mask = model._adjust_attention_mask(attention_mask, seq_length)
        start_idx = model.num_mem_tokens
        end_idx = start_idx + seq_length
        assert new_mask.shape == (1, 1, seq_length + 8, seq_length + 8)
        assert torch.equal(new_mask[..., start_idx:end_idx, start_idx:end_idx], attention_mask)
        assert new_mask[..., start_idx:end_idx, -model.num_mem_tokens :].all()

    def test_rmt_model_rope_extension(self):
        config = MagicMock()
        config.hidden_size = 256
        config.sequence_parallel = False
        config.position_embedding_type = "rope"
        config.multi_latent_attention = False
        config.init_method_std = 0.02

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=16,
            )

        seq_length, batch_size, hidden_size = 512, 2, 256
        decoder_input = torch.randn(seq_length, batch_size, hidden_size)
        input_ids = torch.randint(0, 32000, (batch_size, seq_length))
        position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)

        model.rotary_pos_emb = MagicMock(return_value="rotary")
        preproc_output = (decoder_input, "old_rotary", None, None, None, None)
        with patch(
            "megatron.core.models.rmt.rmt_model.GPTModel._preprocess",
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
