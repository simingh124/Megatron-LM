from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.models.rmt.rmt_model import RMTModel


def _minimal_gpt_init(self, config, transformer_layer_spec, vocab_size, max_sequence_length, **kwargs):
    del transformer_layer_spec, vocab_size, max_sequence_length, kwargs
    torch.nn.Module.__init__(self)
    self.config = config
    self.position_embedding_type = getattr(config, "position_embedding_type", "rope")
    self.mtp_process = False
    self.decoder = MagicMock()


def _make_config(hidden_size=256):
    config = MagicMock()
    config.hidden_size = hidden_size
    config.sequence_parallel = False
    config.position_embedding_type = "rope"
    config.multi_latent_attention = False
    config.init_method_std = 0.02
    return config


class TestRMTModel:
    def test_rmt_model_memory_concat_strip(self):
        config = _make_config(hidden_size=256)

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

    def test_rmt_model_skip_read_memory_from_first_chunk(self):
        config = _make_config(hidden_size=128)

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=4,
            )

        batch_size, seq_length, hidden_size = 2, 16, 128
        decoder_input = torch.randn(seq_length, batch_size, hidden_size)
        model.set_current_chunk_is_first(True)
        model.set_skip_read_memory_for_current_chunk(True)

        concat_hidden = model._concat_memory_embeddings(decoder_input)
        assert concat_hidden.shape == (seq_length + model.num_mem_tokens, batch_size, hidden_size)
        assert torch.equal(concat_hidden[:seq_length], decoder_input)

        stripped = model._strip_memory_tokens(concat_hidden, seq_length)
        assert torch.equal(stripped, decoder_input)

    def test_rmt_model_reset_all_memory(self):
        config = _make_config(hidden_size=256)

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
        config = _make_config(hidden_size=128)

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

    def test_rmt_model_updates_memory_state_when_skip_read_is_enabled(self):
        config = _make_config(hidden_size=64)

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=4,
                tbptt_mode=True,
            )

        model.set_current_chunk_is_first(True)
        model.set_skip_read_memory_for_current_chunk(True)
        hidden_states = torch.randn(20, 2, 64, requires_grad=True)
        model._update_memory_state_from_hidden_states(hidden_states)

        assert model.memory_state.shape == (4, 2, 64)
        assert torch.equal(model.memory_state, hidden_states[-4:].detach())
        assert model.memory_state.requires_grad is False

    def test_rmt_model_attention_mask_inserts_token_block(self):
        config = _make_config(hidden_size=128)

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

    def test_rmt_model_attention_mask_without_read_prefix(self):
        config = _make_config(hidden_size=128)

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=4,
            )

        model.set_current_chunk_is_first(True)
        model.set_skip_read_memory_for_current_chunk(True)
        seq_length = 6
        attention_mask = torch.triu(torch.ones(1, 1, seq_length, seq_length), diagonal=1).bool()
        new_mask = model._adjust_attention_mask(attention_mask, seq_length)

        assert new_mask.shape == (1, 1, seq_length + 4, seq_length + 4)
        assert torch.equal(new_mask[..., :seq_length, :seq_length], attention_mask)
        assert new_mask[..., :seq_length, -model.num_mem_tokens :].all()

    def test_rmt_model_forward_registers_layer_output_callback_for_mem_token_cosine_monitoring(
        self,
    ):
        config = _make_config(hidden_size=2)

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=2,
            )

        input_ids = torch.randint(0, 100, (1, 1))
        position_ids = torch.zeros((1, 1), dtype=torch.long)
        attention_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool)
        decoder_input = torch.zeros((1, 1, 2))
        layer_hidden_states = torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        )
        final_hidden_states = torch.tensor(
            [
                [[3.0, 0.0]],
                [[4.0, 0.0]],
                [[0.0, 5.0]],
                [[1.0, 0.0]],
                [[0.0, 2.0]],
            ]
        )

        def _decoder_side_effect(**kwargs):
            kwargs["layer_output_callback"](1, layer_hidden_states)
            return final_hidden_states

        model.decoder.side_effect = _decoder_side_effect

        with (
            patch.object(
                model,
                "_preprocess",
                return_value=(decoder_input, None, None, None, None, None),
            ),
            patch.object(model, "_postprocess", side_effect=lambda hidden_states, **_: hidden_states),
        ):
            output = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        assert torch.equal(output, final_hidden_states[2:3])

        metrics = model.consume_all_monitoring_metrics()
        assert float(metrics["rmt/token/read_mem_token_cosine_mean"]) == pytest.approx(1.0)
        assert float(metrics["rmt/token/write_mem_token_cosine_mean"]) == pytest.approx(0.0)

    def test_rmt_model_monitoring_helpers_separate_first_chunk_metrics(self):
        config = _make_config(hidden_size=2)

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=1,
            )

        model.reset_all_monitoring_stats()
        model.set_current_chunk_is_first(True)
        model.set_skip_read_memory_for_current_chunk(False)
        first_chunk_hidden_states = torch.tensor(
            [
                [[3.0, 0.0]],
                [[6.0, 0.0]],
                [[0.0, 10.0]],
                [[12.0, 0.0]],
            ]
        )
        model._update_monitoring_stats_from_hidden_states(first_chunk_hidden_states, seq_len=2)

        model.set_current_chunk_is_first(False)
        second_chunk_hidden_states = torch.tensor(
            [
                [[4.0, 0.0]],
                [[8.0, 0.0]],
                [[0.0, 12.0]],
                [[14.0, 0.0]],
            ]
        )
        model._update_monitoring_stats_from_hidden_states(second_chunk_hidden_states, seq_len=2)
        metrics = model.consume_all_monitoring_metrics()

        assert float(metrics["rmt/token/read_mem_token_norm_mean_first_chunk"]) == pytest.approx(3.0)
        assert float(metrics["rmt/token/read_ctx_norm_ratio_first_chunk"]) == pytest.approx(3.0 / 8.0)
        assert float(metrics["rmt/token/read_mem_token_norm_mean"]) == pytest.approx(4.0)
        assert float(metrics["rmt/token/read_ctx_norm_ratio"]) == pytest.approx(4.0 / 10.0)
        assert float(metrics["rmt/token/write_mem_token_norm_mean"]) == pytest.approx(13.0)
        assert float(metrics["rmt/token/context_token_norm_mean"]) == pytest.approx(9.0)
        assert float(metrics["rmt/token/write_ctx_norm_ratio"]) == pytest.approx(13.0 / 9.0)
        assert float(metrics["rmt/token/read_write_norm_ratio"]) == pytest.approx(3.5 / 13.0)

    def test_rmt_model_monitoring_skips_first_chunk_read_metrics_when_prefix_is_disabled(self):
        config = _make_config(hidden_size=2)

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=1,
            )

        model.reset_all_monitoring_stats()
        model.set_current_chunk_is_first(True)
        model.set_skip_read_memory_for_current_chunk(True)
        hidden_states = torch.tensor(
            [
                [[3.0, 4.0]],
                [[0.0, 12.0]],
                [[5.0, 0.0]],
            ]
        )
        model._update_monitoring_stats_from_hidden_states(hidden_states, seq_len=2)
        metrics = model.consume_all_monitoring_metrics()

        assert "rmt/token/read_mem_token_norm_mean_first_chunk" not in metrics
        assert "rmt/token/read_ctx_norm_ratio_first_chunk" not in metrics
        assert float(metrics["rmt/token/context_token_norm_mean"]) == pytest.approx(8.5)
        assert float(metrics["rmt/token/write_mem_token_norm_mean"]) == pytest.approx(5.0)
        assert float(metrics["rmt/token/write_ctx_norm_ratio"]) == pytest.approx(5.0 / 8.5)
        assert "rmt/token/read_write_norm_ratio" not in metrics

    def test_rmt_model_mem_token_cosine_metrics_are_layer_averaged(self):
        config = _make_config(hidden_size=2)

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=2,
            )

        model.reset_all_monitoring_stats()
        model.set_current_chunk_is_first(True)
        model.set_skip_read_memory_for_current_chunk(False)

        first_layer_hidden_states = torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        )
        second_layer_hidden_states = torch.tensor(
            [
                [[1.0, 0.0]],
                [[-1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        )

        model._update_mem_token_cosine_monitoring_stats_from_hidden_states(
            first_layer_hidden_states, seq_len=1
        )
        model._update_mem_token_cosine_monitoring_stats_from_hidden_states(
            second_layer_hidden_states, seq_len=1
        )

        model.set_current_chunk_is_first(False)
        rest_layer_hidden_states = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        )
        model._update_mem_token_cosine_monitoring_stats_from_hidden_states(
            rest_layer_hidden_states, seq_len=1
        )
        metrics = model.consume_all_monitoring_metrics()

        assert float(metrics["rmt/token/read_mem_token_cosine_mean_first_chunk"]) == pytest.approx(0.0)
        assert float(metrics["rmt/token/read_mem_token_cosine_max_mean_first_chunk"]) == pytest.approx(0.0)
        assert float(metrics["rmt/token/read_mem_token_cosine_min_mean_first_chunk"]) == pytest.approx(0.0)
        assert float(
            metrics["rmt/token/read_mem_token_cosine_gt_0p8_ratio_mean_first_chunk"]
        ) == pytest.approx(0.5)
        assert float(metrics["rmt/token/read_mem_token_cosine_mean"]) == pytest.approx(0.0)
        assert float(metrics["rmt/token/read_mem_token_cosine_max_mean"]) == pytest.approx(0.0)
        assert float(metrics["rmt/token/read_mem_token_cosine_min_mean"]) == pytest.approx(0.0)
        assert float(metrics["rmt/token/read_mem_token_cosine_gt_0p8_ratio_mean"]) == pytest.approx(
            0.0
        )
        assert float(metrics["rmt/token/write_mem_token_cosine_mean"]) == pytest.approx(1.0 / 3.0)
        assert float(metrics["rmt/token/write_mem_token_cosine_max_mean"]) == pytest.approx(
            1.0 / 3.0
        )
        assert float(metrics["rmt/token/write_mem_token_cosine_min_mean"]) == pytest.approx(
            1.0 / 3.0
        )
        assert float(metrics["rmt/token/write_mem_token_cosine_gt_0p8_ratio_mean"]) == pytest.approx(
            1.0 / 3.0
        )

    def test_rmt_model_mem_token_cosine_metrics_skip_first_chunk_read_when_prefix_is_disabled(self):
        config = _make_config(hidden_size=2)

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=2,
            )

        model.reset_all_monitoring_stats()
        model.set_current_chunk_is_first(True)
        model.set_skip_read_memory_for_current_chunk(True)
        hidden_states = torch.tensor(
            [
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        )
        model._update_mem_token_cosine_monitoring_stats_from_hidden_states(hidden_states, seq_len=1)
        metrics = model.consume_all_monitoring_metrics()

        assert "rmt/token/read_mem_token_cosine_mean_first_chunk" not in metrics
        assert "rmt/token/read_mem_token_cosine_max_mean_first_chunk" not in metrics
        assert "rmt/token/read_mem_token_cosine_min_mean_first_chunk" not in metrics
        assert "rmt/token/read_mem_token_cosine_gt_0p8_ratio_mean_first_chunk" not in metrics
        assert float(metrics["rmt/token/write_mem_token_cosine_mean"]) == pytest.approx(1.0)
        assert float(metrics["rmt/token/write_mem_token_cosine_max_mean"]) == pytest.approx(1.0)
        assert float(metrics["rmt/token/write_mem_token_cosine_min_mean"]) == pytest.approx(1.0)
        assert float(metrics["rmt/token/write_mem_token_cosine_gt_0p8_ratio_mean"]) == pytest.approx(
            1.0
        )

    def test_rmt_model_mem_token_cosine_metrics_skip_single_mem_token(self):
        config = _make_config(hidden_size=2)

        with patch("megatron.core.models.rmt.rmt_model.GPTModel.__init__", new=_minimal_gpt_init):
            model = RMTModel(
                config=config,
                transformer_layer_spec=MagicMock(),
                vocab_size=32000,
                max_sequence_length=2048,
                num_mem_tokens=1,
            )

        model.reset_all_monitoring_stats()
        model.set_current_chunk_is_first(False)
        model.set_skip_read_memory_for_current_chunk(False)
        hidden_states = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 0.0]],
            ]
        )
        model._update_mem_token_cosine_monitoring_stats_from_hidden_states(hidden_states, seq_len=1)
        metrics = model.consume_all_monitoring_metrics()

        assert "rmt/token/read_mem_token_cosine_mean" not in metrics
        assert "rmt/token/read_mem_token_cosine_max_mean" not in metrics
        assert "rmt/token/read_mem_token_cosine_min_mean" not in metrics
        assert "rmt/token/read_mem_token_cosine_gt_0p8_ratio_mean" not in metrics
        assert "rmt/token/write_mem_token_cosine_mean" not in metrics
        assert "rmt/token/write_mem_token_cosine_max_mean" not in metrics
        assert "rmt/token/write_mem_token_cosine_min_mean" not in metrics
        assert "rmt/token/write_mem_token_cosine_gt_0p8_ratio_mean" not in metrics

    def test_rmt_model_rope_extension(self):
        config = _make_config(hidden_size=256)

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
