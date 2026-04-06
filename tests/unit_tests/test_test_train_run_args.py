import pytest

from megatron.training.arguments import parse_args, validate_args


def _build_base_args(**overrides):
    args = parse_args(ignore_unknown_args=True)
    args.num_layers = 4
    args.num_attention_heads = 4
    args.hidden_size = 16
    args.max_position_embeddings = 16
    args.seq_length = 8
    args.micro_batch_size = 1

    for key, value in overrides.items():
        setattr(args, key, value)

    return args


def test_validate_args_disables_artifact_outputs_for_test_train_run():
    args = validate_args(
        _build_base_args(
            test_train_run=True,
            tensorboard_dir="/tmp/tensorboard",
            save="/tmp/checkpoints",
            save_interval=10,
            save_retain_interval=20,
            keep_last_n_checkpoints=2,
            save_wgrads_interval=4,
            save_dgrads_interval=5,
            async_save=True,
            use_persistent_ckpt_worker=True,
            log_progress=True,
            config_logger_dir="/tmp/config_logger",
            wandb_project="proj",
            wandb_exp_name="exp",
            wandb_save_dir="/tmp/wandb",
            wandb_entity="entity",
            enable_one_logger=True,
            non_persistent_ckpt_type="global",
            non_persistent_save_interval=3,
            non_persistent_global_ckpt_dir="/tmp/non_persistent",
            replication=True,
            replication_jump=2,
        )
    )

    assert args.tensorboard_dir is None
    assert args.save is None
    assert args.save_interval is None
    assert args.save_retain_interval is None
    assert args.keep_last_n_checkpoints is None
    assert args.save_wgrads_interval is None
    assert args.save_dgrads_interval is None
    assert args.async_save is False
    assert args.use_persistent_ckpt_worker is False
    assert args.log_progress is False
    assert args.config_logger_dir == ""
    assert args.wandb_project is None
    assert args.wandb_exp_name is None
    assert args.wandb_save_dir is None
    assert args.wandb_entity is None
    assert args.enable_one_logger is False
    assert args.non_persistent_ckpt_type is None
    assert args.non_persistent_save_interval is None
    assert args.non_persistent_global_ckpt_dir is None
    assert args.replication is False
    assert args.replication_jump is None


def test_validate_args_keeps_tensorboard_dir_for_pytorch_profiler_in_test_train_run():
    args = validate_args(
        _build_base_args(
            test_train_run=True,
            profile=True,
            use_pytorch_profiler=True,
            tensorboard_dir="/tmp/tensorboard",
            save="/tmp/checkpoints",
            save_interval=10,
            log_progress=True,
            config_logger_dir="/tmp/config_logger",
            wandb_project="proj",
        )
    )

    assert args.tensorboard_dir == "/tmp/tensorboard"
    assert args.disable_tensorboard_writer is True
    assert args.save is None
    assert args.save_interval is None
    assert args.log_progress is False
    assert args.config_logger_dir == ""
    assert args.wandb_project is None


def test_validate_args_keeps_artifact_outputs_when_test_train_run_disabled():
    args = validate_args(
        _build_base_args(
            test_train_run=False,
            tensorboard_dir="/tmp/tensorboard",
            save="/tmp/checkpoints",
            save_interval=10,
            log_progress=True,
            config_logger_dir="/tmp/config_logger",
            wandb_project="proj",
            wandb_exp_name="exp",
            wandb_save_dir="/tmp/wandb",
            wandb_entity="entity",
            enable_one_logger=True,
            async_save=True,
            use_persistent_ckpt_worker=True,
            non_persistent_ckpt_type="global",
            non_persistent_save_interval=3,
            non_persistent_global_ckpt_dir="/tmp/non_persistent",
        )
    )

    assert args.tensorboard_dir == "/tmp/tensorboard"
    assert args.save == "/tmp/checkpoints"
    assert args.save_interval == 10
    assert args.log_progress is True
    assert args.config_logger_dir == "/tmp/config_logger"
    assert args.wandb_project == "proj"
    assert args.wandb_exp_name == "exp"
    assert args.wandb_save_dir == "/tmp/wandb"
    assert args.wandb_entity == "entity"
    assert args.enable_one_logger is True
    assert args.async_save is True
    assert args.use_persistent_ckpt_worker is True
    assert args.non_persistent_ckpt_type == "global"
    assert args.non_persistent_save_interval == 3
    assert args.non_persistent_global_ckpt_dir == "/tmp/non_persistent"
    assert not getattr(args, "disable_tensorboard_writer", False)


def test_validate_args_rejects_checkpoint_conversion_for_test_train_run():
    with pytest.raises(
        AssertionError,
        match="incompatible with --ckpt-convert-format",
    ):
        validate_args(
            _build_base_args(
                test_train_run=True,
                ckpt_convert_format="torch_dist",
            )
        )


def test_validate_args_disables_artifact_outputs_for_param_stats_only():
    args = validate_args(
        _build_base_args(
            param_stats_only=True,
            tensorboard_dir="/tmp/tensorboard",
            save="/tmp/checkpoints",
            save_interval=10,
            save_retain_interval=20,
            keep_last_n_checkpoints=2,
            save_wgrads_interval=4,
            save_dgrads_interval=5,
            async_save=True,
            use_persistent_ckpt_worker=True,
            log_progress=True,
            config_logger_dir="/tmp/config_logger",
            wandb_project="proj",
            wandb_exp_name="exp",
            wandb_save_dir="/tmp/wandb",
            wandb_entity="entity",
            enable_one_logger=True,
            non_persistent_ckpt_type="global",
            non_persistent_save_interval=3,
            non_persistent_global_ckpt_dir="/tmp/non_persistent",
            replication=True,
            replication_jump=2,
        )
    )

    assert args.tensorboard_dir is None
    assert args.save is None
    assert args.save_interval is None
    assert args.save_retain_interval is None
    assert args.keep_last_n_checkpoints is None
    assert args.save_wgrads_interval is None
    assert args.save_dgrads_interval is None
    assert args.async_save is False
    assert args.use_persistent_ckpt_worker is False
    assert args.log_progress is False
    assert args.config_logger_dir == ""
    assert args.wandb_project is None
    assert args.wandb_exp_name is None
    assert args.wandb_save_dir is None
    assert args.wandb_entity is None
    assert args.enable_one_logger is False
    assert args.non_persistent_ckpt_type is None
    assert args.non_persistent_save_interval is None
    assert args.non_persistent_global_ckpt_dir is None
    assert args.replication is False
    assert args.replication_jump is None


def test_validate_args_rejects_checkpoint_conversion_for_param_stats_only():
    with pytest.raises(
        AssertionError,
        match="incompatible with --ckpt-convert-format",
    ):
        validate_args(
            _build_base_args(
                param_stats_only=True,
                ckpt_convert_format="torch_dist",
            )
        )
