import pytest
from argparse import Namespace

from examples.armt.armt_args import validate_armt_constraints


def test_constraint_pp_must_be_1():
    """约束校验：PP>1 必须报错（ARMT v1 仅支持 PP=1）。"""
    args = Namespace(
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        fp8=None,
        fp8_recipe=None,
        use_armt_tbptt=True,
        position_embedding_type="rope",
        seq_length=2048,
        armt_chunk_size=512,
        num_mem_tokens=16,
        sequence_parallel=False,
    )
    with pytest.raises(ValueError, match="PP=1"):
        validate_armt_constraints(args)


def test_constraint_cp_must_be_1():
    """约束校验：CP>1 必须报错（ARMT v1 禁用 CP）。"""
    args = Namespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        tensor_model_parallel_size=1,
        fp8=None,
        fp8_recipe=None,
        use_armt_tbptt=True,
        position_embedding_type="rope",
        seq_length=2048,
        armt_chunk_size=512,
        num_mem_tokens=16,
        sequence_parallel=False,
    )
    with pytest.raises(ValueError, match="CP=1"):
        validate_armt_constraints(args)


def test_constraint_fp8_forbidden():
    """约束校验：开启 FP8 必须报错（v1 明确禁止 FP8）。"""
    args = Namespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        fp8="e4m3",
        fp8_recipe=None,
        use_armt_tbptt=True,
        position_embedding_type="rope",
        seq_length=2048,
        armt_chunk_size=512,
        num_mem_tokens=16,
        sequence_parallel=False,
    )
    with pytest.raises(ValueError, match="FP8"):
        validate_armt_constraints(args)


def test_constraint_chunk_size_divisibility():
    """约束校验：SP safety-mode 下要求 (chunk_size + M) % TP == 0，否则报错。"""
    args = Namespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tensor_model_parallel_size=2,
        fp8=None,
        fp8_recipe=None,
        use_armt_tbptt=True,
        position_embedding_type="rope",
        seq_length=2048,
        armt_chunk_size=511,
        num_mem_tokens=16,
        sequence_parallel=True,
    )
    with pytest.raises(ValueError, match="armt_chunk_size"):
        validate_armt_constraints(args)


def test_valid_config_passes():
    """约束校验：合法配置应通过（不抛异常）。"""
    args = Namespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        tensor_model_parallel_size=2,
        fp8=None,
        fp8_recipe=None,
        use_armt_tbptt=True,
        position_embedding_type="rope",
        seq_length=2048,
        armt_chunk_size=512,
        num_mem_tokens=16,
        sequence_parallel=True,
    )
    validate_armt_constraints(args)
