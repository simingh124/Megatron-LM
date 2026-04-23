import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cross_attn_launcher_only_uses_cross_attn_memory_hyperparameters():
    script = (
        REPO_ROOT
        / "playground"
        / "rmt"
        / "qwen3_0p6b_armt_cross_attn_0324_fs_wo_tbptt.sh"
    ).read_text()

    assert "ARMT_N_HEADS=" not in script
    assert "--armt-n-heads" not in script
    assert "RECURRENT_SLOT_NUM_HEADS=" in script
    assert "${RECURRENT_SLOT_NUM_HEADS}" in script
    assert "RECURRENT_SLOT_READ_ATTN_BACKEND=${RECURRENT_SLOT_READ_ATTN_BACKEND:-flash}" in script
    assert "--recurrent-slot-read-attn-backend ${RECURRENT_SLOT_READ_ATTN_BACKEND}" in script


def test_cross_attn_w_norm_launcher_has_norm_defaults_and_resume_switch():
    script = (
        REPO_ROOT
        / "playground"
        / "rmt"
        / "qwen3_0p6b_armt_cross_attn_0324_fs_wo_tbptt_w_norm.sh"
    ).read_text()

    assert "RECURRENT_MEM_QK_NORM=${RECURRENT_MEM_QK_NORM:-1}" in script
    assert "RECURRENT_MEMORY_INPUT_PRE_NORM=${RECURRENT_MEMORY_INPUT_PRE_NORM:-1}" in script
    assert "--recurrent-mem-qk-norm" in script
    assert "--recurrent-memory-input-pre-norm" in script
    assert 'if [[ "${ENABLE_RESUME}" == "1" ]]; then' in script
    assert 'CKPT_AND_LOG_ARGS+=(--load "${CHECKPOINT_PATH}")' in script


def test_all_playground_launchers_expose_resume_switch():
    launcher_paths = sorted((REPO_ROOT / "playground").rglob("*.sh"))

    assert launcher_paths

    for path in launcher_paths:
        script = path.read_text()

        assert "ENABLE_RESUME=${ENABLE_RESUME:-0}" in script, path
        assert "latest_checkpointed_iteration.txt" in script, path
        assert 'CKPT_AND_LOG_ARGS+=(--load "${CHECKPOINT_PATH}")' in script, path
        assert 'elif [[ "${ENABLE_RESUME}" == "1" ]]; then' not in script, path

        if re.search(r"^\\s*LOAD_CHECKPOINT_PATH=", script, re.M):
            assert 'CKPT_AND_LOG_ARGS+=(--load "${LOAD_CHECKPOINT_PATH}")' in script, path
            assert "CKPT_AND_LOG_ARGS+=(--no-load-optim)" in script, path
            assert "CKPT_AND_LOG_ARGS+=(--no-load-rng)" in script, path
