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
