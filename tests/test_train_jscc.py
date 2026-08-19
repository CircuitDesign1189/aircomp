import pytest
import torch

from airComp.config import ITEM_TYPES, JSCCConfig, TrainConfig
from airComp.env.negotiation import Pool
from airComp.jscc.dataset import JsccExample, save_dataset
from airComp.jscc.train_jscc import train

INPUT_DIM = 8


def _synthetic_examples(n: int = 12, with_embed_target: bool = False) -> list:
    pool = Pool(counts={t: 4 for t in ITEM_TYPES})
    return [
        JsccExample(
            hidden=torch.randn(INPUT_DIM),
            action_idx=i % 3,
            counts={t: (i + j) % 3 for j, t in enumerate(ITEM_TYPES)},
            aux=float(i % 2),
            pool=pool,
            values={t: float(j + 1) for j, t in enumerate(ITEM_TYPES)},
            embed_target=torch.randn(INPUT_DIM) if with_embed_target else None,
        )
        for i in range(n)
    ]


def test_phase1_unaffected_by_default_utility_weight(tmp_path):
    """utility_loss_weight=0.0 is the default -- Phase 1 behavior must not change."""
    dataset_path = tmp_path / "ds.pt"
    save_dataset(_synthetic_examples(), str(dataset_path))

    result = train(
        str(dataset_path), str(tmp_path / "out.pt"), input_dim=INPUT_DIM,
        jscc_cfg=JSCCConfig(k=8), train_cfg=TrainConfig(epochs=1, batch_size=4),
    )

    assert len(result["loss_history"]) == 1
    assert (tmp_path / "out.pt").exists()


def test_train_runs_with_utility_loss_enabled(tmp_path):
    dataset_path = tmp_path / "ds.pt"
    save_dataset(_synthetic_examples(), str(dataset_path))

    result = train(
        str(dataset_path), str(tmp_path / "out.pt"), input_dim=INPUT_DIM,
        jscc_cfg=JSCCConfig(k=8), train_cfg=TrainConfig(epochs=1, batch_size=4, utility_loss_weight=0.5),
    )

    assert len(result["loss_history"]) == 1
    assert (tmp_path / "out.pt").exists()


def test_train_continues_from_init_checkpoint(tmp_path):
    """The Phase 2 workflow: fine-tune a Phase-1 checkpoint rather than starting fresh."""
    dataset_path = tmp_path / "ds.pt"
    save_dataset(_synthetic_examples(), str(dataset_path))
    jscc_cfg = JSCCConfig(k=8)

    phase1_ckpt = tmp_path / "phase1.pt"
    train(str(dataset_path), str(phase1_ckpt), input_dim=INPUT_DIM, jscc_cfg=jscc_cfg,
          train_cfg=TrainConfig(epochs=1, batch_size=4))

    phase2_ckpt = tmp_path / "phase2.pt"
    result = train(
        str(dataset_path), str(phase2_ckpt), input_dim=INPUT_DIM, jscc_cfg=jscc_cfg,
        train_cfg=TrainConfig(epochs=1, batch_size=4, utility_loss_weight=1.0),
        init_checkpoint=str(phase1_ckpt),
    )

    assert phase2_ckpt.exists()
    assert len(result["loss_history"]) == 1


def test_train_runs_with_embed_loss_enabled(tmp_path):
    dataset_path = tmp_path / "ds.pt"
    save_dataset(_synthetic_examples(with_embed_target=True), str(dataset_path))

    result = train(
        str(dataset_path), str(tmp_path / "out.pt"), input_dim=INPUT_DIM,
        jscc_cfg=JSCCConfig(k=8, embed_dim=INPUT_DIM),
        train_cfg=TrainConfig(epochs=1, batch_size=4, embed_loss_weight=0.5),
    )

    assert len(result["loss_history"]) == 1
    ckpt = torch.load(tmp_path / "out.pt", weights_only=False)
    assert any(name.startswith("embed_head") for name in ckpt["decoder"])


def test_train_rejects_embed_dim_when_dataset_lacks_embed_target(tmp_path):
    dataset_path = tmp_path / "ds.pt"
    save_dataset(_synthetic_examples(with_embed_target=False), str(dataset_path))

    with pytest.raises(ValueError, match="embed_target"):
        train(
            str(dataset_path), str(tmp_path / "out.pt"), input_dim=INPUT_DIM,
            jscc_cfg=JSCCConfig(k=8, embed_dim=INPUT_DIM),
            train_cfg=TrainConfig(epochs=1, batch_size=4, embed_loss_weight=0.5),
        )
