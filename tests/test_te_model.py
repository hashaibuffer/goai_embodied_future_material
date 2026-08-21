from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.distillation.te_dataset import InputStatistics  # noqa: E402
from training.distillation.export_student_onnx import (  # noqa: E402
    portable_artifact_path, portable_training_metadata)
from training.distillation.te_model import StudentConfig, StudentPolicy  # noqa: E402


def statistics():
    mask = np.zeros(441, np.bool_)
    mask[:249] = True
    mask[6:9] = False
    return InputStatistics(
        mean=np.zeros(441, np.float32),
        std=np.ones(441, np.float32),
        normalized_mask=mask,
    )


@pytest.mark.parametrize("architecture", ["branch", "flat", "gated_ensemble"])
def test_learned_student_has_frozen_io_shape(architecture):
    model = StudentPolicy(
        StudentConfig(controller="learned", architecture=architecture), statistics())
    output = model(torch.zeros(7, 441))
    assert output.shape == (7, 16)
    assert torch.isfinite(output).all()


def test_proprio_clone_is_exactly_invariant_to_terrain():
    model = StudentPolicy(StudentConfig(controller="proprio_clone"), statistics()).eval()
    first = torch.randn(5, 441)
    second = first.clone()
    second[:, 57:] = torch.randn(5, 384) * 100
    with torch.no_grad():
        np.testing.assert_array_equal(model(first).numpy(), model(second).numpy())


def test_model_rejects_wrong_observation_width():
    model = StudentPolicy(StudentConfig(), statistics())
    with pytest.raises(ValueError, match="441"):
        model(torch.zeros(2, 440))


def test_constant_training_feature_is_ignored_and_extremes_are_clipped():
    stats = statistics()
    stats.std[10] = 1e-4
    model = StudentPolicy(StudentConfig(architecture="flat"), stats).eval()
    baseline = torch.zeros(1, 441)
    changed_constant = baseline.clone()
    changed_constant[:, 10] = 0.3
    positive_extreme = baseline.clone()
    positive_extreme[:, 0] = 1e6
    clipped_positive = baseline.clone()
    clipped_positive[:, 0] = 10.0
    with torch.no_grad():
        np.testing.assert_array_equal(
            model(baseline).numpy(), model(changed_constant).numpy())
        np.testing.assert_allclose(
            model(positive_extreme).numpy(), model(clipped_positive).numpy())


def test_command_slots_remain_active_raw_values():
    model = StudentPolicy(StudentConfig(architecture="flat"), statistics()).eval()
    baseline = torch.zeros(1, 441)
    command = baseline.clone()
    command[:, 6] = 0.5
    with torch.no_grad():
        assert not np.array_equal(model(baseline).numpy(), model(command).numpy())


def test_gated_ensemble_hard_selects_one_policy_without_blending():
    model = StudentPolicy(
        StudentConfig(architecture="gated_ensemble"), statistics()).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.primary_policy[-1].bias.fill_(1.0)
        model.recovery_policy[-1].bias.fill_(2.0)
        model.gate[-1].bias.fill_(-1.0)
        np.testing.assert_array_equal(
            model(torch.zeros(2, 441)).numpy(), np.ones((2, 16), np.float32))
        model.gate[-1].bias.fill_(1.0)
        np.testing.assert_array_equal(
            model(torch.zeros(2, 441)).numpy(), np.full((2, 16), 2.0, np.float32))


def test_export_metadata_does_not_publish_machine_specific_paths():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "results" / "distillation_runs" / "run" / "best.pt"
    metadata = {
        "training_manifest": root / "results" / "datasets" / "manifest.json",
        "init_checkpoint": checkpoint,
        "extra_shards": [{"path": root / "results" / "dagger.npz", "sha256": "x"}],
        "preserve_checkpoint": checkpoint,
        "preserve_shards": [
            {"path": root / "results" / "preserve.npz", "sha256": "y"}],
        "primary_checkpoint": checkpoint,
        "recovery_checkpoint": checkpoint,
        "positive_shards": [
            {"path": root / "results" / "positive.npz", "sha256": "z"}],
        "negative_shards": [
            {"path": root / "results" / "negative.npz", "sha256": "w"}],
    }
    portable = portable_training_metadata(metadata)
    assert portable_artifact_path(checkpoint) == (
        "results/distillation_runs/run/best.pt")
    assert portable["training_manifest"] == "results/datasets/manifest.json"
    assert portable["init_checkpoint"] == "results/distillation_runs/run/best.pt"
    assert portable["extra_shards"][0]["path"] == "results/dagger.npz"
    assert portable["preserve_checkpoint"] == "results/distillation_runs/run/best.pt"
    assert portable["preserve_shards"][0]["path"] == "results/preserve.npz"
    assert portable["primary_checkpoint"] == "results/distillation_runs/run/best.pt"
    assert portable["recovery_checkpoint"] == "results/distillation_runs/run/best.pt"
    assert portable["positive_shards"][0]["path"] == "results/positive.npz"
    assert portable["negative_shards"][0]["path"] == "results/negative.npz"
    assert str(root) not in str(portable)


def test_export_metadata_preserves_empty_optional_checkpoint_paths():
    portable = portable_training_metadata({
        "init_checkpoint": None,
        "preserve_checkpoint": None,
    })
    assert portable == {"init_checkpoint": None, "preserve_checkpoint": None}
