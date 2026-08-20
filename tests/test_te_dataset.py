from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "s10_terrain_perception"))
from d_priv_dataset import DPrivRecorder  # noqa: E402

from training.distillation.te_dataset import (  # noqa: E402
    ACTION_RAW_LIMIT,
    compute_input_statistics,
    validate_te_shard,
)


TEACHER_SHA = "a" * 64


def write_shard(tmp_path, *, source="privileged_mujoco", labels=True,
                action=0.5, command_mismatch=False):
    recorder = DPrivRecorder(
        teacher_model=None,
        teacher_source=source,
        metadata={
            "labels_usable": labels,
            "teacher_sha256": TEACHER_SHA,
            "terrain_id": "flat",
        },
    )
    for step in range(4):
        observation = np.zeros(441, np.float32)
        observation[249:] = 1.0
        command = np.asarray([0.6, 0.0, 0.0], np.float32)
        observation[6:9] = command
        if command_mismatch:
            observation[6] = 0.5
        recorder.append(
            student_obs=observation,
            teacher_action_raw=np.full(16, action, np.float32),
            command_raw=command,
            waypoint_id=0,
            terrain_id="flat",
            episode_id=1,
            step_id=step,
            base_pose_wxyz=[0, 0, .4, 1, 0, 0, 0],
            privileged_hit_fraction=1.0,
        )
    return recorder.write(tmp_path / "shard.npz")


def test_strict_te_shard_accepts_real_raw_labels(tmp_path):
    path = write_shard(tmp_path)
    report = validate_te_shard(path, expected_teacher_sha256=TEACHER_SHA)
    assert report.samples == 4
    assert report.terrain == "flat"
    assert report.action_abs_max == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"source": "fake_mujoco_smoke"}, "teacher sources"),
        ({"labels": False}, "labels_usable"),
        ({"action": ACTION_RAW_LIMIT + 1}, "exceeds"),
        ({"command_mismatch": True}, "command slot"),
    ],
)
def test_strict_te_shard_rejects_unusable_data(tmp_path, kwargs, match):
    path = write_shard(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        validate_te_shard(path, expected_teacher_sha256=TEACHER_SHA)


def test_input_statistics_leave_validity_binary_space_unchanged():
    observations = np.zeros((8, 441), np.float32)
    observations[:, 0] = np.arange(8)
    observations[:, 57:249] = np.linspace(-0.5, 1.0, 192)
    observations[::2, 249:] = 1.0
    stats = compute_input_statistics(observations)
    np.testing.assert_array_equal(stats.mean[6:9], 0.0)
    np.testing.assert_array_equal(stats.std[6:9], 1.0)
    np.testing.assert_array_equal(stats.mean[249:], 0.0)
    np.testing.assert_array_equal(stats.std[249:], 1.0)
    assert stats.normalized_mask[:6].all()
    assert not stats.normalized_mask[6:9].any()
    assert stats.normalized_mask[9:249].all()
    assert not stats.normalized_mask[249:].any()
