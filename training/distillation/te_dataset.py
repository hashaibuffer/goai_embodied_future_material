"""Strict dataset loading for TE privileged-policy distillation.

Only schema-v2 shards produced by the privileged MuJoCo teacher are accepted.
The older TD datasets use a different command/teacher contract and must never be
silently mixed into TE training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


OBS_DIM = 441
ACTION_DIM = 16
PROPRIO_DIM = 57
HEIGHT_DIM = 192
VALIDITY_DIM = 192
SCHEMA_VERSION = 2
ACTION_RAW_LIMIT = 100.0
REQUIRED_FIELDS = {
    "schema_version",
    "metadata_json",
    "teacher_source",
    "student_obs",
    "teacher_action_raw",
    "command_raw",
    "waypoint_id",
    "terrain_id",
    "episode_id",
    "step_id",
    "base_pose_wxyz",
    "privileged_hit_fraction",
}


@dataclass(frozen=True)
class ShardSpec:
    path: Path
    expected_sha256: str | None = None
    split_group: str | None = None


@dataclass(frozen=True)
class ShardReport:
    path: Path
    samples: int
    terrain: str
    episodes: tuple[int, ...]
    action_abs_max: float
    teacher_sha256: str


@dataclass
class LoadedDataset:
    observations: np.ndarray
    actions: np.ndarray
    terrains: np.ndarray
    episode_ids: np.ndarray
    step_ids: np.ndarray
    shard_ids: np.ndarray
    reports: list[ShardReport]

    def __post_init__(self) -> None:
        count = len(self.observations)
        if self.observations.shape != (count, OBS_DIM):
            raise ValueError(f"observations must be [N,{OBS_DIM}]")
        if self.actions.shape != (count, ACTION_DIM):
            raise ValueError(f"actions must be [N,{ACTION_DIM}]")
        for values in (self.terrains, self.episode_ids, self.step_ids, self.shard_ids):
            if len(values) != count:
                raise ValueError("dataset metadata length mismatch")

    def __len__(self) -> int:
        return len(self.observations)


@dataclass(frozen=True)
class InputStatistics:
    mean: np.ndarray
    std: np.ndarray
    normalized_mask: np.ndarray

    def __post_init__(self) -> None:
        for name, values in (("mean", self.mean), ("std", self.std),
                             ("normalized_mask", self.normalized_mask)):
            if np.asarray(values).shape != (OBS_DIM,):
                raise ValueError(f"{name} must have shape ({OBS_DIM},)")
        if np.any(self.std <= 0) or not np.isfinite(self.std).all():
            raise ValueError("normalization std must be finite and positive")

    def to_json(self) -> dict:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "normalized_mask": self.normalized_mask.astype(bool).tolist(),
        }

    @classmethod
    def from_json(cls, value: Mapping) -> "InputStatistics":
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float32),
            std=np.asarray(value["std"], dtype=np.float32),
            normalized_mask=np.asarray(value["normalized_mask"], dtype=np.bool_),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(data: Mapping) -> dict:
    try:
        value = json.loads(str(data["metadata_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid metadata_json") from exc
    if not isinstance(value, dict):
        raise ValueError("metadata_json must contain an object")
    return value


def validate_te_shard(
    path: Path,
    *,
    expected_teacher_sha256: str,
    expected_sha256: str | None = None,
) -> ShardReport:
    """Validate one shard against the frozen TE raw-action contract."""
    path = Path(path)
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{path}: shard SHA mismatch: {actual_sha256} != {expected_sha256}")

    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_FIELDS - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        if "teacher_obs" in data.files or "teacher_action_norm" in data.files:
            raise ValueError(f"{path}: forbidden legacy/privileged training field present")
        if int(data["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"{path}: expected schema v{SCHEMA_VERSION}")

        observations = data["student_obs"]
        actions = data["teacher_action_raw"]
        commands = data["command_raw"]
        count = len(observations)
        expected = {
            "student_obs": (np.dtype(np.float32), (count, OBS_DIM)),
            "teacher_action_raw": (np.dtype(np.float32), (count, ACTION_DIM)),
            "command_raw": (np.dtype(np.float32), (count, 3)),
            "base_pose_wxyz": (np.dtype(np.float32), (count, 7)),
            "privileged_hit_fraction": (np.dtype(np.float32), (count,)),
        }
        if count == 0:
            raise ValueError(f"{path}: empty shard")
        for name, (dtype, shape) in expected.items():
            values = data[name]
            if values.dtype != dtype or values.shape != shape:
                raise ValueError(
                    f"{path}: {name} must be {dtype} {shape}, got "
                    f"{values.dtype} {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: {name} contains non-finite values")

        metadata = _metadata(data)
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: metadata schema mismatch")
        if metadata.get("labels_usable") is not True:
            raise ValueError(f"{path}: labels_usable is not true")
        if metadata.get("teacher_sha256") != expected_teacher_sha256:
            raise ValueError(f"{path}: privileged teacher SHA mismatch")
        sources = set(data["teacher_source"].tolist())
        if sources != {"privileged_mujoco"}:
            raise ValueError(f"{path}: invalid teacher sources {sorted(sources)}")

        action_abs_max = float(np.max(np.abs(actions)))
        if action_abs_max > ACTION_RAW_LIMIT + 1e-3:
            raise ValueError(
                f"{path}: raw teacher action exceeds +/-{ACTION_RAW_LIMIT}: "
                f"{action_abs_max:.6g}")
        if not np.allclose(observations[:, 6:9], commands, atol=1e-6, rtol=0.0):
            raise ValueError(f"{path}: student command slot is not command_raw")

        validity = observations[:, PROPRIO_DIM + HEIGHT_DIM:]
        if np.any((validity != 0.0) & (validity != 1.0)):
            raise ValueError(f"{path}: validity channel is not binary")
        terrains = set(data["terrain_id"].tolist())
        if len(terrains) != 1 or not next(iter(terrains)):
            raise ValueError(f"{path}: shard must contain one non-empty terrain_id")
        terrain = str(next(iter(terrains)))
        metadata_terrain = metadata.get("terrain_id")
        if metadata_terrain is not None and str(metadata_terrain) != terrain:
            raise ValueError(f"{path}: terrain_id does not match metadata")

        return ShardReport(
            path=path,
            samples=count,
            terrain=terrain,
            episodes=tuple(sorted(set(map(int, data["episode_id"].tolist())))),
            action_abs_max=action_abs_max,
            teacher_sha256=expected_teacher_sha256,
        )


def accepted_training_specs(manifest_path: Path) -> tuple[list[ShardSpec], str]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("training manifest is not schema v2")
    if manifest.get("action_field") != "teacher_action_raw":
        raise ValueError("training manifest does not use teacher_action_raw")
    teacher_sha256 = str(manifest["teacher_onnx_sha256"])
    root = manifest_path.parent
    specs = [
        ShardSpec(
            path=root / item["path"],
            expected_sha256=item["sha256"],
            split_group=item.get("split_group"),
        )
        for item in manifest["accepted"]
    ]
    if len(specs) != int(manifest["accepted_shards"]):
        raise ValueError("accepted shard count does not match manifest")
    if len({spec.path for spec in specs}) != len(specs):
        raise ValueError("training manifest contains duplicate paths")
    return specs, teacher_sha256


def evaluation_specs(
    dataset_root: Path,
    *,
    groups: Sequence[str],
    excluded_paths: Iterable[str] = (),
) -> list[ShardSpec]:
    dataset_root = Path(dataset_root)
    excluded = {Path(value).as_posix() for value in excluded_paths}
    specs = []
    for group in groups:
        for path in sorted((dataset_root / group).rglob("*.npz")):
            relative = path.relative_to(dataset_root).as_posix()
            if relative not in excluded:
                specs.append(ShardSpec(path=path))
    if not specs:
        raise ValueError(f"no evaluation shards found under {groups}")
    return specs


def qc_excluded_paths(qc_manifest_path: Path) -> set[str]:
    manifest = json.loads(Path(qc_manifest_path).read_text(encoding="utf-8"))
    return {
        Path(item["path"]).as_posix()
        for item in manifest.get("excluded_from_default_training", [])
    }


def load_shards(
    specs: Sequence[ShardSpec],
    *,
    expected_teacher_sha256: str,
) -> LoadedDataset:
    observations, actions = [], []
    terrains, episode_ids, step_ids, shard_ids = [], [], [], []
    reports = []
    for shard_index, spec in enumerate(specs):
        report = validate_te_shard(
            spec.path,
            expected_teacher_sha256=expected_teacher_sha256,
            expected_sha256=spec.expected_sha256,
        )
        reports.append(report)
        with np.load(spec.path, allow_pickle=False) as data:
            count = report.samples
            observations.append(data["student_obs"].astype(np.float32, copy=False))
            actions.append(data["teacher_action_raw"].astype(np.float32, copy=False))
            terrains.append(data["terrain_id"].astype(np.str_, copy=False))
            episode_ids.append(data["episode_id"].astype(np.int32, copy=False))
            step_ids.append(data["step_id"].astype(np.int32, copy=False))
            shard_ids.append(np.full(count, shard_index, dtype=np.int32))
    loaded = LoadedDataset(
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        terrains=np.concatenate(terrains),
        episode_ids=np.concatenate(episode_ids),
        step_ids=np.concatenate(step_ids),
        shard_ids=np.concatenate(shard_ids),
        reports=reports,
    )
    return loaded


def compute_input_statistics(observations: np.ndarray) -> InputStatistics:
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != OBS_DIM:
        raise ValueError(f"observations must be [N,{OBS_DIM}]")
    # Normalize sensor and height values. Commands are already bounded in
    # physical policy space, so keep them raw; a constant command in one data
    # collection must not become an unusable or 3000-sigma feature later.
    # Keep binary validity in its original 0/1 representation as well.
    mask = np.zeros(OBS_DIM, dtype=np.bool_)
    mask[:PROPRIO_DIM + HEIGHT_DIM] = True
    mask[6:9] = False
    mean = np.zeros(OBS_DIM, dtype=np.float32)
    std = np.ones(OBS_DIM, dtype=np.float32)
    mean[mask] = observations[:, mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    measured = observations[:, mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std[mask] = np.maximum(measured, np.float32(1e-4))
    return InputStatistics(mean=mean, std=std, normalized_mask=mask)
