"""Versioned, pickle-free D_priv writer for the 441->16 student contract."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DPrivRecorder:
    def __init__(self, *, teacher_model, metadata=None, teacher_source="privileged_mujoco"):
        self.teacher_model = str(Path(teacher_model).resolve()) if teacher_model else "fake_policy"
        self.teacher_sha256 = sha256_file(teacher_model) if teacher_model else "fake_policy"
        self.metadata = dict(metadata or {})
        self.teacher_source = str(teacher_source)
        self.rows = {name: [] for name in (
            "student_obs", "teacher_action_norm", "command_raw", "waypoint_id",
            "terrain_id", "episode_id", "step_id", "base_pose_wxyz", "privileged_hit_fraction")}

    def append(self, *, student_obs, teacher_action_norm, command_raw, waypoint_id,
               terrain_id, episode_id, step_id, base_pose_wxyz, privileged_hit_fraction):
        obs = np.asarray(student_obs, np.float32).reshape(441)
        action = np.asarray(teacher_action_norm, np.float32).reshape(16)
        command = np.asarray(command_raw, np.float32).reshape(3)
        if not np.isfinite(obs).all() or not np.isfinite(action).all():
            raise ValueError("D_priv observation/action contains non-finite values")
        self.rows["student_obs"].append(obs)
        self.rows["teacher_action_norm"].append(action)
        self.rows["command_raw"].append(command)
        self.rows["waypoint_id"].append(int(waypoint_id))
        self.rows["terrain_id"].append(str(terrain_id))
        self.rows["episode_id"].append(int(episode_id))
        self.rows["step_id"].append(int(step_id))
        self.rows["base_pose_wxyz"].append(np.asarray(base_pose_wxyz, np.float32).reshape(7))
        self.rows["privileged_hit_fraction"].append(float(privileged_hit_fraction))

    def write(self, path):
        count = len(self.rows["student_obs"])
        if count == 0:
            raise ValueError("refusing to write an empty D_priv shard")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "teacher_model": self.teacher_model,
            "teacher_sha256": self.teacher_sha256,
            **self.metadata,
        }
        np.savez_compressed(
            output,
            schema_version=np.asarray(SCHEMA_VERSION, np.int32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            teacher_source=np.full(count, self.teacher_source),
            student_obs=np.asarray(self.rows["student_obs"], np.float32),
            teacher_action_norm=np.asarray(self.rows["teacher_action_norm"], np.float32),
            command_raw=np.asarray(self.rows["command_raw"], np.float32),
            waypoint_id=np.asarray(self.rows["waypoint_id"], np.int32),
            terrain_id=np.asarray(self.rows["terrain_id"], np.str_),
            episode_id=np.asarray(self.rows["episode_id"], np.int32),
            step_id=np.asarray(self.rows["step_id"], np.int32),
            base_pose_wxyz=np.asarray(self.rows["base_pose_wxyz"], np.float32),
            privileged_hit_fraction=np.asarray(self.rows["privileged_hit_fraction"], np.float32),
        )
        # np.savez_compressed silently appends .npz when the path lacks that suffix;
        # return the path that was actually written so callers can find the file.
        if not output.suffix == ".npz":
            output = output.with_suffix(output.suffix + ".npz")
        return output


def validate_d_priv(path):
    with np.load(path, allow_pickle=False) as data:
        count = len(data["student_obs"])
        expected = {
            "student_obs": (count, 441), "teacher_action_norm": (count, 16),
            "command_raw": (count, 3), "base_pose_wxyz": (count, 7),
        }
        for name, shape in expected.items():
            if data[name].shape != shape or data[name].dtype != np.float32:
                raise ValueError(f"{name}: expected float32 {shape}, got {data[name].dtype} {data[name].shape}")
        if int(data["schema_version"]) != SCHEMA_VERSION:
            raise ValueError("unsupported D_priv schema")
        return count
