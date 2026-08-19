"""Portable, pickle-free NPZ schema for TD teacher datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
ARRAY_FIELDS = {
    "obs_student": (np.float32, (441,)),
    "obs_teacher": (np.float32, (57,)),
    "heightmap": (np.float32, (2, 16, 12)),
    "action_teacher": (np.float32, (16,)),
    "cmd_raw": (np.float32, (3,)),
    "cmd_terrain": (np.float32, (3,)),
    "risk_features": (np.float32, (8,)),
    "pose": (np.float32, (7,)),
}
SCALAR_FIELDS = {
    "timestamp_ns": np.int64,
    "sequence": np.uint64,
    "episode_id": np.uint32,
    "wp_id": np.int32,
    "next_wp_id": np.int32,
    "teacher_source": np.uint8,
    "failure_code": np.uint8,
    "success": np.bool_,
    "pre_failure": np.bool_,
    "focus_segment": np.bool_,
    "post_teleport": np.bool_,
    "contrast_label": np.uint8,
    "heightmap_valid": np.bool_,
    "heightmap_age_ms": np.float32,
}


def validate_record(record):
    for name, (dtype, shape) in ARRAY_FIELDS.items():
        value = np.asarray(record[name], dtype=dtype)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
    if not np.array_equal(
        np.asarray(record["obs_student"], dtype=np.float32)[6:9],
        np.clip(np.asarray(record["cmd_raw"], dtype=np.float32),
                [-1.0, -0.6, -1.0], [1.0, 0.6, 1.0]),
    ):
        raise ValueError("student observation command slot is not cmd_raw")
    if not np.array_equal(
        np.asarray(record["obs_teacher"], dtype=np.float32)[6:9],
        np.clip(np.asarray(record["cmd_terrain"], dtype=np.float32),
                [-1.0, -0.6, -1.0], [1.0, 0.6, 1.0]),
    ):
        raise ValueError("teacher observation command slot is not cmd_terrain")


class ChunkedDatasetWriter:
    def __init__(self, output, metadata, chunk_size=5000):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata, schema_version=SCHEMA_VERSION)
        self.chunk_size = int(chunk_size)
        self.records = []
        self.part = 0
        self.paths = []

    def append(self, record):
        validate_record(record)
        self.records.append(record)
        if len(self.records) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.records:
            return None
        suffix = "" if self.part == 0 and len(self.records) < self.chunk_size else f"_part{self.part:04d}"
        path = self.output.with_name(self.output.stem + suffix + ".npz")
        temp = path.with_suffix(path.suffix + ".part")
        arrays = {}
        for name, (dtype, shape) in ARRAY_FIELDS.items():
            arrays[name] = np.stack([
                np.asarray(item[name], dtype=dtype).reshape(shape) for item in self.records
            ])
        for name, dtype in SCALAR_FIELDS.items():
            arrays[name] = np.asarray([item[name] for item in self.records], dtype=dtype)
        arrays["metadata_json"] = np.asarray(
            json.dumps(self.metadata, sort_keys=True, ensure_ascii=False))
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(path)
        self.paths.append(path)
        self.records.clear()
        self.part += 1
        return path

    def close(self):
        self.flush()
        return list(self.paths)


def validate_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        required = set(ARRAY_FIELDS) | set(SCALAR_FIELDS) | {"metadata_json"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"missing fields: {sorted(missing)}")
        count = len(data["timestamp_ns"])
        for name, (_, shape) in ARRAY_FIELDS.items():
            if data[name].shape != (count, *shape):
                raise ValueError(f"wrong shape for {name}: {data[name].shape}")
        if not np.isfinite(data["obs_student"]).all() or not np.isfinite(data["action_teacher"]).all():
            raise ValueError("dataset contains non-finite values")
        if np.any(data["success"] & data["pre_failure"]):
            raise ValueError("success and pre_failure labels must be mutually exclusive")
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported schema version")
    return count
