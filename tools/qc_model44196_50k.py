#!/usr/bin/env python3
"""Build the strict 50k QC manifest for model_44196 raw-action D_priv."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "results" / "datasets" / "model_44196_raw_v2"
OUTPUT = DATASET_ROOT / "QC_50K_MANIFEST.json"
TEACHER_SHA256 = "857f2d59c04b6ee979e3fc776e2ca00e593c4d009297ed3b84427b0a5752b9be"

sys.path.insert(0, str(ROOT / "src" / "s10_terrain_perception"))
from d_priv_dataset import validate_d_priv  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_paths():
    legacy = json.loads((DATASET_ROOT / "QC_MANIFEST.json").read_text())
    for item in legacy["default_training_shards"]:
        yield DATASET_ROOT / item["path"]
    yield from sorted((DATASET_ROOT / "CANDIDATE_50K").glob("*/*.npz"))
    yield from sorted((DATASET_ROOT / "SUPPLEMENT_50K").glob("*/*.npz"))


def metrics(path: Path):
    validate_d_priv(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        pose = data["base_pose_wxyz"]
        pos = pose[:, :3]
        quat = pose[:, 3:7]
        x, y = quat[:, 1], quat[:, 2]
        tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))
        action_abs = np.abs(data["teacher_action_raw"])
        command = data["command_raw"]
        sources = set(data["teacher_source"].tolist())
        return {
            "samples": len(pos),
            "episode_id": int(data["episode_id"][0]),
            "terrain": str(data["terrain_id"][0]),
            "seed": int(metadata["seed"]),
            "start_xyzyaw": list(map(float, metadata["start_xyzyaw"])),
            "labels_usable": metadata.get("labels_usable") is True,
            "teacher_source_ok": sources == {"privileged_mujoco"},
            "teacher_sha_ok": metadata.get("teacher_sha256") == TEACHER_SHA256,
            "vx_0p6": bool(np.allclose(command[:, 0], 0.6)),
            "displacement_m": float(np.linalg.norm(pos[-1, :2] - pos[0, :2])),
            "z_start_m": float(pos[0, 2]),
            "z_min_m": float(pos[:, 2].min()),
            "z_max_m": float(pos[:, 2].max()),
            "z_final_m": float(pos[-1, 2]),
            "tilt_max_deg": float(tilt.max()),
            "action_abs_max": float(action_abs.max()),
            "action_abs_p99": float(np.quantile(action_abs, 0.99)),
            "action_saturated_values": int((action_abs >= 99.999).sum()),
            "hit_min": float(data["privileged_hit_fraction"].min()),
        }


def rejection_reasons(item):
    reasons = []
    if item["samples"] != 1000:
        reasons.append("samples != 1000")
    if not item["labels_usable"] or not item["teacher_source_ok"]:
        reasons.append("not real privileged labels")
    if not item["teacher_sha_ok"]:
        reasons.append("teacher SHA mismatch")
    if not item["vx_0p6"]:
        reasons.append("vx is not uniformly 0.6 m/s")
    if item["hit_min"] < 0.99:
        reasons.append("privileged hit fraction < 0.99")
    if item["tilt_max_deg"] > 35.0:
        reasons.append("max tilt > 35 deg")
    if item["action_saturated_values"]:
        reasons.append("action reaches +/-100")
    if item["action_abs_max"] > 30.0:
        reasons.append("action abs max > 30")

    terrain = item["terrain"]
    disp = item["displacement_m"]
    z0, zmin, zmax, zf = (
        item["z_start_m"], item["z_min_m"], item["z_max_m"], item["z_final_m"])
    if terrain == "start_long_ramp":
        if disp < 4.0:
            reasons.append("long-ramp displacement < 4 m")
        if zmin < 0.30:
            reasons.append("long-ramp base_z < 0.30 m")
    elif terrain == "mild_up":
        if disp < 8.0 or zmax - z0 < 0.15:
            reasons.append("mild-up coverage insufficient")
    elif terrain in {"flat", "low_plateau", "high_plateau"}:
        if disp < 7.0:
            reasons.append("flat/plateau displacement < 7 m")
    elif terrain == "gentle_slope":
        if disp < 6.0 or z0 - zmin < 0.25:
            reasons.append("gentle-slope coverage insufficient")
    elif terrain == "regular_stairs_up":
        if disp < 4.0 or zmax - z0 < 0.50 or zf < 1.35:
            reasons.append("regular stairs not completed")
    elif terrain in {"steep_down", "long_down"}:
        if disp < 6.0 or z0 - zmin < 0.50:
            reasons.append("down-slope coverage insufficient")
    elif terrain == "high_step":
        if disp < 2.0 or zmax - z0 < 0.50 or zf < 2.0:
            reasons.append("high step not completed")
    elif terrain == "high_step_recovery":
        if disp < 1.0:
            reasons.append("recovery displacement < 1 m")
    return reasons


def rounded(value):
    return round(value, 6)


def main():
    accepted, rejected = [], []
    for path in dict.fromkeys(candidate_paths()):
        item = metrics(path)
        reasons = rejection_reasons(item)
        start = item["start_xyzyaw"]
        record = {
            "path": str(path.relative_to(DATASET_ROOT)),
            "sha256": sha256(path),
            "terrain": item["terrain"],
            "episode_id": item["episode_id"],
            "seed": item["seed"],
            "samples": item["samples"],
            "start_xyzyaw": [rounded(x) for x in start],
            "split_group": (
                f"{item['terrain']}:{start[0]:.2f}:{start[1]:.2f}:{start[3]:.3f}"),
            "displacement_m": rounded(item["displacement_m"]),
            "z_min_m": rounded(item["z_min_m"]),
            "z_max_m": rounded(item["z_max_m"]),
            "z_final_m": rounded(item["z_final_m"]),
            "tilt_max_deg": rounded(item["tilt_max_deg"]),
            "action_abs_max": rounded(item["action_abs_max"]),
            "action_abs_p99": rounded(item["action_abs_p99"]),
            "privileged_hit_fraction_min": rounded(item["hit_min"]),
        }
        if reasons:
            record["reasons"] = reasons
            rejected.append(record)
        else:
            accepted.append(record)

    terrain_shards = Counter(x["terrain"] for x in accepted)
    terrain_samples = Counter()
    for item in accepted:
        terrain_samples[item["terrain"]] += item["samples"]
    report = {
        "model": "model_44196.pt",
        "schema_version": 2,
        "action_field": "teacher_action_raw",
        "action_contract": "raw actor output; clip limit 100; never clip to [-1,1]",
        "teacher_onnx": "artifacts/teacher_model_44196_1413_raw100.onnx",
        "teacher_onnx_sha256": TEACHER_SHA256,
        "recommended_vx_mps": 0.6,
        "accepted_shards": len(accepted),
        "accepted_samples": sum(x["samples"] for x in accepted),
        "rejected_shards": len(rejected),
        "terrain_shards": dict(sorted(terrain_shards.items())),
        "terrain_samples": dict(sorted(terrain_samples.items())),
        "te_contract_warning": (
            "Consume teacher_action_raw. The current TE-distill.md still names the old "
            "teacher_action_norm field and must be corrected before training."),
        "split_rule": (
            "Keep every identical split_group wholly in train or validation; never split "
            "time-correlated rows randomly."),
        "accepted": sorted(accepted, key=lambda x: (x["terrain"], x["episode_id"])),
        "rejected": sorted(rejected, key=lambda x: x["episode_id"]),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "accepted_shards", "accepted_samples", "rejected_shards",
        "terrain_shards", "terrain_samples")}, indent=2, ensure_ascii=False))
    if report["accepted_shards"] != 50 or report["accepted_samples"] != 50000:
        raise SystemExit("strict accepted set is not exactly 50 shards / 50000 samples")


if __name__ == "__main__":
    main()
