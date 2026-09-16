#!/usr/bin/env python3
"""Create a hash-pinned four-Actor bundle for GoAI MuJoCo playback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROLES = ("NORMAL", "HIGH_CLIMB", "LOW_STEP_SEQUENCE", "RECOVERY")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--high-climb", type=Path, required=True)
    parser.add_argument("--low-step-sequence", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height-split-m", type=float, default=0.16)
    parser.add_argument("--low-forward-command-max-mps", type=float, default=0.6)
    parser.add_argument(
        "--low-command-adapter",
        choices=("model99_latched_world_direction_v1",),
        default=None,
    )
    args = parser.parse_args()
    if args.height_split_m <= 0.0:
        parser.error("--height-split-m must be positive")
    if args.low_forward_command_max_mps <= 0.0:
        parser.error("--low-forward-command-max-mps must be positive")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = {
        "NORMAL": args.normal,
        "HIGH_CLIMB": args.high_climb,
        "LOW_STEP_SEQUENCE": args.low_step_sequence,
        "RECOVERY": args.recovery,
    }
    skills = {}
    for role in ROLES:
        path = sources[role].expanduser().resolve()
        if not path.is_file():
            parser.error(f"{role} ONNX does not exist: {path}")
        sidecar = path.with_suffix(path.suffix + ".json")
        if not sidecar.is_file():
            parser.error(f"{role} ONNX sidecar does not exist: {sidecar}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("protocol") != "s10-asymmetric-cnn-gru-1413-v1":
            parser.error(f"{role} has incompatible protocol: {metadata.get('protocol')!r}")
        actual_hash = sha256(path)
        if metadata.get("onnx_sha256") != actual_hash:
            parser.error(f"{role} ONNX hash does not match its sidecar")
        skills[role] = {
            "path": os.path.relpath(path, output.parent),
            "sha256": actual_hash,
        }

    payload = {
        "schema_version": 1,
        "kind": "s10-goai-teacher-router-bundle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actor_protocol": "s10-asymmetric-cnn-gru-1413-v1",
        # The scene is deliberately absent. --xml is mandatory at runtime.
        "router": {
            "height_split_m": args.height_split_m,
            "low_forward_command_max_mps": args.low_forward_command_max_mps,
            "forward_min_x": 0.1,
            "max_abs_y": 0.1,
            "max_abs_yaw": 0.1,
            "detector_confirm_s": 0.04,
            "attempt_timeout_s": 12.0,
            "successor_search_s": 0.20,
            "recovery_hold_s": 0.20,
            "recovery_timeout_s": 5.0,
            "recovery_base_height_m": 0.35,
        },
        "detector": {
            "front_x_range_m": [0.30, 1.20],
            "corridor_half_width_m": 0.25,
            "min_height_m": 0.04,
            "max_height_m": 0.45,
            "min_landing_width_m": 0.42,
            "min_landing_depth_m": 0.20,
            "landing_height_tolerance_m": 0.04,
            "landing_coverage_min": 0.80,
            "clearance_start_x_m": 0.25,
            "clearance_range_m": 1.20,
            "clearance_lateral_m": [-0.22, 0.0, 0.22],
            # Low ray distinguishes a solid riser from free space under an
            # overhang; middle/upper rays prove body/head obstruction.
            "clearance_height_above_base_m": [-0.25, 0.0, 0.15],
        },
        "skills": skills,
    }
    if args.low_command_adapter is not None:
        payload["router"].update({
            "low_command_adapter": args.low_command_adapter,
            "low_command_yaw_gain": 0.5,
            "low_command_yaw_limit": 0.5,
            "low_command_smoothing_tau_s": 0.20,
        })
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote GoAI teacher Router bundle: {output}")


if __name__ == "__main__":
    main()
