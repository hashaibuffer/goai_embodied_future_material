#!/usr/bin/env python3
"""Print and validate a D_priv shard without loading pickle data."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "s10_terrain_perception"))
from d_priv_dataset import validate_d_priv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", type=Path)
parser.add_argument(
    "--require-training-labels", action="store_true",
    help="Fail unless metadata and every source mark the shard as real privileged labels")
args = parser.parse_args()
count = validate_d_priv(args.path)
with np.load(args.path, allow_pickle=False) as data:
    metadata = json.loads(str(data["metadata_json"]))
    sources = set(data["teacher_source"].tolist())
    if args.require_training_labels:
        if metadata.get("labels_usable") is not True:
            raise SystemExit("refusing shard: metadata.labels_usable is not true")
        if sources != {"privileged_mujoco"}:
            raise SystemExit(
                f"refusing shard: expected teacher_source=privileged_mujoco, got {sorted(sources)}")
    report = {
        "samples": count,
        "metadata": metadata,
        "fields": {name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)}
                   for name in data.files},
        "action_abs_max": float(np.max(np.abs(data["teacher_action_norm"]))),
        "privileged_hit_fraction": {
            "min": float(np.min(data["privileged_hit_fraction"])),
            "mean": float(np.mean(data["privileged_hit_fraction"])),
        },
    }
print(json.dumps(report, indent=2, sort_keys=True))
