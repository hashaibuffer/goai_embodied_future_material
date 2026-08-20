#!/usr/bin/env python3
"""Collect the 44 candidate shards that extend model_44196 D_priv to 50k."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts" / "collect_mujoco_d_priv.py"
TEACHER = ROOT / "artifacts" / "teacher_model_44196_1413_raw100.onnx"
OUTPUT_ROOT = ROOT / "results" / "datasets" / "model_44196_raw_v2" / "CANDIDATE_50K"
RECOVERY_SCHEDULE = ROOT / "configs" / "collect" / "high_step_recovery_v06.csv"


# name, x, y, surface z, yaw, number of 1000-sample shards, offset scale
SEGMENTS = [
    ("start_long_ramp", 0.0000, -1.7250, 0.200, 1.6240, 5, 1.0),
    ("mild_up", -15.6000, 23.2800, 0.475, 1.5150, 5, 1.0),
    ("high_plateau", -20.6550, 47.7825, 1.165, 0.0113, 5, 1.0),
    ("low_plateau", -12.9225, 34.6275, 0.475, -0.0062, 5, 1.0),
    ("gentle_slope", 2.7225, 34.5300, 0.475, -0.1793, 5, 1.0),
    ("regular_stairs_up", -15.0200, 36.5000, 0.850, 1.5700, 5, 0.7),
    ("steep_down", -13.5000, 41.5500, 1.165, -1.4880, 5, 0.7),
    ("long_down", 33.9225, 24.4275, 2.360, 3.1300, 4, 0.7),
    ("high_step", 21.6600, 29.5700, 1.340, 0.0500, 3, 0.35),
    ("high_step_recovery", 21.6600, 29.5700, 1.340, 0.0500, 2, 0.25),
]

# longitudinal metres, lateral metres, yaw radians
VARIATIONS = [
    (-0.20, -0.12, -0.015),
    (-0.10, 0.12, 0.015),
    (0.10, -0.06, 0.025),
    (0.20, 0.06, -0.025),
    (-0.30, 0.00, 0.000),
]


def jobs():
    episode = 7001
    for name, x, y, z, yaw, count, scale in SEGMENTS:
        for longitudinal, lateral, yaw_delta in VARIATIONS[:count]:
            longitudinal *= scale
            lateral *= scale
            sx = x + longitudinal * math.cos(yaw) - lateral * math.sin(yaw)
            sy = y + longitudinal * math.sin(yaw) + lateral * math.cos(yaw)
            syaw = yaw + yaw_delta * scale
            output = OUTPUT_ROOT / name / f"D_priv_model_44196_raw_v2_{name}_{episode}.npz"
            yield {
                "name": name,
                "episode": episode,
                "seed": episode,
                "start": (sx, sy, z, syaw),
                "output": output,
                "recovery": name == "high_step_recovery",
            }
            episode += 1


def collect(job, force=False):
    output = job["output"]
    if output.exists() and not force:
        return job, "SKIP", "already exists"
    output.parent.mkdir(parents=True, exist_ok=True)
    x, y, z, yaw = job["start"]
    cmd = [
        sys.executable, str(COLLECTOR),
        "--teacher-onnx", str(TEACHER),
        "--start", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{yaw:.6f}",
        "--command", "0.6", "0", "0",
        "--samples", "1000",
        "--terrain-id", job["name"],
        "--episode-id", str(job["episode"]),
        "--seed", str(job["seed"]),
        "--output", str(output),
        "--log-every", "500",
    ]
    if job["recovery"]:
        cmd.extend(["--command-schedule", str(RECOVERY_SCHEDULE)])
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
    return job, "PASS" if proc.returncode == 0 else "FAIL", tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    planned = list(jobs())
    print(f"planned_shards={len(planned)} planned_samples={len(planned) * 1000}", flush=True)
    if args.dry_run:
        for job in planned:
            print(job)
        return
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect, job, args.force) for job in planned]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            job, status, tail = future.result()
            print(
                f"[{index:02d}/{len(planned)}] {status} "
                f"episode={job['episode']} terrain={job['name']}\n{tail}",
                flush=True,
            )
            if status == "FAIL":
                failures.append(job["episode"])
    if failures:
        raise SystemExit(f"failed episodes: {failures}")


if __name__ == "__main__":
    main()
