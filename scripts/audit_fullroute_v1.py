#!/usr/bin/env python3
"""Audit a fullroute_v1 collection plan and emit a training-safe manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_text(value) -> str:
    return str(np.asarray(value).reshape(()).item())


def audit_shard(job: dict, teacher_sha: str, thresholds: dict) -> tuple[dict, list[str]]:
    path = Path(job["output"])
    failures = []
    metrics = {"path": str(path), "scenario": job["scenario"]}
    if not path.is_file():
        return metrics, ["missing shard"]
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "student_obs", "teacher_action_raw", "command_raw", "base_pose_wxyz",
                "teacher_source", "privileged_hit_fraction", "schema_version",
                "metadata_json",
            }
            missing = sorted(required - set(data.files))
            if missing:
                return metrics, [f"missing arrays: {missing}"]
            count = len(data["student_obs"])
            metrics["samples"] = count
            if count != int(job["samples"]):
                failures.append(f"samples {count} != planned {job['samples']}")
            if int(np.asarray(data["schema_version"]).reshape(())) != 2:
                failures.append("schema_version is not 2")
            metadata = json.loads(scalar_text(data["metadata_json"]))
            if metadata.get("teacher_sha256") != teacher_sha:
                failures.append("teacher SHA-256 mismatch")
            if metadata.get("collection_config_sha256") != job.get("config_sha256"):
                failures.append("collection config SHA-256 mismatch")
            if metadata.get("collection_scenario") != job["scenario"]:
                failures.append("collection scenario mismatch")
            if metadata.get("collection_phase") != job["phase"]:
                failures.append("collection phase mismatch")
            sources = set(data["teacher_source"].astype(str).tolist())
            if sources != {"privileged_mujoco"}:
                failures.append(f"teacher_source={sorted(sources)}")
            arrays = (
                data["student_obs"], data["teacher_action_raw"], data["command_raw"],
                data["base_pose_wxyz"], data["privileged_hit_fraction"])
            if any(not np.isfinite(array).all() for array in arrays):
                failures.append("non-finite values")
            commands = np.asarray(data["command_raw"], np.float32)
            poses = np.asarray(data["base_pose_wxyz"], np.float32)
            hits = np.asarray(data["privileged_hit_fraction"], np.float32)
            actions = np.asarray(data["teacher_action_raw"], np.float32)
            metrics["mean_privileged_hit_fraction"] = float(hits.mean())
            metrics["min_base_z_m"] = float(poses[:, 2].min())
            metrics["action_abs_max"] = float(np.abs(actions).max())
            metrics["command_vx_mean"] = float(commands[:, 0].mean())
            metrics["command_vx_max"] = float(commands[:, 0].max())
            metrics["command_wz_abs_mean"] = float(np.abs(commands[:, 2]).mean())
            if metrics["mean_privileged_hit_fraction"] < float(
                    thresholds.get("min_mean_privileged_hit_fraction", .50)):
                failures.append("privileged height hit fraction too low")
            if metrics["min_base_z_m"] < float(thresholds.get("min_base_z_m", .08)):
                failures.append("base collapsed below minimum z")
            if metrics["action_abs_max"] > 100.001:
                failures.append("teacher action exceeded raw limit")

            if count > 1:
                qw, qx, qy, qz = (poses[:, 3], poses[:, 4], poses[:, 5], poses[:, 6])
                yaw = np.arctan2(
                    2 * (qw * qz + qx * qy),
                    1 - 2 * (qy * qy + qz * qz))
                delta = np.diff(poses[:, :2], axis=0)
                forward_speed = (
                    delta[:, 0] * np.cos(yaw[:-1])
                    + delta[:, 1] * np.sin(yaw[:-1])) / .02
                requested = commands[:-1, 0] > .40
                denom = max(1, int(requested.sum()))
                reverse_fraction = float(
                    np.logical_and(requested, forward_speed < -.05).sum() / denom)
                stalled_fraction = float(
                    np.logical_and(requested, np.abs(forward_speed) < .03).sum() / denom)
                metrics["reverse_fraction_when_fast"] = reverse_fraction
                metrics["stalled_fraction_when_fast"] = stalled_fraction
                if requested.sum() >= 25 and reverse_fraction > float(
                        thresholds.get("max_reverse_fraction_when_fast", .15)):
                    failures.append("excessive reverse motion under forward command")
                if requested.sum() >= 25 and stalled_fraction > float(
                        thresholds.get("max_stalled_fraction_when_fast", .75)):
                    failures.append("mostly stationary under fast forward command")
    except Exception as exc:
        failures.append(f"read/validation error: {exc}")
        return metrics, failures

    if job["mode"] == "route":
        summary_path = path.with_suffix(path.suffix + ".summary.json")
        if not summary_path.is_file():
            failures.append("missing route summary")
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics["route_complete"] = summary.get("route_complete")
            if summary.get("route_complete") is not True:
                failures.append(
                    f"route incomplete; next waypoint={summary.get('route_next_waypoint')}")
            if summary.get("waypoint_reach_radius_m") != .20:
                failures.append("route did not use official 0.20 m reach radius")
            if summary.get("teacher_scan_every") != 1:
                failures.append("teacher scan was not refreshed every policy frame")
    metrics["sha256"] = sha256_file(path)
    return metrics, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config_path = Path(plan["config"])
    if sha256_file(config_path) != plan["config_sha256"]:
        parser.error("config hash differs from the frozen collection plan")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for relative, expected in plan.get("code_files_sha256", {}).items():
        code_path = Path(__file__).resolve().parents[1] / relative
        if not code_path.is_file() or sha256_file(code_path) != expected:
            parser.error(f"collection code hash differs from plan: {relative}")
    teacher_sha = plan["teacher_sha256"]
    thresholds = cfg.get("quality_gates", {})

    accepted, rejected = [], []
    for job in plan["jobs"]:
        job = {**job, "config_sha256": plan["config_sha256"]}
        metrics, failures = audit_shard(job, teacher_sha, thresholds)
        record = {"job": job, "metrics": metrics}
        if failures:
            record["failures"] = failures
            rejected.append(record)
            print(f"[REJECT] {job['scenario']}#{job['shard']}: {'; '.join(failures)}")
        else:
            accepted.append(record)
            print(f"[ACCEPT] {job['scenario']}#{job['shard']}")

    by_phase = Counter()
    by_category = Counter()
    for record in accepted:
        job = record["job"]
        by_phase[job["phase"]] += int(record["metrics"]["samples"])
        by_category[f"{job['phase']}:{job['category']}"] += int(
            record["metrics"]["samples"])
    manifest = {
        "schema_version": 1,
        "dataset_id": cfg["dataset_id"],
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "config_sha256": plan["config_sha256"],
        "teacher_sha256": teacher_sha,
        "student_sha256": plan.get("student_sha256"),
        "accepted_shards": [
            {"path": item["metrics"]["path"], "sha256": item["metrics"]["sha256"],
             "samples": item["metrics"]["samples"], "phase": item["job"]["phase"],
             "category": item["job"]["category"], "scenario": item["job"]["scenario"]}
            for item in accepted
        ],
        "rejected": rejected,
        "accepted_samples_by_phase": dict(sorted(by_phase.items())),
        "accepted_samples_by_phase_category": dict(sorted(by_category.items())),
    }
    manifest_path = (args.manifest or (
        plan_path.parent / (plan_path.stem + "__manifest.json"))).resolve()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted_shards": len(accepted),
        "rejected_shards": len(rejected),
        "accepted_samples_by_phase": dict(sorted(by_phase.items())),
        "manifest": str(manifest_path),
    }, indent=2, sort_keys=True))
    if rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
