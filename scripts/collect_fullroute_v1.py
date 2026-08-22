#!/usr/bin/env python3
"""Deterministic, config-driven full-route D_priv collection orchestrator.

Dry-run is the default. Pass --execute only after the privileged teacher has
completed the frozen 33-waypoint probe and its summary is supplied as a gate.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "collect" / "fullroute_v1.json"
FORMAL_PHASES = ("train", "validation", "test", "dagger_r1", "dagger_r2", "dagger_r3")
CODE_FILES = (
    "scripts/collect_mujoco_d_priv.py",
    "scripts/collect_fullroute_v1.py",
    "src/s10_terrain_perception/d_priv_dataset.py",
    "src/s10_terrain_perception/mujoco_teacher.py",
    "src/s10_terrain_perception/mujoco_lidar.py",
    "src/s10_terrain_perception/heightmap.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(master_seed: int, dataset_id: str, scenario_id: str, shard: int) -> int:
    payload = f"{master_seed}:{dataset_id}:{scenario_id}:{shard}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 1:
        raise ValueError("collection config schema_version must be 1")
    required = ("dataset_id", "master_seed", "samples_per_shard", "profiles", "scenarios")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"collection config missing keys: {missing}")
    if not cfg["dataset_id"] or int(cfg["samples_per_shard"]) <= 0:
        raise ValueError("dataset_id must be non-empty and samples_per_shard positive")
    for name, rows in cfg["profiles"].items():
        if not rows or any(len(row) != 4 for row in rows):
            raise ValueError(f"profile {name!r} must contain [fraction,vx,vy,wz] rows")
        fractions = [float(row[0]) for row in rows]
        if fractions[0] != 0.0 or fractions != sorted(fractions):
            raise ValueError(f"profile {name!r} must start at 0 and be sorted")
        if fractions[-1] >= 1.0:
            raise ValueError(f"profile {name!r} fractions must be less than 1")
        for _, vx, vy, wz in rows:
            if not (-1 <= vx <= 1 and -.6 <= vy <= .6 and -1 <= wz <= 1):
                raise ValueError(f"profile {name!r} exceeds collector command limits")
    seen = set()
    for scenario in cfg["scenarios"]:
        sid = scenario["id"]
        if sid in seen:
            raise ValueError(f"duplicate scenario id: {sid}")
        seen.add(sid)
        if scenario["phase"] not in ("probe",) + FORMAL_PHASES:
            raise ValueError(f"unknown phase in {sid}: {scenario['phase']}")
        if int(scenario["shards"]) <= 0:
            raise ValueError(f"{sid}: shards must be positive")
        mode = scenario["mode"]
        if mode == "schedule":
            if not scenario.get("profiles") or not scenario.get("start_poses"):
                raise ValueError(f"{sid}: schedule mode needs profiles and start_poses")
            unknown = set(scenario["profiles"]) - set(cfg["profiles"])
            if unknown:
                raise ValueError(f"{sid}: unknown profiles {sorted(unknown)}")
            if any(len(pose) != 4 for pose in scenario["start_poses"]):
                raise ValueError(f"{sid}: start poses must be [x,y,z,yaw]")
        elif mode == "route":
            if not scenario.get("waypoint_ranges") or not scenario.get("speeds"):
                raise ValueError(f"{sid}: route mode needs waypoint_ranges and speeds")
            for first, last in scenario["waypoint_ranges"]:
                if not 0 <= first <= last <= 32:
                    raise ValueError(f"{sid}: invalid waypoint range {first}..{last}")
            if any(not .4 <= float(v) <= 1.0 for v in scenario["speeds"]):
                raise ValueError(f"{sid}: route speeds must be in [0.4, 1.0] m/s")
        else:
            raise ValueError(f"{sid}: mode must be schedule or route")
    return cfg


def expand_plan(cfg: dict, phases: set[str]) -> list[dict]:
    jobs = []
    episode_id = 0
    default_samples = int(cfg["samples_per_shard"])
    for scenario in cfg["scenarios"]:
        if scenario["phase"] not in phases:
            continue
        samples = int(scenario.get("samples", default_samples))
        for shard in range(int(scenario["shards"])):
            job = {
                "dataset_id": cfg["dataset_id"],
                "scenario": scenario["id"],
                "phase": scenario["phase"],
                "category": scenario["category"],
                "terrain_id": scenario.get("terrain_id", scenario["category"]),
                "mode": scenario["mode"],
                "shard": shard,
                "samples": samples,
                "episode_id": episode_id,
                "seed": stable_seed(
                    int(cfg["master_seed"]), cfg["dataset_id"], scenario["id"], shard),
            }
            episode_id += 1
            if scenario["mode"] == "schedule":
                job["profile"] = scenario["profiles"][shard % len(scenario["profiles"])]
                job["start"] = scenario["start_poses"][shard % len(scenario["start_poses"])]
            else:
                job["waypoint_range"] = scenario["waypoint_ranges"][
                    shard % len(scenario["waypoint_ranges"])]
                job["autonav_vx"] = float(
                    scenario["speeds"][shard % len(scenario["speeds"])])
                job["start_waypoint"] = int(job["waypoint_range"][0])
            jobs.append(job)
    return jobs


def summarize(jobs: list[dict]) -> dict:
    by_phase = Counter()
    by_category = Counter()
    for job in jobs:
        by_phase[job["phase"]] += job["samples"]
        by_category[f"{job['phase']}:{job['category']}"] += job["samples"]
    return {
        "jobs": len(jobs),
        "samples": sum(job["samples"] for job in jobs),
        "samples_by_phase": dict(sorted(by_phase.items())),
        "samples_by_phase_category": dict(sorted(by_category.items())),
    }


def schedule_rows(profile: list[list[float]], samples: int) -> str:
    rows = []
    last_start = -1
    for fraction, vx, vy, wz in profile:
        start = int(round(float(fraction) * samples))
        start = max(last_start + 1, start) if rows else 0
        if start >= samples:
            raise ValueError("scaled profile has a start outside the shard")
        rows.append(f"{start},{float(vx):.6f},{float(vy):.6f},{float(wz):.6f}")
        last_start = start
    return "\n".join(rows) + "\n"


def output_path(output_root: Path, job: dict) -> Path:
    name = f"{job['scenario']}__{job['shard']:03d}__seed{job['seed']}.npz"
    return output_root / job["phase"] / job["category"] / name


def build_command(
    cfg: dict, job: dict, teacher: Path, student: Path | None,
    output_root: Path, schedule_dir: Path, config_sha256: str,
) -> list[str]:
    collector = ROOT / cfg.get("collector", "scripts/collect_mujoco_d_priv.py")
    command = [
        sys.executable, str(collector), "--teacher-onnx", str(teacher),
        "--output", str(output_path(output_root, job)),
        "--samples", str(job["samples"]),
        "--terrain-id", job["terrain_id"],
        "--episode-id", str(job["episode_id"]),
        "--seed", str(job["seed"]),
        "--collection-config-sha256", config_sha256,
        "--collection-scenario", job["scenario"],
        "--collection-phase", job["phase"],
        "--log-every", str(cfg.get("log_every", 250)),
    ]
    if cfg.get("xml"):
        command += ["--xml", str((ROOT / cfg["xml"]).resolve())]
    if job["phase"].startswith("dagger_"):
        if student is None:
            raise ValueError(f"{job['phase']} requires --student")
        command += ["--rollout-policy", str(student)]
    if job["mode"] == "schedule":
        schedule_path = schedule_dir / (
            f"{job['scenario']}__{job['shard']:03d}__seed{job['seed']}.csv")
        command += ["--command-schedule", str(schedule_path), "--start"]
        command += [str(value) for value in job["start"]]
    else:
        first, last = job["waypoint_range"]
        command += [
            "--waypoint-range", str(first), str(last),
            "--start-waypoint", str(job["start_waypoint"]),
            "--autonav-vx", str(job["autonav_vx"]),
        ]
    return command


def validate_gate(path: Path, teacher_sha256: str) -> None:
    gate = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "same teacher SHA-256": gate.get("teacher_sha256") == teacher_sha256,
        "all 33 official waypoints": gate.get("waypoint_range") == [0, 32]
            and gate.get("route_complete") is True,
        "exact privileged scan": gate.get("teacher_scan_every") == 1,
        "official reach radius": abs(float(
            gate.get("waypoint_reach_radius_m", -1)) - .20) < 1e-9,
        "frozen controller": gate.get("autonav_controller") == "official_route_v1",
        "usable labels": gate.get("labels_usable") is True,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("teacher gate rejected: " + ", ".join(failed))


def run_one(command: list[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode, str(log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path)
    parser.add_argument(
        "--phase", action="append",
        choices=("probe",) + FORMAL_PHASES,
        help="Repeat to select phases; default is train+validation+test")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--allow-ungated-probe", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip shards that already have both .npz and .summary.json")
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    config_sha = sha256_file(config_path)
    phases = set(args.phase or ("train", "validation", "test"))
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if "probe" in phases and len(phases) != 1:
        parser.error("probe must be run by itself")
    if any(p.startswith("dagger_") for p in phases) and args.student is None:
        parser.error("DAgger phases require --student")
    teacher = args.teacher.resolve()
    if not teacher.is_file():
        parser.error(f"teacher not found: {teacher}")
    student = args.student.resolve() if args.student else None
    if student is not None and not student.is_file():
        parser.error(f"student not found: {student}")

    output_root = (args.output_root or (ROOT / cfg.get(
        "output_root", f"results/datasets/{cfg['dataset_id']}"))).resolve()
    jobs = expand_plan(cfg, phases)
    report = summarize(jobs)
    teacher_sha = sha256_file(teacher)
    print(json.dumps({
        "mode": "EXECUTE" if args.execute else "DRY-RUN",
        "config": str(config_path),
        "config_sha256": config_sha,
        "teacher": str(teacher),
        "teacher_sha256": teacher_sha,
        "student": str(student) if student else None,
        "output_root": str(output_root),
        **report,
    }, indent=2, sort_keys=True))

    if not args.execute:
        print("Dry-run only. Add --execute after reviewing totals and the teacher gate.")
        return
    if phases == {"probe"}:
        if not args.allow_ungated_probe:
            parser.error("probe execution requires --allow-ungated-probe")
    else:
        if args.gate_report is None:
            parser.error("formal collection requires --gate-report from a completed probe")
        validate_gate(args.gate_report.resolve(), teacher_sha)

    schedule_dir = output_root / "_schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    plan_jobs = []
    commands = []
    for job in jobs:
        output = output_path(output_root, job)
        summary_path = output.with_suffix(output.suffix + ".summary.json")
        completed = output.exists() and summary_path.exists()
        if output.exists() or summary_path.exists():
            if not (args.resume and completed):
                raise FileExistsError(
                    f"refusing to overwrite existing shard: {output}; "
                    "use --resume or a new dataset id")
        if job["mode"] == "schedule":
            schedule = schedule_dir / (
                f"{job['scenario']}__{job['shard']:03d}__seed{job['seed']}.csv")
            expected_schedule = schedule_rows(
                cfg["profiles"][job["profile"]], job["samples"])
            if schedule.exists() and schedule.read_text(encoding="utf-8") != expected_schedule:
                raise ValueError(f"existing schedule differs from config: {schedule}")
            if not schedule.exists():
                schedule.write_text(expected_schedule, encoding="utf-8")
        command = build_command(
            cfg, job, teacher, student, output_root, schedule_dir, config_sha)
        plan_jobs.append({**job, "output": str(output), "command": command})
        if not completed:
            commands.append((command, output_root / "_logs" / (
                f"{job['scenario']}__{job['shard']:03d}.log")))

    plan = {
        "schema_version": 1,
        "config": str(config_path),
        "config_sha256": config_sha,
        "code_files_sha256": {
            relative: sha256_file(ROOT / relative) for relative in CODE_FILES},
        "teacher": str(teacher),
        "teacher_sha256": teacher_sha,
        "student": str(student) if student else None,
        "student_sha256": sha256_file(student) if student else None,
        "phases": sorted(phases),
        "summary": report,
        "jobs": plan_jobs,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / (
        "collection_plan__" + "_".join(sorted(phases)) + ".json")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, command, log): (command, log)
            for command, log in commands
        }
        for future in concurrent.futures.as_completed(futures):
            code, log = future.result()
            print(f"[{'PASS' if code == 0 else 'FAIL'}] {log}", flush=True)
            if code:
                failures.append(log)
    if failures:
        raise SystemExit(f"{len(failures)} collection job(s) failed; inspect: {failures}")
    print(f"Collection finished. Audit with scripts/audit_fullroute_v1.py {plan_path}")


if __name__ == "__main__":
    main()
