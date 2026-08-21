#!/usr/bin/env python3
"""Collect balanced turning-v2 teacher demonstrations and student DAgger shards."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts" / "collect_mujoco_d_priv.py"
OUTPUT_ROOT = ROOT / "results/datasets/model_44196_turning_v2"

# start_sample, vx, vy, wz. Commands stay inside the frozen deployment contract.
COURSES = {
    "yaw_left": [(0,0,0,0),(80,0,0,.15),(220,0,0,.30),(380,0,0,.45),(560,0,0,.60),(760,0,0,.30),(900,0,0,0)],
    "yaw_right": [(0,0,0,0),(80,0,0,-.15),(220,0,0,-.30),(380,0,0,-.45),(560,0,0,-.60),(760,0,0,-.30),(900,0,0,0)],
    "arc_left": [(0,.2,0,0),(100,.25,0,.15),(260,.35,0,.30),(440,.45,0,.45),(620,.3,0,.60),(780,.45,0,.25),(920,.5,0,0)],
    "arc_right": [(0,.2,0,0),(100,.25,0,-.15),(260,.35,0,-.30),(440,.45,0,-.45),(620,.3,0,-.60),(780,.45,0,-.25),(920,.5,0,0)],
    "switchback": [(0,.25,0,0),(120,.2,0,.30),(280,.35,0,-.30),(440,.2,0,.50),(600,.35,0,-.50),(760,.45,0,.20),(880,.45,0,-.20),(960,.4,0,0)],
    "turn_then_go": [(0,0,0,.20),(160,0,0,.40),(320,0,0,.60),(500,.2,0,.40),(650,.4,0,.20),(780,.5,0,0),(900,.3,0,-.20),(970,.4,0,0)],
    "lateral_left": [(0,.2,0,0),(120,.25,.10,.15),(300,.3,.20,.30),(480,.25,.10,.45),(650,.35,0,.25),(800,.4,-.10,-.15),(930,.4,0,0)],
    "lateral_right": [(0,.2,0,0),(120,.25,-.10,-.15),(300,.3,-.20,-.30),(480,.25,-.10,-.45),(650,.35,0,-.25),(800,.4,.10,.15),(930,.4,0,0)],
}

# x, y, base start z, nominal yaw. These cover the official start and two
# broad route platforms without placing the robot on stairs or an edge.
STARTS = {
    "official": (0.0, -2.5, .2, 0.0),
    "high_plateau": (-20.655, 47.7825, 1.165, 0.0113),
    "low_plateau": (-12.9225, 34.6275, .475, -0.0062),
}

VARIATIONS = [(-.06, -.04, -.08), (.05, .04, .08)]


def schedule_file(course: str, directory: Path) -> Path:
    path = directory / f"{course}.csv"
    path.write_text("\n".join(",".join(map(str, row)) for row in COURSES[course]) + "\n")
    return path


def planned_jobs(phase: str):
    if phase == "probe":
        for i, course in enumerate(COURSES):
            yield dict(course=course, start_name="official", role="teacher", episode=8301+i,
                       variation=(0,0,0), samples=400)
        return
    episode = 8401
    start_names = list(STARTS)
    for course_index, course in enumerate(COURSES):
        # Two locations and two perturbations per course: 32 teacher/DAgger shards.
        selected = [start_names[course_index % 3], start_names[(course_index + 1) % 3]]
        for start_name, variation in zip(selected, VARIATIONS):
            for role in ("teacher", "dagger"):
                yield dict(course=course, start_name=start_name, role=role,
                           episode=episode, variation=variation, samples=1000)
                episode += 1


def collect(job, teacher: Path, student: Path | None, schedule: Path, force: bool):
    x, y, z, yaw = STARTS[job["start_name"]]
    dx, dy, dyaw = job["variation"]
    group = "PROBE" if job["samples"] < 1000 else "FORMAL"
    name = f'D_priv_turnv2_{job["course"]}_{job["start_name"]}_{job["role"]}_{job["episode"]}.npz'
    output = OUTPUT_ROOT / group / job["role"] / name
    if output.exists() and not force:
        return job, "SKIP", "already exists"
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(COLLECTOR), "--teacher-onnx", str(teacher),
           "--output", str(output), "--samples", str(job["samples"]),
           "--command-schedule", str(schedule), "--terrain-id",
           f'turnv2_{job["course"]}_{job["start_name"]}_{job["role"]}',
           "--episode-id", str(job["episode"]), "--seed", str(job["episode"]),
           "--start", str(x+dx), str(y+dy), str(z), str(yaw+dyaw),
           "--log-every", str(job["samples"])]
    if job["role"] == "dagger":
        if student is None:
            raise ValueError("--student is required for formal DAgger collection")
        cmd[cmd.index("--output"):cmd.index("--output")] = ["--rollout-policy", str(student)]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-10:])
    return job, "PASS" if proc.returncode == 0 else "FAIL", tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--phase", choices=("probe", "formal"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = list(planned_jobs(args.phase))
    print(json.dumps({"phase": args.phase, "shards": len(jobs),
                      "planned_samples": sum(j["samples"] for j in jobs)}, indent=2), flush=True)
    if args.dry_run:
        print(json.dumps(jobs, indent=2)); return
    failures=[]
    with tempfile.TemporaryDirectory(prefix="turning-v2-") as temp:
        schedules={name:schedule_file(name,Path(temp)) for name in COURSES}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(collect,j,args.teacher,args.student,schedules[j["course"]],args.force) for j in jobs]
            for index,future in enumerate(concurrent.futures.as_completed(futures),1):
                job,status,tail=future.result()
                print(f'[{index:02d}/{len(jobs)}] {status} {job["role"]} {job["course"]} {job["start_name"]}\n{tail}',flush=True)
                if status == "FAIL": failures.append(job)
    if failures: raise SystemExit(f"failed jobs: {failures}")


if __name__ == "__main__":
    main()
