"""TD completion audit across one or more portable NPZ dataset chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from focus_segments import (
    PRIVILEGED_SEGMENTS, PRIVILEGED_TARGET_WAYPOINTS, ROUTE_TARGET_WAYPOINTS,
    load_fail_segments, official_focus_waypoints,
)
from schema import validate_dataset

ROOT = Path(__file__).resolve().parents[2]


def audit_coverage(paths, focus_ids, require_full_route=True):
    all_fail_ids = tuple(sorted(set(int(value) for value in focus_ids)))
    focus_ids = official_focus_waypoints(all_fail_ids)
    observed_route_targets = set()
    privileged_targets = set(PRIVILEGED_TARGET_WAYPOINTS) | {next_wp for _, next_wp in PRIVILEGED_SEGMENTS}
    privileged_samples = {target: 0 for target in sorted(privileged_targets)}
    per_waypoint = {
        wp: {"samples": 0, "pre_failure": 0, "success": 0} for wp in focus_ids
    }
    totals = {
        "samples": 0,
        "flat": 0,
        "danger": 0,
        "rewritten": 0,
        "post_teleport": 0,
    }
    command_contract_ok = True
    official_teacher_only = True
    for path in paths:
        validate_dataset(path)
        with np.load(path, allow_pickle=False) as data:
            count = len(data["timestamp_ns"])
            official_teacher_only &= bool(np.all(data["teacher_source"] == 0))
            observed_route_targets.update(
                int(value) for value in np.unique(data["next_wp_id"])
                if int(value) in ROUTE_TARGET_WAYPOINTS
            )
            for target in privileged_samples:
                selected_privileged = data["next_wp_id"] == target
                matching_segments = {wp for wp, next_wp in PRIVILEGED_SEGMENTS if next_wp == target}
                if matching_segments:
                    selected_privileged &= np.isin(data["wp_id"], tuple(matching_segments))
                privileged_samples[target] += int(np.count_nonzero(selected_privileged))
            totals["samples"] += count
            risk = data["risk_features"][:, 7]
            valid = data["heightmap_valid"]
            totals["flat"] += int(np.count_nonzero((risk < 0.1) & valid))
            totals["danger"] += int(np.count_nonzero((risk >= 0.5) & valid))
            difference = np.max(np.abs(data["cmd_raw"] - data["cmd_terrain"]), axis=1)
            totals["rewritten"] += int(np.count_nonzero(difference > 1e-4))
            totals["post_teleport"] += int(np.count_nonzero(data["post_teleport"]))
            command_contract_ok &= bool(np.allclose(
                data["obs_student"][:, 6:9], data["cmd_raw"], atol=1e-6))
            command_contract_ok &= bool(np.allclose(
                data["obs_teacher"][:, 6:9], data["cmd_terrain"], atol=1e-6))
            for wp in focus_ids:
                selected = data["wp_id"] == wp
                per_waypoint[wp]["samples"] += int(np.count_nonzero(selected))
                per_waypoint[wp]["pre_failure"] += int(np.count_nonzero(
                    selected & data["pre_failure"]))
                authentic_success = (
                    selected
                    & data["success"]
                    & ~data["pre_failure"]
                    & ~data["post_teleport"]
                )
                per_waypoint[wp]["success"] += int(np.count_nonzero(authentic_success))
    missing_route_targets = sorted(ROUTE_TARGET_WAYPOINTS - observed_route_targets)
    missing_privileged_coverage = [
        target for target, count in privileged_samples.items() if count == 0
    ]
    missing_samples = [wp for wp, value in per_waypoint.items() if value["samples"] == 0]
    missing_failure = [wp for wp, value in per_waypoint.items() if value["pre_failure"] == 0]
    missing_success = [wp for wp, value in per_waypoint.items() if value["success"] == 0]
    complete = (
        totals["flat"] > 0
        and totals["danger"] > 0
        and totals["rewritten"] > 0
        and command_contract_ok
        and official_teacher_only
        and (not require_full_route or not missing_route_targets)
        and (not require_full_route or not missing_privileged_coverage)
        and not missing_samples
        and not missing_failure
        and not missing_success
    )
    return {
        "complete": complete,
        "command_contract_ok": command_contract_ok,
        "official_teacher_only": official_teacher_only,
        "require_full_route": require_full_route,
        "totals": totals,
        "all_fail_waypoints": list(all_fail_ids),
        "official_focus_waypoints": list(focus_ids),
        "privileged_segments": [list(segment) for segment in sorted(PRIVILEGED_SEGMENTS)],
        "privileged_target_waypoints": sorted(PRIVILEGED_TARGET_WAYPOINTS),
        "privileged_coverage_samples": {str(k): v for k, v in privileged_samples.items()},
        "missing_route_targets": missing_route_targets,
        "missing_privileged_coverage": missing_privileged_coverage,
        "per_waypoint": {str(key): value for key, value in per_waypoint.items()},
        "missing_samples": missing_samples,
        "missing_failure": missing_failure,
        "missing_success": missing_success,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument(
        "--fail-segments",
        default=str(ROOT / "results/fail_segments.md"),
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    report = audit_coverage(args.datasets, load_fail_segments(args.fail_segments))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["complete"] and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
