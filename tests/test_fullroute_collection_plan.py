from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "collect" / "fullroute_v1.json"
SCRIPT = ROOT / "scripts" / "collect_fullroute_v1.py"

spec = importlib.util.spec_from_file_location("collect_fullroute_v1", SCRIPT)
collector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(collector)


def test_formal_plan_has_frozen_fresh_dataset_totals():
    cfg = collector.load_config(CONFIG)
    jobs = collector.expand_plan(cfg, {"train", "validation", "test"})
    summary = collector.summarize(jobs)
    assert cfg["dataset_id"] == "fullroute_v1_20260822"
    assert summary["samples_by_phase"] == {
        "test": 8000, "train": 140000, "validation": 12000}
    assert summary["samples_by_phase_category"]["train:flat"] == 30000
    assert summary["samples_by_phase_category"]["train:ramp"] == 20000
    assert summary["samples_by_phase_category"]["train:stairs"] == 30000
    assert summary["samples_by_phase_category"]["train:turning"] == 25000
    assert summary["samples_by_phase_category"]["train:official"] == 25000
    assert summary["samples_by_phase_category"]["train:recovery"] == 10000
    assert all("turning_v2" not in str(job) for job in jobs)


def test_dagger_round_totals_are_30k_25k_18k():
    cfg = collector.load_config(CONFIG)
    jobs = collector.expand_plan(cfg, {"dagger_r1", "dagger_r2", "dagger_r3"})
    assert collector.summarize(jobs)["samples_by_phase"] == {
        "dagger_r1": 30000, "dagger_r2": 25000, "dagger_r3": 18000}


def test_plan_expansion_is_deterministic_and_route_contract_is_official():
    cfg = collector.load_config(CONFIG)
    one = collector.expand_plan(cfg, {"train"})
    two = collector.expand_plan(cfg, {"train"})
    assert one == two
    assert len({job["seed"] for job in one}) == len(one)
    route_jobs = [job for job in one if job["mode"] == "route"]
    assert route_jobs
    assert all(.6 <= job["autonav_vx"] <= 1.0 for job in route_jobs)
    assert all(job["start_waypoint"] == job["waypoint_range"][0]
               for job in route_jobs)
    assert cfg["fixed_contract"]["waypoint_reach_radius_m"] == .20
    assert cfg["fixed_contract"]["privileged_scan_every_policy_frames"] == 1


def test_gate_requires_matching_teacher_and_full_33_waypoint_pass(tmp_path):
    teacher_sha = "a" * 64
    good = {
        "teacher_sha256": teacher_sha,
        "waypoint_range": [0, 32],
        "route_complete": True,
        "teacher_scan_every": 1,
        "waypoint_reach_radius_m": .20,
        "autonav_controller": "official_route_v1",
        "labels_usable": True,
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    collector.validate_gate(path, teacher_sha)
    good["route_complete"] = False
    path.write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(ValueError, match="all 33 official waypoints"):
        collector.validate_gate(path, teacher_sha)


def test_scaled_command_schedule_starts_at_zero():
    text = collector.schedule_rows(
        [[0, .6, 0, 0], [.25, .75, 0, .2], [.5, .9, 0, 0]], 1000)
    assert text.splitlines() == [
        "0,0.600000,0.000000,0.000000",
        "250,0.750000,0.000000,0.200000",
        "500,0.900000,0.000000,0.000000",
    ]
