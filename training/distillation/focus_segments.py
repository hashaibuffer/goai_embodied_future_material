"""Parse TA's fail_segments.md hand-off without owning navigation logic."""

from __future__ import annotations

import re
from pathlib import Path

FAIL_SEGMENT_RE = re.compile(r"\bwp\s*~=\s*(\d+)\b")
# Discrete stair/descent and wall-maze segments are TH's privileged-teacher
# scope. TD retains route coverage there, but does not require official
# success labels the blind policy cannot reliably provide.
PRIVILEGED_SEGMENTS = frozenset({
    (6, 7),    # stepped section confirmed by elevation profile
    (15, 16),  # stepped section confirmed by elevation profile
    (16, 17),  # high-step hand-off
    (17, 18),  # climb A
    (20, 21),  # stepped descent
    (22, 23),  # climb B / Issue #5 high-stair regression
    (23, 24),  # stepped continuation confirmed by elevation profile
    (27, 28),  # climb C entrance
})
PRIVILEGED_TARGET_WAYPOINTS = frozenset(range(28, 33))
ROUTE_TARGET_WAYPOINTS = frozenset(range(33))


def requires_privileged(wp_id, next_wp_id):
    pair = (int(wp_id), int(next_wp_id))
    return pair in PRIVILEGED_SEGMENTS or pair[1] in PRIVILEGED_TARGET_WAYPOINTS


def official_focus_waypoints(waypoint_ids):
    return tuple(sorted(
        wp for wp in {int(value) for value in waypoint_ids}
        if 0 <= wp < 32 and not requires_privileged(wp, wp + 1)
    ))


def load_fail_segments(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TA fail segment hand-off not found: {path}")
    waypoint_ids = {
        int(match.group(1)) for match in FAIL_SEGMENT_RE.finditer(path.read_text())
    }
    if not waypoint_ids:
        raise ValueError(f"no 'wp~=N' entries found in {path}")
    return tuple(sorted(waypoint_ids))
