"""Parse TA's fail_segments.md hand-off without owning navigation logic."""

from __future__ import annotations

import re
from pathlib import Path

FAIL_SEGMENT_RE = re.compile(r"\bwp\s*~=\s*(\d+)\b")
PRIVILEGED_SEGMENTS = frozenset({(16, 17)})
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
