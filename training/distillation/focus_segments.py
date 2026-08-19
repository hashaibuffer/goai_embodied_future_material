"""Parse TA's fail_segments.md hand-off without owning navigation logic."""

from __future__ import annotations

import re
from pathlib import Path

FAIL_SEGMENT_RE = re.compile(r"\bwp\s*~=\s*(\d+)\b")


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
