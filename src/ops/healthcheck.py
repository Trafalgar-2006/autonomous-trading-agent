"""
Liveness check — exit 0 if the agent's heartbeat is fresh, 1 otherwise.

The agent touches `data/heartbeat.txt` at the end of every cycle. Used by the
Docker HEALTHCHECK and by the watchdog. Tolerance defaults to 90 minutes
(3x the default 30-minute scan interval) and is overridable with
HEARTBEAT_MAX_AGE (seconds).

    python -m src.ops.healthcheck
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

HEARTBEAT_PATH = Path("data/heartbeat.txt")
DEFAULT_MAX_AGE = 5400  # 90 minutes


def heartbeat_age_seconds(path: Path = HEARTBEAT_PATH) -> Optional[float]:
    """Seconds since the last heartbeat, or None if missing/unreadable."""
    if not path.exists():
        return None
    try:
        ts = datetime.fromisoformat(path.read_text().strip())
    except Exception:
        return None
    return (datetime.utcnow() - ts).total_seconds()


def check(max_age: Optional[int] = None, path: Path = HEARTBEAT_PATH) -> tuple[bool, str]:
    """Return (healthy, message)."""
    limit = max_age if max_age is not None else int(
        os.getenv("HEARTBEAT_MAX_AGE", DEFAULT_MAX_AGE))
    age = heartbeat_age_seconds(path)
    if age is None:
        return False, "no heartbeat recorded (agent has not completed a cycle)"
    if age > limit:
        return False, f"heartbeat stale: {age / 60:.1f} min old (limit {limit / 60:.0f} min)"
    return True, f"healthy: heartbeat {age / 60:.1f} min old"


def main() -> int:
    healthy, message = check()
    print(message)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
