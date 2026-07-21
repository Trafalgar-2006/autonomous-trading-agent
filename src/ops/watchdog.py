"""
Dead-man's switch — alerts you (via Telegram) if the agent stops running.

Run it from cron/Task Scheduler on a schedule (e.g. hourly) on the host, or as
a sidecar. It checks the heartbeat and, if stale, sends one alert. It never
trades and never restarts anything — it only tells you something is wrong.

    python -m src.ops.watchdog
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from .healthcheck import check

# Only re-alert once per staleness episode.
_ALERT_FLAG = Path("data/.watchdog_alerted")


async def _notify(message: str) -> None:
    from ..monitoring.alerts import TelegramAlerts
    await TelegramAlerts().send(message)


def main() -> int:
    healthy, message = check()

    if healthy:
        # Recovered — clear the flag and say so once.
        if _ALERT_FLAG.exists():
            _ALERT_FLAG.unlink(missing_ok=True)
            asyncio.run(_notify(f"<b>Trading Agent recovered</b>\n{message}"))
        print(message)
        return 0

    print(f"UNHEALTHY: {message}")
    if not _ALERT_FLAG.exists():
        try:
            _ALERT_FLAG.parent.mkdir(parents=True, exist_ok=True)
            _ALERT_FLAG.write_text("alerted")
        except Exception:
            pass
        asyncio.run(_notify(
            f"<b>Trading Agent DOWN</b>\n{message}\n"
            f"The agent is not completing cycles — check the host/container."
        ))
    return 1


if __name__ == "__main__":
    sys.exit(main())
