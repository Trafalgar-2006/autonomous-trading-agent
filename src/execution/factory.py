"""Broker factory — pick the broker implementation from config."""

from __future__ import annotations

import logging

from ..core.config import Config
from .base import BaseBroker

logger = logging.getLogger(__name__)


def make_broker() -> BaseBroker:
    """
    Return the broker named by `general.broker` in settings:
      "alpaca" (default) — Alpaca live/paper via API keys
      "paper"            — the local simulated PaperBroker (no account needed)
    """
    kind = Config().settings.get("general", {}).get("broker", "alpaca").lower()
    if kind == "paper":
        from .paper_broker import PaperBroker
        logger.info("Using PaperBroker (local simulation)")
        return PaperBroker()
    if kind not in ("alpaca", ""):
        logger.warning(f"Unknown broker '{kind}' — falling back to Alpaca")
    from .broker import AlpacaBroker
    return AlpacaBroker()
