"""
Base Strategy — abstract interface for all trading strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..core.config import Config
from ..core.models import Signal


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name
        config = Config()
        strategy_config = config.strategies.get(name, {})
        self.enabled = strategy_config.get("enabled", True)
        self.weight = strategy_config.get("weight", 1.0)
        self.params = strategy_config.get("parameters", {})

    @abstractmethod
    def analyze(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        """Analyze a symbol's data and optionally return a signal."""
        ...
