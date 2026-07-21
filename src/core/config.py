"""
Configuration loader — reads YAML configs and environment variables.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Load .env file
load_dotenv()


class Config:
    """Centralized configuration manager."""

    _instance = None
    _settings: dict = {}
    _risk: dict = {}
    _strategies: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_all()
        return cls._instance

    def _load_all(self):
        """Load all configuration files."""
        config_dir = Path(__file__).parent.parent.parent / "config"

        # Load main settings
        self._settings = self._load_yaml(config_dir / "settings.yaml")

        # Load risk config
        self._risk = self._load_yaml(config_dir / "risk.yaml")

        # Load strategy configs
        self._strategies = {}
        strategies_dir = config_dir / "strategies"
        if strategies_dir.exists():
            for f in strategies_dir.glob("*.yaml"):
                data = self._load_yaml(f)
                name = data.get("strategy", {}).get("name", f.stem)
                self._strategies[name] = data.get("strategy", {})

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Load a YAML file."""
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ── Accessors ──────────────────────────────────────────────

    @property
    def settings(self) -> dict:
        return self._settings

    @property
    def risk(self) -> dict:
        return self._risk.get("risk_management", {})

    @property
    def strategies(self) -> dict[str, dict]:
        return self._strategies

    @property
    def symbols(self) -> list[str]:
        return self._settings.get("universe", {}).get("symbols", [])

    @property
    def timeframe(self) -> str:
        return self._settings.get("universe", {}).get("timeframe", "1Day")

    @property
    def lookback_days(self) -> int:
        return self._settings.get("universe", {}).get("lookback_days", 365)

    @property
    def mode(self) -> str:
        return self._settings.get("general", {}).get("mode", "paper")

    @property
    def execution_mode(self) -> str:
        """'auto' = execute APPROVED trades; 'propose' = memos only, human executes."""
        return self._settings.get("general", {}).get("execution_mode", "auto").lower()

    @property
    def strategy_mode(self) -> str:
        """'ensemble' (TA strategies) or 'xs_momentum' (cross-sectional momentum)."""
        return self._settings.get("general", {}).get("strategy_mode", "ensemble").lower()

    @property
    def xs(self) -> dict:
        return self._settings.get("cross_sectional", {})

    @property
    def db_path(self) -> str:
        return self._settings.get("database", {}).get("path", "data/trading.db")

    # ── Environment Variables ──────────────────────────────────

    @property
    def alpaca_api_key(self) -> str:
        return os.getenv("ALPACA_API_KEY", "")

    @property
    def alpaca_secret_key(self) -> str:
        return os.getenv("ALPACA_SECRET_KEY", "")

    @property
    def is_paper(self) -> bool:
        return os.getenv("ALPACA_PAPER", "true").lower() == "true"

    @property
    def telegram_bot_token(self) -> str:
        return os.getenv("TELEGRAM_BOT_TOKEN", "")

    @property
    def telegram_chat_id(self) -> str:
        return os.getenv("TELEGRAM_CHAT_ID", "")

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")

    # ── Risk Helpers ───────────────────────────────────────────

    _live_equity: float = 0.0  # Updated at runtime from broker

    @property
    def initial_capital(self) -> float:
        """Returns live equity if set, otherwise falls back to YAML config."""
        if self._live_equity > 0:
            return self._live_equity
        return self.risk.get("capital", {}).get("initial", 100000.0)

    def set_live_equity(self, equity: float):
        """Update live equity from broker account data."""
        if equity > 0:
            self._live_equity = equity

    @property
    def max_risk_per_trade(self) -> float:
        return self.risk.get("max_risk_per_trade", 0.02)

    @property
    def max_position_size(self) -> float:
        return self.risk.get("max_position_size", 0.15)

    @property
    def max_total_exposure(self) -> float:
        return self.risk.get("max_total_exposure", 0.70)

    @property
    def max_correlated_exposure(self) -> float:
        return self.risk.get("max_correlated_exposure", 0.30)

    @property
    def correlation_threshold(self) -> float:
        return self.risk.get("correlation_threshold", 0.70)

    @property
    def max_open_positions(self) -> int:
        return self.risk.get("max_open_positions", 5)

    @property
    def max_daily_loss(self) -> float:
        return self.risk.get("max_daily_loss", 0.05)

    @property
    def max_weekly_loss(self) -> float:
        return self.risk.get("max_weekly_loss", 0.10)

    @property
    def max_consecutive_losses(self) -> int:
        return self.risk.get("max_consecutive_losses", 4)

    @property
    def cooldown_hours(self) -> int:
        return self.risk.get("cooldown_hours", 24)

    @property
    def min_risk_reward(self) -> float:
        """Minimum reward:risk for a setup to be APPROVED (else WATCHLIST)."""
        return self.risk.get("min_risk_reward", 1.5)

    @property
    def strong_confidence(self) -> float:
        """Confidence at/above which a setup is considered strong enough to auto-trade."""
        return self.risk.get("strong_confidence", 0.6)

    @property
    def market_filter(self) -> bool:
        """Only open new longs when SPY is above its SMA (validated risk reducer)."""
        return bool(self.risk.get("market_filter", False))

    @property
    def market_filter_sma(self) -> int:
        return int(self.risk.get("market_filter_sma", 200))

    @property
    def vol_target(self):
        """Annualized per-position volatility target for sizing (None = disabled)."""
        return self.risk.get("vol_target", None)

    def get(self, key: str, default: Any = None) -> Any:
        """Generic getter with dot notation (e.g., 'general.mode')."""
        keys = key.split(".")
        val = self._settings
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    def reload(self):
        """Force reload all configs."""
        self._load_all()
