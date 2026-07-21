"""Tests for ops: config validation and the heartbeat healthcheck."""

from datetime import datetime, timedelta

from src.core.config import Config
from src.core.validate import validate_config
from src.ops.healthcheck import check, heartbeat_age_seconds


class _FakeConfig:
    """Minimal stand-in exposing the fields validate_config reads."""
    def __init__(self, **over):
        self.settings = {"general": {"strategy_mode": over.pop("strategy_mode", "ensemble")}}
        self.execution_mode = over.pop("execution_mode", "auto")
        self.max_risk_per_trade = over.pop("max_risk_per_trade", 0.01)
        self.max_position_size = over.pop("max_position_size", 0.10)
        self.max_total_exposure = over.pop("max_total_exposure", 0.80)
        self.max_daily_loss = over.pop("max_daily_loss", 0.02)
        self.max_weekly_loss = over.pop("max_weekly_loss", 0.05)
        self.max_correlated_exposure = over.pop("max_correlated_exposure", 0.30)
        self.max_open_positions = over.pop("max_open_positions", 8)
        self.vol_target = over.pop("vol_target", None)
        self.is_paper = over.pop("is_paper", True)
        self.market_filter = over.pop("market_filter", True)


# ── Config validation ─────────────────────────────────────────────

def test_real_config_is_valid():
    errors, _ = validate_config(Config())
    assert errors == [], f"shipped config should be valid, got: {errors}"


def test_percent_typo_is_an_error():
    # 10 instead of 0.10 — the classic footgun
    errors, _ = validate_config(_FakeConfig(max_position_size=10))
    assert any("max_position_size" in e for e in errors)


def test_unknown_strategy_mode_is_an_error():
    errors, _ = validate_config(_FakeConfig(strategy_mode="moon_phase"))
    assert any("strategy_mode" in e for e in errors)


def test_position_size_above_total_exposure_is_an_error():
    errors, _ = validate_config(_FakeConfig(max_position_size=0.9, max_total_exposure=0.5))
    assert any("exceeds max_total_exposure" in e for e in errors)


def test_live_mode_warns():
    _, warnings_ = validate_config(_FakeConfig(is_paper=False))
    assert any("LIVE" in w for w in warnings_)


def test_market_filter_off_warns():
    _, warnings_ = validate_config(_FakeConfig(market_filter=False))
    assert any("market filter" in w for w in warnings_)


# ── Heartbeat healthcheck ─────────────────────────────────────────

def test_missing_heartbeat_is_unhealthy(tmp_path):
    healthy, msg = check(path=tmp_path / "nope.txt")
    assert healthy is False
    assert "no heartbeat" in msg


def test_fresh_heartbeat_is_healthy(tmp_path):
    hb = tmp_path / "heartbeat.txt"
    hb.write_text(datetime.utcnow().isoformat())
    healthy, msg = check(max_age=3600, path=hb)
    assert healthy is True
    assert "healthy" in msg


def test_stale_heartbeat_is_unhealthy(tmp_path):
    hb = tmp_path / "heartbeat.txt"
    hb.write_text((datetime.utcnow() - timedelta(hours=5)).isoformat())
    healthy, msg = check(max_age=3600, path=hb)
    assert healthy is False
    assert "stale" in msg


def test_heartbeat_age_reads_timestamp(tmp_path):
    hb = tmp_path / "heartbeat.txt"
    hb.write_text((datetime.utcnow() - timedelta(seconds=120)).isoformat())
    age = heartbeat_age_seconds(hb)
    assert age is not None and 100 < age < 200
