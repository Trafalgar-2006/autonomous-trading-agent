"""Tests for portfolio-level risk: drawdown, sectors, beta, trailing stop."""

from datetime import datetime

from src.core.models import Position, Side
from src.risk.manager import RiskManager
from src.risk.portfolio import (
    equity_drawdown,
    portfolio_beta,
    sector_exposure,
    sector_exposure_for,
    sector_of,
)


def pos(symbol, value, price=100.0):
    """Position whose market_value works out to `value`."""
    return Position(symbol=symbol, side=Side.BUY, quantity=value / price,
                    entry_price=price, entry_time=datetime.utcnow(),
                    current_price=price)


# ── Drawdown ──────────────────────────────────────────────────────

def test_no_drawdown_at_peak():
    assert equity_drawdown(100_000, 100_000) == 0.0


def test_drawdown_from_peak():
    assert abs(equity_drawdown(75_000, 100_000) - 0.25) < 1e-9


def test_above_peak_is_not_negative():
    assert equity_drawdown(120_000, 100_000) == 0.0


# ── Sectors ───────────────────────────────────────────────────────

def test_sector_lookup():
    assert sector_of("JPM") == "financials"
    assert sector_of("NVDA") == "semis"
    assert sector_of("BTC/USD") == "crypto"
    assert sector_of("WEIRDCO") == "other"


def test_sector_exposure_aggregates():
    positions = [pos("JPM", 20_000), pos("BAC", 20_000), pos("AAPL", 10_000)]
    exposure = sector_exposure(positions, equity=100_000)
    assert abs(exposure["financials"] - 0.40) < 1e-9
    assert abs(exposure["tech"] - 0.10) < 1e-9


def test_sector_exposure_for_symbol():
    positions = [pos("JPM", 30_000)]
    # GS is also financials, so it sees the existing financials exposure
    assert abs(sector_exposure_for("GS", positions, 100_000) - 0.30) < 1e-9
    # Unknown sector never inherits concentration
    assert sector_exposure_for("WEIRDCO", positions, 100_000) == 0.0


# ── Beta ──────────────────────────────────────────────────────────

def test_portfolio_beta_weighted():
    positions = [pos("AAPL", 50_000), pos("KO", 50_000)]
    beta = portfolio_beta(positions, {"AAPL": 1.4, "KO": 0.6}, equity=100_000)
    assert abs(beta - 1.0) < 1e-9


def test_missing_beta_defaults_to_market():
    positions = [pos("AAPL", 100_000)]
    assert abs(portfolio_beta(positions, {}, 100_000) - 1.0) < 1e-9


# ── Trailing equity stop ──────────────────────────────────────────

def test_trailing_stop_trips_beyond_limit():
    r = RiskManager()
    r.update_equity(100_000)                      # sets the peak
    limit = r.config.max_drawdown_pct
    r.update_equity(100_000 * (1 - limit - 0.02))  # clearly beyond
    assert r.status["cooldown_active"] is True


def test_trailing_stop_quiet_within_limit():
    r = RiskManager()
    r.update_equity(100_000)
    r.update_equity(100_000 * (1 - r.config.max_drawdown_pct / 2))
    assert r.status["cooldown_active"] is False


def test_peak_equity_ratchets_up():
    r = RiskManager()
    r.update_equity(100_000)
    r.update_equity(120_000)
    r.update_equity(110_000)
    assert r.status["peak_equity"] == 120_000
    assert abs(r.status["drawdown"] - (10_000 / 120_000)) < 1e-9
