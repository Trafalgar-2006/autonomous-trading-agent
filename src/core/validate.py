"""
Startup config validation — fail fast on nonsensical or dangerous settings.

Catches the class of mistake that silently ruins a run: a risk limit typo'd
into a 10x position, an unknown strategy mode, LIVE mode enabled by accident.
Returns (errors, warnings); the agent refuses to start on errors.
"""

from __future__ import annotations

VALID_STRATEGY_MODES = {"ensemble", "xs_momentum"}
VALID_EXECUTION_MODES = {"auto", "propose"}
VALID_BROKERS = {"alpaca", "paper"}


def validate_config(config) -> tuple[list[str], list[str]]:
    """Validate a Config instance. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    general = config.settings.get("general", {})
    mode = general.get("strategy_mode", "ensemble")
    if mode not in VALID_STRATEGY_MODES:
        errors.append(f"general.strategy_mode '{mode}' is not one of {sorted(VALID_STRATEGY_MODES)}")

    if config.execution_mode not in VALID_EXECUTION_MODES:
        errors.append(f"general.execution_mode '{config.execution_mode}' is not one of "
                      f"{sorted(VALID_EXECUTION_MODES)}")

    broker = general.get("broker", "alpaca")
    if broker not in VALID_BROKERS:
        errors.append(f"general.broker '{broker}' is not one of {sorted(VALID_BROKERS)}")

    # ── Risk sanity: fractions must be sane fractions ──────────────
    fractions = {
        "max_risk_per_trade": config.max_risk_per_trade,
        "max_position_size": config.max_position_size,
        "max_total_exposure": config.max_total_exposure,
        "max_daily_loss": config.max_daily_loss,
        "max_weekly_loss": config.max_weekly_loss,
        "max_correlated_exposure": config.max_correlated_exposure,
    }
    for name, value in fractions.items():
        if value is None:
            errors.append(f"risk_management.{name} is missing")
        elif not (0 < value <= 1):
            errors.append(f"risk_management.{name}={value} must be a fraction in (0, 1] "
                          f"— 0.10 means 10%, not 10")

    if config.max_open_positions <= 0:
        errors.append(f"risk_management.max_open_positions={config.max_open_positions} must be >= 1")

    # ── Cross-field coherence ──────────────────────────────────────
    if (config.max_position_size or 0) > (config.max_total_exposure or 1):
        errors.append("max_position_size exceeds max_total_exposure — a single position "
                      "could breach the portfolio cap")
    if (config.max_daily_loss or 0) > (config.max_weekly_loss or 1):
        warnings.append("max_daily_loss > max_weekly_loss — the weekly breaker can never trip first")

    vol_target = config.vol_target
    if vol_target is not None and not (0 < vol_target <= 2):
        errors.append(f"risk_management.vol_target={vol_target} should be an annualized "
                      f"volatility like 0.20, or null to disable")

    # ── Warnings (allowed, but you should know) ────────────────────
    if not config.is_paper:
        warnings.append("ALPACA_PAPER is false — this is configured for LIVE trading with real money")
    if (config.max_risk_per_trade or 0) > 0.05:
        warnings.append(f"max_risk_per_trade={config.max_risk_per_trade:.0%} is aggressive (>5% per trade)")
    if not config.market_filter:
        warnings.append("market filter is OFF — drawdowns were ~2x worse without it in walk-forward")

    return errors, warnings
