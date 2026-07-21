"""
Market-data quality checks.

Bad data silently produces bad trades: a zero price makes position sizing
explode, a stale feed makes yesterday's signal look like today's, a 10x tick
triggers a phantom breakout. These checks catch that before the data reaches
the strategies.

Severity:
  "critical" — do not trade this symbol on this data
  "warning"  — usable, but worth surfacing
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_bars(df: pd.DataFrame, symbol: str = "", max_stale_days: int = 5,
                  max_jump: float = 0.5, min_bars: int = 30) -> list[dict]:
    """
    Check one symbol's OHLCV frame. Returns a list of issue dicts:
    {severity, code, detail}. An empty list means the data looks sane.
    """
    issues: list[dict] = []

    def add(severity: str, code: str, detail: str):
        issues.append({"severity": severity, "code": code,
                       "detail": f"{symbol}: {detail}" if symbol else detail})

    if df is None or len(df) == 0:
        add("critical", "empty", "no bars returned")
        return issues

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        add("critical", "missing_columns", f"missing columns {missing}")
        return issues

    if len(df) < min_bars:
        add("warning", "insufficient_history",
            f"only {len(df)} bars (< {min_bars})")

    # Non-positive prices break sizing and returns outright.
    price_cols = ["open", "high", "low", "close"]
    if (df[price_cols] <= 0).any().any():
        add("critical", "non_positive_price", "contains zero/negative prices")

    if df[price_cols].isna().any().any():
        add("warning", "nan_prices", "contains NaN prices")

    # OHLC must be internally consistent.
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl:
        add("critical", "high_below_low", f"{bad_hl} bars with high < low")

    bad_range = ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
    if bad_range:
        add("critical", "close_outside_range",
            f"{bad_range} bars with close outside [low, high]")

    # Implausible single-bar moves usually mean a bad tick or an unadjusted split.
    returns = df["close"].pct_change().abs()
    jumps = int((returns > max_jump).sum())
    if jumps:
        add("warning", "extreme_jump",
            f"{jumps} bar(s) moved > {max_jump:.0%} (possible bad tick or split)")

    # Staleness — the feed may be serving us yesterday's world.
    try:
        last = pd.to_datetime(df.index.max())
        age_days = (datetime.utcnow() - last.to_pydatetime().replace(tzinfo=None)).days
        if age_days > max_stale_days:
            add("critical", "stale", f"newest bar is {age_days} days old")
    except Exception:
        add("warning", "unparsable_index", "could not read the timestamp index")

    if (df["volume"] <= 0).all():
        add("warning", "no_volume", "all volumes are zero")

    return issues


def filter_tradeable(data: dict[str, pd.DataFrame], **kwargs) -> tuple[dict, list[dict]]:
    """
    Split a {symbol: df} map into (clean_data, all_issues).

    Symbols with any critical issue are dropped — better to skip a name than to
    trade it on data we know is broken.
    """
    clean: dict[str, pd.DataFrame] = {}
    all_issues: list[dict] = []
    for symbol, df in data.items():
        issues = validate_bars(df, symbol=symbol, **kwargs)
        all_issues.extend(issues)
        if not any(i["severity"] == "critical" for i in issues):
            clean[symbol] = df
    return clean, all_issues
