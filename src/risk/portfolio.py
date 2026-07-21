"""
Portfolio-level risk — pure functions, no side effects.

Position-level risk (per-trade stops, sizing) lives in manager.py. This module
covers risk that only exists at the portfolio level: total drawdown from peak,
sector concentration, and beta to the market.
"""

from __future__ import annotations

# Sector map for the traded universe. Anything unlisted falls into "other",
# which is deliberately NOT capped — an unknown symbol shouldn't be silently
# lumped in with an unrelated sector.
SECTOR_MAP: dict[str, str] = {
    # Technology / semis
    "AAPL": "tech", "MSFT": "tech", "NVDA": "semis", "AMD": "semis",
    "INTC": "semis", "CRM": "tech", "ORCL": "tech", "ADBE": "tech",
    "CSCO": "tech", "QCOM": "semis", "AVGO": "semis", "TXN": "semis",
    "MU": "semis", "AMAT": "semis", "PLTR": "tech",
    # Communication / media
    "GOOGL": "communication", "META": "communication", "NFLX": "communication",
    "DIS": "communication", "T": "communication", "VZ": "communication",
    "CMCSA": "communication",
    # Consumer
    "AMZN": "consumer", "TSLA": "consumer", "HD": "consumer", "NKE": "consumer",
    "SBUX": "consumer", "MCD": "consumer", "LOW": "consumer", "COST": "consumer",
    "TGT": "consumer", "UBER": "consumer",
    # Financials
    "JPM": "financials", "BAC": "financials", "GS": "financials",
    "MS": "financials", "WFC": "financials", "C": "financials",
    "AXP": "financials", "SCHW": "financials", "COIN": "financials",
    # Healthcare
    "JNJ": "healthcare", "PFE": "healthcare", "MRK": "healthcare",
    "ABBV": "healthcare", "UNH": "healthcare", "GILD": "healthcare",
    "AMGN": "healthcare", "CVS": "healthcare",
    # Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "KMI": "energy", "OXY": "energy", "XLE": "energy",
    # Industrials / materials / utilities / staples
    "CAT": "industrials", "BA": "industrials", "GE": "industrials",
    "HON": "industrials", "UPS": "industrials", "DE": "industrials",
    "LIN": "materials", "FCX": "materials", "NEM": "materials",
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
    "PG": "staples", "KO": "staples", "PEP": "staples", "WMT": "staples",
    # Broad index ETFs — their own bucket (they're the market, not a sector)
    "SPY": "index", "QQQ": "index", "IWM": "index", "XLF": "financials",
    # Crypto
    "BTC/USD": "crypto", "ETH/USD": "crypto",
}


def sector_of(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "other")


def equity_drawdown(current_equity: float, peak_equity: float) -> float:
    """Fractional drawdown from the peak (0.0 when at/above the peak)."""
    if not peak_equity or peak_equity <= 0:
        return 0.0
    return max(0.0, (peak_equity - current_equity) / peak_equity)


def sector_exposure(positions, equity: float) -> dict[str, float]:
    """Fraction of equity held in each sector: {sector: fraction}."""
    if not equity or equity <= 0:
        return {}
    out: dict[str, float] = {}
    for p in positions:
        sector = sector_of(p.symbol)
        out[sector] = out.get(sector, 0.0) + (getattr(p, "market_value", 0.0) or 0.0) / equity
    return out


def sector_exposure_for(symbol: str, positions, equity: float) -> float:
    """Current exposure fraction of the sector `symbol` belongs to."""
    if sector_of(symbol) == "other":
        return 0.0  # unknown sector — don't group with anything
    return sector_exposure(positions, equity).get(sector_of(symbol), 0.0)


def portfolio_beta(positions, betas: dict[str, float], equity: float) -> float:
    """
    Equity-weighted beta of the book. Symbols missing from `betas` are
    assumed to have beta 1.0 (market-like) rather than being ignored.
    """
    if not equity or equity <= 0 or not positions:
        return 0.0
    total = 0.0
    for p in positions:
        weight = (getattr(p, "market_value", 0.0) or 0.0) / equity
        total += weight * betas.get(p.symbol, 1.0)
    return total
