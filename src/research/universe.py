"""
Broader research universe for cross-sectional experiments.

A wider, sector-diversified set of liquid US equities. Cross-sectional
momentum benefits from a bigger ranking pool than the ~24 core symbols. These
are all liquid large/mid caps with history back to at least 2019 (the free
Alpaca IEX feed's practical floor). Truly illiquid small caps are omitted —
this data source can't reliably support them, which is a real limitation to
note before drawing conclusions about small-cap edge.
"""

BROAD_UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AMD", "INTC", "CRM", "ORCL", "ADBE", "CSCO",
    "QCOM", "AVGO", "TXN", "MU", "AMAT",
    # Communication / media
    "GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "NKE", "SBUX", "MCD", "LOW", "COST",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "C", "AXP", "SCHW",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "UNH", "GILD", "AMGN",
    # Energy
    "XOM", "CVX", "COP", "SLB", "KMI", "OXY",
    # Industrials
    "CAT", "BA", "GE", "HON", "UPS", "DE",
    # Materials / utilities / staples
    "LIN", "FCX", "NEM", "NEE", "DUK", "PG", "KO", "PEP", "WMT",
    # Benchmark / ETFs (kept for regime context)
    "SPY", "QQQ", "IWM", "XLE", "XLF",
]
