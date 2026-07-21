"""Tests for the market-data feed's crypto/equity routing (no network)."""

from src.data.feed import MarketDataFeed, is_crypto
from src.core.config import Config


def test_is_crypto():
    assert is_crypto("BTC/USD")
    assert is_crypto("ETH/USD")
    assert not is_crypto("AAPL")
    assert not is_crypto("SPY")


def test_get_bars_multi_routes_crypto_and_equities():
    # Bypass __init__ so no real Alpaca clients are constructed.
    feed = MarketDataFeed.__new__(MarketDataFeed)
    feed.config = Config()
    feed._fetch_stock = lambda syms, tf, start: {s: "STOCK" for s in syms}
    feed._fetch_crypto = lambda syms, tf, start: {s: "CRYPTO" for s in syms}

    out = feed.get_bars_multi(["AAPL", "BTC/USD", "MSFT", "ETH/USD"], days=30)
    assert set(out) == {"AAPL", "BTC/USD", "MSFT", "ETH/USD"}
    assert out["AAPL"] == "STOCK" and out["MSFT"] == "STOCK"
    assert out["BTC/USD"] == "CRYPTO" and out["ETH/USD"] == "CRYPTO"


def test_get_bars_multi_equities_only():
    feed = MarketDataFeed.__new__(MarketDataFeed)
    feed.config = Config()
    feed._fetch_stock = lambda syms, tf, start: {s: "STOCK" for s in syms}
    feed._fetch_crypto = lambda syms, tf, start: {s: "CRYPTO" for s in syms}
    out = feed.get_bars_multi(["AAPL", "SPY"], days=30)
    assert set(out) == {"AAPL", "SPY"}
    assert all(v == "STOCK" for v in out.values())
