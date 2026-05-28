"""
Market Scanner — discovers trading candidates from the entire market.

Scans all tradable assets on Alpaca and filters by:
- Price range
- Volume
- Fractional share support
"""

from __future__ import annotations

import logging
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from ..core.config import Config
from .feed import MarketDataFeed

logger = logging.getLogger(__name__)


class MarketScanner:
    """Scans the market for trading candidates."""

    def __init__(self, feed: MarketDataFeed):
        self.config = Config()
        self.feed = feed
        self.scanner_config = self.config.settings.get("scanner", {})

        api_key = self.config.alpaca_api_key
        secret_key = self.config.alpaca_secret_key

        if api_key and api_key != "your_api_key_here":
            self.trading_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=self.config.is_paper,
            )
        else:
            self.trading_client = None

    async def scan(self) -> Optional[list[str]]:
        """Scan the market and return a list of symbols to trade."""
        if self.trading_client is None:
            logger.warning("Scanner: No API keys configured")
            return None

        try:
            # Get all tradable US equities
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            assets = self.trading_client.get_all_assets(request)

            min_price = self.scanner_config.get("min_price", 2.0)
            max_price = self.scanner_config.get("max_price", 1500.0)
            min_volume = self.scanner_config.get("min_avg_volume", 500_000)
            max_candidates = self.scanner_config.get("max_candidates", 100)
            require_fractional = self.scanner_config.get("require_fractional", True)

            # Filter assets
            candidates = []
            for asset in assets:
                if not asset.tradable:
                    continue
                if require_fractional and not asset.fractionable:
                    continue
                if asset.exchange in ("OTC",):
                    continue
                candidates.append(asset.symbol)

            logger.info(f"Scanner: {len(candidates)} assets pass basic filters")

            # Fetch recent bars to filter by price and volume
            # Process in batches to avoid API limits
            batch_size = 200
            filtered = []

            for i in range(0, len(candidates), batch_size):
                batch = candidates[i : i + batch_size]
                data = self.feed.get_bars_multi(batch, days=30)

                for symbol, df in data.items():
                    if df.empty or len(df) < 10:
                        continue

                    last_price = df["close"].iloc[-1]
                    avg_volume = df["volume"].tail(20).mean()

                    if min_price <= last_price <= max_price and avg_volume >= min_volume:
                        filtered.append(symbol)

                if len(filtered) >= max_candidates:
                    break

            # Always include core symbols
            core = set(self.config.symbols)
            result = list(core | set(filtered[:max_candidates]))

            logger.info(f"Scanner: Final universe = {len(result)} symbols "
                       f"({len(core)} core + {len(filtered)} discovered)")
            return result

        except Exception as e:
            logger.error(f"Scanner error: {e}")
            return None
