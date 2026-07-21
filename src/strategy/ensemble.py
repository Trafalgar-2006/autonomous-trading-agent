"""
Signal Ensemble — aggregates signals from multiple strategies.

Uses weighted voting to combine signals and filter for high-confidence trades.
Integrates the ML regime classifier to dynamically adjust strategy weights.
Applies a weekly trend filter to block counter-trend BUY signals.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..core.config import Config
from ..core.models import MarketRegime, Signal, SignalAction
from .base import BaseStrategy
from .breakout import BreakoutStrategy
from .mean_reversion import MeanReversionStrategy
from .ml_classifier import RegimeClassifier
from .momentum import MomentumStrategy

logger = logging.getLogger(__name__)


def performance_multiplier(stats: dict, min_trades: int = 10) -> float:
    """
    Map a strategy's realized performance to a weight multiplier in [0.5, 1.5].

    Strategies with a strong win rate get up-weighted; poor performers get
    down-weighted. Below ``min_trades`` closed trades there isn't enough
    evidence, so the multiplier stays neutral (1.0).
    """
    trades = stats.get("trades", 0) or 0
    if trades < min_trades:
        return 1.0
    win_rate = stats.get("win_rate", 0.5) or 0.0
    # win_rate in [0,1] maps linearly to [0.5, 1.5]; 0.5 -> 1.0 (neutral).
    return max(0.5, min(1.5, 0.5 + win_rate))


class SignalEnsemble:
    """
    Aggregates signals from multiple strategies using weighted voting.
    
    Features:
    - Regime-aware: dynamically adjusts strategy weights via ML classifier
    - Multi-timeframe: uses weekly trend to filter daily signals
    - Ensemble voting: boosts confidence when strategies agree
    """

    def __init__(self, min_confidence: float = 0.45):
        self.min_confidence = min_confidence
        self.strategies: list[BaseStrategy] = []
        self.classifier = RegimeClassifier()
        self.current_regime: MarketRegime = MarketRegime.LOW_VOLATILITY
        self.regime_weights: dict[str, float] = {}

        # Per-strategy performance multipliers (learned from realized P&L).
        # Default neutral until update_performance_weights() is called.
        self.performance_weights: dict[str, float] = {}

        # Volatility-target sizing (from config; None = off). Applied as a
        # per-signal size multiplier consumed by the risk manager.
        self._vol_target = Config().vol_target

        # Weekly trend cache: symbol -> {"bullish": bool, "rsi": float, "macd_bullish": bool}
        self._weekly_trends: dict[str, dict] = {}

        # Initialize all strategies
        self._init_strategies()

    def _init_strategies(self):
        """Initialize and register all enabled strategies."""
        # Keep every strategy instance so research can force-enable any subset
        # regardless of the config `enabled` flags.
        self._all_strategies = [
            MeanReversionStrategy(),
            MomentumStrategy(),
            BreakoutStrategy(),
        ]

        self.strategies = [s for s in self._all_strategies if s.enabled]
        logger.info(f"Ensemble initialized with {len(self.strategies)} strategies: "
                     f"{[s.name for s in self.strategies]}")

    def set_active_strategies(self, names):
        """Override which strategies run, ignoring config `enabled` flags.

        Used by the research/experiment harness to A/B specific strategy sets
        (e.g. with vs without momentum)."""
        wanted = set(names)
        self.strategies = [s for s in self._all_strategies if s.name in wanted]
        return self.strategies

    def _detect_regime(self, df: pd.DataFrame):
        """Update the current market regime from the data."""
        self.current_regime = self.classifier.classify(df)
        self.regime_weights = self.classifier.get_strategy_weights(self.current_regime)

    def compute_weekly_trend(self, symbol: str, df: pd.DataFrame):
        """
        Compute weekly trend indicators from daily data.
        
        Resamples daily bars to weekly, then checks:
        - Weekly RSI (above/below 50)
        - Weekly MACD (bullish/bearish)
        - Weekly EMA alignment (10-week above 40-week)
        
        Result: A "bullish" flag that gates daily BUY signals.
        """
        if df.empty or len(df) < 100:
            self._weekly_trends[symbol] = {"bullish": True, "neutral": True}
            return

        try:
            # Resample daily to weekly
            weekly = df.resample("W").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

            if len(weekly) < 20:
                self._weekly_trends[symbol] = {"bullish": True, "neutral": True}
                return

            close = weekly["close"]

            # Weekly RSI (14-period)
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            weekly_rsi = 100 - (100 / (1 + rs))
            current_rsi = weekly_rsi.iloc[-1] if not weekly_rsi.empty else 50

            # Weekly MACD
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_bullish = bool(macd.iloc[-1] > macd_signal.iloc[-1]) if len(macd) > 0 else True

            # Weekly EMA alignment (10-week vs 40-week)
            ema_10w = close.ewm(span=10, adjust=False).mean()
            ema_40w = close.ewm(span=40, adjust=False).mean()
            ema_aligned = bool(ema_10w.iloc[-1] > ema_40w.iloc[-1]) if len(ema_10w) > 0 else True

            # Trend determination
            bullish_count = sum([
                current_rsi > 50,
                macd_bullish,
                ema_aligned,
            ])

            is_bullish = bullish_count >= 2  # At least 2 of 3 confirm
            is_bearish = bullish_count == 0  # All 3 bearish

            self._weekly_trends[symbol] = {
                "bullish": is_bullish,
                "bearish": is_bearish,
                "neutral": not is_bullish and not is_bearish,
                "rsi": round(float(current_rsi), 1),
                "macd_bullish": macd_bullish,
                "ema_aligned": ema_aligned,
            }

            logger.debug(
                f"Weekly trend {symbol}: "
                f"{'BULL' if is_bullish else 'BEAR' if is_bearish else 'NEUTRAL'} "
                f"(RSI={current_rsi:.1f}, MACD={'up' if macd_bullish else 'down'}, "
                f"EMA={'up' if ema_aligned else 'down'})"
            )

        except Exception as e:
            logger.warning(f"Error computing weekly trend for {symbol}: {e}")
            self._weekly_trends[symbol] = {"bullish": True, "neutral": True}

    def compute_weekly_trend_series(self, df: pd.DataFrame) -> dict:
        """
        Precompute the weekly-trend dict for every daily bar (date -> dict).

        Uses only COMPLETED weeks (each daily bar sees the most recently closed
        weekly bar, never its own in-progress week) so there is no intra-week
        look-ahead. Used by the research backtester.
        """
        out: dict = {}
        if df is None or df.empty or len(df) < 100:
            return out
        try:
            weekly = df.resample("W").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
            if len(weekly) < 20:
                return out

            close = weekly["close"]
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            macd_sig = macd.ewm(span=9, adjust=False).mean()
            macd_bull = macd > macd_sig

            ema_10 = close.ewm(span=10, adjust=False).mean()
            ema_40 = close.ewm(span=40, adjust=False).mean()
            ema_aligned = ema_10 > ema_40

            bull_count = (rsi > 50).astype(int) + macd_bull.astype(int) + ema_aligned.astype(int)
            wk = pd.DataFrame({
                "rsi": rsi,
                "bullish": bull_count >= 2,
                "bearish": bull_count == 0,
            })

            # Weekly index is labelled at each week's right edge (Sunday), which
            # is in the future for intra-week days, so reindex+ffill naturally
            # gives each daily bar the last COMPLETED week's values.
            daily = wk.reindex(df.index, method="ffill")
            for date, row in daily.iterrows():
                if pd.isna(row["rsi"]):
                    out[date] = {"bullish": True, "neutral": True}
                else:
                    b = bool(row["bullish"])
                    br = bool(row["bearish"])
                    out[date] = {
                        "bullish": b,
                        "bearish": br,
                        "neutral": not b and not br,
                        "rsi": round(float(row["rsi"]), 1),
                    }
            return out
        except Exception as e:
            logger.warning(f"Error computing weekly trend series: {e}")
            return out

    def train_classifier(self, data: dict[str, pd.DataFrame], save: bool = True) -> bool:
        """Train the regime classifier on enriched historical data."""
        return self.classifier.train(data, save=save)

    def update_performance_weights(self, store, min_trades: int = 10):
        """
        Refresh per-strategy weight multipliers from realized trade history.

        Call at startup and periodically (e.g. daily) so strategies that are
        actually making money get more say, and losers get down-weighted.
        """
        try:
            perf = store.get_strategy_performance()
        except Exception as e:
            logger.warning(f"Could not load strategy performance: {e}")
            return

        weights = {}
        for strategy in self.strategies:
            stats = perf.get(strategy.name, {})
            weights[strategy.name] = performance_multiplier(stats, min_trades)

        self.performance_weights = weights
        if any(w != 1.0 for w in weights.values()):
            logger.info(f"Strategy performance weights updated: "
                        f"{ {k: round(v, 2) for k, v in weights.items()} }")

    def generate_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        weekly: dict | None = None,
        regime: MarketRegime | None = None,
    ) -> list[Signal]:
        """
        Run all strategies on a symbol and return aggregated signals.

        Pipeline:
        1. Compute weekly trend (multi-timeframe filter)
        2. Detect current market regime (ML classifier)
        3. Run strategies with regime-adjusted weights
        4. Apply weekly trend filter to BUY signals

        `weekly` and `regime` may be injected (precomputed) — the research
        backtester does this to avoid recomputing them per bar. When omitted
        they are computed here (the live path).
        """
        # Step 1: Weekly trend
        if weekly is None:
            self.compute_weekly_trend(symbol, df)
            weekly = self._weekly_trends.get(symbol, {})

        # Step 2: Regime + its strategy weights
        if regime is None:
            regime = self.classifier.classify(df)
        regime_weights = self.classifier.get_strategy_weights(regime)
        # keep instance state in sync for logging / scan_universe summaries
        self.current_regime = regime
        self.regime_weights = regime_weights

        signals: list[Signal] = []

        for strategy in self.strategies:
            try:
                signal = strategy.analyze(symbol, df)
                if signal and signal.action != SignalAction.HOLD:
                    # Apply regime-based weight adjustment
                    regime_mult = regime_weights.get(strategy.name, 1.0)
                    signal.confidence = min(1.0, signal.confidence * regime_mult)
                    signal.regime = regime

                    # Add context to reasoning
                    if signal.reasoning:
                        signal.reasoning["regime"] = regime.value
                        signal.reasoning["regime_weight"] = regime_mult
                        signal.reasoning["weekly_trend"] = (
                            "bullish" if weekly.get("bullish")
                            else "bearish" if weekly.get("bearish")
                            else "neutral"
                        )
                        signal.reasoning["weekly_rsi"] = weekly.get("rsi", 50)

                    # Volatility-target sizing multiplier (for BUY entries).
                    if (self._vol_target and signal.action == SignalAction.BUY
                            and "volatility_20d" in df.columns):
                        vol = df["volatility_20d"].iloc[-1]
                        if pd.notna(vol) and vol > 0:
                            signal.reasoning["size_mult"] = float(
                                min(2.0, max(0.5, self._vol_target / float(vol))))

                    # Average daily dollar volume — drives the liquidity guard.
                    if "volume_sma_20" in df.columns:
                        adv = df["volume_sma_20"].iloc[-1]
                        px = df["close"].iloc[-1]
                        if pd.notna(adv) and pd.notna(px) and adv > 0:
                            signal.reasoning["adv_notional"] = float(adv) * float(px)

                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in strategy {strategy.name} for {symbol}: {e}")

        if not signals:
            return []

        # Step 3: Apply weekly trend filter
        buy_signals = [s for s in signals if s.action == SignalAction.BUY]
        sell_signals = [s for s in signals if s.action == SignalAction.SELL]

        result = []

        # Process BUY signals — only blocked if weekly trend is STRONGLY bearish (RSI < 40)
        if buy_signals:
            weekly_rsi = weekly.get("rsi", 50)
            if weekly.get("bearish") and weekly_rsi < 40:
                logger.info(
                    f"Blocking {len(buy_signals)} BUY signal(s) for {symbol} "
                    f"(weekly trend is STRONGLY BEARISH: RSI={weekly_rsi})"
                )
            else:
                best_buy = self._select_best(buy_signals)
                if best_buy and best_buy.confidence >= self.min_confidence:
                    # Boost if weekly confirms
                    if weekly.get("bullish"):
                        best_buy.confidence = min(1.0, best_buy.confidence + 0.05)
                        best_buy.reasoning["weekly_confirmed"] = True

                    # Boost if multiple strategies agree
                    if len(buy_signals) > 1:
                        best_buy.confidence = min(1.0, best_buy.confidence + 0.1)
                        best_buy.reasoning["ensemble_agreement"] = len(buy_signals)

                    result.append(best_buy)

        # Process SELL signals — always allowed
        if sell_signals:
            best_sell = self._select_best(sell_signals)
            if best_sell:
                if best_sell.confidence >= self.min_confidence * 0.8:
                    # Boost sell confidence if weekly is bearish
                    if weekly.get("bearish"):
                        best_sell.confidence = min(1.0, best_sell.confidence + 0.05)
                    result.append(best_sell)

        return result

    def _select_best(self, signals: list[Signal]) -> Signal | None:
        """Select the best signal from a list, weighted by strategy weight and confidence."""
        if not signals:
            return None

        strategy_weights = {s.name: s.weight for s in self.strategies}

        scored = []
        for sig in signals:
            base_weight = strategy_weights.get(sig.strategy, 1.0)
            perf_weight = self.performance_weights.get(sig.strategy, 1.0)
            score = sig.confidence * base_weight * perf_weight
            scored.append((score, sig))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def scan_universe(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        """
        Scan all symbols in the universe and collect signals.
        """
        all_signals: list[Signal] = []

        for symbol, df in data.items():
            try:
                signals = self.generate_signals(symbol, df)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        # Sort by confidence (highest first)
        all_signals.sort(key=lambda s: s.confidence, reverse=True)

        if all_signals:
            logger.info(f"Universe scan found {len(all_signals)} signals "
                       f"(regime={self.current_regime.value}):")
            for sig in all_signals:
                logger.info(f"  {sig.action.value.upper()} {sig.symbol} "
                           f"({sig.strategy}, conf={sig.confidence:.2f})")

        return all_signals
