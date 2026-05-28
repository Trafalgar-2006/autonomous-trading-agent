"""
Signal Ensemble — aggregates signals from multiple strategies.

Uses weighted voting to combine signals and filter for high-confidence trades.
Integrates the ML regime classifier to dynamically adjust strategy weights.
Applies a weekly trend filter to block counter-trend BUY signals.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..core.models import Signal, SignalAction, MarketRegime
from .base import BaseStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .breakout import BreakoutStrategy
from .ml_classifier import RegimeClassifier

logger = logging.getLogger(__name__)


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
        
        # Weekly trend cache: symbol -> {"bullish": bool, "rsi": float, "macd_bullish": bool}
        self._weekly_trends: dict[str, dict] = {}
        
        # Initialize all strategies
        self._init_strategies()

    def _init_strategies(self):
        """Initialize and register all enabled strategies."""
        all_strategies = [
            MeanReversionStrategy(),
            MomentumStrategy(),
            BreakoutStrategy(),
        ]
        
        self.strategies = [s for s in all_strategies if s.enabled]
        logger.info(f"Ensemble initialized with {len(self.strategies)} strategies: "
                     f"{[s.name for s in self.strategies]}")

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

    def train_classifier(self, data: dict[str, pd.DataFrame]) -> bool:
        """Train the regime classifier on enriched historical data."""
        return self.classifier.train(data)

    def generate_signals(self, symbol: str, df: pd.DataFrame) -> list[Signal]:
        """
        Run all strategies on a symbol and return aggregated signals.
        
        Pipeline:
        1. Compute weekly trend (multi-timeframe filter)
        2. Detect current market regime (ML classifier)
        3. Run strategies with regime-adjusted weights
        4. Apply weekly trend filter to BUY signals
        """
        # Step 1: Weekly trend
        self.compute_weekly_trend(symbol, df)
        weekly = self._weekly_trends.get(symbol, {})

        # Step 2: Detect regime
        self._detect_regime(df)

        signals: list[Signal] = []

        for strategy in self.strategies:
            try:
                signal = strategy.analyze(symbol, df)
                if signal and signal.action != SignalAction.HOLD:
                    # Apply regime-based weight adjustment
                    regime_mult = self.regime_weights.get(strategy.name, 1.0)
                    signal.confidence = min(1.0, signal.confidence * regime_mult)
                    signal.regime = self.current_regime
                    
                    # Add context to reasoning
                    if signal.reasoning:
                        signal.reasoning["regime"] = self.current_regime.value
                        signal.reasoning["regime_weight"] = regime_mult
                        signal.reasoning["weekly_trend"] = (
                            "bullish" if weekly.get("bullish")
                            else "bearish" if weekly.get("bearish")
                            else "neutral"
                        )
                        signal.reasoning["weekly_rsi"] = weekly.get("rsi", 50)
                    
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

    def _select_best(self, signals: list[Signal]) -> Optional[Signal]:
        """Select the best signal from a list, weighted by strategy weight and confidence."""
        if not signals:
            return None

        strategy_weights = {s.name: s.weight for s in self.strategies}

        scored = []
        for sig in signals:
            weight = strategy_weights.get(sig.strategy, 1.0)
            score = sig.confidence * weight
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
