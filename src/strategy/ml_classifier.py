"""
ML Regime Classifier — detects market regime using machine learning.

Classifies the market into regimes:
- Trending Up
- Trending Down
- Mean Reverting
- High Volatility
- Low Volatility

Uses a Random Forest trained on technical features.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ..core.models import MarketRegime

logger = logging.getLogger(__name__)

MODEL_PATH = Path("data/regime_model.pkl")

# Features used for classification
FEATURES = [
    "rsi_14", "adx", "macd_hist", "bb_width", "bb_pct",
    "atr_pct", "volume_ratio", "return_1d", "return_5d", "return_20d",
    "volatility_20d",
]

# Default strategy weights per regime
DEFAULT_WEIGHTS = {
    MarketRegime.TRENDING_UP: {
        "momentum": 1.4,
        "mean_reversion": 0.6,
        "breakout": 1.2,
    },
    MarketRegime.TRENDING_DOWN: {
        "momentum": 1.2,
        "mean_reversion": 0.8,
        "breakout": 0.8,
    },
    MarketRegime.MEAN_REVERTING: {
        "momentum": 0.6,
        "mean_reversion": 1.4,
        "breakout": 0.6,
    },
    MarketRegime.HIGH_VOLATILITY: {
        "momentum": 0.8,
        "mean_reversion": 0.6,
        "breakout": 1.2,
    },
    MarketRegime.LOW_VOLATILITY: {
        "momentum": 1.0,
        "mean_reversion": 1.0,
        "breakout": 1.0,
    },
}


class RegimeClassifier:
    """ML-based market regime classifier."""

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self._load_model()

    def _load_model(self):
        """Load a pre-trained model if available."""
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    saved = pickle.load(f)
                self.model = saved["model"]
                self.scaler = saved["scaler"]
                logger.info("Loaded pre-trained regime classifier")
            except Exception as e:
                logger.warning(f"Could not load regime model: {e}")

    def _save_model(self):
        """Save the trained model to disk."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        logger.info(f"Regime model saved to {MODEL_PATH}")

    def _label_regime(self, df: pd.DataFrame) -> pd.Series:
        """Create training labels from price data."""
        labels = pd.Series(index=df.index, dtype=str)
        
        returns_20d = df["close"].pct_change(20)
        volatility = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
        vol_median = volatility.median()
        
        for i in range(len(df)):
            ret = returns_20d.iloc[i] if pd.notna(returns_20d.iloc[i]) else 0
            vol = volatility.iloc[i] if pd.notna(volatility.iloc[i]) else vol_median
            adx_val = df["adx"].iloc[i] if "adx" in df.columns and pd.notna(df["adx"].iloc[i]) else 20

            if vol > vol_median * 1.5:
                labels.iloc[i] = MarketRegime.HIGH_VOLATILITY.value
            elif vol < vol_median * 0.5:
                labels.iloc[i] = MarketRegime.LOW_VOLATILITY.value
            elif ret > 0.05 and adx_val > 25:
                labels.iloc[i] = MarketRegime.TRENDING_UP.value
            elif ret < -0.05 and adx_val > 25:
                labels.iloc[i] = MarketRegime.TRENDING_DOWN.value
            else:
                labels.iloc[i] = MarketRegime.MEAN_REVERTING.value

        return labels

    def _extract_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract feature matrix from DataFrame."""
        available = [f for f in FEATURES if f in df.columns]
        if len(available) < len(FEATURES) * 0.7:
            return None

        features = df[available].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()
        return features

    def train(self, data: dict[str, pd.DataFrame], save: bool = True) -> bool:
        """Train the regime classifier on enriched historical data.

        Set save=False during walk-forward research so per-fold models don't
        overwrite the production model on disk.
        """
        all_features = []
        all_labels = []

        for symbol, df in data.items():
            features = self._extract_features(df)
            if features is None or len(features) < 60:
                continue

            labels = self._label_regime(df)
            labels = labels.loc[features.index]

            all_features.append(features)
            all_labels.append(labels)

        if not all_features:
            logger.warning("Not enough data to train regime classifier")
            return False

        X = pd.concat(all_features)
        y = pd.concat(all_labels)

        # Remove any remaining NaNs
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            logger.warning(f"Only {len(X)} samples — too few for training")
            return False

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)

        # Log accuracy
        accuracy = self.model.score(X_scaled, y)
        logger.info(f"Regime classifier trained: {len(X)} samples, accuracy={accuracy:.2%}")

        # Save model (skipped during walk-forward research to avoid clobbering
        # the production model).
        if save:
            self._save_model()
        return True

    def classify(self, df: pd.DataFrame) -> MarketRegime:
        """Classify the current market regime."""
        if self.model is None or self.scaler is None:
            return self._heuristic_classify(df)

        features = self._extract_features(df)
        if features is None or features.empty:
            return self._heuristic_classify(df)

        try:
            latest = features.iloc[[-1]]
            X = self.scaler.transform(latest)
            prediction = self.model.predict(X)[0]
            return MarketRegime(prediction)
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return self._heuristic_classify(df)

    def classify_series(self, df: pd.DataFrame) -> dict:
        """
        Classify every bar at once (date -> MarketRegime).

        Used by the research backtester to precompute regimes rather than
        calling classify() once per bar. Uses the same model/scaler (or the
        same heuristic) as classify(), so results are consistent.
        """
        out: dict = {}
        if df is None or df.empty:
            return out

        # ── Model path ────────────────────────────────────────
        if self.model is not None and self.scaler is not None:
            feats = self._extract_features(df)
            if feats is not None and not feats.empty:
                try:
                    preds = self.model.predict(self.scaler.transform(feats))
                    for date, p in zip(feats.index, preds):
                        out[date] = MarketRegime(p)
                    return out
                except Exception as e:
                    logger.warning(f"Batch classification failed: {e}")

        # ── Heuristic path (row-wise, matches _heuristic_classify) ──
        returns = df["close"].pct_change()
        ret_20d = df["close"].pct_change(20)
        vol = returns.rolling(20).std() * np.sqrt(252)
        adx = df["adx"] if "adx" in df.columns else pd.Series(20.0, index=df.index)
        for date in df.index:
            r = ret_20d.get(date)
            v = vol.get(date)
            a = adx.get(date)
            r = r if pd.notna(r) else 0.0
            v = v if pd.notna(v) else 0.2
            a = a if pd.notna(a) else 20.0
            if v > 0.4:
                out[date] = MarketRegime.HIGH_VOLATILITY
            elif v < 0.1:
                out[date] = MarketRegime.LOW_VOLATILITY
            elif r > 0.05 and a > 25:
                out[date] = MarketRegime.TRENDING_UP
            elif r < -0.05 and a > 25:
                out[date] = MarketRegime.TRENDING_DOWN
            else:
                out[date] = MarketRegime.MEAN_REVERTING
        return out

    def _heuristic_classify(self, df: pd.DataFrame) -> MarketRegime:
        """Fallback heuristic classification when ML model is unavailable."""
        if df is None or len(df) < 20:
            return MarketRegime.LOW_VOLATILITY

        returns = df["close"].pct_change()
        ret_20d = df["close"].pct_change(20).iloc[-1] if len(df) >= 20 else 0
        vol = returns.tail(20).std() * np.sqrt(252) if len(df) >= 20 else 0.2
        adx_val = df["adx"].iloc[-1] if "adx" in df.columns and pd.notna(df["adx"].iloc[-1]) else 20

        if vol > 0.4:
            return MarketRegime.HIGH_VOLATILITY
        elif vol < 0.1:
            return MarketRegime.LOW_VOLATILITY
        elif ret_20d > 0.05 and adx_val > 25:
            return MarketRegime.TRENDING_UP
        elif ret_20d < -0.05 and adx_val > 25:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.MEAN_REVERTING

    def get_strategy_weights(self, regime: MarketRegime) -> dict[str, float]:
        """Get strategy weights for a given regime."""
        return DEFAULT_WEIGHTS.get(regime, DEFAULT_WEIGHTS[MarketRegime.LOW_VOLATILITY])
