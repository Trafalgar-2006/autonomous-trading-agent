"""Tests for the per-strategy performance feedback loop."""

from src.strategy.ensemble import SignalEnsemble, performance_multiplier


def test_multiplier_neutral_below_min_trades():
    assert performance_multiplier({"trades": 3, "win_rate": 1.0}) == 1.0


def test_multiplier_neutral_at_even_win_rate():
    assert performance_multiplier({"trades": 50, "win_rate": 0.5}) == 1.0


def test_multiplier_caps():
    assert performance_multiplier({"trades": 50, "win_rate": 1.0}) == 1.5
    assert performance_multiplier({"trades": 50, "win_rate": 0.0}) == 0.5


class _FakeStore:
    def get_strategy_performance(self):
        # Covers every strategy the ensemble might load; assertions below only
        # rely on ones that are enabled (momentum is disabled by config).
        return {
            "momentum": {"trades": 50, "win_rate": 0.9},
            "mean_reversion": {"trades": 50, "win_rate": 0.9},   # strong -> up-weight
            "breakout": {"trades": 50, "win_rate": 0.1},         # weak -> down-weight
        }


def test_update_weights_from_store():
    e = SignalEnsemble()
    e.update_performance_weights(_FakeStore(), min_trades=10)
    # Only enabled strategies get weights; check the two that are always enabled.
    assert e.performance_weights.get("mean_reversion", 1.0) > 1.0
    assert e.performance_weights.get("breakout", 1.0) < 1.0


def test_too_few_trades_stays_neutral():
    class Store:
        def get_strategy_performance(self):
            return {"mean_reversion": {"trades": 3, "win_rate": 1.0}}
    e = SignalEnsemble()
    e.update_performance_weights(Store(), min_trades=10)
    assert e.performance_weights.get("mean_reversion", 1.0) == 1.0


def test_update_weights_survives_bad_store():
    class Broken:
        def get_strategy_performance(self):
            raise RuntimeError("db down")

    e = SignalEnsemble()
    e.update_performance_weights(Broken())  # must not raise
