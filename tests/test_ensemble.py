"""Tests for the per-strategy performance feedback loop."""

from src.strategy.ensemble import performance_multiplier, SignalEnsemble


def test_multiplier_neutral_below_min_trades():
    assert performance_multiplier({"trades": 3, "win_rate": 1.0}) == 1.0


def test_multiplier_neutral_at_even_win_rate():
    assert performance_multiplier({"trades": 50, "win_rate": 0.5}) == 1.0


def test_multiplier_caps():
    assert performance_multiplier({"trades": 50, "win_rate": 1.0}) == 1.5
    assert performance_multiplier({"trades": 50, "win_rate": 0.0}) == 0.5


class _FakeStore:
    def get_strategy_performance(self):
        return {
            "momentum": {"trades": 50, "win_rate": 0.9},
            "mean_reversion": {"trades": 50, "win_rate": 0.1},
            "breakout": {"trades": 2, "win_rate": 1.0},  # too few -> neutral
        }


def test_update_weights_from_store():
    e = SignalEnsemble()
    e.update_performance_weights(_FakeStore(), min_trades=10)
    assert e.performance_weights.get("momentum", 1.0) > 1.0
    assert e.performance_weights.get("mean_reversion", 1.0) < 1.0
    assert e.performance_weights.get("breakout", 1.0) == 1.0


def test_update_weights_survives_bad_store():
    class Broken:
        def get_strategy_performance(self):
            raise RuntimeError("db down")

    e = SignalEnsemble()
    e.update_performance_weights(Broken())  # must not raise
