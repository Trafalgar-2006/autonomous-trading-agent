"""Tests for the pure stop-loss / trailing-stop math."""

from src.risk.stops import compute_stop_level


def test_base_stop_is_tighter_of_hard_cap_and_atr():
    # entry 100, atr 2 -> ATR stop = 96; hard cap 5% -> 95. max() = 96 (tighter).
    stop, profit_atr = compute_stop_level(entry=100, price=100, atr=2.0,
                                          high_watermark=100, max_loss_pct=0.05)
    assert stop == 96.0
    assert profit_atr == 0.0


def test_hard_cap_used_when_atr_wide():
    # wide ATR (10) -> ATR stop 80; hard cap 5% -> 95. max() = 95 (hard cap wins).
    stop, _ = compute_stop_level(entry=100, price=100, atr=10.0,
                                 high_watermark=100, max_loss_pct=0.05)
    assert stop == 95.0


def test_moves_to_breakeven_after_1x_atr_profit():
    # price 102 = +1 ATR -> stop raised to entry + 0.1*atr = 100.2
    stop, profit_atr = compute_stop_level(entry=100, price=102, atr=2.0,
                                          high_watermark=102, max_loss_pct=0.05)
    assert profit_atr == 1.0
    assert stop == 100.2


def test_trails_below_high_after_2x_atr_profit():
    # +3 ATR profit, high watermark 106 -> trail = 106 - 1.5*2 = 103
    stop, profit_atr = compute_stop_level(entry=100, price=106, atr=2.0,
                                          high_watermark=106, max_loss_pct=0.05)
    assert profit_atr == 3.0
    assert stop == 103.0


def test_stop_never_loosens():
    # A later, lower price must not drop the stop below the trailed level given
    # the same high watermark.
    high_stop, _ = compute_stop_level(entry=100, price=106, atr=2.0,
                                      high_watermark=106, max_loss_pct=0.05)
    pullback_stop, _ = compute_stop_level(entry=100, price=104, atr=2.0,
                                          high_watermark=106, max_loss_pct=0.05)
    assert pullback_stop == high_stop  # still trailing off the same high


def test_zero_atr_falls_back_to_pct():
    # atr 0 -> falls back to 2% of entry (=2). Base stop = max(95, 100-2*2=96)=96
    stop, _ = compute_stop_level(entry=100, price=100, atr=0.0,
                                 high_watermark=100, max_loss_pct=0.05)
    assert stop == 96.0
