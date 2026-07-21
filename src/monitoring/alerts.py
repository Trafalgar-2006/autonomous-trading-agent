"""
Telegram Alerts — sends notifications via Telegram bot.
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from ..core.config import Config
from ..core.models import Signal

logger = logging.getLogger(__name__)


class TelegramAlerts:
    """Sends trading alerts via Telegram bot."""

    def __init__(self):
        config = Config()
        self.bot_token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if self.enabled:
            logger.info("Telegram alerts enabled")
        else:
            logger.info("Telegram alerts disabled (no bot token or chat ID)")

    async def send(self, message: str):
        """Send a message via Telegram."""
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(f"Telegram API error: {resp.status} - {body}")
        except asyncio.TimeoutError:
            logger.warning("Telegram send timed out")
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")

    async def notify_signal(self, signal: Signal):
        """Send a formatted signal notification."""
        emoji = "BUY" if signal.action.value == "buy" else "SELL"
        regime = signal.regime.value if signal.regime else "unknown"
        
        msg = (
            f"<b>{emoji} Signal: {signal.symbol}</b>\n"
            f"Strategy: {signal.strategy}\n"
            f"Confidence: {signal.confidence:.0%}\n"
            f"Price: ${signal.entry_price:.2f}\n"
            f"Regime: {regime}\n"
        )
        
        if signal.stop_loss:
            msg += f"Stop Loss: ${signal.stop_loss:.2f}\n"
        if signal.take_profit:
            msg += f"Take Profit: ${signal.take_profit:.2f}\n"
        
        await self.send(msg)

    async def notify_error(self, error: str):
        """Send an error notification."""
        await self.send(f"<b>ERROR</b>\n{error}")

    async def notify_trade_closed(self, symbol: str, pnl: float, pnl_pct: float, reason: str):
        """Send a trade closed notification."""
        result = "WIN" if pnl > 0 else "LOSS"
        msg = (
            f"<b>Trade Closed ({result}): {symbol}</b>\n"
            f"P&L: ${pnl:.2f} ({pnl_pct:.2%})\n"
            f"Reason: {reason}\n"
        )
        await self.send(msg)

    async def notify_eod_summary(
        self,
        account: dict,
        positions: list,
        risk_status: dict,
        stats: dict,
    ):
        """Send an end-of-day summary: equity, open positions, P&L, risk state."""
        equity = account.get("equity", 0) or 0
        cash = account.get("cash", 0) or 0
        open_pnl = sum(getattr(p, "unrealized_pnl", 0) or 0 for p in positions)

        total_trades = stats.get("total_trades") or 0
        wins = stats.get("wins") or 0
        win_rate = (wins / total_trades) if total_trades else 0.0
        total_pnl = stats.get("total_pnl") or 0.0

        cooldown = risk_status.get("cooldown_active", False)

        lines = [
            "<b>End-of-Day Summary</b>",
            f"Equity: ${equity:,.2f}",
            f"Cash: ${cash:,.2f}",
            f"Open positions: {len(positions)} (unrealized ${open_pnl:,.2f})",
            f"Realized P&L (all-time): ${total_pnl:,.2f}",
            f"Closed trades: {total_trades} | Win rate: {win_rate:.0%}",
            f"Daily P&L: ${risk_status.get('daily_pnl', 0):,.2f}",
            f"Circuit breaker: {'ACTIVE' if cooldown else 'OK'}",
        ]
        if positions:
            lines.append("")
            for p in positions[:15]:
                pnl = getattr(p, "unrealized_pnl", 0) or 0
                lines.append(f"  {p.symbol}: {p.quantity:.2f} @ ${p.entry_price:.2f} "
                             f"(${pnl:,.2f})")

        await self.send("\n".join(lines))
