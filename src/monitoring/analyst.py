"""
AI Analyst — Claude-powered narrative layer.

Turns the agent's quantitative state into plain-English morning briefs and
end-of-day narratives (the two daily messages), plus optional trade
explanations. This is where the project becomes a genuine *AI* agent rather
than only an ML bot — an LLM explains *why*, in words, what the numbers did.

Uses the Anthropic SDK. Degrades gracefully: with no ANTHROPIC_API_KEY (or the
`anthropic` package missing), it silently disables and every method returns
None, so the rest of the agent is unaffected.
"""

from __future__ import annotations

import logging
import os

from ..core.config import Config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a concise, risk-aware market analyst assisting an autonomous PAPER-trading "
    "agent that trades US equities. You translate the agent's quantitative signals and "
    "portfolio state into plain English for the human overseeing it. Rules: this is paper "
    "trading, not advice; never promise profit or predict prices; never invent numbers or "
    "facts beyond the data you are given; be brief and scannable (a few short lines or "
    "bullets); note risk plainly. If the data is thin, say so rather than embellishing."
)


# ── Pure prompt builders (testable without the API) ────────────────

def build_morning_prompt(account: dict, positions: list, decisions: list) -> str:
    equity = account.get("equity", 0) or 0
    cash = account.get("cash", 0) or 0
    lines = [
        "Write a short MORNING BRIEF for the trader. Cover, in 3-5 short bullets:",
        "current portfolio state, the notable open positions, and what the agent is "
        "watching today. End with a one-line risk reminder.",
        "",
        f"Account equity: ${equity:,.2f}; cash: ${cash:,.2f}.",
        f"Open positions ({len(positions)}):",
    ]
    for p in positions[:12]:
        pnl = getattr(p, "unrealized_pnl", 0) or 0
        lines.append(f"  - {p.symbol}: {p.quantity:.2f} @ ${p.entry_price:.2f} "
                     f"(unrealized ${pnl:,.2f})")
    if not positions:
        lines.append("  - none")

    watch = [d for d in decisions if d.get("status") in ("approved", "watchlist")][:8]
    if watch:
        lines.append("")
        lines.append("Latest candidates from the agent's decision funnel:")
        for d in watch:
            rr = f"{d['risk_reward']:.2f}" if d.get("risk_reward") else "n/a"
            lines.append(f"  - {d['status'].upper()} {d['action'].upper()} {d['symbol']} "
                         f"({d.get('strategy', '?')}, R:R {rr})")
    return "\n".join(lines)


def build_decision_prompt(memo) -> str:
    parts = [
        "In 1-2 sentences, explain to a trader why the agent reached this verdict. "
        "Plain English, no jargon dumps.",
        "",
        f"Symbol: {memo.symbol} | Action: {memo.action.value.upper()} | "
        f"Verdict: {memo.status.value.upper()}",
        f"Strategy: {memo.strategy} | Signal strength: {memo.signal_strength:.0%} | "
        f"Risk level: {memo.risk_level}",
    ]
    if memo.entry:
        parts.append(f"Entry ${memo.entry:.2f}, stop "
                     f"${memo.stop:.2f}" if memo.stop else f"Entry ${memo.entry:.2f}")
    if memo.risk_reward:
        parts.append(f"Reward:risk {memo.risk_reward:.2f}")
    if memo.reasons:
        parts.append("Basis: " + "; ".join(memo.reasons))
    return "\n".join(parts)


def build_eod_prompt(account: dict, positions: list, risk_status: dict, stats: dict) -> str:
    equity = account.get("equity", 0) or 0
    open_pnl = sum(getattr(p, "unrealized_pnl", 0) or 0 for p in positions)
    total = stats.get("total_trades") or 0
    wins = stats.get("wins") or 0
    win_rate = (wins / total) if total else 0.0
    lines = [
        "Write a short END-OF-DAY note for the trader: 3-4 bullets on how the day went "
        "and the current risk posture. Factual and calm.",
        "",
        f"Equity: ${equity:,.2f}.",
        f"Open positions: {len(positions)} (unrealized ${open_pnl:,.2f}).",
        f"Daily P&L: ${risk_status.get('daily_pnl', 0):,.2f}. "
        f"Circuit breaker: {'ACTIVE' if risk_status.get('cooldown_active') else 'OK'}.",
        f"Lifetime closed trades: {total}; win rate {win_rate:.0%}; "
        f"realized P&L ${stats.get('total_pnl') or 0:,.2f}.",
    ]
    return "\n".join(lines)


class AIAnalyst:
    """Claude-backed narrative generator (optional, graceful without a key)."""

    def __init__(self):
        cfg = Config()
        self.settings = cfg.settings.get("ai_analyst", {})
        self.model = self.settings.get("model", "claude-opus-4-8")
        self.effort = self.settings.get("effort", "low")
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._client = None
        self.enabled = False

        if self.settings.get("enabled", True) and self.api_key:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
                self.enabled = True
                logger.info(f"AI Analyst enabled (model={self.model})")
            except Exception as e:
                logger.warning(f"AI Analyst unavailable ({e}) — running without it")
        else:
            logger.info("AI Analyst disabled (no ANTHROPIC_API_KEY or disabled in config)")

    async def _complete(self, user: str, max_tokens: int = 1024) -> str | None:
        if not self.enabled:
            return None
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user}])
        # `effort` is supported on Opus/Sonnet but errors on Haiku — only send it otherwise.
        if self.effort and not self.model.startswith("claude-haiku"):
            kwargs["output_config"] = {"effort": self.effort}
        try:
            resp = await self._client.messages.create(**kwargs)
            if resp.stop_reason == "refusal":
                return None
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return text or None
        except Exception as e:
            logger.warning(f"AI Analyst request failed: {e}")
            return None

    async def morning_brief(self, account, positions, decisions) -> str | None:
        return await self._complete(build_morning_prompt(account, positions, decisions))

    async def explain_decision(self, memo) -> str | None:
        return await self._complete(build_decision_prompt(memo), max_tokens=400)

    async def eod_narrative(self, account, positions, risk_status, stats) -> str | None:
        return await self._complete(build_eod_prompt(account, positions, risk_status, stats))
