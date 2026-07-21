"""
Event Bus — simple pub/sub for decoupled communication between components.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable

from .models import Event, EventType

logger = logging.getLogger(__name__)


class EventBus:
    """Simple async event bus for component communication."""

    _instance = None
    _handlers: dict[EventType, list[Callable]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._handlers = defaultdict(list)
            cls._instance._history = []
        return cls._instance

    def subscribe(self, event_type: EventType, handler: Callable[[Event], Awaitable[None]]):
        """Subscribe a handler to an event type."""
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe a handler from an event type."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: Event):
        """Publish an event to all subscribers."""
        self._history.append(event)

        # Keep history bounded
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        handlers = self._handlers.get(event.type, [])
        if handlers:
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler {handler.__name__}: {e}")

        logger.debug(f"Event published: {event.type.value} from {event.source}")

    @property
    def history(self) -> list[Event]:
        """Get event history."""
        return list(self._history)

    def clear_history(self):
        """Clear event history."""
        self._history.clear()
