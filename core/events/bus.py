import asyncio
from typing import Callable, Dict, List
from core.events.types import BaseEvent

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[BaseEvent], None]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[BaseEvent], None]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event: BaseEvent):
        event_type = event.event_type
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                # Dispatch synchronously for in-memory model (could use asyncio.create_task for async callbacks)
                try:
                    callback(event)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Error in event handler for {event_type}: {e}")
