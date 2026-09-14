---
date: "2026-04-06"
type: lld-question
difficulty: medium
tags: [lld, interview-prep, event-bus, observer-pattern]
---

# Design an Event/Message Bus

## 1. Problem Statement
Design an in-process event bus for decoupled component communication with sync/async delivery, wildcard subscriptions, and event filtering.

## 2. Key Implementation (Python)

```python
import threading
from typing import Callable, Dict, List, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fnmatch

class Event:
    def __init__(self, topic: str, data: Any = None):
        self.topic = topic
        self.data = data

class EventBus:
    def __init__(self, async_mode: bool = False, workers: int = 4):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._async = async_mode
        self._executor = ThreadPoolExecutor(max_workers=workers) if async_mode else None

    def subscribe(self, topic_pattern: str, handler: Callable):
        """Subscribe to a topic. Supports wildcards: 'order.*' matches 'order.created'"""
        with self._lock:
            self._subscribers[topic_pattern].append(handler)

    def unsubscribe(self, topic_pattern: str, handler: Callable):
        with self._lock:
            if topic_pattern in self._subscribers:
                self._subscribers[topic_pattern].remove(handler)

    def publish(self, event: Event):
        with self._lock:
            handlers = []
            for pattern, subs in self._subscribers.items():
                if fnmatch.fnmatch(event.topic, pattern) or pattern == event.topic:
                    handlers.extend(subs)

        for handler in handlers:
            if self._async:
                self._executor.submit(handler, event)
            else:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Handler error: {e}")

    def shutdown(self):
        if self._executor:
            self._executor.shutdown(wait=True)

if __name__ == "__main__":
    bus = EventBus()

    def on_order(event):
        print(f"📦 Order handler: {event.data}")

    def on_any(event):
        print(f"🔔 Audit log: [{event.topic}] {event.data}")

    bus.subscribe("order.created", on_order)
    bus.subscribe("order.*", on_any)

    bus.publish(Event("order.created", {"id": "O123", "total": 99.99}))
    bus.publish(Event("order.shipped", {"id": "O123"}))
```

## 3. Patterns: **Observer** (core pub/sub) | **Mediator** (bus decouples publishers and subscribers)
## 4. Follow-ups: **Dead letter queue?** Failed events go to DLQ for retry | **Event replay?** Store events for replay/reprocessing | **Priority events?** Priority queue for urgent events.

---
**Related:** [[02 - Observer Pattern]] | [[Design a Pub-Sub Messaging System]]
