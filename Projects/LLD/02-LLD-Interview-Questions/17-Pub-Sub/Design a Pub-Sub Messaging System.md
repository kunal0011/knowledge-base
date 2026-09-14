---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, pub-sub, observer-pattern]
---

# Design a Pub-Sub Messaging System

## 1. Problem Statement
Design a publish-subscribe messaging system where publishers send messages to topics and subscribers receive them asynchronously.

## 2. Class Design

```mermaid
classDiagram
    class MessageBroker {
        -Map~String,Topic~ topics
        +createTopic(name) Topic
        +publish(topicName, message)
        +subscribe(topicName, subscriber)
    }
    class Topic {
        -String name
        -List~Subscriber~ subscribers
        -Queue~Message~ messages
        +publish(message)
        +addSubscriber(sub)
    }
    class Message {
        -String id
        -String body
        -DateTime timestamp
    }
    class Subscriber {
        <<interface>>
        +onMessage(message)*
    }

    MessageBroker *-- Topic
    Topic --> Subscriber
    Topic --> Message
```

## 3. Key Implementation (Python)

```python
from abc import ABC, abstractmethod
from typing import Dict, List
from datetime import datetime
from collections import deque
import threading, uuid

class Message:
    def __init__(self, body: str):
        self.id = str(uuid.uuid4())[:8]
        self.body = body
        self.timestamp = datetime.now()

class Subscriber(ABC):
    @abstractmethod
    def on_message(self, topic: str, message: Message): pass

class PrintSubscriber(Subscriber):
    def __init__(self, name: str):
        self.name = name
    def on_message(self, topic: str, message: Message):
        print(f"  [{self.name}] Got from '{topic}': {message.body}")

class Topic:
    def __init__(self, name: str):
        self.name = name
        self._subscribers: List[Subscriber] = []
        self._lock = threading.Lock()

    def subscribe(self, subscriber: Subscriber):
        with self._lock:
            self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: Subscriber):
        with self._lock:
            self._subscribers.remove(subscriber)

    def publish(self, message: Message):
        with self._lock:
            subs = list(self._subscribers)
        for sub in subs:
            sub.on_message(self.name, message)

class MessageBroker:
    def __init__(self):
        self._topics: Dict[str, Topic] = {}

    def create_topic(self, name: str) -> Topic:
        if name not in self._topics:
            self._topics[name] = Topic(name)
        return self._topics[name]

    def publish(self, topic_name: str, message: Message):
        topic = self._topics.get(topic_name)
        if topic:
            topic.publish(message)

    def subscribe(self, topic_name: str, subscriber: Subscriber):
        topic = self.create_topic(topic_name)
        topic.subscribe(subscriber)

if __name__ == "__main__":
    broker = MessageBroker()
    s1 = PrintSubscriber("Service-A")
    s2 = PrintSubscriber("Service-B")

    broker.subscribe("orders", s1)
    broker.subscribe("orders", s2)
    broker.subscribe("payments", s1)

    broker.publish("orders", Message("Order #123 created"))
    broker.publish("payments", Message("Payment received"))
```

## 4. Patterns: **Observer** (core pub-sub) | **Strategy** (delivery guarantees)

## 5. Follow-ups
- **At-most-once vs at-least-once vs exactly-once?** Acknowledgment + dedup.
- **Message persistence?** Write-ahead log for durability.
- **Consumer groups?** Round-robin delivery among group members.
- **Backpressure?** Bounded queue with overflow strategies.

---

**Related:** [[02 - Observer Pattern]] | [[01 - Strategy Pattern]]
