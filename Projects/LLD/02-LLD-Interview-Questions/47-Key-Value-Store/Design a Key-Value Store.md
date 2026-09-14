---
date: "2026-04-06"
type: lld-question
difficulty: easy
tags: [lld, interview-prep, key-value-store]
---

# Design a Key-Value Store

## 1. Problem Statement
Design an in-memory key-value store with TTL, persistence, and basic data types (string, list, hash).

## 2. Key Implementation (Python)

```python
import time, json, threading
from typing import Any, Optional, Dict

class Entry:
    def __init__(self, value: Any, ttl_seconds: float = None):
        self.value = value
        self.created_at = time.time()
        self.expiry = time.time() + ttl_seconds if ttl_seconds else None

    def is_expired(self) -> bool:
        return self.expiry is not None and time.time() > self.expiry

class KeyValueStore:
    def __init__(self):
        self._store: Dict[str, Entry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._store[key]
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl: float = None):
        with self._lock:
            self._store[key] = Entry(value, ttl)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def keys(self, pattern: str = "*") -> list:
        with self._lock:
            self._cleanup_expired()
            if pattern == "*":
                return list(self._store.keys())
            import fnmatch
            return [k for k in self._store.keys() if fnmatch.fnmatch(k, pattern)]

    # ──── List operations ────
    def lpush(self, key: str, *values):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._store[key] = Entry(list(values))
            else:
                entry.value = list(values) + entry.value

    def rpush(self, key: str, *values):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._store[key] = Entry(list(values))
            else:
                entry.value.extend(values)

    def lrange(self, key: str, start: int, end: int) -> list:
        val = self.get(key)
        if val is None: return []
        return val[start:end + 1] if end >= 0 else val[start:]

    # ──── Hash operations ────
    def hset(self, key: str, field: str, value: Any):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._store[key] = Entry({field: value})
            else:
                entry.value[field] = value

    def hget(self, key: str, field: str) -> Optional[Any]:
        val = self.get(key)
        return val.get(field) if isinstance(val, dict) else None

    # ──── Persistence ────
    def save(self, filepath: str):
        with self._lock:
            data = {k: {"value": e.value, "expiry": e.expiry}
                    for k, e in self._store.items() if not e.is_expired()}
            with open(filepath, 'w') as f:
                json.dump(data, f)

    def load(self, filepath: str):
        with open(filepath) as f:
            data = json.load(f)
        with self._lock:
            for k, v in data.items():
                ttl = v["expiry"] - time.time() if v["expiry"] else None
                if ttl is None or ttl > 0:
                    self._store[k] = Entry(v["value"], ttl)

    def _cleanup_expired(self):
        expired = [k for k, e in self._store.items() if e.is_expired()]
        for k in expired:
            del self._store[k]
```

## 3. Follow-ups: **Pub/sub?** Observer pattern for key change events | **Transactions?** MULTI/EXEC with command buffering | **Cluster?** Consistent hashing for sharding.

---
**Related:** [[06 - Singleton Pattern]] | [[02 - Observer Pattern]]
