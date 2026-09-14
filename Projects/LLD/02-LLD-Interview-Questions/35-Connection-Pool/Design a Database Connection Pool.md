---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, connection-pool, singleton]
---

# Design a Database Connection Pool

## 1. Problem Statement
Design a thread-safe connection pool that manages reusable database connections with configurable min/max size, timeout, and health checks.

## 2. Key Implementation (Python)

```python
import threading, time, queue
from typing import Optional

class Connection:
    """Simulated database connection"""
    _counter = 0
    def __init__(self):
        Connection._counter += 1
        self.id = Connection._counter
        self.created_at = time.time()
        self.is_valid = True

    def execute(self, sql: str) -> str:
        return f"[Conn-{self.id}] Executed: {sql}"

    def close(self):
        self.is_valid = False

class ConnectionPool:
    def __init__(self, min_size: int = 2, max_size: int = 10,
                 timeout: float = 5.0, max_idle_time: float = 300):
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        self.max_idle_time = max_idle_time
        self._pool = queue.Queue(maxsize=max_size)
        self._active_count = 0
        self._lock = threading.Lock()

        # Pre-populate with min connections
        for _ in range(min_size):
            self._pool.put(Connection())
            self._active_count += 1

    def get_connection(self) -> Optional[Connection]:
        """Get a connection from the pool (blocking with timeout)"""
        try:
            conn = self._pool.get(timeout=0.1)
            if conn.is_valid:
                return conn
            # Connection invalid, create new
            return self._create_connection()
        except queue.Empty:
            return self._create_connection()

    def release_connection(self, conn: Connection):
        """Return connection to the pool"""
        if conn.is_valid:
            try:
                self._pool.put_nowait(conn)
            except queue.Full:
                conn.close()
                with self._lock:
                    self._active_count -= 1
        else:
            with self._lock:
                self._active_count -= 1

    def _create_connection(self) -> Optional[Connection]:
        with self._lock:
            if self._active_count < self.max_size:
                conn = Connection()
                self._active_count += 1
                return conn
        # Pool exhausted, wait
        try:
            return self._pool.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError("Connection pool exhausted")

    @property
    def available(self) -> int:
        return self._pool.qsize()

    @property
    def active(self) -> int:
        return self._active_count

    def shutdown(self):
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()

class PooledConnection:
    """Context manager for auto-release"""
    def __init__(self, pool: ConnectionPool):
        self._pool = pool
        self._conn = None

    def __enter__(self) -> Connection:
        self._conn = self._pool.get_connection()
        return self._conn

    def __exit__(self, *args):
        if self._conn:
            self._pool.release_connection(self._conn)

if __name__ == "__main__":
    pool = ConnectionPool(min_size=2, max_size=5)

    # Context manager usage
    with PooledConnection(pool) as conn:
        print(conn.execute("SELECT * FROM users"))
    print(f"Available: {pool.available}, Active: {pool.active}")
```

## 3. Patterns: **Singleton** (pool instance) | **Factory** (connection creation) | **Proxy** (PooledConnection wraps Connection)

## 4. Follow-ups
- **Health checks?** Background thread pinging idle connections.
- **Connection leak detection?** Track checkout time, log if not returned.
- **Read/write splitting?** Separate pools for read replicas and primary.

---

**Related:** [[06 - Singleton Pattern]] | [[14 - Proxy Pattern]]
