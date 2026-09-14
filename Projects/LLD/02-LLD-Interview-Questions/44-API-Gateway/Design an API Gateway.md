---
date: "2026-04-06"
type: lld-question
difficulty: medium
tags: [lld, interview-prep, api-gateway, proxy-pattern, decorator-pattern]
---

# Design an API Gateway

## 1. Problem Statement
Design an API gateway handling routing, authentication, rate limiting, request transformation, and load balancing.

## 2. Key Implementation (Python)

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Optional
import random

class Request:
    def __init__(self, method: str, path: str, headers: Dict = None, body: str = ""):
        self.method = method; self.path = path
        self.headers = headers or {}; self.body = body

class Response:
    def __init__(self, status: int, body: str, headers: Dict = None):
        self.status = status; self.body = body; self.headers = headers or {}

# ──── Middleware Chain (Decorator Pattern) ────

class Middleware(ABC):
    @abstractmethod
    def process(self, request: Request, next_handler: Callable) -> Response: pass

class AuthMiddleware(Middleware):
    def __init__(self, valid_tokens: set):
        self._tokens = valid_tokens

    def process(self, request, next_handler) -> Response:
        token = request.headers.get("Authorization", "")
        if token not in self._tokens:
            return Response(401, "Unauthorized")
        return next_handler(request)

class RateLimitMiddleware(Middleware):
    def __init__(self, max_rps: int = 100):
        self._counts: Dict[str, int] = {}
        self._max = max_rps

    def process(self, request, next_handler) -> Response:
        client = request.headers.get("X-Client-ID", "unknown")
        self._counts[client] = self._counts.get(client, 0) + 1
        if self._counts[client] > self._max:
            return Response(429, "Too Many Requests")
        return next_handler(request)

class LoggingMiddleware(Middleware):
    def process(self, request, next_handler) -> Response:
        print(f"→ {request.method} {request.path}")
        response = next_handler(request)
        print(f"← {response.status}")
        return response

# ──── Router + Load Balancer ────

class ServiceInstance:
    def __init__(self, host: str, port: int):
        self.host = host; self.port = port
    def handle(self, request: Request) -> Response:
        return Response(200, f"Handled by {self.host}:{self.port}")

class LoadBalancer:
    def __init__(self, instances: List[ServiceInstance]):
        self._instances = instances; self._idx = 0

    def round_robin(self) -> ServiceInstance:
        instance = self._instances[self._idx % len(self._instances)]
        self._idx += 1
        return instance

class APIGateway:
    def __init__(self):
        self._routes: Dict[str, LoadBalancer] = {}
        self._middlewares: List[Middleware] = []

    def add_route(self, path_prefix: str, instances: List[ServiceInstance]):
        self._routes[path_prefix] = LoadBalancer(instances)

    def add_middleware(self, middleware: Middleware):
        self._middlewares.append(middleware)

    def handle(self, request: Request) -> Response:
        # Build middleware chain
        def final_handler(req: Request) -> Response:
            for prefix, lb in self._routes.items():
                if req.path.startswith(prefix):
                    instance = lb.round_robin()
                    return instance.handle(req)
            return Response(404, "Not Found")

        handler = final_handler
        for mw in reversed(self._middlewares):
            prev = handler
            handler = lambda req, m=mw, p=prev: m.process(req, p)

        return handler(request)

if __name__ == "__main__":
    gateway = APIGateway()
    gateway.add_middleware(LoggingMiddleware())
    gateway.add_middleware(AuthMiddleware({"token123"}))
    gateway.add_route("/api/users", [ServiceInstance("host1", 8080), ServiceInstance("host2", 8080)])

    req = Request("GET", "/api/users/1", headers={"Authorization": "token123"})
    resp = gateway.handle(req)
    print(f"Response: {resp.status} — {resp.body}")
```

## 3. Patterns: **Proxy** (gateway proxies backend) | **Decorator/Chain of Responsibility** (middleware) | **Strategy** (load balancing algorithms)
## 4. Follow-ups: **Circuit breaker?** Track failure rate, open circuit on threshold | **Caching?** Cache GET responses with TTL | **Request aggregation?** Combine multiple backend calls into one response.

---
**Related:** [[14 - Proxy Pattern]] | [[03 - Decorator Pattern]] | [[01 - Strategy Pattern]]
