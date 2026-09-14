"""
Distributed Rate Limiter & Traffic Shaper Reference Lab
======================================================
Production-grade, zero-dependency Python implementation of:
1. Generic Cell Rate Algorithm (GCRA / Leaky Bucket as a Meter)
2. Two-Tier Hierarchical Rate Limiter with Local In-Memory Token Batching
3. Classic Token Bucket & Sliding Window Counter (for comparative benchmarking)
4. Mock Distributed Redis Engine with atomic Lua simulation & latency injection
5. Chaos Engineering Simulator (Network partition, fail-open/fail-closed tests)
6. Concurrency Benchmark Suite & Verification Tests
"""

import time
import threading
import math
import random
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import unittest


# ============================================================================
# 1. MOCK DISTRIBUTED REDIS STORE (WITH ATOMIC LUA SCRIPT SIMULATION)
# ============================================================================

class MockRedisCluster:
    """
    Thread-safe in-memory simulation of a Redis Cluster.
    Supports atomic Lua script execution, key expiration, latency injection,
    and simulated network partition/outage modes.
    """
    def __init__(self, latency_ms: float = 0.0):
        self._store: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._latency_ms = latency_ms
        self._is_partitioned = False
        self.total_commands = 0
        self.lua_invocations = 0

    def set_latency(self, latency_ms: float):
        with self._lock:
            self._latency_ms = latency_ms

    def set_partitioned(self, partitioned: bool):
        with self._lock:
            self._is_partitioned = partitioned

    def _simulate_network(self):
        if self._is_partitioned:
            raise ConnectionError("Redis Cluster connection timeout (simulated network partition)")
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)

    def get(self, key: str) -> Optional[str]:
        self._simulate_network()
        with self._lock:
            self.total_commands += 1
            entry = self._store.get(key)
            if entry is None:
                return None
            val, expire_at = entry
            if expire_at and time.time() > expire_at:
                del self._store[key]
                return None
            return str(val)

    def set(self, key: str, value: Any, ex_seconds: Optional[float] = None) -> bool:
        self._simulate_network()
        with self._lock:
            self.total_commands += 1
            expire_at = time.time() + ex_seconds if ex_seconds else None
            self._store[key] = (value, expire_at)
            return True

    def eval_gcra_lua(self, key: str, limit: int, period: float, cost: int = 1) -> Tuple[bool, int, float]:
        """
        Simulates the atomic Redis GCRA Lua script execution.
        Emission interval I = period / limit
        Delay variation tolerance L = period
        Returns: (allowed: bool, remaining_tokens: int, retry_after_seconds: float)
        """
        self._simulate_network()
        with self._lock:
            self.total_commands += 1
            self.lua_invocations += 1

            now = time.time()
            emission_interval = period / limit
            tolerance = period

            entry = self._store.get(key)
            if entry is None or (entry[1] and now > entry[1]):
                tat = now
            else:
                tat = float(entry[0])

            # New TAT calculation
            new_tat = max(now, tat) + (cost * emission_interval)
            allow_at = new_tat - tolerance

            if now >= allow_at:
                # Allowed: store new TAT with TTL
                ttl = max(1.0, math.ceil(new_tat - now))
                self._store[key] = (new_tat, now + ttl)
                remaining = max(0, int((tolerance - (new_tat - now)) / emission_interval))
                return True, remaining, 0.0
            else:
                # Throttled: calculate retry-after
                retry_after = round(allow_at - now, 3)
                remaining = 0
                return False, remaining, max(0.001, retry_after)


# ============================================================================
# 2. GENERIC CELL RATE ALGORITHM (GCRA) RATE LIMITER
# ============================================================================

@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_after: float
    retry_after: float
    source: str = "central_redis"


class GCRARateLimiter:
    """
    Generic Cell Rate Algorithm (GCRA / Leaky Bucket as a Meter)
    Standardized in ATM networks and widely used in production (e.g., Stripe, Envoy).
    Tracks only a single timestamp: Theoretical Arrival Time (TAT).
    Memory: Exactly 1 scalar value per active key. Zero sliding-log bloat.
    """
    def __init__(self, redis: MockRedisCluster, limit: int, period: float):
        self.redis = redis
        self.limit = limit
        self.period = period

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        try:
            allowed, remaining, retry_after = self.redis.eval_gcra_lua(
                key=key, limit=self.limit, period=self.period, cost=cost
            )
            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_after=self.period,
                retry_after=retry_after,
                source="redis_gcra"
            )
        except ConnectionError:
            # Default fail-open for unclassified errors
            return RateLimitResult(
                allowed=True,
                remaining=1,
                reset_after=self.period,
                retry_after=0.0,
                source="fail_open_fallback"
            )


# ============================================================================
# 3. TWO-TIER HIERARCHICAL RATE LIMITER (LOCAL BATCHING + REDIS SYNC)
# ============================================================================

class LocalBatchRateLimiter:
    """
    Two-Tier Rate Limiter (Envoy / Stripe Architecture).
    Solves the Centralization Bottleneck:
    - Tier 1: Local in-memory token cache evaluated in-process (< 5 microseconds).
    - Tier 2: Asynchronous batch replenishment from Redis when local quota is low.
    Reduces centralized Redis QPS by a factor of BATCH_SIZE (e.g. 100x).
    """
    def __init__(
        self,
        redis: MockRedisCluster,
        limit: int,
        period: float,
        batch_size: int = 50,
        fail_closed_keys: Optional[set] = None
    ):
        self.redis = redis
        self.limit = limit
        self.period = period
        self.batch_size = batch_size
        self.fail_closed_keys = fail_closed_keys or set()

        # Local in-process cache: key -> [remaining_local_tokens, expiry_timestamp]
        self._local_cache: Dict[str, list] = {}
        self._lock = threading.Lock()
        self.local_hits = 0
        self.remote_syncs = 0
        self.fail_open_events = 0
        self.fail_closed_events = 0

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        now = time.time()

        # Step 1: Check Local Token Cache
        with self._lock:
            if key in self._local_cache:
                entry = self._local_cache[key]
                tokens, expires_at = entry[0], entry[1]
                if now < expires_at and tokens >= cost:
                    entry[0] -= cost
                    self.local_hits += 1
                    return RateLimitResult(
                        allowed=True,
                        remaining=entry[0],
                        reset_after=max(0.0, expires_at - now),
                        retry_after=0.0,
                        source="local_memory_cache"
                    )

        # Step 2: Local cache exhausted or expired -> Fetch Batch from Redis
        try:
            self.remote_syncs += 1
            # Request a batch of tokens from Redis GCRA
            allowed, remaining, retry_after = self.redis.eval_gcra_lua(
                key=key, limit=self.limit, period=self.period, cost=self.batch_size
            )

            if allowed:
                # Successfully acquired batch: replenish local cache
                with self._lock:
                    allocated = self.batch_size - cost
                    self._local_cache[key] = [allocated, now + self.period]
                return RateLimitResult(
                    allowed=True,
                    remaining=allocated,
                    reset_after=self.period,
                    retry_after=0.0,
                    source="redis_batch_acquired"
                )
            else:
                # Upstream quota exhausted: try single unit fallback
                single_allowed, single_rem, single_retry = self.redis.eval_gcra_lua(
                    key=key, limit=self.limit, period=self.period, cost=cost
                )
                return RateLimitResult(
                    allowed=single_allowed,
                    remaining=single_rem,
                    reset_after=self.period,
                    retry_after=single_retry,
                    source="redis_single_fallback"
                )

        except ConnectionError:
            # Step 3: Redis Network Partition / Outage Handling
            is_critical = any(prefix in key for prefix in self.fail_closed_keys)
            if is_critical:
                self.fail_closed_events += 1
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after=self.period,
                    retry_after=1.0,
                    source="fail_closed_security_enforced"
                )
            else:
                self.fail_open_events += 1
                return RateLimitResult(
                    allowed=True,
                    remaining=1,
                    reset_after=self.period,
                    retry_after=0.0,
                    source="fail_open_availability_fallback"
                )


# ============================================================================
# 4. CLASSIC COMPARATIVE ALGORITHMS (TOKEN BUCKET & SLIDING WINDOW)
# ============================================================================

class LocalTokenBucket:
    """Thread-safe classic Token Bucket for comparison."""
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def check(self, tokens_requested: int = 1) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens_requested:
                self.tokens -= tokens_requested
                return True
            return False


class LocalSlidingWindowCounter:
    """Thread-safe Sliding Window Counter using two adjacent sub-windows."""
    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window_seconds = window_seconds
        self.current_window_start = time.time()
        self.prev_count = 0
        self.curr_count = 0
        self.lock = threading.Lock()

    def check(self) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.current_window_start

            if elapsed >= self.window_seconds * 2:
                self.prev_count = 0
                self.curr_count = 0
                self.current_window_start = now
                elapsed = 0
            elif elapsed >= self.window_seconds:
                self.prev_count = self.curr_count
                self.curr_count = 0
                self.current_window_start += self.window_seconds
                elapsed = now - self.current_window_start

            # Calculate weighted average
            weight = max(0.0, 1.0 - (elapsed / self.window_seconds))
            estimated_count = (self.prev_count * weight) + self.curr_count

            if estimated_count < self.limit:
                self.curr_count += 1
                return True
            return False


# ============================================================================
# 5. CHAOS SIMULATOR & RESILIENCE VALIDATOR
# ============================================================================

class ChaosSimulator:
    """
    Validates rate limiter behavior under severe fault injection:
    - High Redis latency injection (e.g. 20ms WAN lag)
    - Redis Cluster network partition (ConnectionError)
    - Mixed traffic evaluation: Fail-Open for public APIs vs Fail-Closed for auth/payments
    """
    @staticmethod
    def run_fault_injection_scenario(duration_seconds: float = 2.0) -> Dict[str, Any]:
        redis = MockRedisCluster(latency_ms=0.5)
        # Auth and payment endpoints must FAIL-CLOSED
        fail_closed_routes = {"auth:", "payment:"}
        limiter = LocalBatchRateLimiter(
            redis=redis,
            limit=100,
            period=1.0,
            batch_size=10,
            fail_closed_keys=fail_closed_routes
        )

        results = {
            "public_requests": 0,
            "public_allowed": 0,
            "auth_requests": 0,
            "auth_allowed": 0,
            "partition_induced": False,
        }

        start_time = time.time()
        step = 0

        while time.time() - start_time < duration_seconds:
            step += 1
            # Inject partition at midpoint
            if step == 200:
                redis.set_partitioned(True)
                results["partition_induced"] = True

            # Simulate public API traffic
            results["public_requests"] += 1
            res_pub = limiter.check("public:search:user_123")
            if res_pub.allowed:
                results["public_allowed"] += 1

            # Simulate critical payment traffic
            results["auth_requests"] += 1
            res_auth = limiter.check("payment:checkout:user_456")
            if res_auth.allowed:
                results["auth_allowed"] += 1

            time.sleep(0.002)

        # Restore
        redis.set_partitioned(False)
        results["local_hits"] = limiter.local_hits
        results["remote_syncs"] = limiter.remote_syncs
        results["fail_open_events"] = limiter.fail_open_events
        results["fail_closed_events"] = limiter.fail_closed_events
        return results


# ============================================================================
# 6. HIGH-CONCURRENCY BENCHMARK RUNNER
# ============================================================================

def run_concurrency_benchmark(num_threads: int = 8, requests_per_thread: int = 1000):
    """
    Benchmarks throughput and P99 latency of LocalBatchRateLimiter vs Centralized GCRA.
    """
    print(f"\n==================================================================")
    print(f"  RUNNING HIGH-CONCURRENCY RATE LIMITER BENCHMARK")
    print(f"  Threads: {num_threads} | Total Requests: {num_threads * requests_per_thread}")
    print(f"==================================================================")

    # 1. Benchmark Centralized GCRA (Every request hits Redis)
    redis_central = MockRedisCluster(latency_ms=0.1)  # 0.1ms network hop
    gcra_central = GCRARateLimiter(redis=redis_central, limit=50000, period=1.0)

    latencies_central = []
    lock_central = threading.Lock()

    def worker_central():
        local_lats = []
        for _ in range(requests_per_thread):
            t0 = time.perf_counter()
            gcra_central.check("tenant_benchmark_key")
            local_lats.append((time.perf_counter() - t0) * 1000)
        with lock_central:
            latencies_central.extend(local_lats)

    threads = [threading.Thread(target=worker_central) for _ in range(num_threads)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_central_total = time.perf_counter() - t_start

    latencies_central.sort()
    p50_c = latencies_central[int(len(latencies_central) * 0.50)]
    p99_c = latencies_central[int(len(latencies_central) * 0.99)]
    qps_central = (num_threads * requests_per_thread) / t_central_total

    # 2. Benchmark Local Batch Rate Limiter (Local memory batching)
    redis_batch = MockRedisCluster(latency_ms=0.1)
    batch_limiter = LocalBatchRateLimiter(
        redis=redis_batch, limit=50000, period=1.0, batch_size=100
    )

    latencies_batch = []
    lock_batch = threading.Lock()

    def worker_batch():
        local_lats = []
        for _ in range(requests_per_thread):
            t0 = time.perf_counter()
            batch_limiter.check("tenant_benchmark_key")
            local_lats.append((time.perf_counter() - t0) * 1000)
        with lock_batch:
            latencies_batch.extend(local_lats)

    threads = [threading.Thread(target=worker_batch) for _ in range(num_threads)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_batch_total = time.perf_counter() - t_start

    latencies_batch.sort()
    p50_b = latencies_batch[int(len(latencies_batch) * 0.50)]
    p99_b = latencies_batch[int(len(latencies_batch) * 0.99)]
    qps_batch = (num_threads * requests_per_thread) / t_batch_total

    print(f"\n[1] Centralized Redis GCRA (Pure Remote):")
    print(f"    Total Throughput:  {qps_central:,.0f} req/sec")
    print(f"    P50 Latency:       {p50_c:.3f} ms")
    print(f"    P99 Latency:       {p99_c:.3f} ms")
    print(f"    Redis Commands:    {redis_central.total_commands:,}")

    print(f"\n[2] Two-Tier Local Batching (Envoy/Stripe Pattern):")
    print(f"    Total Throughput:  {qps_batch:,.0f} req/sec")
    print(f"    P50 Latency:       {p50_b:.3f} ms")
    print(f"    P99 Latency:       {p99_b:.3f} ms")
    print(f"    Redis Commands:    {redis_batch.total_commands:,} (Reduction: {redis_central.total_commands / max(1, redis_batch.total_commands):.1f}x)")
    print(f"    Local Cache Hits:  {batch_limiter.local_hits:,}")


# ============================================================================
# 7. UNIT AND INTEGRATION TESTS
# ============================================================================

class TestRateLimiters(unittest.TestCase):

    def setUp(self):
        self.redis = MockRedisCluster(latency_ms=0.0)

    def test_gcra_burst_and_throttle(self):
        # Limit 5 requests per 1.0 second
        limiter = GCRARateLimiter(self.redis, limit=5, period=1.0)
        key = "test_user_burst"

        # First 5 should succeed
        for i in range(5):
            res = limiter.check(key)
            self.assertTrue(res.allowed, f"Request {i+1} should be allowed")

        # 6th should be throttled
        res_throttled = limiter.check(key)
        self.assertFalse(res_throttled.allowed, "Request 6 should be throttled")
        self.assertGreater(res_throttled.retry_after, 0.0, "Retry-After must be positive")

    def test_gcra_token_replenishment(self):
        limiter = GCRARateLimiter(self.redis, limit=5, period=0.2)
        key = "test_user_refill"

        # Exhaust tokens
        for _ in range(5):
            limiter.check(key)
        self.assertFalse(limiter.check(key).allowed)

        # Wait for replenishment interval (0.2s / 5 = 0.04s)
        time.sleep(0.06)
        res = limiter.check(key)
        self.assertTrue(res.allowed, "Token should have replenished")

    def test_local_batching_efficiency(self):
        limiter = LocalBatchRateLimiter(self.redis, limit=100, period=1.0, batch_size=20)
        key = "test_batch_key"

        # Send 15 requests
        for _ in range(15):
            res = limiter.check(key)
            self.assertTrue(res.allowed)

        # Redis should only have been called ONCE for the initial batch of 20
        self.assertEqual(self.redis.lua_invocations, 1)
        self.assertEqual(limiter.local_hits, 14)

    def test_fail_open_vs_fail_closed_under_partition(self):
        fail_closed_keys = {"auth:", "billing:"}
        limiter = LocalBatchRateLimiter(
            self.redis, limit=10, period=1.0, batch_size=5, fail_closed_keys=fail_closed_keys
        )

        # Disconnect Redis
        self.redis.set_partitioned(True)

        # Public route should FAIL-OPEN
        res_public = limiter.check("public:search:user_1")
        self.assertTrue(res_public.allowed)
        self.assertEqual(res_public.source, "fail_open_availability_fallback")

        # Auth route should FAIL-CLOSED
        res_auth = limiter.check("auth:login:user_1")
        self.assertFalse(res_auth.allowed)
        self.assertEqual(res_auth.source, "fail_closed_security_enforced")


if __name__ == "__main__":
    # 1. Run Unit Tests
    print("==================================================================")
    print("  RUNNING SUITE OF 4 UNIT & INTEGRATION TESTS")
    print("==================================================================")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRateLimiters)
    runner = unittest.TextTestRunner(verbosity=2)
    test_res = runner.run(suite)

    if test_res.wasSuccessful():
        # 2. Run Fault Injection / Chaos Lab
        print("\n==================================================================")
        print("  RUNNING CHAOS INJECTION SIMULATION (Network Partition & Failover)")
        print("==================================================================")
        chaos_report = ChaosSimulator.run_fault_injection_scenario(duration_seconds=1.0)
        print(f"  Public Requests:     {chaos_report['public_requests']} (Allowed: {chaos_report['public_allowed']})")
        print(f"  Auth Requests:       {chaos_report['auth_requests']} (Allowed: {chaos_report['auth_allowed']})")
        print(f"  Fail-Open Events:    {chaos_report['fail_open_events']}")
        print(f"  Fail-Closed Events:  {chaos_report['fail_closed_events']}")
        print(f"  Partition Injected:  {chaos_report['partition_induced']}")

        # 3. Run High-Concurrency Benchmark
        run_concurrency_benchmark(num_threads=8, requests_per_thread=2500)
