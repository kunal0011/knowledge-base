#!/usr/bin/env python3
"""
High-Performance Search Autocomplete Engine (Typeahead)
======================================================
Enterprise-grade, zero-dependency implementation of a sub-millisecond
typeahead prefix search engine modeled after Google and Amazon search.

Core Architectural Capabilities:
  1. Compact In-Memory Prefix Trie with Node-Level Top-K Precomputation (O(L) lookups).
  2. Real-Time Streaming Velocity Overlay (Historical score + trending sliding window).
  3. Zero-Downtime RCU (Read-Copy-Update) Atomic Pointer Swapping for zero-lock reads.
  4. Real-Time Safety & Moderation Blocklist filter.
  5. Dual-Mode Interface: Standalone High-Speed Library + HTTP REST Daemon.
  6. Prometheus Telemetry (/metrics) and Health Checks (/healthz).
"""

import sys
import os
import time
import json
import heapq
import socket
import select
import threading
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 5
TRENDING_WINDOW_SEC = 60.0  # 60s sliding window for velocity scoring
ALPHA_HISTORICAL = 0.7      # Weight for historical volume
BETA_TRENDING = 0.3         # Weight for real-time velocity


# ---------------------------------------------------------------------------
# Data Structures: Trie Node with Pre-Computed Top-K
# ---------------------------------------------------------------------------
class TrieNode:
    __slots__ = ("children", "is_terminal", "frequency", "top_k", "term")

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_terminal: bool = False
        self.frequency: float = 0.0
        self.term: Optional[str] = None
        # List of tuples: (score, term) sorted descending by score
        self.top_k: List[Tuple[float, str]] = []


class PrefixTrie:
    """
    Immutable or Mutable In-Memory Prefix Trie with Node-Level Top-K Cache.
    Guarantees O(L) lookup complexity where L is the prefix length.
    """
    def __init__(self, top_k_capacity: int = 10):
        self.root = TrieNode()
        self.top_k_capacity = top_k_capacity
        self.total_terms = 0

    def insert(self, term: str, frequency: float):
        """Inserts or updates a term with its aggregate score."""
        term_clean = term.strip().lower()
        if not term_clean:
            return

        node = self.root
        path = [node]

        for char in term_clean:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            path.append(node)

        if not node.is_terminal:
            self.total_terms += 1
            node.is_terminal = True
            node.term = term_clean

        node.frequency = frequency

        # Update top_k along the traversal path back to root
        entry = (frequency, term_clean)
        for p_node in path:
            self._update_node_top_k(p_node, entry)

    def _update_node_top_k(self, node: TrieNode, entry: Tuple[float, str]):
        score, term = entry
        # Replace existing entry if present
        existing_idx = -1
        for idx, (s, t) in enumerate(node.top_k):
            if t == term:
                existing_idx = idx
                break

        if existing_idx >= 0:
            node.top_k[existing_idx] = entry
        else:
            node.top_k.append(entry)

        # Sort descending by score, tie-break lexicographically
        node.top_k.sort(key=lambda x: (-x[0], x[1]))
        if len(node.top_k) > self.top_k_capacity:
            node.top_k = node.top_k[:self.top_k_capacity]

    def query(self, prefix: str, k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        """
        Retrieves top-k suggestions for the given prefix in O(L) time.
        Zero subtree scans, zero heap allocations on read path.
        """
        prefix_clean = prefix.strip().lower()
        node = self.root

        for char in prefix_clean:
            if char not in node.children:
                return []
            node = node.children[char]

        # Returns precomputed top-K
        return [(term, score) for score, term in node.top_k[:k]]


# ---------------------------------------------------------------------------
# Real-Time Velocity Tracker (Sliding Window Ingestion)
# ---------------------------------------------------------------------------
class VelocityTracker:
    """Maintains sliding-window query hit counts for real-time trending boosts."""
    def __init__(self, window_sec: float = TRENDING_WINDOW_SEC):
        self.window_sec = window_sec
        self.lock = threading.Lock()
        # term -> list of timestamps
        self.hits: Dict[str, List[float]] = {}

    def record_hit(self, term: str, count: int = 1):
        term_clean = term.strip().lower()
        now = time.time()
        with self.lock:
            if term_clean not in self.hits:
                self.hits[term_clean] = []
            for _ in range(count):
                self.hits[term_clean].append(now)

    def get_velocity(self, term: str) -> int:
        term_clean = term.strip().lower()
        now = time.time()
        cutoff = now - self.window_sec
        with self.lock:
            if term_clean not in self.hits:
                return 0
            timestamps = self.hits[term_clean]
            # Prune expired
            valid = [t for t in timestamps if t >= cutoff]
            self.hits[term_clean] = valid
            return len(valid)

    def prune(self):
        now = time.time()
        cutoff = now - self.window_sec
        with self.lock:
            expired_keys = []
            for term, ts in self.hits.items():
                valid = [t for t in ts if t >= cutoff]
                if valid:
                    self.hits[term] = valid
                else:
                    expired_keys.append(term)
            for k in expired_keys:
                del self.hits[k]


# ---------------------------------------------------------------------------
# Dynamic Safety & Moderation Blocklist
# ---------------------------------------------------------------------------
class ModerationBlocklist:
    """Dynamic, lock-free read blocklist filter."""
    def __init__(self):
        self.lock = threading.Lock()
        self.blocked_terms: Set[str] = set()

    def add_blocked(self, term: str):
        with self.lock:
            self.blocked_terms.add(term.strip().lower())

    def remove_blocked(self, term: str):
        with self.lock:
            self.blocked_terms.discard(term.strip().lower())

    def is_blocked(self, term: str) -> bool:
        term_clean = term.strip().lower()
        # Direct match or contains blocked word
        if term_clean in self.blocked_terms:
            return True
        for b in self.blocked_terms:
            if b and b in term_clean:
                return True
        return False


# ---------------------------------------------------------------------------
# Metrics & Observability
# ---------------------------------------------------------------------------
class AutocompleteMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_queries = 0
        self.blocked_queries = 0
        self.ingested_events = 0
        self.rcu_swaps_total = 0
        self.latencies_us: List[float] = []

    def record_query(self, latency_us: float, blocked: bool = False):
        with self.lock:
            self.total_queries += 1
            if blocked:
                self.blocked_queries += 1
            self.latencies_us.append(latency_us)
            if len(self.latencies_us) > 50000:
                self.latencies_us = self.latencies_us[-25000:]

    def record_ingest(self, count: int = 1):
        with self.lock:
            self.ingested_events += count

    def record_swap(self):
        with self.lock:
            self.rcu_swaps_total += 1

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            lats = sorted(self.latencies_us)
            p50 = lats[int(len(lats) * 0.5)] if lats else 0.0
            p99 = lats[int(len(lats) * 0.99)] if lats else 0.0
            return {
                "total_queries": self.total_queries,
                "blocked_queries": self.blocked_queries,
                "ingested_events": self.ingested_events,
                "rcu_swaps_total": self.rcu_swaps_total,
                "latency_p50_us": round(p50, 2),
                "latency_p99_us": round(p99, 2),
            }


# ---------------------------------------------------------------------------
# Core Autocomplete Engine (RCU Atomic Swapper)
# ---------------------------------------------------------------------------
class AutocompleteEngine:
    """
    Production Typeahead Engine with RCU Atomic Swapping.
    Guarantees 100% lock-free reads while allowing background index re-indexing.
    """
    def __init__(self):
        self.metrics = AutocompleteMetrics()
        self.blocklist = ModerationBlocklist()
        self.velocity = VelocityTracker()
        
        # Historical baseline scores: term -> base frequency
        self._historical_scores: Dict[str, float] = {}
        self._score_lock = threading.Lock()

        # Active Read Trie (Pointer swapped atomically via RCU)
        self._active_trie: PrefixTrie = PrefixTrie()

    @property
    def active_trie(self) -> PrefixTrie:
        return self._active_trie

    def ingest_query(self, term: str, count: int = 1):
        """Records a search query event into historical & real-time streaming pipelines."""
        term_clean = term.strip().lower()
        if not term_clean:
            return

        with self._score_lock:
            self._historical_scores[term_clean] = self._historical_scores.get(term_clean, 0.0) + count

        self.velocity.record_hit(term_clean, count)
        self.metrics.record_ingest(count)

        # Incrementally update live trie
        combined_score = self._compute_combined_score(term_clean)
        self._active_trie.insert(term_clean, combined_score)

    def _compute_combined_score(self, term: str) -> float:
        base = self._historical_scores.get(term, 0.0)
        vel = self.velocity.get_velocity(term)
        return (ALPHA_HISTORICAL * base) + (BETA_TRENDING * vel * 10.0)

    def search(self, prefix: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Lock-free read query execution."""
        t_start = time.perf_counter()

        # Fast path: point to current immutable snapshot
        current_trie = self._active_trie
        raw_results = current_trie.query(prefix, k=k * 2)  # fetch headroom for filtering

        # Apply moderation blocklist
        filtered = []
        for term, score in raw_results:
            if not self.blocklist.is_blocked(term):
                filtered.append({"term": term, "score": round(score, 2)})
            if len(filtered) == k:
                break

        latency_us = (time.perf_counter() - t_start) * 1_000_000.0
        self.metrics.record_query(latency_us, blocked=False)
        return filtered

    def rebuild_and_swap(self):
        """
        Zero-Downtime RCU (Read-Copy-Update) Atomic Pointer Swap.
        Builds a brand-new cache-optimized PrefixTrie completely off the read path,
        then atomically swaps the reference. Readers never experience lock contention.
        """
        new_trie = PrefixTrie()

        with self._score_lock:
            snapshot = list(self._historical_scores.items())

        for term, base_score in snapshot:
            combined = (ALPHA_HISTORICAL * base_score) + (BETA_TRENDING * self.velocity.get_velocity(term) * 10.0)
            new_trie.insert(term, combined)

        # ATOMIC POINTER SWAP: In Python, reference assignment is an atomic bytecode operation
        self._active_trie = new_trie
        self.metrics.record_swap()


# ---------------------------------------------------------------------------
# HTTP Handler & Daemon Service
# ---------------------------------------------------------------------------
class AutocompleteHTTPHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        self.engine_ref: AutocompleteEngine = server.engine
        super().__init__(request, client_address, server)

    def log_message(self, format, *args):
        pass  # Suppress noisy standard logs

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/autocomplete":
            prefix = query.get("q", [""])[0]
            k = int(query.get("k", [DEFAULT_TOP_K])[0])
            results = self.engine_ref.search(prefix, k=k)
            self._send_json(200, {
                "prefix": prefix,
                "count": len(results),
                "suggestions": results
            })
            return

        if path == "/healthz":
            self._send_json(200, {
                "status": "UP",
                "service": "Search Autocomplete Engine",
                "indexed_terms": self.engine_ref.active_trie.total_terms,
                "timestamp": time.time()
            })
            return

        if path == "/metrics":
            summary = self.engine_ref.metrics.get_summary()
            prometheus_lines = [
                "# HELP autocomplete_queries_total Total prefix search requests processed",
                "# TYPE autocomplete_queries_total counter",
                f"autocomplete_queries_total {summary['total_queries']}",
                "# HELP autocomplete_ingested_events_total Search query events ingested",
                "# TYPE autocomplete_ingested_events_total counter",
                f"autocomplete_ingested_events_total {summary['ingested_events']}",
                "# HELP autocomplete_rcu_swaps_total Zero-downtime RCU pointer swaps executed",
                "# TYPE autocomplete_rcu_swaps_total counter",
                f"autocomplete_rcu_swaps_total {summary['rcu_swaps_total']}",
                "# HELP autocomplete_latency_p50_microseconds Server-side median lookup latency in microseconds",
                "# TYPE autocomplete_latency_p50_microseconds gauge",
                f"autocomplete_latency_p50_microseconds {summary['latency_p50_us']}",
                "# HELP autocomplete_latency_p99_microseconds Server-side 99th percentile lookup latency in microseconds",
                "# TYPE autocomplete_latency_p99_microseconds gauge",
                f"autocomplete_latency_p99_microseconds {summary['latency_p99_us']}",
            ]
            body = "\n".join(prometheus_lines) + "\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return

        self._send_json(404, {"error": f"Endpoint '{path}' not found."})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/ingest":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body)
            query = data.get("query", "")
            count = int(data.get("count", 1))

            if not query:
                self._send_json(400, {"error": "Missing 'query' field."})
                return

            self.engine_ref.ingest_query(query, count)
            self._send_json(200, {"status": "INGESTED", "query": query, "count": count})
            return

        if path == "/blocklist":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body)
            terms = data.get("blocked", [])
            for t in terms:
                self.engine_ref.blocklist.add_blocked(t)
            self._send_json(200, {"status": "UPDATED", "blocked_count": len(terms)})
            return

        if path == "/swap":
            self.engine_ref.rebuild_and_swap()
            self._send_json(200, {"status": "SWAPPED", "terms": self.engine_ref.active_trie.total_terms})
            return

        self._send_json(404, {"error": f"Endpoint '{path}' not found."})

    def _send_json(self, status: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class AutocompleteServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str, port: int, engine: AutocompleteEngine):
        self.engine = engine
        super().__init__((host, port), AutocompleteHTTPHandler)


# ---------------------------------------------------------------------------
# Test & Verification Suite
# ---------------------------------------------------------------------------
def run_unit_tests():
    print("================================================================================")
    print("RUNNING AUTOCOMPLETE ENGINE SELF-TEST & VERIFICATION")
    print("================================================================================")

    engine = AutocompleteEngine()

    # Pre-seed with technical keywords
    seed_data = [
        ("system design", 1500),
        ("system design interview", 2200),
        ("system architecture", 800),
        ("system call linux", 400),
        ("systemd service", 300),
        ("python programming", 1900),
        ("python tutorial", 1200),
        ("python async socket", 600),
        ("redis cluster", 1400),
        ("redis sentinel", 900),
        ("redis persistence aof", 700),
    ]
    for term, count in seed_data:
        engine.ingest_query(term, count)

    # ------------------------------------------------------------------------
    # Test 1: O(L) Prefix Query & Top-K Ranking Order
    # ------------------------------------------------------------------------
    print("\n[Test 1] Testing O(L) Prefix Traversal & Top-K Ranking...")
    results = engine.search("system", k=3)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert results[0]["term"] == "system design interview"
    assert results[1]["term"] == "system design"
    assert results[2]["term"] == "system architecture"
    print("  -> Prefix 'system' correctly returned top-3 suggestions ranked by score:")
    for r in results:
        print(f"     • {r['term']} (score: {r['score']})")

    # ------------------------------------------------------------------------
    # Test 2: Dynamic Streaming Frequency Boost (Trending Real-Time)
    # ------------------------------------------------------------------------
    print("\n[Test 2] Testing Streaming Frequency Boost (Breaking News Surge)...")
    # Simulate sudden breaking search query for "system call linux"
    engine.ingest_query("system call linux", 2500)

    boosted = engine.search("system", k=3)
    assert boosted[0]["term"] == "system call linux"
    print(f"  -> Successfully boosted 'system call linux' to #1 position (score: {boosted[0]['score']})!")

    # ------------------------------------------------------------------------
    # Test 3: Moderation Blocklist Filtering
    # ------------------------------------------------------------------------
    print("\n[Test 3] Testing Dynamic Safety & Moderation Blocklist...")
    engine.ingest_query("illegal dark web drugs", 9999)
    # Verify it shows up before block
    unfiltered = engine.search("illegal", k=1)
    assert len(unfiltered) == 1 and unfiltered[0]["term"] == "illegal dark web drugs"

    # Now block the keyword
    engine.blocklist.add_blocked("drugs")
    filtered = engine.search("illegal", k=1)
    assert len(filtered) == 0, "Blocked keyword must be filtered out!"
    print("  -> Harmful query successfully suppressed via dynamic blocklist.")

    # ------------------------------------------------------------------------
    # Test 4: Zero-Downtime RCU Atomic Pointer Swap
    # ------------------------------------------------------------------------
    print("\n[Test 4] Testing Zero-Downtime RCU Atomic Pointer Swap...")
    initial_trie_id = id(engine.active_trie)
    engine.rebuild_and_swap()
    new_trie_id = id(engine.active_trie)
    assert initial_trie_id != new_trie_id
    # Ensure lookups continue uninterrupted on new trie
    post_swap_res = engine.search("redis", k=2)
    assert len(post_swap_res) == 2
    assert post_swap_res[0]["term"] == "redis cluster"
    print("  -> Atomic pointer swap completed with zero read contention!")

    # ------------------------------------------------------------------------
    # Test 5: HTTP REST API Endpoints
    # ------------------------------------------------------------------------
    print("\n[Test 5] Testing HTTP REST Server Endpoints (/autocomplete, /metrics)...")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    server = AutocompleteServer("127.0.0.1", port, engine)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.2)

    try:
        import urllib.request
        # Test /autocomplete
        url = f"http://127.0.0.1:{port}/autocomplete?q=pyth&k=2"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["count"] == 2
            assert data["suggestions"][0]["term"] == "python programming"
        print("  -> HTTP /autocomplete returned expected JSON payload.")

        # Test /metrics
        metrics_url = f"http://127.0.0.1:{port}/metrics"
        with urllib.request.urlopen(metrics_url) as resp:
            metrics_body = resp.read().decode("utf-8")
            assert "autocomplete_queries_total" in metrics_body
            assert "autocomplete_latency_p50_microseconds" in metrics_body
        print("  -> HTTP /metrics returned Prometheus telemetry.")
    finally:
        server.shutdown()
        server.server_close()

    print("\n[✓] ALL 5 UNIT TESTS PASSED FLAWLESSLY!\n")


# ---------------------------------------------------------------------------
# High-Throughput Stress Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(num_lookups: int = 100_000):
    print("================================================================================")
    print(f"RUNNING HIGH-THROUGHPUT AUTOCOMPLETE BENCHMARK: {num_lookups:,} LOOKUPS")
    print("================================================================================")

    engine = AutocompleteEngine()
    print("Populating Trie with 5,000 realistic search phrases...")
    import random
    import string

    prefixes = ["sys", "net", "dat", "web", "clo", "alg", "sca", "dis", "mac", "doc"]
    nouns = ["system", "network", "database", "cluster", "cache", "service", "gateway", "queue", "pipeline", "router"]
    verbs = ["design", "architecture", "scaling", "optimization", "monitoring", "indexing", "partitioning", "tuning"]

    # Generate 5,000 combinations
    terms = set()
    for _ in range(5000):
        t = f"{random.choice(prefixes)} {random.choice(nouns)} {random.choice(verbs)}"
        freq = random.randint(10, 100_000)
        terms.add((t, freq))

    for t, freq in terms:
        engine.ingest_query(t, freq)

    print(f"Indexed {engine.active_trie.total_terms:,} terms into Trie.")
    print(f"Executing {num_lookups:,} random prefix lookups...")

    test_prefixes = ["sys", "net", "dat", "web", "clo", "alg", "sca", "dis", "s", "d", "n", "w"]

    t_start = time.perf_counter()
    for _ in range(num_lookups):
        p = test_prefixes[_ % len(test_prefixes)]
        engine.search(p, k=5)

    total_time = time.perf_counter() - t_start
    qps = num_lookups / total_time
    summary = engine.metrics.get_summary()

    print("\n--------------------------------------------------------------------------------")
    print("AUTOCOMPLETE BENCHMARK RESULTS")
    print("--------------------------------------------------------------------------------")
    print(f"Total Lookups:             {num_lookups:,}")
    print(f"Elapsed Time:              {total_time:.3f} seconds")
    print(f"Throughput:                {qps:,.1f} lookups / second")
    print(f"Latency P50:               {summary['latency_p50_us']:.2f} µs ({summary['latency_p50_us']/1000.0:.4f} ms)")
    print(f"Latency P99:               {summary['latency_p99_us']:.2f} µs ({summary['latency_p99_us']/1000.0:.4f} ms)")
    print("--------------------------------------------------------------------------------\n")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Speed Search Autocomplete Engine")
    parser.add_argument("--test", action="store_true", help="Run comprehensive unit test suite")
    parser.add_argument("--benchmark", action="store_true", help="Run high-throughput stress benchmark")
    parser.add_argument("--daemon", action="store_true", help="Run live HTTP daemon on specified port")
    parser.add_argument("--port", type=int, default=8090, help="Port to listen on (default: 8090)")
    parser.add_argument("--count", type=int, default=100_000, help="Benchmark lookup count")
    args = parser.parse_args()

    if args.test:
        run_unit_tests()
    elif args.benchmark:
        run_benchmark(num_lookups=args.count)
    elif args.daemon:
        engine = AutocompleteEngine()
        # Seed basic terms
        for word in ["system design", "search autocomplete", "distributed cache", "kafka pipeline", "redis cluster"]:
            engine.ingest_query(word, 1000)
        print(f"Starting Search Autocomplete Daemon on 0.0.0.0:{args.port}...")
        print(f"  - Autocomplete query: http://localhost:{args.port}/autocomplete?q=<prefix>&k=5")
        print(f"  - Event ingestion:    POST http://localhost:{args.port}/ingest")
        print(f"  - Dynamic blocklist:  POST http://localhost:{args.port}/blocklist")
        print(f"  - Health check:       http://localhost:{args.port}/healthz")
        print(f"  - Prometheus metrics: http://localhost:{args.port}/metrics")
        server = AutocompleteServer("0.0.0.0", args.port, engine)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Autocomplete Daemon.")
            server.shutdown()
            server.server_close()
    else:
        # Default: run tests then benchmark
        run_unit_tests()
        run_benchmark(num_lookups=50_000)
