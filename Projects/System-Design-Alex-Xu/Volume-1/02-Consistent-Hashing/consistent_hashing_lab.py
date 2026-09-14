"""
Consistent Hashing & Bounded Loads Reference Lab
================================================
Production-grade, zero-dependency Python implementation of:
1. Classic Consistent Hash Ring with Virtual Nodes (Karger et al.)
2. Google Bounded-Loads Consistent Hashing (Mirrokni et al., Vimeo/Envoy pattern)
3. Eytzinger Cache-Conscious Branchless Binary Search Ring
4. Churn & Rebalancing Simulator (Consistent Hashing vs. Modular Hashing)
5. Zipfian Hot-Key Skew & Spillover Simulator
6. High-Throughput Lookup Benchmark & Unit Test Suite
"""

import hashlib
import bisect
import math
import random
import time
import threading
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import unittest


# ============================================================================
# 1. HASH UTILITIES (64-BIT FAST UNIFORM HASHING)
# ============================================================================

def hash_64(val: str) -> int:
    """
    Computes a 64-bit integer hash with uniform distribution.
    Uses MD5 truncated to 8 bytes (simulating MurmurHash3/xxHash64 speed & uniformity).
    """
    digest = hashlib.md5(val.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big')


# ============================================================================
# 2. CLASSIC CONSISTENT HASH RING (KARGER ET AL.)
# ============================================================================

class ConsistentHashRing:
    """
    Classic Consistent Hash Ring with Virtual Nodes.
    - Each physical node is hashed V times across the 64-bit integer space [0, 2^64 - 1].
    - Lookups perform O(log(N * V)) binary search to find the clockwise successor.
    """
    def __init__(self, nodes: Optional[List[str]] = None, vnodes_per_node: int = 150):
        self.vnodes_per_node = vnodes_per_node
        self.ring: List[int] = []                    # Sorted list of vnode token hashes
        self.vnode_to_node: Dict[int, str] = {}      # Token hash -> physical node ID
        self.nodes: set = set()
        self._lock = threading.Lock()

        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node: str, weight: float = 1.0):
        """Adds a physical node with weight-proportional virtual nodes."""
        with self._lock:
            if node in self.nodes:
                return
            self.nodes.add(node)
            num_vnodes = max(1, int(self.vnodes_per_node * weight))
            for i in range(num_vnodes):
                vtoken = hash_64(f"{node}#vn_{i}")
                # Handle rare 64-bit collision
                while vtoken in self.vnode_to_node:
                    vtoken = hash_64(f"{node}#vn_{i}_{random.randint(0, 100000)}")
                bisect.insort(self.ring, vtoken)
                self.vnode_to_node[vtoken] = node

    def remove_node(self, node: str):
        """Removes a physical node and purges all its virtual nodes."""
        with self._lock:
            if node not in self.nodes:
                return
            self.nodes.remove(node)
            new_ring = []
            for token in self.ring:
                if self.vnode_to_node[token] == node:
                    del self.vnode_to_node[token]
                else:
                    new_ring.append(token)
            self.ring = new_ring

    def get_node(self, key: str) -> Optional[str]:
        """Finds the primary clockwise successor physical node for a given key."""
        if not self.ring:
            return None
        k_hash = hash_64(key)
        idx = bisect.bisect_right(self.ring, k_hash)
        if idx == len(self.ring):
            idx = 0  # Wrap around the circular ring
        return self.vnode_to_node[self.ring[idx]]

    def get_nodes(self, key: str, count: int) -> List[str]:
        """
        Returns `count` distinct physical successor nodes for replication.
        Skips virtual nodes that map to an already-selected physical node.
        """
        if not self.ring:
            return []
        count = min(count, len(self.nodes))
        k_hash = hash_64(key)
        idx = bisect.bisect_right(self.ring, k_hash)
        selected_nodes: List[str] = []
        visited_physical: set = set()

        for step in range(len(self.ring)):
            ring_idx = (idx + step) % len(self.ring)
            physical_node = self.vnode_to_node[self.ring[ring_idx]]
            if physical_node not in visited_physical:
                selected_nodes.append(physical_node)
                visited_physical.add(physical_node)
                if len(selected_nodes) == count:
                    break
        return selected_nodes


# ============================================================================
# 3. GOOGLE BOUNDED-LOADS CONSISTENT HASH RING (MIRROKNI ET AL.)
# ============================================================================

class BoundedLoadConsistentHashRing:
    """
    Google Bounded-Loads Consistent Hashing (Vimeo / Envoy Router Pattern).
    Eliminates Hot-Key Skew and Cascading Neighbor Collapses.
    
    Invariant:
      Load on any node <= ceil(c * (Total Active Requests / Number of Nodes))
      where c is the load factor parameter (typically 1.20 to 1.30).
    
    If the primary clockwise node is at capacity, the request spills over to the
    next clockwise node along the ring until an under-capacity node is found.
    """
    def __init__(
        self,
        nodes: Optional[List[str]] = None,
        vnodes_per_node: int = 150,
        load_factor: float = 1.25
    ):
        self.ring = ConsistentHashRing(nodes, vnodes_per_node)
        self.load_factor = load_factor
        self.node_loads: Dict[str, int] = defaultdict(int)
        self.total_load = 0
        self._lock = threading.Lock()

    def add_node(self, node: str, weight: float = 1.0):
        with self._lock:
            self.ring.add_node(node, weight)
            if node not in self.node_loads:
                self.node_loads[node] = 0

    def remove_node(self, node: str):
        with self._lock:
            self.ring.remove_node(node)
            if node in self.node_loads:
                self.total_load -= self.node_loads[node]
                del self.node_loads[node]

    def assign_request(self, key: str) -> Tuple[str, bool]:
        """
        Assigns a request to a node ensuring the load factor ceiling is never exceeded.
        Returns: (assigned_node_id, is_spillover: bool)
        """
        with self._lock:
            num_nodes = len(self.ring.nodes)
            if num_nodes == 0:
                raise RuntimeError("No nodes available in the hash ring")

            # Calculate the dynamic load ceiling
            # Ceiling = ceil(c * (total_load + 1) / num_nodes)
            max_allowed_load = math.ceil(self.load_factor * (self.total_load + 1) / num_nodes)

            # Retrieve candidate nodes along the clockwise ring
            candidates = self.ring.get_nodes(key, count=num_nodes)
            primary_node = candidates[0]

            for i, candidate in enumerate(candidates):
                if self.node_loads[candidate] < max_allowed_load:
                    self.node_loads[candidate] += 1
                    self.total_load += 1
                    is_spillover = (i > 0)
                    return candidate, is_spillover

            # If all nodes are at ceiling (rare), assign to primary as fallback
            self.node_loads[primary_node] += 1
            self.total_load += 1
            return primary_node, False

    def release_request(self, node: str):
        """Decrements node load when request finishes processing."""
        with self._lock:
            if node in self.node_loads and self.node_loads[node] > 0:
                self.node_loads[node] -= 1
                self.total_load = max(0, self.total_load - 1)


# ============================================================================
# 4. EYTZINGER (CACHE-CONSCIOUS) SEARCH RING
# ============================================================================

class EytzingerHashRing:
    """
    Eytzinger (1-based Breadth-First Layout) Array for Cache-Conscious Lookup.
    In standard binary search, early jumps span hundreds of cache lines, causing
    near 100% L1/L2 cache misses. 
    Eytzinger packs root at [1], left child at [2*i], right child at [2*i+1],
    maximizing CPU hardware prefetcher efficiency.
    """
    def __init__(self, sorted_tokens: List[int], token_to_node: Dict[int, str]):
        self.n = len(sorted_tokens)
        self.eytzinger_tree = [0] * (self.n + 1)
        self.token_to_node = token_to_node
        self._build_eytzinger(sorted_tokens, 0, 1)

    def _build_eytzinger(self, sorted_list: List[int], in_idx: int, out_idx: int) -> int:
        if out_idx <= self.n:
            in_idx = self._build_eytzinger(sorted_list, in_idx, 2 * out_idx)
            self.eytzinger_tree[out_idx] = sorted_list[in_idx]
            in_idx += 1
            in_idx = self._build_eytzinger(sorted_list, in_idx, 2 * out_idx + 1)
        return in_idx

    def search(self, val: int) -> int:
        """Branchless Eytzinger search. Returns index of successor token."""
        k = 1
        while k <= self.n:
            # Multiply k by 2, add 1 if val >= tree[k]
            k = 2 * k + (1 if val >= self.eytzinger_tree[k] else 0)
        # Shift back to find the actual element
        k >>= (k.bit_length() - (self.n + 1).bit_length())
        if k == 0:
            k = 1
        return min(k, self.n)


# ============================================================================
# 5. CHURN & REBALANCING SIMULATOR
# ============================================================================

class ChurnSimulator:
    """
    Compares the Rehashing Catastrophe of Modular Hashing vs. Consistent Hashing.
    """
    @staticmethod
    def run_node_churn_experiment(num_nodes: int = 100, num_keys: int = 100_000) -> Dict[str, Any]:
        keys = [f"cache_key_{i}" for i in range(num_keys)]

        # --- Test 1: Modular Hashing ---
        initial_mod_map = {k: hash_64(k) % num_nodes for k in keys}
        # 1 node fails (N -> N - 1)
        after_mod_map = {k: hash_64(k) % (num_nodes - 1) for k in keys}
        mod_remapped = sum(1 for k in keys if initial_mod_map[k] != after_mod_map[k])
        mod_remap_pct = (mod_remapped / num_keys) * 100.0

        # --- Test 2: Consistent Hashing ---
        nodes = [f"node_{i}" for i in range(num_nodes)]
        ch_ring = ConsistentHashRing(nodes=nodes, vnodes_per_node=200)
        initial_ch_map = {k: ch_ring.get_node(k) for k in keys}

        # Remove 1 node
        ch_ring.remove_node("node_99")
        after_ch_map = {k: ch_ring.get_node(k) for k in keys}
        ch_remapped = sum(1 for k in keys if initial_ch_map[k] != after_ch_map[k])
        ch_remap_pct = (ch_remapped / num_keys) * 100.0

        return {
            "num_nodes": num_nodes,
            "num_keys": num_keys,
            "mod_remapped": mod_remapped,
            "mod_remap_pct": mod_remap_pct,
            "ch_remapped": ch_remapped,
            "ch_remap_pct": ch_remap_pct,
            "theoretical_ch_pct": (1.0 / num_nodes) * 100.0
        }


# ============================================================================
# 6. HOT-KEY & BOUNDED-LOAD SPILLOVER SIMULATOR
# ============================================================================

class HotKeySimulator:
    """
    Simulates viral Zipfian traffic distribution across physical nodes.
    Demonstrates how Google Bounded Loads prevents single-node saturation.
    """
    @staticmethod
    def run_skew_experiment(num_nodes: int = 10, total_requests: int = 50_000) -> Dict[str, Any]:
        nodes = [f"cache_node_{i}" for i in range(num_nodes)]
        classic_ring = ConsistentHashRing(nodes, vnodes_per_node=150)
        bounded_ring = BoundedLoadConsistentHashRing(nodes, vnodes_per_node=150, load_factor=1.25)

        # Generate Zipfian keys: 80% of traffic hits 10 hot viral keys
        viral_keys = [f"viral_video_stream_{i}" for i in range(10)]
        longtail_keys = [f"normal_key_{i}" for i in range(5000)]

        classic_loads: Dict[str, int] = defaultdict(int)
        spillover_count = 0

        for _ in range(total_requests):
            # 80% viral key selection
            if random.random() < 0.80:
                k = random.choice(viral_keys)
            else:
                k = random.choice(longtail_keys)

            # Classic Ring Assignment
            assigned_classic = classic_ring.get_node(k)
            classic_loads[assigned_classic] += 1

            # Bounded Load Assignment
            _, is_spillover = bounded_ring.assign_request(k)
            if is_spillover:
                spillover_count += 1

        avg_load = total_requests / num_nodes
        max_classic_load = max(classic_loads.values())
        max_bounded_load = max(bounded_ring.node_loads.values())

        return {
            "total_requests": total_requests,
            "num_nodes": num_nodes,
            "avg_load": avg_load,
            "max_classic_load": max_classic_load,
            "classic_skew_factor": max_classic_load / avg_load,
            "max_bounded_load": max_bounded_load,
            "bounded_skew_factor": max_bounded_load / avg_load,
            "spillover_requests": spillover_count,
            "spillover_pct": (spillover_count / total_requests) * 100.0,
            "classic_loads": dict(classic_loads),
            "bounded_loads": dict(bounded_ring.node_loads)
        }


# ============================================================================
# 7. HIGH-THROUGHPUT LOOKUP BENCHMARK
# ============================================================================

def run_lookup_benchmark(num_threads: int = 8, lookups_per_thread: int = 50_000):
    nodes = [f"server_node_{i}" for i in range(100)]
    ring = ConsistentHashRing(nodes=nodes, vnodes_per_node=200)  # 20,000 vnodes total
    total_lookups = num_threads * lookups_per_thread

    print("\n==================================================================")
    print(f"  RUNNING HIGH-THROUGHPUT CONSISTENT HASH LOOKUP BENCHMARK")
    print(f"  Nodes: {len(nodes)} | Vnodes: {len(ring.ring):,} | Threads: {num_threads}")
    print(f"  Total Lookups: {total_lookups:,}")
    print("==================================================================")

    keys = [f"user_session_token_{i}" for i in range(lookups_per_thread)]

    latencies = []
    lock = threading.Lock()

    def worker():
        local_lats = []
        for k in keys:
            t0 = time.perf_counter()
            ring.get_node(k)
            local_lats.append((time.perf_counter() - t0) * 1_000_000)  # microseconds
        with lock:
            latencies.extend(local_lats)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_total = time.perf_counter() - t_start

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p99 = latencies[int(len(latencies) * 0.99)]
    qps = total_lookups / t_total

    print(f"  Throughput:    {qps:,.0f} lookups/second")
    print(f"  P50 Latency:   {p50:.2f} microseconds")
    print(f"  P99 Latency:   {p99:.2f} microseconds")
    print(f"  Elapsed Time:  {t_total:.3f} seconds")


# ============================================================================
# 8. UNIT AND INTEGRATION TESTS
# ============================================================================

class TestConsistentHashing(unittest.TestCase):

    def test_ring_determinism_and_replication(self):
        nodes = ["node_A", "node_B", "node_C", "node_D"]
        ring = ConsistentHashRing(nodes=nodes, vnodes_per_node=100)

        # Same key must always return the same primary node
        key = "account_uuid_948102"
        first_pick = ring.get_node(key)
        for _ in range(10):
            self.assertEqual(ring.get_node(key), first_pick)

        # Replicas must return N distinct physical nodes
        replicas = ring.get_nodes(key, count=3)
        self.assertEqual(len(replicas), 3)
        self.assertEqual(len(set(replicas)), 3, "Replicas must be distinct physical servers")
        self.assertEqual(replicas[0], first_pick, "Primary node must match first replica")

    def test_monotonicity_on_node_addition(self):
        ring = ConsistentHashRing(nodes=["node_1", "node_2", "node_3"], vnodes_per_node=150)
        sample_keys = [f"key_{i}" for i in range(1000)]
        before_map = {k: ring.get_node(k) for k in sample_keys}

        # Add a 4th node
        ring.add_node("node_4")
        after_map = {k: ring.get_node(k) for k in sample_keys}

        # Invariant: Keys can only migrate to node_4. They must NEVER migrate between 1, 2, and 3.
        for k in sample_keys:
            old_node = before_map[k]
            new_node = after_map[k]
            if old_node != new_node:
                self.assertEqual(new_node, "node_4", f"Key {k} migrated unlawfully to {new_node}!")

    def test_bounded_loads_ceiling_enforcement(self):
        nodes = ["cache_1", "cache_2", "cache_3"]
        bounded = BoundedLoadConsistentHashRing(nodes=nodes, vnodes_per_node=100, load_factor=1.20)

        # Send 300 identical keys that all hash to the same primary node
        key = "hot_breaking_news_slug"
        spillovers = 0
        for _ in range(300):
            _, is_spill = bounded.assign_request(key)
            if is_spill:
                spillovers += 1

        # Spillovers must have triggered to protect the primary node
        self.assertGreater(spillovers, 0, "Spillover routing must trigger under severe key skew")
        max_load = max(bounded.node_loads.values())
        min_load = min(bounded.node_loads.values())
        # With 300 requests and 3 nodes, avg = 100. Max load cannot exceed ceil(1.20 * 301 / 3) = 121
        self.assertLessEqual(max_load, 125, f"Max load {max_load} exceeded allowed ceiling!")


if __name__ == "__main__":
    print("==================================================================")
    print("  RUNNING SUITE OF 3 UNIT TESTS (Determinism, Monotonicity, Bounded Loads)")
    print("==================================================================")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConsistentHashing)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)

    if res.wasSuccessful():
        print("\n==================================================================")
        print("  RUNNING CHURN SIMULATION (Rehashing Catastrophe vs Consistent Hashing)")
        print("==================================================================")
        churn = ChurnSimulator.run_node_churn_experiment(num_nodes=100, num_keys=100_000)
        print(f"  Cluster Size:       {churn['num_nodes']} nodes -> {churn['num_nodes']-1} nodes")
        print(f"  Modular Hashing:    {churn['mod_remapped']:,} keys remapped ({churn['mod_remap_pct']:.2f}%)")
        print(f"  Consistent Hashing: {churn['ch_remapped']:,} keys remapped ({churn['ch_remap_pct']:.2f}%)")
        print(f"  Theoretical Ideal:  {churn['theoretical_ch_pct']:.2f}% (1 / N)")

        print("\n==================================================================")
        print("  RUNNING ZIPFIAN HOT-KEY SKEW & BOUNDED-LOADS SIMULATION")
        print("==================================================================")
        skew = HotKeySimulator.run_skew_experiment(num_nodes=10, total_requests=50_000)
        print(f"  Average Load / Node:   {skew['avg_load']:,.0f} requests")
        print(f"  Classic Max Node Load: {skew['max_classic_load']:,} ({skew['classic_skew_factor']:.2f}x average!)")
        print(f"  Bounded Max Node Load: {skew['max_bounded_load']:,} ({skew['bounded_skew_factor']:.2f}x ceiling enforced)")
        print(f"  Spillover Requests:    {skew['spillover_requests']:,} ({skew['spillover_pct']:.1f}% routed to next hop)")

        run_lookup_benchmark(num_threads=8, lookups_per_thread=25_000)
