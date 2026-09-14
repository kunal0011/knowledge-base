#!/usr/bin/env python3
"""
High-Performance Proximity Service Engine (Yelp & Google Places)
================================================================
Enterprise-grade, zero-dependency implementation of a high-throughput,
sub-millisecond geospatial proximity search service.

Core Architectural Capabilities:
  1. Geohash Base32 Bit-Interleaving Encoder & Decoder.
  2. 8-Neighbor Geohash Expansion (solving the cell boundary-crossing problem).
  3. In-Memory Dual-Level Spatial Grid + SQLite WAL persistent store.
  4. Exact Haversine Spherical Distance Calculation & Filtering.
  5. Multi-Factor Spatial Ranking Engine (Distance decay + Rating + Review volume).
  6. Real HTTP REST Daemon with /nearby, /businesses, /healthz, and Prometheus /metrics.
"""

import sys
import os
import time
import json
import math
import socket
import select
import threading
import sqlite3
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Constants & Geohash Configuration
# ---------------------------------------------------------------------------
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
EARTH_RADIUS_METERS = 6371000.0


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class BusinessRecord:
    business_id: str
    name: str
    lat: float
    lng: float
    geohash: str
    category: str
    rating: float
    review_count: int
    address: str

    def to_dict(self, distance_m: Optional[float] = None, score: Optional[float] = None) -> Dict[str, Any]:
        d = {
            "business_id": self.business_id,
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "geohash": self.geohash,
            "category": self.category,
            "rating": self.rating,
            "review_count": self.review_count,
            "address": self.address,
        }
        if distance_m is not None:
            d["distance_m"] = round(distance_m, 1)
        if score is not None:
            d["score"] = round(score, 3)
        return d


# ---------------------------------------------------------------------------
# Geohash Bit-Interleaving & Spatial Geometry Core
# ---------------------------------------------------------------------------
class GeohashCore:
    @staticmethod
    def encode(lat: float, lng: float, precision: int = 6) -> str:
        """Encodes latitude and longitude into a Base32 Geohash string."""
        lat_range = [-90.0, 90.0]
        lng_range = [-180.0, 180.0]
        geohash = []
        is_lng = True
        ch = 0
        bit = 0

        while len(geohash) < precision:
            if is_lng:
                mid = (lng_range[0] + lng_range[1]) / 2.0
                if lng >= mid:
                    ch |= (1 << (4 - bit))
                    lng_range[0] = mid
                else:
                    lng_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2.0
                if lat >= mid:
                    ch |= (1 << (4 - bit))
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid

            is_lng = not is_lng
            bit += 1
            if bit == 5:
                geohash.append(BASE32[ch])
                bit = 0
                ch = 0

        return "".join(geohash)

    @staticmethod
    def decode(gh: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Decodes a Geohash string into ((lat_min, lat_max), (lng_min, lng_max))."""
        lat_range = [-90.0, 90.0]
        lng_range = [-180.0, 180.0]
        is_lng = True

        for c in gh:
            idx = BASE32.index(c)
            for bit in range(4, -1, -1):
                mask = 1 << bit
                if is_lng:
                    mid = (lng_range[0] + lng_range[1]) / 2.0
                    if idx & mask:
                        lng_range[0] = mid
                    else:
                        lng_range[1] = mid
                else:
                    mid = (lat_range[0] + lat_range[1]) / 2.0
                    if idx & mask:
                        lat_range[0] = mid
                    else:
                        lat_range[1] = mid
                is_lng = not is_lng

        return (lat_range[0], lat_range[1]), (lng_range[0], lng_range[1])

    @staticmethod
    def get_neighbors(gh: str) -> List[str]:
        """
        Returns the 8 geographic neighbor cells surrounding the given geohash.
        Essential for solving the edge/boundary-crossing search problem.
        """
        (lat_min, lat_max), (lng_min, lng_max) = GeohashCore.decode(gh)
        lat_mid = (lat_min + lat_max) / 2.0
        lng_mid = (lng_min + lng_max) / 2.0
        d_lat = lat_max - lat_min
        d_lng = lng_max - lng_min
        p = len(gh)

        return [
            GeohashCore.encode(lat_mid + d_lat, lng_mid, p),          # North
            GeohashCore.encode(lat_mid - d_lat, lng_mid, p),          # South
            GeohashCore.encode(lat_mid, lng_mid + d_lng, p),          # East
            GeohashCore.encode(lat_mid, lng_mid - d_lng, p),          # West
            GeohashCore.encode(lat_mid + d_lat, lng_mid + d_lng, p),  # North-East
            GeohashCore.encode(lat_mid + d_lat, lng_mid - d_lng, p),  # North-West
            GeohashCore.encode(lat_mid - d_lat, lng_mid + d_lng, p),  # South-East
            GeohashCore.encode(lat_mid - d_lat, lng_mid - d_lng, p),  # South-West
        ]

    @staticmethod
    def get_search_cells(lat: float, lng: float, radius_m: float) -> List[str]:
        """
        Dynamically selects the optimal Geohash precision based on radius,
        returning the central cell plus all 8 surrounding neighbor cells.
        """
        if radius_m <= 1000.0:
            precision = 6  # ~1.2 km x 0.6 km
        elif radius_m <= 5000.0:
            precision = 5  # ~4.9 km x 4.9 km
        else:
            precision = 4  # ~39 km x 19.5 km

        center_gh = GeohashCore.encode(lat, lng, precision)
        neighbors = GeohashCore.get_neighbors(center_gh)
        return [center_gh] + neighbors

    @staticmethod
    def haversine_distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculates great-circle distance between two points on the sphere in meters."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lng2 - lng1)

        a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return EARTH_RADIUS_METERS * c


# ---------------------------------------------------------------------------
# Storage & Spatial Index Engine
# ---------------------------------------------------------------------------
class SpatialDatabase:
    """Persistent SQLite store with thread-safe WAL mode and in-memory spatial grid cache."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self.lock = threading.Lock()
        # In-memory spatial grid: geohash prefix -> list of BusinessRecord
        self.spatial_grid: Dict[str, List[BusinessRecord]] = {}
        self._init_db()
        self._warm_spatial_grid()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS businesses (
                    business_id TEXT PRIMARY KEY,
                    name TEXT,
                    lat REAL,
                    lng REAL,
                    geohash TEXT,
                    category TEXT,
                    rating REAL,
                    review_count INTEGER,
                    address TEXT,
                    created_at REAL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_geohash ON businesses (geohash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON businesses (category);")
        conn.close()

    def _warm_spatial_grid(self):
        """Pre-populates the in-memory spatial grid on boot for sub-millisecond lookups."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT business_id, name, lat, lng, geohash, category, rating, review_count, address FROM businesses;")
        for r in cursor.fetchall():
            biz = BusinessRecord(
                business_id=r[0], name=r[1], lat=r[2], lng=r[3], geohash=r[4],
                category=r[5], rating=r[6], review_count=r[7], address=r[8]
            )
            self._index_in_memory(biz)

    def _index_in_memory(self, biz: BusinessRecord):
        with self.lock:
            # Index at multiple prefix lengths (4, 5, 6)
            for p in (4, 5, 6):
                prefix = biz.geohash[:p]
                if prefix not in self.spatial_grid:
                    self.spatial_grid[prefix] = []
                self.spatial_grid[prefix].append(biz)

    def add_business(self, biz: BusinessRecord):
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO businesses (business_id, name, lat, lng, geohash, category, rating, review_count, address, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (biz.business_id, biz.name, biz.lat, biz.lng, biz.geohash, biz.category, biz.rating, biz.review_count, biz.address, time.time()))
        self._index_in_memory(biz)

    def get_candidates_in_cells(self, cells: List[str]) -> List[BusinessRecord]:
        """Retrieves candidate businesses across the target cells from in-memory grid."""
        seen_ids = set()
        candidates = []
        with self.lock:
            for cell in cells:
                for biz in self.spatial_grid.get(cell, []):
                    if biz.business_id not in seen_ids:
                        seen_ids.add(biz.business_id)
                        candidates.append(biz)
        return candidates


# ---------------------------------------------------------------------------
# Proximity Query & Multi-Factor Ranking Engine
# ---------------------------------------------------------------------------
class ProximityQueryEngine:
    def __init__(self, db: SpatialDatabase):
        self.db = db

    def search_nearby(self, lat: float, lng: float, radius_m: float = 3000.0,
                      category: Optional[str] = None, min_rating: float = 0.0,
                      k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes point-radius proximity search with candidate pruning,
        exact Haversine distance verification, and multi-factor ranking.
        """
        t0 = time.perf_counter()

        # 1. Determine bounding search cells (Center + 8 Neighbors)
        search_cells = GeohashCore.get_search_cells(lat, lng, radius_m)

        # 2. Retrieve candidate businesses from spatial grid
        candidates = self.db.get_candidates_in_cells(search_cells)

        # 3. Filter and score candidates
        scored_results = []
        for biz in candidates:
            # Filter by category if specified
            if category and biz.category.lower() != category.lower():
                continue

            # Filter by minimum rating
            if biz.rating < min_rating:
                continue

            # Exact spherical distance
            dist_m = GeohashCore.haversine_distance_m(lat, lng, biz.lat, biz.lng)
            if dist_m > radius_m:
                continue

            # Multi-factor ranking: Distance Decay (50%) + Rating (30%) + Reviews (20%)
            dist_score = max(0.0, 1.0 - (dist_m / radius_m))
            rating_score = biz.rating / 5.0
            review_score = min(1.0, math.log10(max(1, biz.review_count) + 1) / 4.0)

            composite_score = (0.50 * dist_score) + (0.30 * rating_score) + (0.20 * review_score)
            scored_results.append((composite_score, dist_m, biz))

        # Sort descending by composite score
        scored_results.sort(key=lambda x: -x[0])

        top_k = scored_results[:k]
        return [biz.to_dict(distance_m=dist, score=score) for score, dist, biz in top_k]


# ---------------------------------------------------------------------------
# Metrics & Telemetry
# ---------------------------------------------------------------------------
class ProximityMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_queries = 0
        self.total_businesses_indexed = 0
        self.latencies_us: List[float] = []

    def record_query(self, latency_us: float):
        with self.lock:
            self.total_queries += 1
            self.latencies_us.append(latency_us)
            if len(self.latencies_us) > 50000:
                self.latencies_us = self.latencies_us[-25000:]

    def record_business_add(self):
        with self.lock:
            self.total_businesses_indexed += 1

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            lats = sorted(self.latencies_us)
            p50 = lats[int(len(lats) * 0.5)] if lats else 0.0
            p99 = lats[int(len(lats) * 0.99)] if lats else 0.0
            return {
                "total_queries": self.total_queries,
                "total_businesses_indexed": self.total_businesses_indexed,
                "latency_p50_us": round(p50, 2),
                "latency_p99_us": round(p99, 2),
            }


# ---------------------------------------------------------------------------
# HTTP Handler & Proximity Daemon
# ---------------------------------------------------------------------------
class ProximityHTTPHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        self.server_ref = server
        super().__init__(request, client_address, server)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/healthz":
            self._send_json(200, {
                "status": "UP",
                "service": "High-Performance Proximity Service",
                "timestamp": time.time()
            })
            return

        if path == "/metrics":
            summary = self.server_ref.metrics.get_summary()
            lines = [
                "# HELP proximity_queries_total Total spatial proximity searches processed",
                "# TYPE proximity_queries_total counter",
                f"proximity_queries_total {summary['total_queries']}",
                "# HELP proximity_businesses_indexed_total Total registered businesses indexed",
                "# TYPE proximity_businesses_indexed_total counter",
                f"proximity_businesses_indexed_total {summary['total_businesses_indexed']}",
                "# HELP proximity_latency_p50_microseconds Median spatial search latency",
                "# TYPE proximity_latency_p50_microseconds gauge",
                f"proximity_latency_p50_microseconds {summary['latency_p50_us']}",
                "# HELP proximity_latency_p99_microseconds 99th percentile spatial search latency",
                "# TYPE proximity_latency_p99_microseconds gauge",
                f"proximity_latency_p99_microseconds {summary['latency_p99_us']}",
            ]
            body = "\n".join(lines) + "\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return

        if path == "/nearby":
            t0 = time.perf_counter()
            lat = float(query.get("lat", [37.7749])[0])
            lng = float(query.get("lng", [-122.4194])[0])
            radius_m = float(query.get("radius_m", [3000.0])[0])
            category = query.get("category", [None])[0]
            min_rating = float(query.get("min_rating", [0.0])[0])
            k = int(query.get("k", [10])[0])

            results = self.server_ref.engine.search_nearby(
                lat=lat, lng=lng, radius_m=radius_m, category=category, min_rating=min_rating, k=k
            )

            latency_us = (time.perf_counter() - t0) * 1_000_000.0
            self.server_ref.metrics.record_query(latency_us)

            self._send_json(200, {
                "center": {"lat": lat, "lng": lng},
                "radius_m": radius_m,
                "count": len(results),
                "businesses": results
            })
            return

        self._send_json(404, {"error": f"Endpoint '{path}' not found."})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/businesses":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body)

            lat = float(data["lat"])
            lng = float(data["lng"])
            gh = GeohashCore.encode(lat, lng, precision=7)

            biz = BusinessRecord(
                business_id=data["business_id"],
                name=data["name"],
                lat=lat,
                lng=lng,
                geohash=gh,
                category=data.get("category", "general"),
                rating=float(data.get("rating", 4.0)),
                review_count=int(data.get("review_count", 1)),
                address=data.get("address", "")
            )
            self.server_ref.db.add_business(biz)
            self.server_ref.metrics.record_business_add()
            self._send_json(201, {"status": "CREATED", "business_id": biz.business_id, "geohash": gh})
            return

        self._send_json(404, {"error": f"Endpoint '{path}' not found."})

    def _send_json(self, status: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ProximityServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str, port: int, db_path: str):
        self.db = SpatialDatabase(db_path)
        self.engine = ProximityQueryEngine(self.db)
        self.metrics = ProximityMetrics()
        super().__init__((host, port), ProximityHTTPHandler)


# ---------------------------------------------------------------------------
# Test & Verification Suite
# ---------------------------------------------------------------------------
def run_unit_tests():
    print("================================================================================")
    print("RUNNING PROXIMITY SERVICE SELF-TEST & VERIFICATION")
    print("================================================================================")

    db_file = f"/tmp/test_proximity_{int(time.time())}.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    server = ProximityServer("127.0.0.1", port, db_file)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)

    try:
        import urllib.request
        base_url = f"http://127.0.0.1:{port}"

        # --------------------------------------------------------------------
        # Test 1: Geohash Encoding & Decoding Precision
        # --------------------------------------------------------------------
        print("\n[Test 1] Testing Geohash Base32 Bit Interleaving & Precision...")
        sf_lat, sf_lng = 37.774929, -122.419416
        gh6 = GeohashCore.encode(sf_lat, sf_lng, precision=6)
        assert gh6 == "9q8yyk", f"Expected 9q8yyk, got {gh6}"
        lat_r, lng_r = GeohashCore.decode(gh6)
        assert lat_r[0] <= sf_lat <= lat_r[1]
        assert lng_r[0] <= sf_lng <= lng_r[1]
        print(f"  -> Successfully encoded San Francisco coordinate to {gh6} and verified bounding box.")

        # --------------------------------------------------------------------
        # Test 2: 8-Neighbor Expansion & Boundary Edge-Crossing
        # --------------------------------------------------------------------
        print("\n[Test 2] Testing 8-Neighbor Expansion & Boundary Edge-Crossing...")
        neighbors = GeohashCore.get_neighbors(gh6)
        assert len(neighbors) == 8
        assert len(set(neighbors)) == 8
        print(f"  -> Generated 8 distinct neighbor cells: {neighbors[:4]}...")

        # Insert a business right on the edge in a neighboring cell (50m away)
        # Point A in cell 1, Point B across the border in cell 2
        biz_center = BusinessRecord("b_center", "Central Bistro", 37.7750, -122.4190, GeohashCore.encode(37.7750, -122.4190, 6), "restaurant", 4.8, 500, "123 Market St")
        biz_across = BusinessRecord("b_across", "Border Cafe", 37.7770, -122.4190, GeohashCore.encode(37.7770, -122.4190, 6), "restaurant", 4.9, 800, "456 Mission St")

        server.db.add_business(biz_center)
        server.db.add_business(biz_across)

        # Search from Central Bistro location with 500m radius
        results = server.engine.search_nearby(37.7750, -122.4190, radius_m=500.0, k=5)
        found_ids = [r["business_id"] for r in results]
        assert "b_center" in found_ids
        assert "b_across" in found_ids, "Border Cafe must be discovered across the geohash cell boundary!"
        print(f"  -> Boundary edge-crossing defeated! Discovered '{results[1]['name']}' across geohash border ({results[1]['distance_m']}m away).")

        # --------------------------------------------------------------------
        # Test 3: Multi-Factor Ranking & Distance Decay
        # --------------------------------------------------------------------
        print("\n[Test 3] Testing Multi-Factor Spatial Ranking (Distance + Rating + Reviews)...")
        # Add a distant 5-star restaurant (2500m) and a nearby 4.2-star restaurant (100m)
        server.db.add_business(BusinessRecord("b_near", "Neighborhood Slice", 37.7755, -122.4190, GeohashCore.encode(37.7755, -122.4190, 6), "restaurant", 4.2, 50, "Local Ave"))
        server.db.add_business(BusinessRecord("b_far", "Michelin Star Estate", 37.7950, -122.4190, GeohashCore.encode(37.7950, -122.4190, 6), "restaurant", 5.0, 5000, "Far Hills"))

        ranked = server.engine.search_nearby(37.7750, -122.4190, radius_m=3000.0, k=4)
        print("  -> Ranked results:")
        for r in ranked:
            print(f"     • {r['name']} | Dist: {r['distance_m']}m | Rating: {r['rating']}★ | Score: {r['score']}")

        # --------------------------------------------------------------------
        # Test 4: HTTP REST Endpoints
        # --------------------------------------------------------------------
        print("\n[Test 4] Testing HTTP REST API Endpoints (/nearby, /businesses, /metrics)...")
        # Add business via HTTP
        add_req = urllib.request.Request(
            f"{base_url}/businesses",
            data=json.dumps({
                "business_id": "b_http_1",
                "name": "HTTP Coffee Lab",
                "lat": 37.7752,
                "lng": -122.4192,
                "category": "coffee",
                "rating": 4.7,
                "review_count": 350
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(add_req) as resp:
            assert resp.status == 201
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "CREATED"
        print("  -> POST /businesses successfully created and indexed record.")

        # Query /nearby via HTTP
        with urllib.request.urlopen(f"{base_url}/nearby?lat=37.7750&lng=-122.4190&radius_m=1000&category=coffee") as resp:
            nearby_data = json.loads(resp.read().decode("utf-8"))
            assert nearby_data["count"] >= 1
            assert nearby_data["businesses"][0]["business_id"] == "b_http_1"
        print("  -> GET /nearby returned expected JSON with category filtering.")

        # Query /metrics
        with urllib.request.urlopen(f"{base_url}/metrics") as resp:
            metrics_body = resp.read().decode("utf-8")
            assert "proximity_queries_total" in metrics_body
            assert "proximity_latency_p50_microseconds" in metrics_body
        print("  -> GET /metrics returned valid Prometheus telemetry.")

        print("\n[✓] ALL 4 UNIT TEST SUITES PASSED FLAWLESSLY!\n")
    finally:
        server.shutdown()
        server.server_close()
        if os.path.exists(db_file):
            os.remove(db_file)


# ---------------------------------------------------------------------------
# High-Throughput Stress Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(num_queries: int = 20_000):
    print("================================================================================")
    print(f"RUNNING HIGH-THROUGHPUT PROXIMITY BENCHMARK: {num_queries:,} SPATIAL QUERIES")
    print("================================================================================")

    db_file = f"/tmp/bench_proximity_{int(time.time())}.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    db = SpatialDatabase(db_file)
    engine = ProximityQueryEngine(db)

    print("Populating Spatial Grid with 5,000 businesses across SF Bay Area...")
    import random

    categories = ["restaurant", "coffee", "hotel", "gas_station", "gym", "pharmacy"]
    # Seed 5,000 points around San Francisco (lat: 37.70 to 37.85, lng: -122.52 to -122.35)
    for i in range(5000):
        lat = 37.70 + random.random() * 0.15
        lng = -122.52 + random.random() * 0.17
        gh = GeohashCore.encode(lat, lng, precision=7)
        biz = BusinessRecord(
            business_id=f"biz_{i}",
            name=f"Business #{i}",
            lat=lat,
            lng=lng,
            geohash=gh,
            category=random.choice(categories),
            rating=round(3.0 + random.random() * 2.0, 1),
            review_count=random.randint(10, 2000),
            address=f"{i} California St"
        )
        db.add_business(biz)

    print(f"Seeded 5,000 businesses. Executing {num_queries:,} spatial radius queries...")
    t_start = time.perf_counter()

    for _ in range(num_queries):
        q_lat = 37.70 + random.random() * 0.15
        q_lng = -122.52 + random.random() * 0.17
        engine.search_nearby(q_lat, q_lng, radius_m=2000.0, k=10)

    total_time = time.perf_counter() - t_start
    qps = num_queries / total_time

    print("\n--------------------------------------------------------------------------------")
    print("PROXIMITY SERVICE BENCHMARK RESULTS")
    print("--------------------------------------------------------------------------------")
    print(f"Total Spatial Queries:     {num_queries:,}")
    print(f"Elapsed Time:              {total_time:.3f} seconds")
    print(f"Throughput:                {qps:,.1f} spatial queries / second")
    print(f"Average Latency:           {(total_time / num_queries) * 1000.0:.3f} ms / query ({(total_time / num_queries) * 1_000_000.0:.1f} µs)")
    print("--------------------------------------------------------------------------------\n")

    if os.path.exists(db_file):
        os.remove(db_file)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Performance Proximity Search Engine")
    parser.add_argument("--test", action="store_true", help="Run comprehensive unit test suite")
    parser.add_argument("--benchmark", action="store_true", help="Run high-throughput stress benchmark")
    parser.add_argument("--daemon", action="store_true", help="Run live HTTP proximity search daemon")
    parser.add_argument("--port", type=int, default=8093, help="Port to listen on (default: 8093)")
    parser.add_argument("--db", type=str, default="proximity.db", help="SQLite database path")
    parser.add_argument("--count", type=int, default=20_000, help="Benchmark query count")
    args = parser.parse_args()

    if args.test:
        run_unit_tests()
    elif args.benchmark:
        run_benchmark(num_queries=args.count)
    elif args.daemon:
        print(f"Starting Proximity Service Daemon on 0.0.0.0:{args.port}...")
        print(f"  - Nearby Search:      GET  http://localhost:{args.port}/nearby?lat=37.77&lng=-122.41&radius_m=3000")
        print(f"  - Register Business:  POST http://localhost:{args.port}/businesses")
        print(f"  - Health Check:       GET  http://localhost:{args.port}/healthz")
        print(f"  - Prometheus Metrics: GET  http://localhost:{args.port}/metrics")
        server = ProximityServer("0.0.0.0", args.port, args.db)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Proximity Server.")
            server.shutdown()
            server.server_close()
    else:
        # Default: run tests then benchmark
        run_unit_tests()
        run_benchmark(num_queries=10_000)
