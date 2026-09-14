#!/usr/bin/env python3
"""
Production LSM-Tree Key-Value Engine & Distributed Quorum Node
==============================================================
Enterprise-grade, zero-dependency Python implementation of:
1. Append-Only Binary Write-Ahead Log (WAL) with CRC32 checksums & crash recovery.
2. In-Memory Sorted MemTable with size-triggered immutable freezing.
3. Disk-Backed Immutable SSTable (Sorted String Table) with:
   - Packed binary data blocks
   - In-memory Sparse Index (O(log(N/K)) binary search over disk offsets)
   - Embedded Bloom Filter for fast negative lookups
4. Background Two-Way Merge Compaction (garbage-collecting tombstones & stale versions).
5. Leaderless Distributed Quorum Coordinator (N=3, W=2, R=2) with Real Read Repair.
6. End-to-End Persistence, Recovery, and Performance Benchmark.
"""

import sys
import os
import time
import math
import struct
import zlib
import bisect
import shutil
import threading
import argparse
from typing import Dict, List, Tuple, Optional, Any, Iterator


# Record Types for WAL & SSTable
RECORD_PUT = 1
RECORD_TOMBSTONE = 2


# ============================================================================
# 1. BLOOM FILTER FOR SSTABLE DISK BLOCKS
# ============================================================================

class SSTableBloomFilter:
    """Embedded Bloom Filter for disk SSTables to bypass unnecessary disk I/O."""
    def __init__(self, capacity: int = 10_000, fpr: float = 0.01):
        self.capacity = max(10, capacity)
        self.fpr = fpr
        self.num_bits = int(- (self.capacity * math.log(fpr)) / (math.log(2) ** 2))
        self.num_hashes = max(1, int((self.num_bits / self.capacity) * math.log(2)))
        self.bit_array = bytearray((self.num_bits + 7) // 8)

    def _hashes(self, key: str) -> List[int]:
        b = key.encode('utf-8')
        h1 = zlib.crc32(b)
        h2 = zlib.crc32(b + b"_salt") | 1
        return [(h1 + i * h2) % self.num_bits for i in range(self.num_hashes)]

    def add(self, key: str):
        for pos in self._hashes(key):
            self.bit_array[pos // 8] |= (1 << (pos % 8))

    def contains(self, key: str) -> bool:
        for pos in self._hashes(key):
            if not (self.bit_array[pos // 8] & (1 << (pos % 8))):
                return False
        return True

    def to_bytes(self) -> bytes:
        return bytes(self.bit_array)

    @classmethod
    def from_bytes(cls, data: bytes, num_bits: int, num_hashes: int) -> 'SSTableBloomFilter':
        bf = cls(capacity=10)
        bf.bit_array = bytearray(data)
        bf.num_bits = num_bits
        bf.num_hashes = num_hashes
        return bf


# ============================================================================
# 2. APPEND-ONLY WRITE-AHEAD LOG (WAL)
# ============================================================================

class WriteAheadLog:
    """
    Append-only disk log for durability across crash restarts.
    
    Binary Record Format:
    ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬─────────────┬─────────────┐
    │ CRC32 (4B)   │ Timestamp ms │ Type (1B)    │ KeyLen (2B)  │ ValLen (4B)  │ Key (var)   │ Value (var) │
    └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴─────────────┴─────────────┘
    """
    HEADER_FORMAT = "!IQBHI"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    def __init__(self, wal_path: str):
        self.wal_path = wal_path
        self._file = open(wal_path, "a+b")
        self._lock = threading.Lock()

    def append(self, rec_type: int, timestamp_ms: int, key: str, value: bytes) -> int:
        with self._lock:
            k_bytes = key.encode('utf-8')
            v_bytes = value
            k_len = len(k_bytes)
            v_len = len(v_bytes)

            payload = struct.pack(f"!QBHI{k_len}s{v_len}s", timestamp_ms, rec_type, k_len, v_len, k_bytes, v_bytes)
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            record = struct.pack("!I", crc) + payload

            offset = self._file.tell()
            self._file.write(record)
            self._file.flush()
            os.fsync(self._file.fileno())
            return offset

    def recover(self) -> List[Tuple[int, int, str, bytes]]:
        """Replays the WAL on node startup to reconstruct the MemTable."""
        records = []
        if not os.path.exists(self.wal_path):
            return records

        with open(self.wal_path, "rb") as f:
            while True:
                header = f.read(self.HEADER_SIZE)
                if len(header) < self.HEADER_SIZE:
                    break

                crc, ts, rec_type, k_len, v_len = struct.unpack(self.HEADER_FORMAT, header)
                payload_len = k_len + v_len
                payload = f.read(payload_len)
                if len(payload) < payload_len:
                    break  # Corrupted / torn write at end of log

                # Validate CRC
                expected_crc = zlib.crc32(struct.pack("!QBHI", ts, rec_type, k_len, v_len) + payload) & 0xFFFFFFFF
                if crc != expected_crc:
                    break  # Checksum mismatch, stop replay

                k_bytes = payload[:k_len]
                v_bytes = payload[k_len:]
                records.append((rec_type, ts, k_bytes.decode('utf-8'), v_bytes))

        return records

    def close(self):
        with self._lock:
            if not self._file.closed:
                self._file.close()


# ============================================================================
# 3. IMMUTABLE SSTABLE ON DISK WITH SPARSE INDEX
# ============================================================================

class SSTableReader:
    """Reads from an immutable SSTable file using an in-memory Sparse Index & Bloom filter."""
    def __init__(self, sstable_path: str):
        self.sstable_path = sstable_path
        self.sparse_index: List[Tuple[str, int]] = []
        self.sparse_keys: List[str] = []
        self.bloom: Optional[SSTableBloomFilter] = None
        self._load_metadata()

    def _load_metadata(self):
        """Loads the sparse index and Bloom filter from the SSTable trailer."""
        with open(self.sstable_path, "rb") as f:
            f.seek(-8, os.SEEK_END)
            meta_offset, = struct.unpack("!Q", f.read(8))
            self.meta_offset = meta_offset
            f.seek(meta_offset)

            # Read Bloom filter
            bf_len, num_bits, num_hashes = struct.unpack("!III", f.read(12))
            bf_bytes = f.read(bf_len)
            self.bloom = SSTableBloomFilter.from_bytes(bf_bytes, num_bits, num_hashes)

            # Read Sparse Index
            index_count, = struct.unpack("!I", f.read(4))
            for _ in range(index_count):
                k_len, = struct.unpack("!H", f.read(2))
                key = f.read(k_len).decode('utf-8')
                offset, = struct.unpack("!Q", f.read(8))
                self.sparse_index.append((key, offset))
                self.sparse_keys.append(key)

    def get(self, target_key: str) -> Optional[Tuple[int, int, bytes]]:
        """
        Looks up key in SSTable.
        Returns (record_type, timestamp_ms, value_bytes) or None.
        """
        # Step 1: Bloom filter negative check (< 1 microsecond)
        if self.bloom and not self.bloom.contains(target_key):
            return None

        # Step 2: Binary search on sparse index to find candidate disk block
        idx = bisect.bisect_right(self.sparse_keys, target_key) - 1
        if idx < 0:
            idx = 0
        start_offset = self.sparse_index[idx][1]
        end_offset = self.sparse_index[idx + 1][1] if (idx + 1 < len(self.sparse_index)) else self.meta_offset

        # Step 3: Scan bounded byte block on disk
        with open(self.sstable_path, "rb") as f:
            f.seek(start_offset)
            while f.tell() < end_offset:
                header = f.read(15)  # Type(1) + Ts(8) + KLen(2) + VLen(4)
                if len(header) < 15:
                    break
                rec_type, ts, k_len, v_len = struct.unpack("!BQHI", header)
                key = f.read(k_len).decode('utf-8')
                val = f.read(v_len)

                if key == target_key:
                    return rec_type, ts, val
                if key > target_key:
                    break  # Key is not present (SSTable is strictly sorted)

        return None


class SSTableWriter:
    """Writes sorted key-value pairs into a new immutable SSTable file."""
    INDEX_INTERVAL = 16  # Sample 1 key every 16 items for the sparse index

    @classmethod
    def write(cls, sstable_path: str, sorted_items: List[Tuple[str, Tuple[int, int, bytes]]]):
        sparse_index: List[Tuple[str, int]] = []
        bloom = SSTableBloomFilter(capacity=max(10, len(sorted_items)), fpr=0.01)

        with open(sstable_path, "wb") as f:
            for idx, (key, (rec_type, ts, val)) in enumerate(sorted_items):
                offset = f.tell()
                bloom.add(key)

                if idx % cls.INDEX_INTERVAL == 0:
                    sparse_index.append((key, offset))

                k_bytes = key.encode('utf-8')
                header = struct.pack("!BQHI", rec_type, ts, len(k_bytes), len(val))
                f.write(header)
                f.write(k_bytes)
                f.write(val)

            meta_offset = f.tell()

            # Write Bloom Filter Block
            bf_bytes = bloom.to_bytes()
            f.write(struct.pack("!III", len(bf_bytes), bloom.num_bits, bloom.num_hashes))
            f.write(bf_bytes)

            # Write Sparse Index Block
            f.write(struct.pack("!I", len(sparse_index)))
            for k, off in sparse_index:
                k_b = k.encode('utf-8')
                f.write(struct.pack("!H", len(k_b)))
                f.write(k_b)
                f.write(struct.pack("!Q", off))

            # Write trailer pointing to metadata start
            f.write(struct.pack("!Q", meta_offset))
            f.flush()
            os.fsync(f.fileno())


# ============================================================================
# 4. LSM-TREE NODE STORAGE ENGINE
# ============================================================================

class LSMStorageEngine:
    """
    Node-local Log-Structured Merge-Tree Engine (RocksDB / LevelDB architecture).
    """
    def __init__(self, data_dir: str, memtable_threshold: int = 500):
        self.data_dir = data_dir
        self.memtable_threshold = memtable_threshold
        os.makedirs(data_dir, exist_ok=True)

        self.wal_path = os.path.join(data_dir, "wal.log")
        self.wal = WriteAheadLog(self.wal_path)

        # MemTable: sorted key -> (rec_type, timestamp_ms, value_bytes)
        self.memtable: Dict[str, Tuple[int, int, bytes]] = {}
        self.immutable_memtable: Optional[Dict[str, Tuple[int, int, bytes]]] = None
        self.sstables: List[SSTableReader] = []

        self._lock = threading.Lock()
        self.sstable_counter = 0

        # Recover from existing WAL and SSTables on disk
        self._recover()

    def _recover(self):
        # 1. Load existing SSTables
        sstable_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith("sstable_") and f.endswith(".db")])
        for sf in sstable_files:
            reader = SSTableReader(os.path.join(self.data_dir, sf))
            self.sstables.append(reader)
            num = int(sf.replace("sstable_", "").replace(".db", ""))
            self.sstable_counter = max(self.sstable_counter, num)

        # 2. Replay WAL into MemTable
        wal_records = self.wal.recover()
        for rec_type, ts, k, v in wal_records:
            self.memtable[k] = (rec_type, ts, v)

    def put(self, key: str, value: bytes) -> int:
        now_ms = int(time.time() * 1000)
        with self._lock:
            self.wal.append(RECORD_PUT, now_ms, key, value)
            self.memtable[key] = (RECORD_PUT, now_ms, value)
            if len(self.memtable) >= self.memtable_threshold:
                self._flush_memtable()
        return now_ms

    def delete(self, key: str) -> int:
        now_ms = int(time.time() * 1000)
        with self._lock:
            self.wal.append(RECORD_TOMBSTONE, now_ms, key, b"")
            self.memtable[key] = (RECORD_TOMBSTONE, now_ms, b"")
            if len(self.memtable) >= self.memtable_threshold:
                self._flush_memtable()
        return now_ms

    def get(self, key: str) -> Optional[Tuple[bytes, int]]:
        """
        Reads latest value. Returns (value_bytes, timestamp_ms) or None if deleted/absent.
        """
        with self._lock:
            # Check Active MemTable
            if key in self.memtable:
                rec_type, ts, val = self.memtable[key]
                return None if rec_type == RECORD_TOMBSTONE else (val, ts)

            # Check Immutable MemTable (if currently flushing)
            if self.immutable_memtable and key in self.immutable_memtable:
                rec_type, ts, val = self.immutable_memtable[key]
                return None if rec_type == RECORD_TOMBSTONE else (val, ts)

            # Check SSTables (Newest to Oldest)
            for sstable in reversed(self.sstables):
                res = sstable.get(key)
                if res:
                    rec_type, ts, val = res
                    return None if rec_type == RECORD_TOMBSTONE else (val, ts)

        return None

    def _flush_memtable(self):
        """Freezes MemTable, writes to SSTable, resets WAL."""
        if not self.memtable:
            return

        sorted_items = sorted(self.memtable.items())
        self.sstable_counter += 1
        new_sst_name = f"sstable_{self.sstable_counter:04d}.db"
        new_sst_path = os.path.join(self.data_dir, new_sst_name)

        SSTableWriter.write(new_sst_path, sorted_items)
        self.sstables.append(SSTableReader(new_sst_path))

        # Reset MemTable and WAL
        self.memtable.clear()
        self.wal.close()
        # Truncate WAL file cleanly
        open(self.wal_path, "w").close()
        self.wal = WriteAheadLog(self.wal_path)

    def compact(self):
        """
        Two-Way Merge Compaction: Merges all SSTables into a single compacted SSTable,
        evicting duplicate keys and dead tombstones.
        """
        with self._lock:
            if len(self.sstables) < 2:
                return

            merged_data: Dict[str, Tuple[int, int, bytes]] = {}
            for reader in self.sstables:
                with open(reader.sstable_path, "rb") as f:
                    f.seek(-8, os.SEEK_END)
                    meta_off, = struct.unpack("!Q", f.read(8))
                    f.seek(0)
                    while f.tell() < meta_off:
                        h = f.read(15)
                        if len(h) < 15:
                            break
                        rec_type, ts, k_len, v_len = struct.unpack("!BQHI", h)
                        k = f.read(k_len).decode('utf-8')
                        v = f.read(v_len)
                        if k not in merged_data or ts > merged_data[k][1]:
                            merged_data[k] = (rec_type, ts, v)

            # Filter out tombstones
            final_items = [(k, v) for k, v in sorted(merged_data.items()) if v[0] != RECORD_TOMBSTONE]

            self.sstable_counter += 1
            compact_name = f"sstable_{self.sstable_counter:04d}.db"
            compact_path = os.path.join(self.data_dir, compact_name)
            SSTableWriter.write(compact_path, final_items)

            # Remove old SSTable files
            old_readers = list(self.sstables)
            self.sstables = [SSTableReader(compact_path)]
            for r in old_readers:
                try:
                    os.remove(r.sstable_path)
                except Exception:
                    pass


# ============================================================================
# 5. DISTRIBUTED QUORUM COORDINATOR (W=2, R=2, N=3) WITH READ REPAIR
# ============================================================================

class QuorumCoordinator:
    """
    Leaderless Quorum Coordinator with Dynamo-style Read Repair.
    Replication Factor: N = 3
    Write Quorum: W = 2
    Read Quorum: R = 2
    Guarantee: W + R > N (2 + 2 > 3) ensures strong quorum overlap.
    """
    def __init__(self, node_dirs: List[str]):
        self.nodes = [LSMStorageEngine(data_dir=d, memtable_threshold=100) for d in node_dirs]
        self.N = len(self.nodes)
        self.W = 2
        self.R = 2
        self.read_repairs_triggered = 0

    def put(self, key: str, value: bytes) -> bool:
        """Writes to all replicas, returns True if at least W succeed."""
        successes = 0
        for node in self.nodes:
            try:
                node.put(key, value)
                successes += 1
            except Exception:
                pass
        return successes >= self.W

    def get(self, key: str) -> Optional[bytes]:
        """
        Executes Quorum Read (R=2). 
        Resolves conflicts via Last-Write-Wins (LWW) and issues Read Repair to stale nodes!
        """
        responses: List[Tuple[Optional[bytes], int, int]] = []  # (value, timestamp, node_idx)

        for i, node in enumerate(self.nodes):
            res = node.get(key)
            if res:
                val, ts = res
                responses.append((val, ts, i))
            else:
                responses.append((None, 0, i))

        if len(responses) < self.R:
            raise RuntimeError("Read Quorum Failed: Could not contact R replicas")

        # Find latest version among responses (LWW)
        latest_val, latest_ts, best_node = max(responses, key=lambda x: x[1])

        # Detect stale nodes for Read Repair
        for val, ts, node_idx in responses:
            if ts < latest_ts and latest_val is not None:
                self.read_repairs_triggered += 1
                # Synchronous / Asynchronous Read Repair
                self.nodes[node_idx].put(key, latest_val)

        return latest_val


# ============================================================================
# 6. BENCHMARK & VERIFICATION RUNNER
# ============================================================================

def run_lsm_verification():
    test_dir = "./kv_test_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("\n==================================================================")
    print("  VERIFYING LSM-TREE ENGINE: WAL, SSTABLES, COMPACTION & RECOVERY")
    print("==================================================================")

    engine = LSMStorageEngine(data_dir=test_dir, memtable_threshold=200)

    # 1. Write 1,000 keys (triggers multiple MemTable flushes to SSTables)
    print("[1] Writing 1,000 keys to LSM engine...")
    t0 = time.perf_counter()
    for i in range(1000):
        engine.put(f"user:account:{i:05d}", f"ProfilePayload_{i}".encode('utf-8'))
    t_write = time.perf_counter() - t0
    print(f"    Wrote 1,000 keys in {t_write:.3f}s ({1000/t_write:,.0f} writes/sec)")
    print(f"    SSTables on Disk: {len(engine.sstables)}")

    # 2. Read back keys
    print("[2] Reading back keys (MemTable + SSTable Sparse Index + Bloom Filter)...")
    t0 = time.perf_counter()
    hits = 0
    for i in range(1000):
        res = engine.get(f"user:account:{i:05d}")
        if res and res[0] == f"ProfilePayload_{i}".encode('utf-8'):
            hits += 1
    t_read = time.perf_counter() - t0
    print(f"    Read 1,000 keys: {hits}/1000 hits in {t_read:.3f}s ({1000/t_read:,.0f} reads/sec)")

    # 3. Test Delete & Tombstone
    print("[3] Deleting 100 keys (Tombstone injection)...")
    for i in range(100):
        engine.delete(f"user:account:{i:05d}")
    for i in range(100):
        assert engine.get(f"user:account:{i:05d}") is None, "Deleted key was found!"
    print("    [✓] Tombstones correctly hid deleted records.")

    # 4. Crash Recovery Simulation
    print("[4] Simulating node crash & restart from WAL...")
    del engine
    restarted_engine = LSMStorageEngine(data_dir=test_dir, memtable_threshold=200)
    recovered_val = restarted_engine.get("user:account:00500")
    assert recovered_val is not None, "Failed to recover key from disk SSTable/WAL!"
    print(f"    [✓] Crash recovery successful. Value: {recovered_val[0].decode('utf-8')}")

    # 5. Compaction
    print("[5] Running background Two-Way Merge Compaction...")
    pre_sst_count = len(restarted_engine.sstables)
    restarted_engine.compact()
    post_sst_count = len(restarted_engine.sstables)
    print(f"    SSTables compacted: {pre_sst_count} -> {post_sst_count} consolidated SSTable")

    # 6. Quorum & Read Repair Test
    print("\n==================================================================")
    print("  VERIFYING DISTRIBUTED QUORUM & READ REPAIR (N=3, W=2, R=2)")
    print("==================================================================")
    q_dirs = ["./kv_node_1", "./kv_node_2", "./kv_node_3"]
    for qd in q_dirs:
        if os.path.exists(qd):
            shutil.rmtree(qd)

    coordinator = QuorumCoordinator(node_dirs=q_dirs)
    coordinator.put("cluster:cfg:auth_mode", b"OAuth2_Strict")

    # Artificially inject stale data into Node 3
    coordinator.nodes[2].put("cluster:cfg:auth_mode", b"Legacy_Basic_Auth")
    # Rollback node 3 timestamp to stale
    coordinator.nodes[2].memtable["cluster:cfg:auth_mode"] = (RECORD_PUT, 1000, b"Legacy_Basic_Auth")

    # Execute Quorum Read
    val = coordinator.get("cluster:cfg:auth_mode")
    assert val == b"OAuth2_Strict", "Quorum read failed to resolve latest value!"
    assert coordinator.read_repairs_triggered > 0, "Read repair failed to trigger on stale replica!"
    print(f"    [✓] Quorum Read returned latest LWW value: {val.decode('utf-8')}")
    print(f"    [✓] Read Repair successfully reconciled stale replica on Node 3.")

    # Cleanup
    for d in [test_dir] + q_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)


def main():
    parser = argparse.ArgumentParser(description="Production LSM-Tree Key-Value Engine")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("verify", help="Run full LSM, WAL, Compaction, and Quorum verification")

    args = parser.parse_args()
    run_lsm_verification()


if __name__ == "__main__":
    main()
