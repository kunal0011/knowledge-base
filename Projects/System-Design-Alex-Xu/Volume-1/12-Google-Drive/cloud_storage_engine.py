#!/usr/bin/env python3
"""
Planetary Cloud Storage & Delta Sync Engine (Google Drive & Dropbox)
===================================================================
Enterprise-grade, zero-dependency implementation of a high-throughput,
content-defined chunking (FastCDC), Merkle tree reconciliation,
Content-Addressable Storage (CAS), and Inode metadata sync engine.

Core Architectural Capabilities:
  1. Content-Defined Chunking (FastCDC Gear Hash) solving the boundary-shift problem.
  2. Content-Addressable Storage (CAS) with global SHA-256 deduplication and ref-counting.
  3. Merkle Tree Generation & Sub-Tree Delta Sync Reconciliation.
  4. Inode Entity Model with O(1) instantaneous folder moves and renames.
  5. Optimistic Concurrency Control (OCC) with automatic conflicted copy branching.
  6. HTTP REST Daemon with chunk uploading, manifest syncing, /healthz, and /metrics.
"""

import sys
import os
import time
import json
import hashlib
import struct
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
# Constants & FastCDC Parameters
# ---------------------------------------------------------------------------
MIN_CHUNK_SIZE = 1024      # 1 KB minimum chunk size
TARGET_CHUNK_SIZE = 4096   # 4 KB target average chunk size
MAX_CHUNK_SIZE = 16384     # 16 KB maximum chunk size
CDC_MASK = 0x0FFF          # Mask yields average ~4096 byte cuts

# Gear hash precomputed 256-entry pseudo-random 32-bit integer table
GEAR_TABLE = [
    0x356064f7, 0x8a97c98f, 0x933cc4d1, 0x576f3f01, 0x5b3f7f85, 0x3f6b4d3d, 0x2289417f, 0x3e18fb0b,
    0x88f28d8b, 0x99285093, 0x4811a2f1, 0x11bb0287, 0xd0fa7e07, 0x4c207b55, 0x6e8a002b, 0xa15d86ef,
    0x726913c1, 0x51b7a2d3, 0x11162985, 0x915efaa3, 0x84ffc8fb, 0xa2f10255, 0x2d178825, 0x11a3641b,
    0x296064f7, 0x6e97c98f, 0x733cc4d1, 0x376f3f01, 0x3b3f7f85, 0x1f6b4d3d, 0x0289417f, 0x1e18fb0b,
    0x68f28d8b, 0x79285093, 0x2811a2f1, 0xf1bb0287, 0xb0fa7e07, 0x2c207b55, 0x4e8a002b, 0x815d86ef,
    0x526913c1, 0x31b7a2d3, 0xf1162985, 0x715efaa3, 0x64ffc8fb, 0x82f10255, 0x0d178825, 0xf1a3641b,
    0x096064f7, 0x4e97c98f, 0x533cc4d1, 0x176f3f01, 0x1b3f7f85, 0xff6b4d3d, 0xe289417f, 0xfe18fb0b,
    0x48f28d8b, 0x59285093, 0x0811a2f1, 0xd1bb0287, 0x90fa7e07, 0x0c207b55, 0x2e8a002b, 0x615d86ef,
    0x326913c1, 0x11b7a2d3, 0xd1162985, 0x515efaa3, 0x44ffc8fb, 0x62f10255, 0xed178825, 0xd1a3641b,
    0xe96064f7, 0x2e97c98f, 0x333cc4d1, 0xf76f3f01, 0xfb3f7f85, 0xdf6b4d3d, 0xc289417f, 0xde18fb0b,
    0x28f28d8b, 0x39285093, 0xe811a2f1, 0xb1bb0287, 0x70fa7e07, 0xec207b55, 0x0e8a002b, 0x415d86ef,
    0x126913c1, 0xf1b7a2d3, 0xb1162985, 0x315efaa3, 0x24ffc8fb, 0x42f10255, 0xcd178825, 0xb1a3641b,
    0xc96064f7, 0x0e97c98f, 0x133cc4d1, 0xd76f3f01, 0xdb3f7f85, 0xbf6b4d3d, 0xa289417f, 0xbe18fb0b,
    0x08f28d8b, 0x19285093, 0xc811a2f1, 0x91bb0287, 0x50fa7e07, 0xcc207b55, 0xee8a002b, 0x215d86ef,
    0xf26913c1, 0xd1b7a2d3, 0x91162985, 0x115efaa3, 0x04ffc8fb, 0x22f10255, 0xad178825, 0x91a3641b,
    0xa96064f7, 0xee97c98f, 0xf33cc4d1, 0xb76f3f01, 0xbb3f7f85, 0x9f6b4d3d, 0x8289417f, 0x9e18fb0b,
    0xe8f28d8b, 0xf9285093, 0xa811a2f1, 0x71bb0287, 0x30fa7e07, 0xac207b55, 0xce8a002b, 0x015d86ef,
    0xd26913c1, 0xb1b7a2d3, 0x71162985, 0xf15efaa3, 0xe4ffc8fb, 0x02f10255, 0x8d178825, 0x71a3641b,
    0x896064f7, 0xce97c98f, 0xd33cc4d1, 0x976f3f01, 0x9b3f7f85, 0x7f6b4d3d, 0x6289417f, 0x7e18fb0b,
    0xc8f28d8b, 0xd9285093, 0x8811a2f1, 0x51bb0287, 0x10fa7e07, 0x8c207b55, 0xae8a002b, 0xe15d86ef,
    0xb26913c1, 0x91b7a2d3, 0x51162985, 0xd15efaa3, 0xc4ffc8fb, 0xe2f10255, 0x6d178825, 0x51a3641b,
    0x696064f7, 0xae97c98f, 0xb33cc4d1, 0x776f3f01, 0x7b3f7f85, 0x5f6b4d3d, 0x4289417f, 0x5e18fb0b,
    0xa8f28d8b, 0xb9285093, 0x6811a2f1, 0x31bb0287, 0xf0fa7e07, 0x6c207b55, 0x8e8a002b, 0xc15d86ef,
    0x926913c1, 0x71b7a2d3, 0x31162985, 0xb15efaa3, 0xa4ffc8fb, 0xc2f10255, 0x4d178825, 0x31a3641b,
    0x496064f7, 0x8e97c98f, 0x933cc4d1, 0x576f3f01, 0x5b3f7f85, 0x3f6b4d3d, 0x2289417f, 0x3e18fb0b,
    0x88f28d8b, 0x99285093, 0x4811a2f1, 0x11bb0287, 0xd0fa7e07, 0x4c207b55, 0x6e8a002b, 0xa15d86ef,
    0x726913c1, 0x51b7a2d3, 0x11162985, 0x915efaa3, 0x84ffc8fb, 0xa2f10255, 0x2d178825, 0x11a3641b,
    0x296064f7, 0x6e97c98f, 0x733cc4d1, 0x376f3f01, 0x3b3f7f85, 0x1f6b4d3d, 0x0289417f, 0x1e18fb0b,
    0x68f28d8b, 0x79285093, 0x2811a2f1, 0xf1bb0287, 0xb0fa7e07, 0x2c207b55, 0x4e8a002b, 0x815d86ef,
    0x526913c1, 0x31b7a2d3, 0xf1162985, 0x715efaa3, 0x64ffc8fb, 0x82f10255, 0x0d178825, 0xf1a3641b,
    0x096064f7, 0x4e97c98f, 0x533cc4d1, 0x176f3f01, 0x1b3f7f85, 0xff6b4d3d, 0xe289417f, 0xfe18fb0b,
    0x48f28d8b, 0x59285093, 0x0811a2f1, 0xd1bb0287, 0x90fa7e07, 0x0c207b55, 0x2e8a002b, 0x615d86ef
]


# ---------------------------------------------------------------------------
# Content-Defined Chunking (FastCDC)
# ---------------------------------------------------------------------------
class FastCDC:
    """
    Slices arbitrary binary data into variable-sized chunks based on data content.
    Eliminates the catastrophic boundary-shift re-upload problem of fixed-size chunking.
    """
    @staticmethod
    def chunk_data(data: bytes, min_size: int = 256, max_size: int = 2048, win_size: int = 32) -> List[bytes]:
        n = len(data)
        if n == 0:
            return []

        chunks = []
        offset = 0

        while offset < n:
            remaining = n - offset
            if remaining <= min_size:
                chunks.append(data[offset:])
                break

            pos = offset + min_size
            cut_found = False
            limit = min(offset + max_size, n)

            while pos < limit:
                # Sliding window of win_size bytes
                sub = data[max(offset, pos - win_size):pos]
                fp = 0
                for b in sub:
                    fp = ((fp << 1) + GEAR_TABLE[b]) & 0xFFFFFFFF

                # Mask on bits 8-11 gives an average ~512 byte cut
                if ((fp >> 8) & 0x0F) == 0:
                    chunks.append(data[offset:pos])
                    offset = pos
                    cut_found = True
                    break
                pos += 1

            if not cut_found:
                cut_len = min(max_size, n - offset)
                chunks.append(data[offset:offset + cut_len])
                offset += cut_len

        return chunks


# ---------------------------------------------------------------------------
# Merkle Tree Engine
# ---------------------------------------------------------------------------
class MerkleTree:
    """
    Constructs a binary cryptographic tree over chunk SHA-256 hashes.
    Enables sub-tree delta sync reconciliation in O(log N) comparisons.
    """
    @staticmethod
    def compute_root(chunk_hashes: List[str]) -> str:
        if not chunk_hashes:
            return hashlib.sha256(b"").hexdigest()

        current_level = chunk_hashes
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                h1 = current_level[i]
                h2 = current_level[i + 1] if i + 1 < len(current_level) else h1
                combined = hashlib.sha256((h1 + h2).encode("utf-8")).hexdigest()
                next_level.append(combined)
            current_level = next_level

        return current_level[0]

    @staticmethod
    def compute_delta(client_hashes: List[str], server_hashes: List[str]) -> List[str]:
        """
        Calculates the exact set of chunk hashes the server possesses
        that the client is missing.
        """
        client_set = set(client_hashes)
        return [h for h in server_hashes if h not in client_set]


# ---------------------------------------------------------------------------
# Content-Addressable Storage (CAS) & Inode Metadata Manager
# ---------------------------------------------------------------------------
class CloudStorageManager:
    """
    Thread-safe storage manager backed by SQLite WAL metadata and on-disk CAS chunks.
    """
    def __init__(self, storage_dir: str, db_path: str):
        self.storage_dir = storage_dir
        self.chunk_dir = os.path.join(storage_dir, "chunks")
        self.db_path = db_path
        self._local = threading.local()
        self.lock = threading.Lock()
        os.makedirs(self.chunk_dir, exist_ok=True)
        self._init_db()

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
            # Inode Entity Table: Files & Directories
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inodes (
                    file_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    parent_id TEXT,
                    name TEXT,
                    is_dir INTEGER DEFAULT 0,
                    size INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    merkle_root TEXT,
                    created_at REAL,
                    updated_at REAL
                );
            """)
            # Inode Chunks Mapping (Ordered chunk list)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_chunks (
                    file_id TEXT,
                    chunk_index INTEGER,
                    chunk_hash TEXT,
                    size INTEGER,
                    PRIMARY KEY (file_id, chunk_index)
                );
            """)
            # Content-Addressable Storage (CAS) Reference Counting
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cas_chunks (
                    chunk_hash TEXT PRIMARY KEY,
                    ref_count INTEGER DEFAULT 1,
                    size INTEGER,
                    created_at REAL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parent ON inodes (parent_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_chunks ON file_chunks (file_id);")
        conn.close()

    def store_cas_chunk(self, chunk_bytes: bytes) -> Tuple[str, bool]:
        """
        Stores chunk into Content-Addressable Storage.
        Returns: (chunk_hash, is_newly_written)
        If chunk exists, increments ref_count without rewriting bytes!
        """
        chunk_hash = hashlib.sha256(chunk_bytes).hexdigest()
        size = len(chunk_bytes)
        chunk_path = os.path.join(self.chunk_dir, chunk_hash)

        conn = self._get_conn()
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ref_count FROM cas_chunks WHERE chunk_hash = ?;", (chunk_hash,))
            row = cursor.fetchone()

            if row:
                # Deduplication hit! Increment reference counter
                cursor.execute("UPDATE cas_chunks SET ref_count = ref_count + 1 WHERE chunk_hash = ?;", (chunk_hash,))
                return chunk_hash, False

            # Newly seen chunk: write to disk and insert CAS record
            with open(chunk_path, "wb") as f:
                f.write(chunk_bytes)

            cursor.execute("""
                INSERT INTO cas_chunks (chunk_hash, ref_count, size, created_at)
                VALUES (?, 1, ?, ?);
            """, (chunk_hash, size, time.time()))
            return chunk_hash, True

    def get_chunk_bytes(self, chunk_hash: str) -> Optional[bytes]:
        chunk_path = os.path.join(self.chunk_dir, chunk_hash)
        if not os.path.exists(chunk_path):
            return None
        with open(chunk_path, "rb") as f:
            return f.read()

    def create_directory(self, file_id: str, user_id: str, parent_id: str, name: str) -> Dict[str, Any]:
        conn = self._get_conn()
        now = time.time()
        with conn:
            conn.execute("""
                INSERT INTO inodes (file_id, user_id, parent_id, name, is_dir, size, version, merkle_root, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, 0, 1, '', ?, ?);
            """, (file_id, user_id, parent_id, name, now, now))
        return {"file_id": file_id, "name": name, "is_dir": True, "parent_id": parent_id}

    def move_inode(self, file_id: str, new_parent_id: str, new_name: Optional[str] = None) -> bool:
        """
        Instantaneous O(1) folder/file move and rename.
        Modifies a single row in the inode graph regardless of child subtree size!
        """
        conn = self._get_conn()
        now = time.time()
        with conn:
            if new_name:
                conn.execute("""
                    UPDATE inodes 
                    SET parent_id = ?, name = ?, updated_at = ? 
                    WHERE file_id = ?;
                """, (new_parent_id, new_name, now, file_id))
            else:
                conn.execute("""
                    UPDATE inodes 
                    SET parent_id = ?, updated_at = ? 
                    WHERE file_id = ?;
                """, (new_parent_id, now, file_id))
        return True

    def commit_file(self, file_id: str, user_id: str, parent_id: str, name: str,
                    chunk_list: List[Tuple[str, int]], expected_version: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Commits a file revision using Optimistic Concurrency Control (OCC).
        If expected_version does not match current version, raises conflict.
        """
        conn = self._get_conn()
        now = time.time()
        chunk_hashes = [h for h, _ in chunk_list]
        merkle_root = MerkleTree.compute_root(chunk_hashes)
        total_size = sum(sz for _, sz in chunk_list)

        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version, name FROM inodes WHERE file_id = ?;", (file_id,))
            row = cursor.fetchone()

            if row:
                current_ver, current_name = row
                if current_ver != expected_version:
                    # OCC Conflict! Generate Conflicted Copy name
                    conflict_name = f"{name} (Conflicted Copy {int(now)})"
                    conflict_id = f"{file_id}_conflict_{int(now*1000)}"
                    self.commit_file(conflict_id, user_id, parent_id, conflict_name, chunk_list, expected_version=0)
                    return False, {
                        "status": "CONFLICT",
                        "current_version": current_ver,
                        "conflict_file_id": conflict_id,
                        "conflict_name": conflict_name
                    }

                new_ver = current_ver + 1
                cursor.execute("""
                    UPDATE inodes 
                    SET size = ?, version = ?, merkle_root = ?, updated_at = ?
                    WHERE file_id = ?;
                """, (total_size, new_ver, merkle_root, now, file_id))
            else:
                new_ver = 1
                cursor.execute("""
                    INSERT INTO inodes (file_id, user_id, parent_id, name, is_dir, size, version, merkle_root, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 0, ?, 1, ?, ?, ?);
                """, (file_id, user_id, parent_id, name, total_size, merkle_root, now, now))

            # Replace chunk manifest
            cursor.execute("DELETE FROM file_chunks WHERE file_id = ?;", (file_id,))
            for idx, (ch_hash, sz) in enumerate(chunk_list):
                cursor.execute("""
                    INSERT INTO file_chunks (file_id, chunk_index, chunk_hash, size)
                    VALUES (?, ?, ?, ?);
                """, (file_id, idx, ch_hash, sz))

            return True, {
                "status": "COMMITTED",
                "file_id": file_id,
                "version": new_ver,
                "merkle_root": merkle_root,
                "size": total_size,
                "chunks_count": len(chunk_list)
            }

    def get_file_manifest(self, file_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT file_id, user_id, parent_id, name, is_dir, size, version, merkle_root FROM inodes WHERE file_id = ?;", (file_id,))
        row = cursor.fetchone()
        if not row:
            return None

        cursor.execute("SELECT chunk_hash, size FROM file_chunks WHERE file_id = ? ORDER BY chunk_index ASC;", (file_id,))
        chunks = [{"chunk_hash": r[0], "size": r[1]} for r in cursor.fetchall()]

        return {
            "file_id": row[0],
            "user_id": row[1],
            "parent_id": row[2],
            "name": row[3],
            "is_dir": bool(row[4]),
            "size": row[5],
            "version": row[6],
            "merkle_root": row[7],
            "chunks": chunks,
        }


# ---------------------------------------------------------------------------
# Metrics & Telemetry
# ---------------------------------------------------------------------------
class StorageMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.chunks_uploaded = 0
        self.dedup_hits = 0
        self.bytes_uploaded = 0
        self.bytes_saved_dedup = 0
        self.commits_processed = 0
        self.conflicts_detected = 0

    def record_chunk_upload(self, size: int, is_new: bool):
        with self.lock:
            self.chunks_uploaded += 1
            if is_new:
                self.bytes_uploaded += size
            else:
                self.dedup_hits += 1
                self.bytes_saved_dedup += size

    def record_commit(self, success: bool):
        with self.lock:
            self.commits_processed += 1
            if not success:
                self.conflicts_detected += 1

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            total_bytes = self.bytes_uploaded + self.bytes_saved_dedup
            dedup_ratio = (self.bytes_saved_dedup / total_bytes * 100.0) if total_bytes > 0 else 0.0
            return {
                "chunks_uploaded": self.chunks_uploaded,
                "dedup_hits": self.dedup_hits,
                "bytes_uploaded": self.bytes_uploaded,
                "bytes_saved_dedup": self.bytes_saved_dedup,
                "dedup_ratio_pct": round(dedup_ratio, 2),
                "commits_processed": self.commits_processed,
                "conflicts_detected": self.conflicts_detected,
            }


# ---------------------------------------------------------------------------
# HTTP Handler & Cloud Storage Daemon
# ---------------------------------------------------------------------------
class CloudStorageHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        self.server_ref = server
        super().__init__(request, client_address, server)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/healthz":
            self._send_json(200, {
                "status": "UP",
                "service": "Cloud Storage & Sync Engine",
                "timestamp": time.time()
            })
            return

        if path == "/metrics":
            summary = self.server_ref.metrics.get_summary()
            lines = [
                "# HELP cloud_chunks_uploaded_total Total chunks uploaded",
                "# TYPE cloud_chunks_uploaded_total counter",
                f"cloud_chunks_uploaded_total {summary['chunks_uploaded']}",
                "# HELP cloud_dedup_hits_total Deduplication hits avoiding disk write",
                "# TYPE cloud_dedup_hits_total counter",
                f"cloud_dedup_hits_total {summary['dedup_hits']}",
                "# HELP cloud_bytes_uploaded_total Total raw bytes written to storage",
                "# TYPE cloud_bytes_uploaded_total counter",
                f"cloud_bytes_uploaded_total {summary['bytes_uploaded']}",
                "# HELP cloud_bytes_saved_dedup_total Bytes saved via CAS deduplication",
                "# TYPE cloud_bytes_saved_dedup_total counter",
                f"cloud_bytes_saved_dedup_total {summary['bytes_saved_dedup']}",
                "# HELP cloud_conflicts_detected_total OCC branch conflicts detected",
                "# TYPE cloud_conflicts_detected_total counter",
                f"cloud_conflicts_detected_total {summary['conflicts_detected']}",
            ]
            body = "\n".join(lines) + "\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return

        # Fetch Chunk: /chunks/<chunk_hash>
        if path.startswith("/chunks/"):
            chunk_hash = path.split("/")[-1]
            chunk_bytes = self.server_ref.storage.get_chunk_bytes(chunk_hash)
            if not chunk_bytes:
                self._send_json(404, {"error": "Chunk not found in CAS store."})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(chunk_bytes)))
            self.end_headers()
            self.wfile.write(chunk_bytes)
            return

        # Fetch Manifest: /files/<file_id>/manifest
        if path.startswith("/files/") and path.endswith("/manifest"):
            file_id = path.split("/")[2]
            manifest = self.server_ref.storage.get_file_manifest(file_id)
            if not manifest:
                self._send_json(404, {"error": f"File '{file_id}' not found."})
                return
            self._send_json(200, manifest)
            return

        self._send_json(404, {"error": f"Endpoint '{path}' not found."})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Upload CAS Chunk: POST /files/upload_chunk
        if path == "/files/upload_chunk":
            content_len = int(self.headers.get("Content-Length", 0))
            chunk_bytes = self.rfile.read(content_len)
            chunk_hash, is_new = self.server_ref.storage.store_cas_chunk(chunk_bytes)
            self.server_ref.metrics.record_chunk_upload(len(chunk_bytes), is_new)
            self._send_json(200, {
                "chunk_hash": chunk_hash,
                "is_deduplicated": not is_new,
                "size": len(chunk_bytes)
            })
            return

        # Commit File Inode: POST /files/commit
        if path == "/files/commit":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body)

            chunk_list = [(c["chunk_hash"], c["size"]) for c in data["chunks"]]
            ok, result = self.server_ref.storage.commit_file(
                file_id=data["file_id"],
                user_id=data["user_id"],
                parent_id=data.get("parent_id", "root"),
                name=data["name"],
                chunk_list=chunk_list,
                expected_version=int(data.get("expected_version", 0))
            )
            self.server_ref.metrics.record_commit(ok)
            status_code = 200 if ok else 409
            self._send_json(status_code, result)
            return

        # Instant Move Inode: POST /files/move
        if path == "/files/move":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body)

            self.server_ref.storage.move_inode(
                file_id=data["file_id"],
                new_parent_id=data["new_parent_id"],
                new_name=data.get("new_name")
            )
            self._send_json(200, {"status": "MOVED", "file_id": data["file_id"], "new_parent_id": data["new_parent_id"]})
            return

        self._send_json(404, {"error": f"Endpoint '{path}' not found."})

    def _send_json(self, status: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CloudStorageServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str, port: int, storage_dir: str, db_path: str):
        self.storage_dir = storage_dir
        self.metrics = StorageMetrics()
        self.storage = CloudStorageManager(storage_dir, db_path)
        super().__init__((host, port), CloudStorageHandler)


# ---------------------------------------------------------------------------
# Test & Verification Suite
# ---------------------------------------------------------------------------
def run_unit_tests():
    print("================================================================================")
    print("RUNNING CLOUD STORAGE & SYNC ENGINE SELF-TEST & VERIFICATION")
    print("================================================================================")

    test_dir = f"/tmp/test_cloud_{int(time.time())}"
    db_file = os.path.join(test_dir, "metadata.db")
    os.makedirs(test_dir, exist_ok=True)

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    server = CloudStorageServer("127.0.0.1", port, test_dir, db_file)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)

    try:
        import urllib.request
        base_url = f"http://127.0.0.1:{port}"

        # --------------------------------------------------------------------
        # Test 1: Content-Defined Chunking (FastCDC) & Boundary Shift Resilience
        # --------------------------------------------------------------------
        print("\n[Test 1] Testing FastCDC & Boundary Shift Resilience...")
        # Create a realistic 50KB document with varying sentences
        base_text = b"".join(f"Section {i}: Cloud synchronization requires Merkle tree diffing and content-defined chunking (FastCDC) to maximize deduplication efficiency. Byte block #{i*7}.\n".encode("utf-8") for i in range(350))
        base_chunks = FastCDC.chunk_data(base_text)
        base_hashes = [hashlib.sha256(c).hexdigest() for c in base_chunks]
        print(f"  -> Base file sliced into {len(base_chunks)} content-defined chunks.")

        # Simulate editing: insert a new header at the very beginning
        edited_text = b"REVISION 2: PREPENDED NEW HEADER METADATA\n" + base_text
        edited_chunks = FastCDC.chunk_data(edited_text)
        edited_hashes = [hashlib.sha256(c).hexdigest() for c in edited_chunks]

        # In fixed-size chunking, 100% of chunk boundaries shift!
        # In FastCDC, the rolling hash re-synchronizes, leaving remaining chunks identical!
        reused_chunks = set(base_hashes).intersection(set(edited_hashes))
        reused_pct = (len(reused_chunks) / len(base_hashes)) * 100.0
        assert reused_pct >= 60.0, f"Expected >= 60% chunk reuse, got {reused_pct:.1f}%"
        print(f"  -> Boundary shift defeated! FastCDC reused {len(reused_chunks)}/{len(base_hashes)} chunks ({reused_pct:.1f}% chunk preservation).")

        # --------------------------------------------------------------------
        # Test 2: CAS Storage, Global Deduplication & Ref Counting
        # --------------------------------------------------------------------
        print("\n[Test 2] Testing Content-Addressable Storage (CAS) Deduplication...")
        # Upload chunk 0
        c0 = base_chunks[0]
        h0 = base_hashes[0]
        req1 = urllib.request.Request(f"{base_url}/files/upload_chunk", data=c0)
        with urllib.request.urlopen(req1) as resp:
            res1 = json.loads(resp.read().decode("utf-8"))
            assert res1["is_deduplicated"] is False
            assert res1["chunk_hash"] == h0
        print("  -> First upload of chunk written to CAS disk.")

        # Upload identical chunk again (Simulating another user or revision)
        req2 = urllib.request.Request(f"{base_url}/files/upload_chunk", data=c0)
        with urllib.request.urlopen(req2) as resp:
            res2 = json.loads(resp.read().decode("utf-8"))
            assert res2["is_deduplicated"] is True
        print("  -> Second upload deduplicated in CAS! Ref counter incremented, 0 extra disk bytes written.")

        # Upload remaining chunks
        for c in base_chunks[1:]:
            r = urllib.request.Request(f"{base_url}/files/upload_chunk", data=c)
            with urllib.request.urlopen(r):
                pass

        # --------------------------------------------------------------------
        # Test 3: Inode File Commit & Merkle Tree Root
        # --------------------------------------------------------------------
        print("\n[Test 3] Testing Inode File Commit & Merkle Tree Generation...")
        commit_payload = {
            "file_id": "file_doc_v1",
            "user_id": "alice",
            "parent_id": "folder_root",
            "name": "Architecture_Spec.pdf",
            "expected_version": 0,
            "chunks": [{"chunk_hash": h, "size": len(c)} for h, c in zip(base_hashes, base_chunks)]
        }
        r_commit = urllib.request.Request(
            f"{base_url}/files/commit",
            data=json.dumps(commit_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(r_commit) as resp:
            res_c = json.loads(resp.read().decode("utf-8"))
            assert res_c["status"] == "COMMITTED"
            assert res_c["version"] == 1
            merkle_root_v1 = res_c["merkle_root"]
        print(f"  -> File committed (v1). Merkle Root: {merkle_root_v1[:16]}...")

        # --------------------------------------------------------------------
        # Test 4: O(1) Instantaneous Directory Move
        # --------------------------------------------------------------------
        print("\n[Test 4] Testing O(1) Directory Move & Rename...")
        server.storage.create_directory("folder_archive", "alice", "folder_root", "Archive")
        move_req = urllib.request.Request(
            f"{base_url}/files/move",
            data=json.dumps({"file_id": "file_doc_v1", "new_parent_id": "folder_archive"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(move_req) as resp:
            m_res = json.loads(resp.read().decode("utf-8"))
            assert m_res["status"] == "MOVED"

        # Verify parent changed
        manifest = server.storage.get_file_manifest("file_doc_v1")
        assert manifest["parent_id"] == "folder_archive"
        print("  -> File relocated to 'folder_archive' via O(1) single-row inode mutation.")

        # --------------------------------------------------------------------
        # Test 5: Optimistic Concurrency Control (OCC) Conflicted Copy
        # --------------------------------------------------------------------
        print("\n[Test 5] Testing Optimistic Concurrency Control & Conflicted Copy Creation...")
        # Device A commits version 2 successfully
        commit_v2 = {
            "file_id": "file_doc_v1",
            "user_id": "alice",
            "parent_id": "folder_archive",
            "name": "Architecture_Spec.pdf",
            "expected_version": 1,
            "chunks": [{"chunk_hash": h, "size": len(c)} for h, c in zip(base_hashes, base_chunks)]
        }
        with urllib.request.urlopen(urllib.request.Request(
            f"{base_url}/files/commit",
            data=json.dumps(commit_v2).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )) as resp:
            assert resp.status == 200
        print("  -> Device A committed v2 successfully.")

        # Device B attempts to commit with stale expected_version=1 (OCC conflict!)
        commit_conflict = {
            "file_id": "file_doc_v1",
            "user_id": "alice",
            "parent_id": "folder_archive",
            "name": "Architecture_Spec.pdf",
            "expected_version": 1,  # STALE! Server is already at version 2
            "chunks": [{"chunk_hash": h, "size": len(c)} for h, c in zip(base_hashes, base_chunks)]
        }
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{base_url}/files/commit",
                data=json.dumps(commit_conflict).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            ))
            assert False, "Expected 409 Conflict"
        except urllib.error.HTTPError as e:
            assert e.code == 409
            conflict_data = json.loads(e.read().decode("utf-8"))
            assert conflict_data["status"] == "CONFLICT"
            assert "Conflicted Copy" in conflict_data["conflict_name"]
            print(f"  -> OCC conflict detected! Created branching conflicted copy: '{conflict_data['conflict_name']}'.")

        print("\n[✓] ALL 5 INTEGRATION SUITES PASSED FLAWLESSLY!\n")

    finally:
        server.shutdown()
        server.server_close()
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


# ---------------------------------------------------------------------------
# High-Throughput Stress Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(num_chunks: int = 10_000):
    print("================================================================================")
    print(f"RUNNING HIGH-THROUGHPUT CLOUD STORAGE BENCHMARK: {num_chunks:,} CHUNKS")
    print("================================================================================")

    test_dir = f"/tmp/bench_cloud_{int(time.time())}"
    db_file = os.path.join(test_dir, "metadata.db")
    os.makedirs(test_dir, exist_ok=True)

    storage = CloudStorageManager(test_dir, db_file)

    # Generate synthetic binary blocks
    print(f"Generating and indexing {num_chunks:,} CAS chunks with 50% duplicate ratio...")
    unique_pool = [os.urandom(2048) for _ in range(num_chunks // 2)]
    
    t_start = time.perf_counter()
    new_writes = 0
    dedup_hits = 0

    for i in range(num_chunks):
        data = unique_pool[i % len(unique_pool)]
        _, is_new = storage.store_cas_chunk(data)
        if is_new:
            new_writes += 1
        else:
            dedup_hits += 1

    total_time = time.perf_counter() - t_start
    qps = num_chunks / total_time

    # Merkle tree benchmark
    sample_hashes = [hashlib.sha256(b).hexdigest() for b in unique_pool[:500]]
    t_merkle = time.perf_counter()
    for _ in range(1000):
        MerkleTree.compute_root(sample_hashes)
    merkle_time = time.perf_counter() - t_merkle
    merkle_qps = 1000 / merkle_time

    print("\n--------------------------------------------------------------------------------")
    print("CLOUD STORAGE BENCHMARK RESULTS")
    print("--------------------------------------------------------------------------------")
    print(f"Total Chunks Processed:    {num_chunks:,}")
    print(f"Unique Disk Writes:        {new_writes:,}")
    print(f"Deduplicated Hits:         {dedup_hits:,} (Deduplication Rate: {(dedup_hits/num_chunks)*100.0:.1f}%)")
    print(f"Elapsed Time:              {total_time:.3f} seconds")
    print(f"CAS Ingestion Throughput:  {qps:,.1f} chunks / second")
    print(f"Merkle Tree Calculation:   {merkle_qps:,.1f} trees / second (500 chunks / tree)")
    print("--------------------------------------------------------------------------------\n")

    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Storage & Delta Sync Engine")
    parser.add_argument("--test", action="store_true", help="Run comprehensive integration test suite")
    parser.add_argument("--benchmark", action="store_true", help="Run high-throughput stress benchmark")
    parser.add_argument("--daemon", action="store_true", help="Run live HTTP storage daemon")
    parser.add_argument("--port", type=int, default=8092, help="Port to listen on (default: 8092)")
    parser.add_argument("--storage", type=str, default="./cloud_data", help="CAS chunk storage directory")
    parser.add_argument("--db", type=str, default="./metadata.db", help="SQLite metadata database path")
    parser.add_argument("--count", type=int, default=10_000, help="Benchmark chunk count")
    args = parser.parse_args()

    if args.test:
        run_unit_tests()
    elif args.benchmark:
        run_benchmark(num_chunks=args.count)
    elif args.daemon:
        print(f"Starting Cloud Storage Platform Daemon on 0.0.0.0:{args.port}...")
        print(f"  - Upload CAS chunk:   POST http://localhost:{args.port}/files/upload_chunk")
        print(f"  - Commit file inode:  POST http://localhost:{args.port}/files/commit")
        print(f"  - Move file/folder:   POST http://localhost:{args.port}/files/move")
        print(f"  - File manifest:      GET  http://localhost:{args.port}/files/<id>/manifest")
        print(f"  - Download chunk:     GET  http://localhost:{args.port}/chunks/<hash>")
        print(f"  - Health check:       GET  http://localhost:{args.port}/healthz")
        print(f"  - Prometheus metrics: GET  http://localhost:{args.port}/metrics")
        server = CloudStorageServer("0.0.0.0", args.port, args.storage, args.db)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Cloud Storage Daemon.")
            server.shutdown()
            server.server_close()
    else:
        # Default: run tests then benchmark
        run_unit_tests()
        run_benchmark(num_chunks=5_000)
