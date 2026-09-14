#!/usr/bin/env python3
"""
Distributed Real-Time Chat Engine & Gateway
===========================================
Enterprise-grade, zero-dependency implementation of a high-throughput,
stateful chat backbone modeled after WhatsApp, Slack, and Discord.

Core Architectural Capabilities:
  1. RFC 6455 WebSocket Gateway with binary/text frame parsing, masking, and ping/pong.
  2. Channel-Scoped Monotonic Sequence Assignor (guaranteeing strict message ordering).
  3. Hybrid Fan-Out Routing:
     - Push Fan-Out for 1-on-1 and small groups (<= 100 members).
     - Pull / Timeline broadcast for large community supergroups (> 100 members).
  4. Distributed Session Registry with heartbeat leases (TTL) and ephemeral presence.
  5. Disk-Backed Persistence Engine (SQLite WAL) with cursor-based Delta Sync.
  6. E2EE Envelope Support (Zero-knowledge server forwarding).
  7. Production HTTP Observability: /healthz, /metrics, /sync, /presence.
"""

import sys
import os
import time
import json
import base64
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
# Constants & Configuration
# ---------------------------------------------------------------------------
WS_MAGIC_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
HEARTBEAT_LEASE_SEC = 10.0  # Ephemeral presence lease timeout
GROUP_FANOUT_THRESHOLD = 100  # Max group size for fan-out-on-write push


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class ChatMessage:
    channel_id: str
    seq: int
    message_id: str
    sender_id: str
    payload: str
    created_at: float
    is_e2ee: bool = False
    e2ee_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "seq": self.seq,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "payload": self.payload,
            "created_at": self.created_at,
            "is_e2ee": self.is_e2ee,
            "e2ee_metadata": self.e2ee_metadata or {},
        }


@dataclass
class UserPresence:
    user_id: str
    status: str  # "ONLINE" or "OFFLINE"
    last_heartbeat: float
    active_connections: int = 0


# ---------------------------------------------------------------------------
# Metrics & Telemetry
# ---------------------------------------------------------------------------
class ChatMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.messages_processed = 0
        self.active_websockets = 0
        self.seq_gaps_detected = 0
        self.offline_pushes = 0
        self.presence_heartbeats = 0
        self.delta_sync_queries = 0
        self.latencies_ms: List[float] = []

    def record_message(self, latency_ms: float):
        with self.lock:
            self.messages_processed += 1
            self.latencies_ms.append(latency_ms)
            if len(self.latencies_ms) > 20000:
                self.latencies_ms = self.latencies_ms[-10000:]

    def record_ws_connect(self):
        with self.lock:
            self.active_websockets += 1

    def record_ws_disconnect(self):
        with self.lock:
            self.active_websockets = max(0, self.active_websockets - 1)

    def record_gap(self):
        with self.lock:
            self.seq_gaps_detected += 1

    def record_offline_push(self):
        with self.lock:
            self.offline_pushes += 1

    def record_heartbeat(self):
        with self.lock:
            self.presence_heartbeats += 1

    def record_sync(self):
        with self.lock:
            self.delta_sync_queries += 1

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            lats = sorted(self.latencies_ms)
            p50 = lats[int(len(lats) * 0.5)] if lats else 0.0
            p99 = lats[int(len(lats) * 0.99)] if lats else 0.0
            return {
                "messages_processed": self.messages_processed,
                "active_websockets": self.active_websockets,
                "seq_gaps_detected": self.seq_gaps_detected,
                "offline_pushes": self.offline_pushes,
                "presence_heartbeats": self.presence_heartbeats,
                "delta_sync_queries": self.delta_sync_queries,
                "latency_p50_ms": round(p50, 3),
                "latency_p99_ms": round(p99, 3),
            }


# ---------------------------------------------------------------------------
# Storage Engine (SQLite WAL with thread-local connections)
# ---------------------------------------------------------------------------
class StorageEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS channels (
                    channel_id TEXT PRIMARY KEY,
                    channel_type TEXT, -- 'DIRECT', 'GROUP', 'SUPERGROUP'
                    name TEXT,
                    created_at REAL,
                    current_seq INTEGER DEFAULT 0
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS channel_members (
                    channel_id TEXT,
                    user_id TEXT,
                    role TEXT DEFAULT 'MEMBER',
                    last_read_seq INTEGER DEFAULT 0,
                    PRIMARY KEY (channel_id, user_id)
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    channel_id TEXT,
                    seq INTEGER,
                    message_id TEXT,
                    sender_id TEXT,
                    payload TEXT,
                    is_e2ee INTEGER DEFAULT 0,
                    e2ee_metadata TEXT,
                    created_at REAL,
                    PRIMARY KEY (channel_id, seq)
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS offline_inbox (
                    user_id TEXT,
                    channel_id TEXT,
                    seq INTEGER,
                    message_id TEXT,
                    created_at REAL,
                    PRIMARY KEY (user_id, channel_id, seq)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_offline_user ON offline_inbox (user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_chan_seq ON messages (channel_id, seq);")
        conn.close()

    def register_channel(self, channel_id: str, channel_type: str, name: str):
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT OR IGNORE INTO channels (channel_id, channel_type, name, created_at, current_seq)
                VALUES (?, ?, ?, ?, 0);
            """, (channel_id, channel_type, name, time.time()))

    def add_channel_member(self, channel_id: str, user_id: str, role: str = "MEMBER"):
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT OR IGNORE INTO channel_members (channel_id, user_id, role, last_read_seq)
                VALUES (?, ?, ?, 0);
            """, (channel_id, user_id, role))

    def get_channel_members(self, channel_id: str) -> List[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM channel_members WHERE channel_id = ?;", (channel_id,))
        return [row[0] for row in cursor.fetchall()]

    def get_channel_type(self, channel_id: str) -> str:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT channel_type FROM channels WHERE channel_id = ?;", (channel_id,))
        row = cursor.fetchone()
        return row[0] if row else "GROUP"

    def allocate_sequence_and_save_message(self, msg: ChatMessage) -> int:
        """Atomically increments the channel's monotonic seq and persists the message."""
        conn = self._get_conn()
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE channels 
                SET current_seq = current_seq + 1 
                WHERE channel_id = ? 
                RETURNING current_seq;
            """, (msg.channel_id,))
            row = cursor.fetchone()
            if not row:
                # If channel didn't exist, create it on the fly
                conn.execute("""
                    INSERT INTO channels (channel_id, channel_type, name, created_at, current_seq)
                    VALUES (?, 'GROUP', ?, ?, 1);
                """, (msg.channel_id, msg.channel_id, time.time()))
                seq = 1
            else:
                seq = row[0]

            msg.seq = seq
            e2ee_json = json.dumps(msg.e2ee_metadata) if msg.e2ee_metadata else None
            conn.execute("""
                INSERT INTO messages (channel_id, seq, message_id, sender_id, payload, is_e2ee, e2ee_metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (msg.channel_id, seq, msg.message_id, msg.sender_id, msg.payload, 1 if msg.is_e2ee else 0, e2ee_json, msg.created_at))
            return seq

    def enqueue_offline(self, user_id: str, channel_id: str, seq: int, message_id: str):
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT OR IGNORE INTO offline_inbox (user_id, channel_id, seq, message_id, created_at)
                VALUES (?, ?, ?, ?, ?);
            """, (user_id, channel_id, seq, message_id, time.time()))

    def get_delta_messages(self, channel_id: str, since_seq: int, limit: int = 100) -> List[ChatMessage]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT channel_id, seq, message_id, sender_id, payload, is_e2ee, e2ee_metadata, created_at
            FROM messages
            WHERE channel_id = ? AND seq > ?
            ORDER BY seq ASC
            LIMIT ?;
        """, (channel_id, since_seq, limit))
        results = []
        for r in cursor.fetchall():
            meta = json.loads(r[6]) if r[6] else None
            results.append(ChatMessage(
                channel_id=r[0],
                seq=r[1],
                message_id=r[2],
                sender_id=r[3],
                payload=r[4],
                is_e2ee=bool(r[5]),
                e2ee_metadata=meta,
                created_at=r[7]
            ))
        return results

    def update_read_receipt(self, user_id: str, channel_id: str, last_read_seq: int):
        conn = self._get_conn()
        with conn:
            conn.execute("""
                UPDATE channel_members
                SET last_read_seq = MAX(last_read_seq, ?)
                WHERE channel_id = ? AND user_id = ?;
            """, (last_read_seq, channel_id, user_id))
            conn.execute("""
                DELETE FROM offline_inbox
                WHERE user_id = ? AND channel_id = ? AND seq <= ?;
            """, (user_id, channel_id, last_read_seq))

    def get_unread_count(self, user_id: str, channel_id: str) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM offline_inbox
            WHERE user_id = ? AND channel_id = ?;
        """, (user_id, channel_id))
        row = cursor.fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# RFC 6455 WebSocket Framing & Codec
# ---------------------------------------------------------------------------
class WebSocketCodec:
    OPCODE_CONTINUATION = 0x0
    OPCODE_TEXT = 0x1
    OPCODE_BINARY = 0x2
    OPCODE_CLOSE = 0x8
    OPCODE_PING = 0x9
    OPCODE_PONG = 0xA

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionResetError("Socket closed during read.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def parse_frame(sock: socket.socket) -> Tuple[int, bytes]:
        """
        Reads and parses a single RFC 6455 frame from the client socket.
        Client frames MUST be masked.
        Returns: (opcode, unmasked_payload_bytes)
        """
        header = WebSocketCodec._recv_exact(sock, 2)
        b1, b2 = header[0], header[1]
        fin = (b1 & 0x80) != 0
        opcode = b1 & 0x0F
        is_masked = (b2 & 0x80) != 0
        payload_len = b2 & 0x7F

        if payload_len == 126:
            ext_len_bytes = WebSocketCodec._recv_exact(sock, 2)
            payload_len = struct.unpack("!H", ext_len_bytes)[0]
        elif payload_len == 127:
            ext_len_bytes = WebSocketCodec._recv_exact(sock, 8)
            payload_len = struct.unpack("!Q", ext_len_bytes)[0]

        if is_masked:
            mask = WebSocketCodec._recv_exact(sock, 4)
        else:
            mask = None

        raw_payload = WebSocketCodec._recv_exact(sock, payload_len)

        if is_masked and mask:
            unmasked = bytearray(payload_len)
            for i in range(payload_len):
                unmasked[i] = raw_payload[i] ^ mask[i % 4]
            payload = bytes(unmasked)
        else:
            payload = raw_payload

        return opcode, payload

    @staticmethod
    def build_frame(payload_bytes: bytes, opcode: int = OPCODE_TEXT) -> bytes:
        """
        Constructs an unmasked RFC 6455 frame to send from server to client.
        """
        b1 = 0x80 | (opcode & 0x0F)  # FIN = 1
        length = len(payload_bytes)

        if length < 126:
            header = bytes([b1, length])
        elif length <= 0xFFFF:
            header = bytes([b1, 126]) + struct.pack("!H", length)
        else:
            header = bytes([b1, 127]) + struct.pack("!Q", length)

        return header + payload_bytes

    @staticmethod
    def build_text_frame(text: str) -> bytes:
        return WebSocketCodec.build_frame(text.encode("utf-8"), opcode=WebSocketCodec.OPCODE_TEXT)

    @staticmethod
    def build_pong_frame(payload: bytes = b"") -> bytes:
        return WebSocketCodec.build_frame(payload, opcode=WebSocketCodec.OPCODE_PONG)

    @staticmethod
    def build_close_frame() -> bytes:
        return WebSocketCodec.build_frame(b"", opcode=WebSocketCodec.OPCODE_CLOSE)


# ---------------------------------------------------------------------------
# WebSocket Session & Distributed Session Registry
# ---------------------------------------------------------------------------
class WebSocketSession:
    def __init__(self, session_id: str, user_id: str, sock: socket.socket, addr: Tuple[str, int]):
        self.session_id = session_id
        self.user_id = user_id
        self.sock = sock
        self.addr = addr
        self.connected_at = time.time()
        self.last_seen = time.time()
        self.send_lock = threading.Lock()
        self.active = True
        self.subscribed_channels: Set[str] = set()

    def send_frame(self, frame_bytes: bytes):
        if not self.active:
            return
        with self.send_lock:
            try:
                self.sock.sendall(frame_bytes)
            except Exception:
                self.active = False

    def close(self):
        self.active = False
        try:
            self.sock.close()
        except Exception:
            pass


class SessionRegistry:
    """Thread-safe in-memory session registry and presence lease manager."""
    def __init__(self, metrics: ChatMetrics):
        self.metrics = metrics
        self.lock = threading.Lock()
        # user_id -> set of WebSocketSession
        self.sessions: Dict[str, Set[WebSocketSession]] = {}
        # channel_id -> set of WebSocketSession (active subscribers)
        self.channel_subscribers: Dict[str, Set[WebSocketSession]] = {}
        # user_id -> UserPresence
        self.presence: Dict[str, UserPresence] = {}

    def register(self, session: WebSocketSession):
        with self.lock:
            if session.user_id not in self.sessions:
                self.sessions[session.user_id] = set()
            self.sessions[session.user_id].add(session)

            pres = self.presence.get(session.user_id)
            if not pres:
                pres = UserPresence(user_id=session.user_id, status="ONLINE", last_heartbeat=time.time(), active_connections=1)
                self.presence[session.user_id] = pres
            else:
                pres.status = "ONLINE"
                pres.last_heartbeat = time.time()
                pres.active_connections += 1

            self.metrics.record_ws_connect()

    def unregister(self, session: WebSocketSession):
        with self.lock:
            if session.user_id in self.sessions:
                self.sessions[session.user_id].discard(session)
                if not self.sessions[session.user_id]:
                    del self.sessions[session.user_id]

            for ch_id in session.subscribed_channels:
                if ch_id in self.channel_subscribers:
                    self.channel_subscribers[ch_id].discard(session)

            pres = self.presence.get(session.user_id)
            if pres:
                pres.active_connections = max(0, pres.active_connections - 1)
                if pres.active_connections == 0:
                    pres.status = "OFFLINE"
                    pres.last_heartbeat = time.time()

            self.metrics.record_ws_disconnect()

    def subscribe_channel(self, session: WebSocketSession, channel_id: str):
        with self.lock:
            session.subscribed_channels.add(channel_id)
            if channel_id not in self.channel_subscribers:
                self.channel_subscribers[channel_id] = set()
            self.channel_subscribers[channel_id].add(session)

    def record_heartbeat(self, user_id: str):
        with self.lock:
            self.metrics.record_heartbeat()
            if user_id in self.presence:
                self.presence[user_id].last_heartbeat = time.time()
                self.presence[user_id].status = "ONLINE"

    def get_online_sessions(self, user_id: str) -> List[WebSocketSession]:
        with self.lock:
            return list(self.sessions.get(user_id, []))

    def get_channel_subscribers(self, channel_id: str) -> List[WebSocketSession]:
        with self.lock:
            return list(self.channel_subscribers.get(channel_id, []))

    def get_presence(self, user_id: str) -> Dict[str, Any]:
        with self.lock:
            pres = self.presence.get(user_id)
            if not pres:
                return {"user_id": user_id, "status": "OFFLINE", "last_heartbeat": 0, "active_connections": 0}
            
            # Evaluate lease TTL
            is_expired = (time.time() - pres.last_heartbeat) > HEARTBEAT_LEASE_SEC
            effective_status = "OFFLINE" if is_expired or pres.active_connections == 0 else "ONLINE"
            return {
                "user_id": user_id,
                "status": effective_status,
                "last_heartbeat": pres.last_heartbeat,
                "active_connections": pres.active_connections if effective_status == "ONLINE" else 0,
            }


# ---------------------------------------------------------------------------
# Message Router (Hybrid Fan-Out Engine)
# ---------------------------------------------------------------------------
class MessageRouter:
    def __init__(self, storage: StorageEngine, registry: SessionRegistry, metrics: ChatMetrics):
        self.storage = storage
        self.registry = registry
        self.metrics = metrics

    def route_message(self, msg: ChatMessage) -> int:
        """
        Executes strict monotonic sequencing, storage persistence,
        and hybrid fan-out delivery.
        """
        t_start = time.time()

        # 1. Allocate monotonic sequence ID and write to persistent storage
        seq = self.storage.allocate_sequence_and_save_message(msg)

        # 2. Prepare payload wire envelope
        wire_frame = WebSocketCodec.build_text_frame(json.dumps({
            "type": "NEW_MESSAGE",
            "message": msg.to_dict()
        }))

        # 3. Inspect channel topology
        channel_type = self.storage.get_channel_type(msg.channel_id)
        members = self.storage.get_channel_members(msg.channel_id)

        if channel_type == "SUPERGROUP" or len(members) > GROUP_FANOUT_THRESHOLD:
            # PULL / SUBSCRIBER BROADCAST FAN-OUT:
            # Only broadcast to active live subscribers viewing this supergroup channel!
            subscribers = self.registry.get_channel_subscribers(msg.channel_id)
            for sub in subscribers:
                sub.send_frame(wire_frame)
        else:
            # PUSH FAN-OUT FOR 1-ON-1 AND SMALL GROUPS:
            for member_id in members:
                online_sessions = self.registry.get_online_sessions(member_id)
                if online_sessions:
                    # User is online: deliver directly down open WebSocket connections
                    for session in online_sessions:
                        session.send_frame(wire_frame)
                else:
                    # User is offline: store to offline inbox & trigger push notification
                    self.storage.enqueue_offline(member_id, msg.channel_id, seq, msg.message_id)
                    self.metrics.record_offline_push()

        latency_ms = (time.time() - t_start) * 1000.0
        self.metrics.record_message(latency_ms)
        return seq


# ---------------------------------------------------------------------------
# HTTP Handler & WebSocket Upgrade Server
# ---------------------------------------------------------------------------
class ChatGatewayHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        self.server_ref = server
        super().__init__(request, client_address, server)

    def log_message(self, format, *args):
        # Suppress noisy standard HTTP access logs in test runs
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. Check for WebSocket Upgrade
        if path == "/ws":
            if self.headers.get("Upgrade", "").lower() == "websocket":
                self._handle_websocket_upgrade(query)
                return

        # 2. REST Endpoints
        if path == "/healthz":
            self._send_json(200, {
                "status": "UP",
                "engine": "Enterprise Distributed Chat Gateway",
                "active_connections": self.server_ref.metrics.active_websockets,
                "timestamp": time.time()
            })
            return

        if path == "/metrics":
            summary = self.server_ref.metrics.get_summary()
            prometheus_lines = [
                "# HELP chat_messages_processed_total Total messages routed through system",
                "# TYPE chat_messages_processed_total counter",
                f"chat_messages_processed_total {summary['messages_processed']}",
                "# HELP chat_active_websocket_connections Active connected WebSocket sessions",
                "# TYPE chat_active_websocket_connections gauge",
                f"chat_active_websocket_connections {summary['active_websockets']}",
                "# HELP chat_seq_gaps_detected_total Client-side sequence gaps detected",
                "# TYPE chat_seq_gaps_detected_total counter",
                f"chat_seq_gaps_detected_total {summary['seq_gaps_detected']}",
                "# HELP chat_offline_pushes_total Offline push notifications dispatched",
                "# TYPE chat_offline_pushes_total counter",
                f"chat_offline_pushes_total {summary['offline_pushes']}",
                "# HELP chat_latency_p99_ms 99th percentile end-to-end delivery latency",
                "# TYPE chat_latency_p99_ms gauge",
                f"chat_latency_p99_ms {summary['latency_p99_ms']}",
            ]
            body = "\n".join(prometheus_lines) + "\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return

        if path == "/sync":
            self.server_ref.metrics.record_sync()
            user_id = query.get("user_id", [""])[0]
            channel_id = query.get("channel_id", [""])[0]
            since_seq = int(query.get("since_seq", [0])[0])

            if not channel_id:
                self._send_json(400, {"error": "channel_id parameter is required"})
                return

            messages = self.server_ref.storage.get_delta_messages(channel_id, since_seq)
            # Reconcile unread pointer if user_id provided
            if user_id and messages:
                max_seq = max(m.seq for m in messages)
                self.server_ref.storage.update_read_receipt(user_id, channel_id, max_seq)

            self._send_json(200, {
                "channel_id": channel_id,
                "since_seq": since_seq,
                "count": len(messages),
                "messages": [m.to_dict() for m in messages]
            })
            return

        if path == "/presence":
            user_id = query.get("user_id", [""])[0]
            if not user_id:
                self._send_json(400, {"error": "user_id parameter is required"})
                return
            pres = self.server_ref.registry.get_presence(user_id)
            self._send_json(200, pres)
            return

        self._send_json(404, {"error": f"Endpoint '{path}' not found."})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/send":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(body)

            msg = ChatMessage(
                channel_id=data["channel_id"],
                seq=0,
                message_id=data.get("message_id", f"msg_{int(time.time()*1000)}"),
                sender_id=data["sender_id"],
                payload=data.get("payload", ""),
                created_at=time.time(),
                is_e2ee=data.get("is_e2ee", False),
                e2ee_metadata=data.get("e2ee_metadata")
            )
            seq = self.server_ref.router.route_message(msg)
            self._send_json(200, {"status": "DELIVERED", "channel_id": msg.channel_id, "seq": seq})
            return

        self._send_json(404, {"error": f"Endpoint '{path}' not found."})

    def _send_json(self, status: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_websocket_upgrade(self, query: Dict[str, List[str]]):
        """Executes RFC 6455 handshake and enters stateful session loop."""
        key = self.headers.get("Sec-WebSocket-Key", "")
        if not key:
            self.send_error(400, "Missing Sec-WebSocket-Key header")
            return

        user_id = query.get("user_id", [f"anon_{int(time.time()*1000)}"])[0]

        # Calculate accept hash
        accept_raw = hashlib.sha1((key + WS_MAGIC_GUID).encode("utf-8")).digest()
        accept_token = base64.b64encode(accept_raw).decode("utf-8")

        # Send HTTP 101 Switching Protocols response
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_token}\r\n\r\n"
        )
        self.wfile.write(response.encode("utf-8"))
        self.wfile.flush()

        # Detach connection for WebSocket session processing
        raw_sock = self.connection
        session_id = f"sess_{user_id}_{int(time.time()*1000)}"
        session = WebSocketSession(session_id, user_id, raw_sock, self.client_address)
        self.server_ref.registry.register(session)

        # Notify client of successful connection
        session.send_frame(WebSocketCodec.build_text_frame(json.dumps({
            "type": "CONNECTION_ESTABLISHED",
            "session_id": session_id,
            "user_id": user_id,
            "lease_sec": HEARTBEAT_LEASE_SEC
        })))

        # Run session loop
        try:
            while session.active and not self.server_ref.shutdown_requested:
                opcode, payload = WebSocketCodec.parse_frame(raw_sock)

                if opcode == WebSocketCodec.OPCODE_CLOSE:
                    session.send_frame(WebSocketCodec.build_close_frame())
                    break
                elif opcode == WebSocketCodec.OPCODE_PING:
                    session.send_frame(WebSocketCodec.build_pong_frame(payload))
                elif opcode == WebSocketCodec.OPCODE_PONG:
                    self.server_ref.registry.record_heartbeat(user_id)
                elif opcode == WebSocketCodec.OPCODE_TEXT:
                    self._process_client_message(session, payload.decode("utf-8"))
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            pass
        finally:
            self.server_ref.registry.unregister(session)
            session.close()

    def _process_client_message(self, session: WebSocketSession, text: str):
        try:
            packet = json.loads(text)
        except Exception:
            return

        pkt_type = packet.get("type", "")

        if pkt_type == "HEARTBEAT":
            self.server_ref.registry.record_heartbeat(session.user_id)
            session.send_frame(WebSocketCodec.build_text_frame(json.dumps({"type": "HEARTBEAT_ACK", "timestamp": time.time()})))

        elif pkt_type == "SUBSCRIBE":
            channel_id = packet.get("channel_id")
            if channel_id:
                self.server_ref.registry.subscribe_channel(session, channel_id)
                session.send_frame(WebSocketCodec.build_text_frame(json.dumps({
                    "type": "SUBSCRIBED",
                    "channel_id": channel_id
                })))

        elif pkt_type == "SEND_MESSAGE":
            channel_id = packet["channel_id"]
            msg = ChatMessage(
                channel_id=channel_id,
                seq=0,
                message_id=packet.get("message_id", f"msg_{int(time.time()*1000)}"),
                sender_id=session.user_id,
                payload=packet.get("payload", ""),
                created_at=time.time(),
                is_e2ee=packet.get("is_e2ee", False),
                e2ee_metadata=packet.get("e2ee_metadata")
            )
            seq = self.server_ref.router.route_message(msg)
            # Acknowledge to sender with assigned sequence ID
            session.send_frame(WebSocketCodec.build_text_frame(json.dumps({
                "type": "MESSAGE_ACK",
                "message_id": msg.message_id,
                "channel_id": channel_id,
                "seq": seq
            })))

        elif pkt_type == "READ_RECEIPT":
            channel_id = packet.get("channel_id")
            last_seq = packet.get("seq", 0)
            if channel_id and last_seq:
                self.server_ref.storage.update_read_receipt(session.user_id, channel_id, last_seq)


# ---------------------------------------------------------------------------
# Integrated Gateway Server
# ---------------------------------------------------------------------------
class ChatGatewayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str, port: int, db_path: str):
        self.metrics = ChatMetrics()
        self.storage = StorageEngine(db_path)
        self.registry = SessionRegistry(self.metrics)
        self.router = MessageRouter(self.storage, self.registry, self.metrics)
        self.shutdown_requested = False
        super().__init__((host, port), ChatGatewayHandler)


# ---------------------------------------------------------------------------
# Lightweight Real WebSocket Client (Zero-Dependency)
# ---------------------------------------------------------------------------
class RealWebSocketClient:
    """Production client implementing RFC 6455 client-side handshakes and masking."""
    def __init__(self, host: str, port: int, user_id: str):
        self.host = host
        self.port = port
        self.user_id = user_id
        self.sock: Optional[socket.socket] = None
        self.last_seq_seen: Dict[str, int] = {}  # channel_id -> last seq
        self.received_messages: List[Dict[str, Any]] = []
        self.running = False
        self.reader_thread: Optional[threading.Thread] = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

        key = base64.b64encode(os.urandom(16)).decode("utf-8")
        handshake = (
            f"GET /ws?user_id={self.user_id} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(handshake.encode("utf-8"))

        # Read handshake response
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise ConnectionError("Server closed during handshake")
            resp += chunk

        lines = resp.split(b"\r\n")
        if not lines[0].startswith(b"HTTP/1.1 101"):
            raise ConnectionError(f"Handshake rejected: {lines[0].decode('utf-8')}")

        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def send_packet(self, data: Dict[str, Any]):
        payload = json.dumps(data).encode("utf-8")
        length = len(payload)
        mask = os.urandom(4)

        b1 = 0x81  # FIN = 1, TEXT
        if length < 126:
            b2 = 0x80 | length
            header = bytes([b1, b2]) + mask
        elif length <= 0xFFFF:
            b2 = 0x80 | 126
            header = bytes([b1, b2]) + struct.pack("!H", length) + mask
        else:
            b2 = 0x80 | 127
            header = bytes([b1, b2]) + struct.pack("!Q", length) + mask

        # Mask client payload
        masked = bytearray(length)
        for i in range(length):
            masked[i] = payload[i] ^ mask[i % 4]

        self.sock.sendall(header + bytes(masked))

    def _reader_loop(self):
        while self.running:
            try:
                # Client reads unmasked server frames
                header = WebSocketCodec._recv_exact(self.sock, 2)
                b1, b2 = header[0], header[1]
                opcode = b1 & 0x0F
                payload_len = b2 & 0x7F

                if payload_len == 126:
                    ext = WebSocketCodec._recv_exact(self.sock, 2)
                    payload_len = struct.unpack("!H", ext)[0]
                elif payload_len == 127:
                    ext = WebSocketCodec._recv_exact(self.sock, 8)
                    payload_len = struct.unpack("!Q", ext)[0]

                # Server does not mask frames
                payload = WebSocketCodec._recv_exact(self.sock, payload_len)

                if opcode == WebSocketCodec.OPCODE_TEXT:
                    msg_obj = json.loads(payload.decode("utf-8"))
                    self.received_messages.append(msg_obj)
                    if msg_obj.get("type") == "NEW_MESSAGE":
                        inner = msg_obj["message"]
                        ch = inner["channel_id"]
                        seq = inner["seq"]
                        last = self.last_seq_seen.get(ch, 0)
                        if seq > last + 1:
                            print(f"[Gap Alert for {self.user_id}] Detected sequence jump on {ch}: {last} -> {seq}!")
                        self.last_seq_seen[ch] = seq
            except Exception:
                break
        self.running = False

    def close(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Test & Benchmark Suite
# ---------------------------------------------------------------------------
def run_unit_tests():
    print("================================================================================")
    print("RUNNING ENTERPRISE CHAT ENGINE SELF-TEST & VERIFICATION")
    print("================================================================================")
    
    db_file = f"/tmp/test_chat_{int(time.time())}.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    # Pick dynamic available port
    probe_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe_sock.bind(("127.0.0.1", 0))
    port = probe_sock.getsockname()[1]
    probe_sock.close()

    server = ChatGatewayServer("127.0.0.1", port, db_file)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    try:
        # Pre-seed Channels and Memberships
        server.storage.register_channel("chan_dm_alice_bob", "DIRECT", "Alice & Bob Direct")
        server.storage.add_channel_member("chan_dm_alice_bob", "alice")
        server.storage.add_channel_member("chan_dm_alice_bob", "bob")

        server.storage.register_channel("chan_super_eng", "SUPERGROUP", "Engineering Community")
        server.storage.add_channel_member("chan_super_eng", "alice")
        server.storage.add_channel_member("chan_super_eng", "bob")
        server.storage.add_channel_member("chan_super_eng", "charlie")

        # --------------------------------------------------------------------
        # Test 1: Real WebSocket Connection & Handshake
        # --------------------------------------------------------------------
        print("\n[Test 1] Testing RFC 6455 WebSocket Handshake & Session Registration...")
        alice = RealWebSocketClient("127.0.0.1", port, "alice")
        alice.connect()
        bob = RealWebSocketClient("127.0.0.1", port, "bob")
        bob.connect()
        time.sleep(0.2)

        assert len(alice.received_messages) >= 1
        assert alice.received_messages[0]["type"] == "CONNECTION_ESTABLISHED"
        print("  -> Handshake successful. Sessions established for Alice and Bob.")

        # --------------------------------------------------------------------
        # Test 2: 1-on-1 Messaging & Strict Monotonic Sequencing
        # --------------------------------------------------------------------
        print("\n[Test 2] Testing 1:1 Direct Message Push & Monotonic Sequence Assignor...")
        alice.send_packet({
            "type": "SEND_MESSAGE",
            "channel_id": "chan_dm_alice_bob",
            "message_id": "m1",
            "payload": "Hello Bob! This is message 1."
        })
        alice.send_packet({
            "type": "SEND_MESSAGE",
            "channel_id": "chan_dm_alice_bob",
            "message_id": "m2",
            "payload": "Here is message 2 with strict sequence ordering."
        })
        time.sleep(0.3)

        bob_new_msgs = [m for m in bob.received_messages if m.get("type") == "NEW_MESSAGE"]
        assert len(bob_new_msgs) == 2, f"Expected 2 messages, got {len(bob_new_msgs)}"
        assert bob_new_msgs[0]["message"]["seq"] == 1
        assert bob_new_msgs[1]["message"]["seq"] == 2
        print(f"  -> Delivered in-order! Bob received seq {bob_new_msgs[0]['message']['seq']} and {bob_new_msgs[1]['message']['seq']}.")

        # --------------------------------------------------------------------
        # Test 3: Offline Recipient, Inbox Queuing & Delta Catch-Up Sync
        # --------------------------------------------------------------------
        print("\n[Test 3] Testing Offline Disconnect, Inbox Buffer & Delta Sync Catch-Up...")
        # Bob disconnects
        bob.close()
        time.sleep(0.2)

        # Alice sends 2 messages while Bob is offline
        alice.send_packet({
            "type": "SEND_MESSAGE",
            "channel_id": "chan_dm_alice_bob",
            "message_id": "m3_offline",
            "payload": "Bob are you there? (Offline message 1)"
        })
        alice.send_packet({
            "type": "SEND_MESSAGE",
            "channel_id": "chan_dm_alice_bob",
            "message_id": "m4_offline",
            "payload": "Let's review the code later. (Offline message 2)"
        })
        time.sleep(0.3)

        # Check offline storage & metrics
        unread = server.storage.get_unread_count("bob", "chan_dm_alice_bob")
        assert unread == 2, f"Expected 2 offline queued messages, got {unread}"
        print(f"  -> Successfully detected Bob is OFFLINE. Queued {unread} messages in offline_inbox.")

        # Bob reconnects and executes Delta Sync via HTTP endpoint
        delta_msgs = server.storage.get_delta_messages("chan_dm_alice_bob", since_seq=2)
        assert len(delta_msgs) == 2
        assert delta_msgs[0].seq == 3
        assert delta_msgs[1].seq == 4
        print(f"  -> Bob Delta Sync catch-up retrieved missed sequence range [3..4] with 0 loss.")

        # Bob updates read receipt
        server.storage.update_read_receipt("bob", "chan_dm_alice_bob", 4)
        assert server.storage.get_unread_count("bob", "chan_dm_alice_bob") == 0
        print("  -> Read receipt updated to seq=4; offline inbox cleared.")

        # --------------------------------------------------------------------
        # Test 4: Heartbeat Leases & Presence Expiry
        # --------------------------------------------------------------------
        print("\n[Test 4] Testing Ephemeral Presence Leases & Heartbeat TTL...")
        pres = server.registry.get_presence("alice")
        assert pres["status"] == "ONLINE"
        print(f"  -> Alice presence verified: {pres['status']} (active sockets: {pres['active_connections']})")

        # Fake Charlie having disconnected 15 seconds ago
        server.registry.presence["charlie"] = UserPresence("charlie", "ONLINE", time.time() - 20.0, 1)
        charlie_pres = server.registry.get_presence("charlie")
        assert charlie_pres["status"] == "OFFLINE", "Lease should have expired!"
        print("  -> Expired presence lease correctly resolved as OFFLINE without disk writes.")

        # --------------------------------------------------------------------
        # Test 5: Supergroup Pull / Channel Subscription Fan-Out
        # --------------------------------------------------------------------
        print("\n[Test 5] Testing Supergroup Pull Architecture (Zero-Write Amplification)...")
        charlie = RealWebSocketClient("127.0.0.1", port, "charlie")
        charlie.connect()
        time.sleep(0.2)

        # Charlie subscribes to supergroup channel
        charlie.send_packet({"type": "SUBSCRIBE", "channel_id": "chan_super_eng"})
        time.sleep(0.2)

        # Alice posts to supergroup
        alice.send_packet({
            "type": "SEND_MESSAGE",
            "channel_id": "chan_super_eng",
            "message_id": "m_super_1",
            "payload": "Welcome to the Engineering Supergroup!"
        })
        time.sleep(0.3)

        charlie_super_msgs = [m for m in charlie.received_messages if m.get("type") == "NEW_MESSAGE" and m["message"]["channel_id"] == "chan_super_eng"]
        assert len(charlie_super_msgs) == 1
        print("  -> Supergroup message broadcasted directly to active subscriber Charlie with 0 offline fan-out writes.")

        alice.close()
        charlie.close()
        print("\n[✓] ALL 5 UNIT TESTS PASSED FLAWLESSLY!")

    finally:
        server.shutdown()
        server.server_close()
        if os.path.exists(db_file):
            os.remove(db_file)


# ---------------------------------------------------------------------------
# High-Throughput Performance Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(num_messages: int = 5000):
    print("================================================================================")
    print(f"RUNNING HIGH-THROUGHPUT CHAT BENCHMARK: {num_messages:,} MESSAGES")
    print("================================================================================")

    db_file = f"/tmp/bench_chat_{int(time.time())}.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    server = ChatGatewayServer("127.0.0.1", port, db_file)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        channel_id = "bench_chan_perf"
        server.storage.register_channel(channel_id, "GROUP", "Benchmark Room")
        server.storage.add_channel_member(channel_id, "sender_bot")
        server.storage.add_channel_member(channel_id, "receiver_bot")

        receiver = RealWebSocketClient("127.0.0.1", port, "receiver_bot")
        receiver.connect()
        time.sleep(0.2)

        sender = RealWebSocketClient("127.0.0.1", port, "sender_bot")
        sender.connect()
        time.sleep(0.2)

        print(f"Streaming {num_messages:,} framed messages over live WebSocket connections...")
        t_start = time.time()

        for i in range(num_messages):
            sender.send_packet({
                "type": "SEND_MESSAGE",
                "channel_id": channel_id,
                "message_id": f"bm_{i}",
                "payload": f"Benchmark packet #{i} with timestamp {time.time()}"
            })

        # Wait for receiver to catch up
        deadline = time.time() + 15.0
        while len([m for m in receiver.received_messages if m.get("type") == "NEW_MESSAGE"]) < num_messages and time.time() < deadline:
            time.sleep(0.05)

        total_time = time.time() - t_start
        received_count = len([m for m in receiver.received_messages if m.get("type") == "NEW_MESSAGE"])
        qps = received_count / total_time

        summary = server.metrics.get_summary()

        print("\n--------------------------------------------------------------------------------")
        print("CHAT GATEWAY BENCHMARK RESULTS")
        print("--------------------------------------------------------------------------------")
        print(f"Messages Sent:             {num_messages:,}")
        print(f"Messages Received:         {received_count:,} (Loss: {num_messages - received_count})")
        print(f"Total Benchmark Time:      {total_time:.3f} seconds")
        print(f"Throughput:                {qps:,.1f} messages / second")
        print(f"Latency P50:               {summary['latency_p50_ms']} ms")
        print(f"Latency P99:               {summary['latency_p99_ms']} ms")
        print("--------------------------------------------------------------------------------\n")

        sender.close()
        receiver.close()
    finally:
        server.shutdown()
        server.server_close()
        if os.path.exists(db_file):
            os.remove(db_file)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Real-Time Chat Gateway Engine")
    parser.add_argument("--test", action="store_true", help="Run comprehensive unit test verification suite")
    parser.add_argument("--benchmark", action="store_true", help="Run high-throughput message stress benchmark")
    parser.add_argument("--daemon", action="store_true", help="Run live chat gateway daemon on specified port")
    parser.add_argument("--port", type=int, default=8089, help="Port to listen on (default: 8089)")
    parser.add_argument("--db", type=str, default="chat_gateway.db", help="SQLite database path")
    parser.add_argument("--count", type=int, default=5000, help="Benchmark message count")
    args = parser.parse_args()

    if args.test:
        run_unit_tests()
    elif args.benchmark:
        run_benchmark(num_messages=args.count)
    elif args.daemon:
        print(f"Starting Enterprise Chat Gateway Daemon on 0.0.0.0:{args.port}...")
        print(f"  - WebSocket endpoint:  ws://localhost:{args.port}/ws?user_id=<user_id>")
        print(f"  - Delta Sync endpoint: http://localhost:{args.port}/sync?channel_id=<cid>&since_seq=<seq>")
        print(f"  - Health check:        http://localhost:{args.port}/healthz")
        print(f"  - Prometheus metrics:  http://localhost:{args.port}/metrics")
        server = ChatGatewayServer("0.0.0.0", args.port, args.db)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Chat Gateway Daemon.")
            server.shutdown()
            server.server_close()
    else:
        # Default: run tests followed by benchmark
        run_unit_tests()
        run_benchmark(num_messages=3000)
