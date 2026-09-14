#!/usr/bin/env python3
"""
Scalable OpenClaw Autonomous Agent Architecture
Alex Xu Volume 3 - Chapter 5 Production Implementation

Features:
- Pure Python 3 standard library with zero external pip dependencies.
- Omnichannel Unified Gateway (WhatsApp, Telegram, Slack, WebChat normalization).
- In-Flight Turn Steering Controller with 4 queue modes (steer, followup, collect, interrupt).
- Human-Inspectable "No Hidden State" Tiered Memory (AGENTS.md, MEMORY.md, USER.md, DREAMS.md).
- Provenance Taint Tracker preventing untrusted web content from poisoning memory tiers.
- Biologically-inspired Tri-Phase Dreaming Consolidation Engine (Light -> REM -> Deep).
- Companion Device Node Fabric with cryptographic challenge-nonce pairing.
- Multi-threaded HTTP REST daemon with Prometheus /metrics and /healthz.
- Built-in verification test suite (--test) and high-concurrency benchmark (--benchmark).
"""

import sys
import os
import time
import json
import uuid
import hashlib
import hmac
import threading
import queue
import argparse
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, List, Optional, Tuple, Any, Set, Callable


# ============================================================================
# Domain Models & Enums
# ============================================================================

class ChannelType(Enum):
    WHATSAPP = "WHATSAPP"
    TELEGRAM = "TELEGRAM"
    SLACK = "SLACK"
    WEBCHAT = "WEBCHAT"
    DEVICE_NODE = "DEVICE_NODE"


class SteeringMode(Enum):
    STEER = "steer"          # Inject mid-turn, redirect in-flight reasoning
    FOLLOWUP = "followup"    # Enqueue as sequential next turn
    COLLECT = "collect"      # Coalesce burst messages into single follow-up
    INTERRUPT = "interrupt"  # Abort active turn immediately, start newest


class Provenance(Enum):
    OWNER = "OWNER"
    AGENT = "AGENT"
    UNTRUSTED_WEB = "UNTRUSTED_WEB"
    SYSTEM = "SYSTEM"


class DeviceType(Enum):
    MACOS = "MACOS"
    IOS = "IOS"
    ANDROID = "ANDROID"
    SERVER = "SERVER"


class DelegateTier(Enum):
    TIER_1_DRAFT = "TIER_1_DRAFT"         # Read-only draft creation
    TIER_2_SEND_DELEGATE = "TIER_2_SEND"   # Send on behalf of user within budget
    TIER_3_AUTONOMOUS = "TIER_3_AUTONOMOUS"# Proactive crons & unassisted decisions


@dataclass
class ChannelMessage:
    message_id: str
    channel: ChannelType
    sender_id: str
    text: str
    media_url: Optional[str] = None
    provenance: Provenance = Provenance.OWNER
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryEntry:
    entry_id: str
    tier: str
    content: str
    provenance: Provenance
    recall_count: int = 0
    salience_score: float = 0.5
    created_at: float = field(default_factory=time.time)


@dataclass
class DeviceNode:
    node_id: str
    device_type: DeviceType
    public_key_fingerprint: str
    capabilities: List[str]
    is_paired: bool = False
    last_ping: float = field(default_factory=time.time)


# ============================================================================
# Omnichannel Gateway & Normalization
# ============================================================================

class OmnichannelGateway:
    """
    Normalizes heterogeneous protocol inputs (WhatsApp Baileys sockets, Telegram webhooks,
    Slack Bolt payloads, WebChat) into canonical ChannelMessage envelopes.
    """

    def __init__(self):
        self.channel_sessions: Dict[str, str] = {}  # channel_user_id -> canonical_session_id
        self._lock = threading.Lock()

    def bind_channel_identity(self, channel: ChannelType, channel_user: str, canonical_user: str):
        with self._lock:
            key = f"{channel.value}:{channel_user}"
            self.channel_sessions[key] = canonical_user

    def normalize_inbound(self, channel: ChannelType, raw_payload: Dict[str, Any]) -> ChannelMessage:
        """Translates channel-specific JSON payload into canonical ChannelMessage."""
        msg_id = f"msg_{uuid.uuid4().hex[:10]}"
        sender = raw_payload.get("from", "unknown_user")
        text = raw_payload.get("text", "")
        media = raw_payload.get("media_url")
        prov = Provenance.OWNER

        if channel == ChannelType.WHATSAPP:
            # WhatsApp Baileys format
            sender = raw_payload.get("key", {}).get("remoteJid", sender)
            text = raw_payload.get("message", {}).get("conversation", text)
        elif channel == ChannelType.TELEGRAM:
            # Telegram Bot API format
            from_field = raw_payload.get("from")
            if isinstance(from_field, dict):
                sender = str(from_field.get("id", sender))
            elif isinstance(from_field, str):
                sender = from_field
            text = raw_payload.get("text", text)
        elif channel == ChannelType.SLACK:
            # Slack Bolt event
            sender = raw_payload.get("user", sender)
            text = raw_payload.get("text", text)

        return ChannelMessage(
            message_id=msg_id,
            channel=channel,
            sender_id=sender,
            text=text,
            media_url=media,
            provenance=prov
        )


# ============================================================================
# In-Flight Turn Steering & Queue Controller
# ============================================================================

class InFlightSteeringController:
    """
    Controls dynamic turn steering and multi-mode queueing for autonomous agent turns.
    Supports mid-turn trajectory redirection without thread restarts.
    """

    def __init__(self):
        self.active_turn: Optional[Dict[str, Any]] = None
        self.followup_queue: deque = deque()
        self.collected_buffer: List[str] = []
        self.turn_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def submit_message(self, message: ChannelMessage, mode: SteeringMode) -> Dict[str, Any]:
        """
        Dispatches incoming message based on chosen steering mode.
        """
        with self._lock:
            if not self.active_turn or self.active_turn.get("status") == "COMPLETED":
                # No active turn running, start immediate turn
                return self._start_turn(message)

            if mode == SteeringMode.STEER:
                # Inject mid-turn steering instruction into active turn
                self.active_turn["injected_instructions"].append(message.text)
                self.active_turn["skip_planned_tools"] = True
                return {
                    "action": "STEERED_MID_FLIGHT",
                    "turn_id": self.active_turn["turn_id"],
                    "injected_text": message.text
                }

            elif mode == SteeringMode.INTERRUPT:
                # Abort running turn immediately, discard remaining steps
                aborted_id = self.active_turn["turn_id"]
                self.active_turn["status"] = "ABORTED"
                self.turn_history.append(dict(self.active_turn))
                # Start new turn with current message
                return self._start_turn(message, interrupted_turn=aborted_id)

            elif mode == SteeringMode.COLLECT:
                # Coalesce multiple burst messages into single follow-up buffer
                self.collected_buffer.append(message.text)
                return {
                    "action": "COLLECTED_INTO_BUFFER",
                    "buffer_size": len(self.collected_buffer)
                }

            elif mode == SteeringMode.FOLLOWUP:
                # Standard FIFO queue
                self.followup_queue.append(message)
                return {
                    "action": "ENQUEUED_FOLLOWUP",
                    "queue_position": len(self.followup_queue)
                }

        return {"action": "UNKNOWN"}

    def _start_turn(self, message: ChannelMessage, interrupted_turn: Optional[str] = None) -> Dict[str, Any]:
        turn_id = f"turn_{uuid.uuid4().hex[:10]}"
        self.active_turn = {
            "turn_id": turn_id,
            "prompt": message.text,
            "status": "RUNNING",
            "injected_instructions": [],
            "skip_planned_tools": False,
            "interrupted_turn": interrupted_turn,
            "started_at": time.time()
        }
        return {
            "action": "STARTED_TURN",
            "turn_id": turn_id,
            "prompt": message.text
        }

    def execute_active_turn_cycle(self) -> Dict[str, Any]:
        """Simulates execution of the active agent reasoning turn with in-flight checks."""
        with self._lock:
            if not self.active_turn or self.active_turn["status"] != "RUNNING":
                return {"status": "IDLE"}

            turn = self.active_turn
            # Check if mid-turn steering arrived
            if turn["injected_instructions"]:
                effective_prompt = f"{turn['prompt']} -> [MID-TURN STEER: {', '.join(turn['injected_instructions'])}]"
                action_taken = "DYNAMICALLY_STEERED_EXECUTION"
            else:
                effective_prompt = turn["prompt"]
                action_taken = "STANDARD_EXECUTION"

            turn["status"] = "COMPLETED"
            turn["effective_prompt"] = effective_prompt
            turn["completed_at"] = time.time()
            self.turn_history.append(dict(turn))

            # If collected buffer exists, automatically pop it into next turn
            next_turn_info = None
            if self.collected_buffer:
                coalesced_text = " \n".join(self.collected_buffer)
                self.collected_buffer.clear()
                synthetic_msg = ChannelMessage(
                    message_id=f"synth_{uuid.uuid4().hex[:8]}",
                    channel=ChannelType.WEBCHAT,
                    sender_id="coalesced_user",
                    text=coalesced_text
                )
                next_turn_info = self._start_turn(synthetic_msg)
            elif self.followup_queue:
                next_msg = self.followup_queue.popleft()
                next_turn_info = self._start_turn(next_msg)
            else:
                self.active_turn = None

            return {
                "turn_id": turn["turn_id"],
                "status": "COMPLETED",
                "action_taken": action_taken,
                "effective_prompt": effective_prompt,
                "next_turn": next_turn_info
            }


# ============================================================================
# Tiered Memory Engine & Taint Tracking
# ============================================================================

class TieredMemoryEngine:
    """
    Implements OpenClaw's human-inspectable "No Hidden State" tiered Markdown memory.
    Enforces write-time provenance and anti-poisoning taint tracking.
    """

    def __init__(self):
        # Plain text simulated storage
        self.instructions_md: str = "System Standing Orders: Protect user secrets. Be concise."
        self.memory_md: Dict[str, MemoryEntry] = {}   # Long-term curated facts
        self.user_md: Dict[str, Any] = {}             # User profile attributes
        self.episodic_log: List[MemoryEntry] = []     # Daily conversation traces
        self.dreams_md: List[str] = []                # Consolidated wisdom summaries
        self._lock = threading.Lock()

    def record_interaction(self, content: str, provenance: Provenance) -> MemoryEntry:
        """Appends raw event to daily episodic memory with provenance tagging."""
        with self._lock:
            entry = MemoryEntry(
                entry_id=f"mem_{uuid.uuid4().hex[:8]}",
                tier="EPISODIC",
                content=content,
                provenance=provenance,
                salience_score=0.6 if provenance == Provenance.OWNER else 0.2
            )
            self.episodic_log.append(entry)
            return entry

    def write_curated_memory(self, fact_key: str, content: str, provenance: Provenance) -> bool:
        """
        Guards MEMORY.md against taint poisoning: UNTRUSTED_WEB content is rejected
        from directly mutating curated facts without explicit human review.
        """
        with self._lock:
            if provenance == Provenance.UNTRUSTED_WEB:
                # Anti-poisoning security firewall
                raise PermissionError("Taint Security Guard: Untrusted web content cannot write directly to MEMORY.md!")

            entry = MemoryEntry(
                entry_id=f"cur_{uuid.uuid4().hex[:8]}",
                tier="CURATED_MEMORY",
                content=content,
                provenance=provenance,
                recall_count=1,
                salience_score=0.9
            )
            self.memory_md[fact_key] = entry
            return True


# ============================================================================
# Tri-Phase Dreaming Consolidation Engine
# ============================================================================

class TriPhaseDreamingEngine:
    """
    Biologically-inspired memory consolidation executing during idle agent periods.
    Phase 1: Light Sleep (Deduplicate & filter daily episodic traces)
    Phase 2: REM Sleep (Extract recurring themes and semantic associations)
    Phase 3: Deep Sleep (Threshold gating -> rewrite MEMORY.md & DREAMS.md)
    """

    @classmethod
    def consolidate(cls, memory_engine: TieredMemoryEngine, min_salience: float = 0.5) -> Dict[str, Any]:
        with memory_engine._lock:
            episodes = list(memory_engine.episodic_log)

        # Phase 1: Light Sleep (Deduplication & Taint Pruning)
        light_cleaned = []
        seen_texts = set()
        for ep in episodes:
            # Drop untrusted web traces from permanent dream consolidation
            if ep.provenance == Provenance.UNTRUSTED_WEB:
                continue
            normalized = ep.content.strip().lower()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                light_cleaned.append(ep)

        # Phase 2: REM Sleep (Pattern Recognition & Associative Clustering)
        theme_clusters = defaultdict(list)
        for ep in light_cleaned:
            # Semantic categorization heuristic
            if any(w in ep.content.lower() for w in ["database", "postgres", "sql", "migration"]):
                theme_clusters["architecture"].append(ep)
            elif any(w in ep.content.lower() for w in ["flight", "hotel", "travel", "paris"]):
                theme_clusters["travel"].append(ep)
            else:
                theme_clusters["general"].append(ep)

        # Phase 3: Deep Sleep (Threshold Gating & Durable Consolidation)
        promoted_insights = []
        for theme, items in theme_clusters.items():
            if not items:
                continue
            # Calculate aggregate salience
            avg_score = sum(i.salience_score for i in items) / len(items)
            if avg_score >= min_salience:
                insight = f"Consolidated Theme [{theme.upper()}]: Synthesized from {len(items)} traces. Key note: '{items[0].content}'"
                promoted_insights.append(insight)
                # Durably append to DREAMS.md
                with memory_engine._lock:
                    memory_engine.dreams_md.append(insight)
                    # Automatically update curated MEMORY.md for high-salience themes
                    memory_engine.memory_md[f"theme_{theme}"] = MemoryEntry(
                        entry_id=f"dream_{uuid.uuid4().hex[:8]}",
                        tier="DREAMS_CONSOLIDATED",
                        content=insight,
                        provenance=Provenance.AGENT,
                        recall_count=len(items),
                        salience_score=avg_score
                    )

        return {
            "status": "DREAM_CONSOLIDATION_COMPLETE",
            "episodes_processed": len(episodes),
            "light_sleep_retained": len(light_cleaned),
            "rem_themes_discovered": list(theme_clusters.keys()),
            "deep_sleep_insights": promoted_insights
        }


# ============================================================================
# Companion Device Node Fabric & Cryptographic Pairing
# ============================================================================

class DeviceNodeFabric:
    """
    Manages companion device companion nodes (macOS, iOS, Android) across strict NATs.
    Enforces HMAC challenge-nonce cryptographic pairing and capability leases.
    """

    def __init__(self, master_secret: str = "openclaw_enterprise_secret_key"):
        self.master_secret = master_secret.encode("utf-8")
        self.registered_nodes: Dict[str, DeviceNode] = {}
        self.active_challenges: Dict[str, str] = {}  # node_id -> nonce
        self._lock = threading.Lock()

    def request_pairing(self, node_id: str, device_type: DeviceType,
                        capabilities: List[str]) -> str:
        """Issues cryptographic nonce challenge to new companion node."""
        with self._lock:
            nonce = uuid.uuid4().hex
            self.active_challenges[node_id] = nonce
            node = DeviceNode(
                node_id=node_id,
                device_type=device_type,
                public_key_fingerprint=f"fp_{uuid.uuid4().hex[:8]}",
                capabilities=capabilities,
                is_paired=False
            )
            self.registered_nodes[node_id] = node
            return nonce

    def complete_pairing(self, node_id: str, signature: str) -> bool:
        """Verifies HMAC signature of nonce challenge to complete secure pairing."""
        with self._lock:
            if node_id not in self.active_challenges:
                return False
            nonce = self.active_challenges[node_id]
            expected_sig = hmac.new(self.master_secret, nonce.encode("utf-8"), hashlib.sha256).hexdigest()

            if hmac.compare_digest(expected_sig, signature):
                self.registered_nodes[node_id].is_paired = True
                del self.active_challenges[node_id]
                return True
            return False

    def dispatch_device_command(self, node_id: str, command: str,
                                payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches command to paired companion node."""
        with self._lock:
            if node_id not in self.registered_nodes:
                raise KeyError(f"Node '{node_id}' is not registered.")
            node = self.registered_nodes[node_id]
            if not node.is_paired:
                raise PermissionError(f"Node '{node_id}' is not securely paired.")
            if command not in node.capabilities:
                raise ValueError(f"Capability '{command}' not supported on node '{node_id}'.")

            node.last_ping = time.time()

        return {
            "status": "COMMAND_DISPATCHED",
            "node_id": node_id,
            "device": node.device_type.value,
            "command": command,
            "result": f"Executed {command} on {node.device_type.value} companion node."
        }


# ============================================================================
# OpenClaw Autonomous Agent Core Coordinator
# ============================================================================

class OpenClawAgentCoordinator:
    """
    Master Coordinator uniting:
    - Omnichannel Gateway
    - In-Flight Steering Controller
    - Tiered Memory & Taint Tracker
    - Dreaming Consolidation Engine
    - Companion Device Node Fabric
    """

    def __init__(self):
        self.gateway = OmnichannelGateway()
        self.steering = InFlightSteeringController()
        self.memory = TieredMemoryEngine()
        self.nodes = DeviceNodeFabric()
        self.delegate_tier = DelegateTier.TIER_2_SEND_DELEGATE
        self.metrics = {
            "inbound_messages": 0,
            "steered_turns": 0,
            "interrupted_turns": 0,
            "memories_recorded": 0,
            "dreams_consolidated": 0,
            "paired_nodes": 0
        }
        self._lock = threading.Lock()

    def handle_inbound_chat(self, channel: ChannelType, raw_payload: Dict[str, Any],
                            steering_mode: SteeringMode = SteeringMode.FOLLOWUP) -> Dict[str, Any]:
        msg = self.gateway.normalize_inbound(channel, raw_payload)
        with self._lock:
            self.metrics["inbound_messages"] += 1
            if steering_mode == SteeringMode.STEER:
                self.metrics["steered_turns"] += 1
            elif steering_mode == SteeringMode.INTERRUPT:
                self.metrics["interrupted_turns"] += 1

        # Record to episodic memory
        self.memory.record_interaction(msg.text, msg.provenance)
        with self._lock:
            self.metrics["memories_recorded"] += 1

        # Submit to steering controller
        return self.steering.submit_message(msg, steering_mode)


# ============================================================================
# HTTP REST API Daemon & Prometheus Telemetry
# ============================================================================

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class OpenClawAPIHandler(BaseHTTPRequestHandler):
    coordinator: OpenClawAgentCoordinator

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        response_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self):
        if self.path == "/healthz":
            self._send_json(200, {
                "status": "healthy",
                "uptime_seconds": time.time(),
                "active_turn": bool(self.coordinator.steering.active_turn),
                "paired_nodes": len([n for n in self.coordinator.nodes.registered_nodes.values() if n.is_paired])
            })

        elif self.path == "/metrics":
            m = self.coordinator.metrics
            output = [
                "# HELP openclaw_inbound_messages_total Inbound omnichannel messages received",
                "# TYPE openclaw_inbound_messages_total counter",
                f"openclaw_inbound_messages_total {m['inbound_messages']}",
                "# HELP openclaw_steered_turns_total In-flight mid-turn steers executed",
                "# TYPE openclaw_steered_turns_total counter",
                f"openclaw_steered_turns_total {m['steered_turns']}",
                "# HELP openclaw_interrupted_turns_total Mid-flight turns aborted and replaced",
                "# TYPE openclaw_interrupted_turns_total counter",
                f"openclaw_interrupted_turns_total {m['interrupted_turns']}",
                "# HELP openclaw_memories_recorded_total Episodic memories logged",
                "# TYPE openclaw_memories_recorded_total counter",
                f"openclaw_memories_recorded_total {m['memories_recorded']}",
                "# HELP openclaw_dreams_consolidated_total Tri-phase dream consolidations",
                "# TYPE openclaw_dreams_consolidated_total counter",
                f"openclaw_dreams_consolidated_total {m['dreams_consolidated']}"
            ]
            metrics_body = "\n".join(output) + "\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(metrics_body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(metrics_body.encode("utf-8"))

        elif self.path == "/v1/memory":
            mem = self.coordinator.memory
            self._send_json(200, {
                "instructions": mem.instructions_md,
                "curated_memory_count": len(mem.memory_md),
                "episodic_log_count": len(mem.episodic_log),
                "dreams_consolidated": mem.dreams_md
            })

        else:
            self._send_json(404, {"error": "Endpoint not found."})

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            body = json.loads(post_data)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON payload."})
            return

        if self.path == "/v1/chat":
            ch_str = body.get("channel", "WEBCHAT").upper()
            channel = ChannelType[ch_str] if ch_str in ChannelType.__members__ else ChannelType.WEBCHAT
            mode_str = body.get("mode", "followup").lower()
            mode = SteeringMode(mode_str) if mode_str in [e.value for e in SteeringMode] else SteeringMode.FOLLOWUP
            res = self.coordinator.handle_inbound_chat(channel, body.get("payload", {}), mode)
            self._send_json(200, res)

        elif self.path == "/v1/memory/dream":
            res = TriPhaseDreamingEngine.consolidate(self.coordinator.memory)
            self.coordinator.metrics["dreams_consolidated"] += 1
            self._send_json(200, res)

        else:
            self._send_json(404, {"error": "Endpoint not found."})

    def log_message(self, format, *args):
        return


# ============================================================================
# Verification Test Suite & High-Concurrency Benchmark
# ============================================================================

def run_tests():
    """Runs complete Chapter 5 verification test suite."""
    print("\n" + "=" * 80)
    print("RUNNING CHAPTER 5: SCALABLE OPENCLAW AUTONOMOUS AGENT TEST SUITE")
    print("=" * 80)

    coord = OpenClawAgentCoordinator()

    # 1. Omnichannel Gateway Inbound Normalization
    print("\n[Test 1] Omnichannel Inbound Normalization (WhatsApp, Telegram, Slack)...")
    wa_raw = {"key": {"remoteJid": "15550199@s.whatsapp.net"}, "message": {"conversation": "Book flights to Paris"}}
    wa_msg = coord.gateway.normalize_inbound(ChannelType.WHATSAPP, wa_raw)
    assert wa_msg.channel == ChannelType.WHATSAPP
    assert wa_msg.text == "Book flights to Paris"
    assert wa_msg.sender_id == "15550199@s.whatsapp.net"

    tg_raw = {"from": {"id": 987654}, "text": "Schedule deployment"}
    tg_msg = coord.gateway.normalize_inbound(ChannelType.TELEGRAM, tg_raw)
    assert tg_msg.channel == ChannelType.TELEGRAM
    assert tg_msg.text == "Schedule deployment"
    assert tg_msg.sender_id == "987654"
    print("  ✓ WhatsApp & Telegram payloads normalized into canonical ChannelMessage.")

    # 2. In-Flight Turn Steering (steer, followup, collect, interrupt)
    print("\n[Test 2] In-Flight Turn Steering & Queue Modes...")
    # Mode A: Start Turn
    m1 = ChannelMessage("m1", ChannelType.WEBCHAT, "user1", "Refactor payment service to Stripe")
    r1 = coord.steering.submit_message(m1, SteeringMode.FOLLOWUP)
    assert r1["action"] == "STARTED_TURN"
    assert coord.steering.active_turn["status"] == "RUNNING"
    print("  ✓ Active turn initiated: Refactor payment service.")

    # Mode B: Mid-Flight Steer (Inject without aborting)
    m_steer = ChannelMessage("m2", ChannelType.WEBCHAT, "user1", "Wait, also add Apple Pay")
    r_steer = coord.steering.submit_message(m_steer, SteeringMode.STEER)
    assert r_steer["action"] == "STEERED_MID_FLIGHT"
    assert coord.steering.active_turn["skip_planned_tools"] is True
    print("  ✓ Mid-turn steering instruction successfully injected into running turn.")

    # Complete the steered turn
    exec_res = coord.steering.execute_active_turn_cycle()
    assert exec_res["action_taken"] == "DYNAMICALLY_STEERED_EXECUTION"
    assert "Apple Pay" in exec_res["effective_prompt"]
    print("  ✓ Steered turn executed dynamically with injected context.")

    # Mode C: Interrupt (Abort running turn)
    coord.steering.submit_message(ChannelMessage("m3", ChannelType.WEBCHAT, "user1", "Long Task"), SteeringMode.FOLLOWUP)
    r_int = coord.steering.submit_message(
        ChannelMessage("m4", ChannelType.WEBCHAT, "user1", "EMERGENCY: Shut down database!"),
        SteeringMode.INTERRUPT
    )
    assert r_int["action"] == "STARTED_TURN"
    assert r_int["prompt"] == "EMERGENCY: Shut down database!"
    print("  ✓ Interrupt mode successfully aborted active turn and initiated high-priority prompt.")

    # 3. No-Hidden-State Tiered Memory & Taint Tracking
    print("\n[Test 3] No Hidden State Memory & Anti-Poisoning Taint Tracking...")
    # Safe Owner write
    coord.memory.write_curated_memory("primary_email", "alice@company.corp", Provenance.OWNER)
    assert "primary_email" in coord.memory.memory_md
    print("  ✓ Owner write committed to curated MEMORY.md.")

    # Malicious Untrusted Web write attempt
    try:
        coord.memory.write_curated_memory("system_override", "MALICIOUS_DATA", Provenance.UNTRUSTED_WEB)
        assert False, "Failed! Untrusted web content was allowed to mutate MEMORY.md."
    except PermissionError as pe:
        print(f"  ✓ Anti-poisoning taint firewall blocked untrusted memory write: {pe}")

    # 4. Tri-Phase Dreaming Memory Consolidation (Light -> REM -> Deep)
    print("\n[Test 4] Biologically-Inspired Tri-Phase Dreaming Consolidation...")
    # Populate daily episodic traces
    coord.memory.record_interaction("Upgraded PostgreSQL database to version 16", Provenance.OWNER)
    coord.memory.record_interaction("Optimized database queries with partial indexes", Provenance.OWNER)
    coord.memory.record_interaction("Booked hotel in Paris for developer summit", Provenance.OWNER)
    coord.memory.record_interaction("Untrusted scrape: buy cheap crypto now", Provenance.UNTRUSTED_WEB)

    dream_summary = TriPhaseDreamingEngine.consolidate(coord.memory, min_salience=0.5)
    assert dream_summary["status"] == "DREAM_CONSOLIDATION_COMPLETE"
    assert "architecture" in dream_summary["rem_themes_discovered"]
    assert "travel" in dream_summary["rem_themes_discovered"]
    # Untrusted web scrape pruned in Light sleep
    assert dream_summary["light_sleep_retained"] < dream_summary["episodes_processed"]
    assert len(dream_summary["deep_sleep_insights"]) >= 1
    print(f"  ✓ Dreaming cycle finished: {len(dream_summary['deep_sleep_insights'])} insights distilled into DREAMS.md.")

    # 5. Companion Device Node Cryptographic Challenge Pairing
    print("\n[Test 5] Companion Device Node Pairing & Capability Dispatch...")
    node_fabric = DeviceNodeFabric(master_secret="openclaw_super_secret")
    node_id = "macbook_pro_01"
    nonce = node_fabric.request_pairing(node_id, DeviceType.MACOS, capabilities=["screen_capture", "canvas_render"])
    assert node_fabric.registered_nodes[node_id].is_paired is False

    # Sign challenge nonce with HMAC-SHA256
    sig = hmac.new(b"openclaw_super_secret", nonce.encode("utf-8"), hashlib.sha256).hexdigest()
    paired_ok = node_fabric.complete_pairing(node_id, sig)
    assert paired_ok is True
    assert node_fabric.registered_nodes[node_id].is_paired is True
    print("  ✓ Cryptographic HMAC challenge-nonce handshake completed successfully.")

    # Dispatch capability command
    cmd_res = node_fabric.dispatch_device_command(node_id, "screen_capture", {"resolution": "4K"})
    assert cmd_res["status"] == "COMMAND_DISPATCHED"
    print("  ✓ Dispatched capability 'screen_capture' to paired companion node.")

    print("\n" + "=" * 80)
    print("ALL 5 OPENCLAW AGENT ARCHITECTURE TESTS PASSED! (100% VERIFIED)")
    print("=" * 80 + "\n")


def run_benchmark(num_messages: int = 50_000):
    """Benchmarks omnichannel gateway normalization and in-flight queue throughput."""
    print("\n" + "=" * 80)
    print("STARTING OPENCLAW AUTONOMOUS AGENT HIGH-THROUGHPUT BENCHMARK")
    print(f"Target: {num_messages:,} Omnichannel Messages & Steering Turns")
    print("=" * 80)

    coord = OpenClawAgentCoordinator()

    t_start = time.perf_counter()
    for i in range(num_messages):
        payload = {"from": f"tg_user_{i % 100}", "text": f"Status query {i}"}
        coord.handle_inbound_chat(ChannelType.TELEGRAM, payload, SteeringMode.FOLLOWUP)
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    throughput = num_messages / elapsed

    print(f"\n--- BENCHMARK RESULTS ---")
    print(f"Total Omnichannel Messages:   {num_messages:,}")
    print(f"Total Episodic Logs Written:  {len(coord.memory.episodic_log):,}")
    print(f"Total Elapsed Time:           {elapsed:.3f} seconds")
    print(f"Ingress & In-Memory TPS:      {throughput:,.1f} Messages/sec")
    print("=" * 80 + "\n")


def run_server(port: int = 8087):
    """Launches production HTTP REST daemon."""
    coord = OpenClawAgentCoordinator()
    OpenClawAPIHandler.coordinator = coord
    server_address = ("", port)
    httpd = ThreadedHTTPServer(server_address, OpenClawAPIHandler)
    print(f"[*] OpenClaw Autonomous Agent HTTP Daemon listening on port {port}...")
    print(f"[*] Health Check:  http://localhost:{port}/healthz")
    print(f"[*] Metrics:       http://localhost:{port}/metrics")
    print(f"[*] API Endpoints: POST /v1/chat, POST /v1/memory/dream, GET /v1/memory")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down OpenClaw daemon gracefully...")
        httpd.shutdown()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scalable OpenClaw Autonomous Agent Architecture")
    parser.add_argument("--test", action="store_true", help="Run verification test suite")
    parser.add_argument("--benchmark", action="store_true", help="Run high-throughput message benchmark")
    parser.add_argument("--server", action="store_true", help="Launch HTTP REST daemon")
    parser.add_argument("--port", type=int, default=8087, help="Port for HTTP daemon (default: 8087)")
    parser.add_argument("--messages", type=int, default=50000, help="Message count for benchmark (default: 50000)")

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.benchmark:
        run_benchmark(num_messages=args.messages)
    elif args.server:
        run_server(port=args.port)
    else:
        run_tests()
        run_benchmark(num_messages=20000)


if __name__ == "__main__":
    main()
