---
title: "Chapter Hub: Enterprise LLM Gateway & Distributed KV-Cache Serving"
volume: 3
chapter: "17-Distributed-KV-Cache"
difficulty: "Very Hard"
status: "Completed & Verified"
tags: ["llm-gateway", "kv-cache", "vllm", "sglang", "pd-disaggregation", "rocev2", "paged-attention"]
---

# Enterprise LLM Gateway & Distributed KV-Cache Serving — Chapter Hub

> [!important] The Complete System Design Learning Bundle
> This chapter is organized as a unified, self-contained learning module for mastering both the engineering depth and the Staff/Principal interview execution.

---

## 🧭 Navigation & Module Directory

| Resource | Document Link | Description & Key Focus |
|:---|:---|:---|
| 📐 **Architectural Blueprint** | [`01-Architectural-Blueprint.md`](01-Architectural-Blueprint.md) | Full RFC specification, quantitative sizing, schema design, and production topology diagrams. |
| 🎙️ **Interactive Interview Playbook** | [`02-Interactive-Interview-Playbook.md`](02-Interactive-Interview-Playbook.md) | 45-minute live interview roleplay, sparring transcripts, trap cards, and candidate leveling rubrics. |
| 💡 **Deep Explainability Guide** | [`03-Deep-Explainability-Guide.md`](03-Deep-Explainability-Guide.md) | First-principles mental models, technology showdowns ("Why X over Y?"), and mathematical derivations. |
| 🧪 **Runnable Python Lab** | [`kv_cache_gateway_engine.py`](kv_cache_gateway_engine.py) | Production simulation engine with chaos drills, fault injection, and benchmark suite. |

---

## ⚡ 30-Second Executive Pitch

Eliminates GPU memory fragmentation and multi-turn prefill redundancy with Prefill-Decode Disaggregation (Mooncake & Splitwise), 400G RoCEv2 RDMA transfers, and RadixAttention prefix caching.

- **Hyperscale SLA Baseline**: 50,000 requests/minute; 70B parameter models; Time-to-First-Token (TTFT) < 350 ms with cache hit; P99 Inter-Token Latency (ITL) < 25 ms; > 65% prefix cache hit rate.
- **Core Engineering Challenge**: Bridging speed-of-light physical constraints, high-concurrency memory efficiency, and partition resilience without degrading hot-path latency SLAs.

---

## 🧠 Mental Model & Physical Analogy

Imagine a high-speed professional kitchen where 100 chefs make gourmet cakes. In a traditional kitchen (Monolithic GPU serving), the same chef who spends 30 minutes baking the cake layers in the oven (Compute-heavy Prefill) must also spend 5 minutes meticulously piping frosting onto the cake one drop at a time (Memory-bound Decode). Because baking takes so long, frosting stations sit idle. The modern kitchen separates chefs into two specialized rooms: the Oven Room (Prefill GPUs) bakes cakes at maximum temperature, while high-speed motorized carts (400G RDMA network) wheel the baked cakes to the Frosting Room (Decode GPUs) where decorators pipe frosting without ever waiting for an oven.

---

## 🥊 Technology Showdown Snapshot

| Technology Dimension | Primary Choice | Key Trade-Off Reason |
|:---|:---|:---|
| **Core Architecture / Engine** | `Prefill-Decode Disaggregation (Mooncake/Splitwise)` | SOTA STANDARD: Enterprise hyperscale LLM serving |
| **Alternative Evaluated** | `Monolithic GPU Serving (Collocated)` | LEGACY: Single-node / small deployments |
| **Secondary Layer / Sandbox** | `Static Prompt Caching Alone` | INTERMEDIATE: Basic cost reduction |

*(For the complete comprehensive 6-way comparison table and decision matrix, see [`03-Deep-Explainability-Guide.md`](03-Deep-Explainability-Guide.md))*

---

## ⏱️ 3-Minute Pre-Interview Rapid-Fire Cheatsheet

1. **The Golden Formula**:
- **KV-Cache Memory Sizing for Llama-3-70B**:
  Hidden size $d_{\text{model}} = 8192$, Layers $n = 80$, KV Heads $n_{\text{kv}} = 8$, Head Dim $d_{\text{head}} = 128$, Precision = FP16 (2 bytes).
2. **Top Architectural Trap to Avoid**:
   Never choose a single centralized bottleneck for the hot request path. Always articulate a two-tier or decoupled architecture with local caching or asynchronous batching.
3. **The Staff-Level Distinction**:
   Junior candidates jump straight to third-party tools (e.g. "I'll use LangChain/Pinecone"). Staff candidates specify **data layouts, kernel mechanics (`gVisor`/`RDMA`), memory allocation binning, and explicit failure runbooks**.
