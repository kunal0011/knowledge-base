# System Design, Low-Level Design (LLD) & Coding Patterns Master Knowledge Base

[![HLD Systems](https://img.shields.io/badge/System%20Design%20(HLD)-45%20Systems-blue.svg)](#pillar-i-system-design-interview-alex-xu-curriculum-45-chapters)
[![LLD Questions](https://img.shields.io/badge/Low--Level%20Design%20(LLD)-50%20Systems-success.svg)](#pillar-ii-low-level-design-lld--design-patterns-64-topics)
[![LeetCode Problems](https://img.shields.io/badge/Coding%20Patterns-357%20Solutions-brightgreen.svg)](#pillar-iii-coding--leetcode-masterclass-357-problems)
[![Design Patterns](https://img.shields.io/badge/Design%20Patterns-14%20Covered-orange.svg)](#part-1-head-first-design-patterns-14-patterns)
[![Runnable Labs](https://img.shields.io/badge/Simulation%20Labs-45%20Python%20Engines-brightgreen.svg)](#the-4-pillar-chapter-architecture)
[![Target Level](https://img.shields.io/badge/Target%20Level-Senior%20%7C%20Staff%20%7C%20Principal%20(L5--L7)-purple.svg)](#curated-interview-learning-tracks)

> A production-grade, end-to-end engineering knowledge base and interview preparation curriculum. Covers the complete **Alex Xu System Design Series (Volumes 1, 2, and 3 — 45 Production Architectures)**, the **Low-Level Design (LLD) Masterclass (14 Design Patterns + Top 50 Interview Questions)**, and the **Coding Masterclass (357 LeetCode Solutions across 14 Algorithmic Paradigms)** with architectural blueprints, 45-minute verbatim interview sparring transcripts, deep explainability guides, runnable simulation benchmark engines, multi-language implementations (**Python 3, C++, Java**), visual execution walkthroughs, and typed solution guides.

---

## 🏛️ Knowledge Base Architecture

```
knowledge-base/
├── Projects/
│   ├── Coding/                               # Algorithms, Data Structures & LeetCode Patterns
│   │   ├── README.md                         # Coding & Patterns Masterclass Index
│   │   ├── Coding - LeetCode Masterclass Index.md # Master Index (357 Problems)
│   │   ├── 01. Arrays & Hashing/             # 16 Solved & Explained Systems
│   │   ├── 02. Two Pointers/                 # 15 Solved & Explained Systems
│   │   ├── 03. Sliding Window/               # 37 Solved & Explained Systems
│   │   ├── 04. Stack/                        # 21 Solved & Explained Systems
│   │   ├── 05. Queue/                        # 12 Solved & Explained Systems
│   │   ├── 06. Binary Search/                # 14 Solved & Explained Systems
│   │   ├── 07. Linked List/                  # 12 Solved & Explained Systems
│   │   ├── 08. Trees/                        # 22 Solved & Explained Systems
│   │   ├── 09. Heap & Priority Queue/        # 20 Solved & Explained Systems
│   │   ├── 10. Backtracking/                 # 35 Solved & Explained Systems
│   │   ├── 11. Graphs/                       # 22 Solved & Explained Systems
│   │   ├── 12. Dynamic Programming/          # 71 Solved & Explained Systems
│   │   ├── 13. Greedy/                       # 46 Solved & Explained Systems
│   │   └── 14. Bit Manipulation/             # 14 Solved & Explained Systems
│   │
│   ├── System-Design-Alex-Xu/                # High-Level Distributed System Design (HLD)
│   │   ├── System Design Interview - Alex Xu Index.md  # Master HLD Index & Curriculum
│   │   ├── Volume-1/                          # Core Distributed Foundations (12 Chapters)
│   │   ├── Volume-2/                          # Advanced Large-Scale Systems (15 Chapters)
│   │   └── Volume-3/                          # Frontier AI & Agent Platforms (18 Chapters)
│   │
│   ├── LLD/                                  # Low-Level Design & Object-Oriented Architecture
│   │   ├── README.md                          # LLD Masterclass Homepage & Guide
│   │   ├── LLD - Design Patterns & Interview Questions Index.md
│   │   ├── 01-Design-Patterns/                # 14 Head First & GoF Patterns (Python & Java)
│   │   └── 02-LLD-Interview-Questions/        # Top 50 FAANG/MAANG LLD Systems
│   │
│   ├── Data-and-AI-Frameworks/               # High-Performance Data, Distributed & AI Frameworks
│   │   ├── README.md                          # Framework Matrix & Master Curriculum
│   │   ├── NumPy/                             # Vectorized N-D Computing & Buffer Protocols
│   │   ├── Pandas/                            # Tabular Wrangling & Apache Arrow Backends
│   │   ├── PySpark/                           # Distributed Big Data, Catalyst & Tungsten
│   │   └── PyTorch/                           # Deep Learning, Dynamic Autograd & Compilation
│   │
│   └── ai_math/                              # Mathematical Foundations of AI & Deep Learning
│       └── README.md                          # 13 Modules: Linear Algebra to Test-Time Compute
│
├── Templates/                                # System Design & LLD Architectural Templates
└── README.md                                 # Master Repository Portal
```

---

## 🚀 Pillar I: System Design Interview (Alex Xu Curriculum — 45 Chapters)

Every single chapter in the High-Level System Design curriculum is organized into a **self-contained 5-file module**:
1. `README.md` — 30-second executive pitch, quantitative SLAs, and 3-minute pre-interview rapid cheatsheet.
2. `01-Architectural-Blueprint.md` — Complete production RFC specification, data models, and ASCII sequence diagrams.
3. `02-Interactive-Interview-Playbook.md` — 45-minute verbatim candidate-interviewer sparring transcript with pushback defenses and 5 lethal interviewer trap cards.
4. `03-Deep-Explainability-Guide.md` — First-principles physical analogies, 8-dimension technology showdowns ("Why X and NOT Y?"), step-by-step mathematical proofs with numerical worked examples, low-level OS/hardware mechanics, and 03:00 AM production outage triage runbooks.
5. `*_lab.py` — Standalone runnable Python simulation engine with zero external dependencies, automated unit tests, and high-concurrency benchmark suites.

### [Volume 1: Core Distributed Systems Foundations (12 Chapters)](Projects/System-Design-Alex-Xu/Volume-1/README.md)
| Chapter | Topic | Core Breakthroughs & Technologies | Interactive Playbook & Lab |
|:---:|:---|:---|:---|
| **01** | [Distributed Rate Limiter](Projects/System-Design-Alex-Xu/Volume-1/01-Rate-Limiter/01-Architectural-Blueprint.md) | GCRA Theoretical Arrival Time ($TAT$), Two-Tier Local Token Leasing | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/01-Rate-Limiter/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/01-Rate-Limiter/03-Deep-Explainability-Guide.md) |
| **02** | [Consistent Hashing](Projects/System-Design-Alex-Xu/Volume-1/02-Consistent-Hashing/01-Architectural-Blueprint.md) | Google Bounded Loads, Maglev $O(1)$ lookup, Eytzinger branchless search | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/02-Consistent-Hashing/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/02-Consistent-Hashing/03-Deep-Explainability-Guide.md) |
| **03** | [Unique ID Generator](Projects/System-Design-Alex-Xu/Volume-1/03-Unique-ID-Generator/01-Architectural-Blueprint.md) | Snowflake 64-bit bits allocation, UUIDv7 standard, NTP clock-drift bounds | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/03-Unique-ID-Generator/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/03-Unique-ID-Generator/03-Deep-Explainability-Guide.md) |
| **04** | [URL Shortener](Projects/System-Design-Alex-Xu/Volume-1/04-URL-Shortener/01-Architectural-Blueprint.md) | Bijective Base62 encoding, 307 vs 302 redirect telemetry, Bloom filters | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/04-URL-Shortener/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/04-URL-Shortener/03-Deep-Explainability-Guide.md) |
| **05** | [Web Crawler](Projects/System-Design-Alex-Xu/Volume-1/05-Web-Crawler/01-Architectural-Blueprint.md) | Mercator Dual-Queue Frontier, SimHash 4-table near-dedup, non-blocking DNS | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/05-Web-Crawler/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/05-Web-Crawler/03-Deep-Explainability-Guide.md) |
| **06** | [Distributed Key-Value Store](Projects/System-Design-Alex-Xu/Volume-1/06-Key-Value-Store/01-Architectural-Blueprint.md) | Quorum Linearizability ($W+R>N$), LSM Leveled Compaction, RUM conjecture | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/06-Key-Value-Store/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/06-Key-Value-Store/03-Deep-Explainability-Guide.md) |
| **07** | [Notification System](Projects/System-Design-Alex-Xu/Volume-1/07-Notification-System/01-Architectural-Blueprint.md) | 3-tier priority queues, HTTP/2 APNs connection multiplexing, Roaring Bitmaps | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/07-Notification-System/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/07-Notification-System/03-Deep-Explainability-Guide.md) |
| **08** | [News Feed System](Projects/System-Design-Alex-Xu/Volume-1/08-News-Feed/01-Architectural-Blueprint.md) | Push/Pull Fanout hybrid, Read-Your-Writes injection, 3-stage ranking funnel | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/08-News-Feed/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/08-News-Feed/03-Deep-Explainability-Guide.md) |
| **09** | [Chat System](Projects/System-Design-Alex-Xu/Volume-1/09-Chat-System/01-Architectural-Blueprint.md) | 50M C10M WebSockets, Monotonic Sequence IDs, Signal Protocol E2EE | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/09-Chat-System/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/09-Chat-System/03-Deep-Explainability-Guide.md) |
| **10** | [Search Autocomplete](Projects/System-Design-Alex-Xu/Volume-1/10-Search-Autocomplete/01-Architectural-Blueprint.md) | Double-Array Trie (DAT/FST), RCU pointer swap, Spark decay + Flink Heavy Hitters | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/10-Search-Autocomplete/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/10-Search-Autocomplete/03-Deep-Explainability-Guide.md) |
| **11** | [YouTube / Video Streaming](Projects/System-Design-Alex-Xu/Volume-1/11-YouTube/01-Architectural-Blueprint.md) | Resumable chunked Tus uploads, GOP split-and-stitch transcoding grid, CMAF | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/11-YouTube/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/11-YouTube/03-Deep-Explainability-Guide.md) |
| **12** | [Google Drive / Cloud Storage](Projects/System-Design-Alex-Xu/Volume-1/12-Google-Drive/01-Architectural-Blueprint.md) | Content-Defined Chunking (FastCDC), Merkle tree delta sync, 2-phase GC | [Playbook](Projects/System-Design-Alex-Xu/Volume-1/12-Google-Drive/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-1/12-Google-Drive/03-Deep-Explainability-Guide.md) |

---

### [Volume 2: Advanced Hyperscale Production Systems (15 Chapters)](Projects/System-Design-Alex-Xu/Volume-2/README.md)
| Chapter | Topic | Core Breakthroughs & Technologies | Interactive Playbook & Lab |
|:---:|:---|:---|:---|
| **01** | [Proximity Service](Projects/System-Design-Alex-Xu/Volume-2/01-Proximity-Service/01-Architectural-Blueprint.md) | Google S2 64-bit Hilbert integers, B-Tree integer scans, density k-NN | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/01-Proximity-Service/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/01-Proximity-Service/03-Deep-Explainability-Guide.md) |
| **02** | [Nearby Friends](Projects/System-Design-Alex-Xu/Volume-2/02-Nearby-Friends/01-Architectural-Blueprint.md) | Redis 7 Sharded Pub/Sub (`SPUBLISH`), Dead Reckoning, C10M WebSockets | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/02-Nearby-Friends/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/02-Nearby-Friends/03-Deep-Explainability-Guide.md) |
| **03** | [Google Maps](Projects/System-Design-Alex-Xu/Volume-2/03-Google-Maps/01-Architectural-Blueprint.md) | Customizable Contraction Hierarchies (CCH in $<5\text{ms}$), MVT vector tiles | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/03-Google-Maps/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/03-Google-Maps/03-Deep-Explainability-Guide.md) |
| **04** | [Distributed Message Queue](Projects/System-Design-Alex-Xu/Volume-2/04-Distributed-Message-Queue/01-Architectural-Blueprint.md) | Zero-copy DMA `sendfile()`, Linux page cache bypass, KRaft consensus | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/04-Distributed-Message-Queue/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/04-Distributed-Message-Queue/03-Deep-Explainability-Guide.md) |
| **05** | [Metrics Monitoring & TSDB](Projects/System-Design-Alex-Xu/Volume-2/05-Metrics-Monitoring/01-Architectural-Blueprint.md) | Gorilla TSZ delta-of-delta compression, Roaring Bitmaps, PromQL engine | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/05-Metrics-Monitoring/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/05-Metrics-Monitoring/03-Deep-Explainability-Guide.md) |
| **06** | [Ad Click Event Aggregation](Projects/System-Design-Alex-Xu/Volume-2/06-Ad-Click-Event-Aggregation/01-Architectural-Blueprint.md) | Salted Scatter-Gather Pre-Aggregation, Flink Chandy-Lamport 2PC Sink | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/06-Ad-Click-Event-Aggregation/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/06-Ad-Click-Event-Aggregation/03-Deep-Explainability-Guide.md) |
| **07** | [Hotel Reservation System](Projects/System-Design-Alex-Xu/Volume-2/07-Hotel-Reservation/01-Architectural-Blueprint.md) | Two-Tier Concurrency (Redis Hold + PostgreSQL Multi-Date OCC), Overbooking | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/07-Hotel-Reservation/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/07-Hotel-Reservation/03-Deep-Explainability-Guide.md) |
| **08** | [Distributed Email Service](Projects/System-Design-Alex-Xu/Volume-2/08-Distributed-Email-Service/01-Architectural-Blueprint.md) | JMAP over HTTP/3, ScyllaDB TWCS SSTables, S3 SHA-256 Attachment Dedup | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/08-Distributed-Email-Service/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/08-Distributed-Email-Service/03-Deep-Explainability-Guide.md) |
| **09** | [S3-like Object Storage](Projects/System-Design-Alex-Xu/Volume-2/09-S3-Object-Storage/01-Architectural-Blueprint.md) | Reed-Solomon $RS(8,4)$ Galois Field striping, Bitcask append-only chunks | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/09-S3-Object-Storage/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/09-S3-Object-Storage/03-Deep-Explainability-Guide.md) |
| **10** | [Real-Time Gaming Leaderboard](Projects/System-Design-Alex-Xu/Volume-2/10-Gaming-Leaderboard/01-Architectural-Blueprint.md) | In-Memory Fenwick Tree $O(\log B)$ prefix rank, Score-Range Sharding | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/10-Gaming-Leaderboard/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/10-Gaming-Leaderboard/03-Deep-Explainability-Guide.md) |
| **11** | [Payment System](Projects/System-Design-Alex-Xu/Volume-2/11-Payment-System/01-Architectural-Blueprint.md) | Double-Entry Bookkeeping ($\sum \text{Debits} == \sum \text{Credits}$), Two-Tier Idempotency | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/11-Payment-System/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/11-Payment-System/03-Deep-Explainability-Guide.md) |
| **12** | [Digital Wallet](Projects/System-Design-Alex-Xu/Volume-2/12-Digital-Wallet/01-Architectural-Blueprint.md) | LMAX Disruptor lock-free RingBuffer, 64-byte mechanical sympathy padding | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/12-Digital-Wallet/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/12-Digital-Wallet/03-Deep-Explainability-Guide.md) |
| **13** | [Stock Exchange & Matching](Projects/System-Design-Alex-Xu/Volume-2/13-Stock-Exchange/01-Architectural-Blueprint.md) | Single-Threaded Pinned NUMA Matching Core, Solarflare EF_VI Kernel Bypass | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/13-Stock-Exchange/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/13-Stock-Exchange/03-Deep-Explainability-Guide.md) |
| **14** | [Scalable Web Server & Epoll](Projects/System-Design-Alex-Xu/Volume-2/14-Web-Server-FastAPI/01-Architectural-Blueprint.md) | Event-driven non-blocking epoll/kqueue, coroutine task schedulers, HTTP/2/3 | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/14-Web-Server-FastAPI/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/14-Web-Server-FastAPI/03-Deep-Explainability-Guide.md) |
| **15** | [Distributed Task Queue](Projects/System-Design-Alex-Xu/Volume-2/15-Distributed-Task-Queue/01-Architectural-Blueprint.md) | Celery/Temporal architecture, Min-Heap timer wheel, Heartbeat leases | [Playbook](Projects/System-Design-Alex-Xu/Volume-2/15-Distributed-Task-Queue/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-2/15-Distributed-Task-Queue/03-Deep-Explainability-Guide.md) |

---

### [Volume 3: Frontier AI Platforms & Autonomous Agent Architectures (18 Chapters)](Projects/System-Design-Alex-Xu/Volume-3/README.md)
| Chapter | Topic | Core Breakthroughs & Technologies | Interactive Playbook & Lab |
|:---:|:---|:---|:---|
| **01** | [Global Code Search & RAG](Projects/System-Design-Alex-Xu/Volume-3/01-Global-Code-Search-RAG/01-Architectural-Blueprint.md) | Tri-Modal Hybrid Search (Dense, Trigram, SCIP AST), libgit2 tree diff | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/01-Global-Code-Search-RAG/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/01-Global-Code-Search-RAG/03-Deep-Explainability-Guide.md) |
| **02** | [Agentic Trace Loop Learning](Projects/System-Design-Alex-Xu/Volume-3/02-Agentic-Trace-Learning/01-Architectural-Blueprint.md) | OpenTelemetry Ingestion, Tri-Hybrid Trajectory Evaluators, DSPy MIPROv2 | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/02-Agentic-Trace-Learning/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/02-Agentic-Trace-Learning/03-Deep-Explainability-Guide.md) |
| **03** | [Multi-Agent Orchestration](Projects/System-Design-Alex-Xu/Volume-3/03-Multi-Agent-Orchestration/01-Architectural-Blueprint.md) | Event-Sourced Durable State Machines (Temporal), Supervisor-Worker trees | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/03-Multi-Agent-Orchestration/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/03-Multi-Agent-Orchestration/03-Deep-Explainability-Guide.md) |
| **04** | [Agentic Workflow Runner](Projects/System-Design-Alex-Xu/Volume-3/04-Agentic-Workflow-Runner/01-Architectural-Blueprint.md) | Visual DAG Compiler, Weighted Fair Queueing (WFQ), V8 Isolate Sandboxing | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/04-Agentic-Workflow-Runner/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/04-Agentic-Workflow-Runner/03-Deep-Explainability-Guide.md) |
| **05** | [OpenClaw Autonomous Agent](Projects/System-Design-Alex-Xu/Volume-3/05-OpenClaw-Autonomous-Agent/01-Architectural-Blueprint.md) | Omnichannel Gateway, Device Node Fabric, In-Flight Turn Steering, Dreaming | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/05-OpenClaw-Autonomous-Agent/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/05-OpenClaw-Autonomous-Agent/03-Deep-Explainability-Guide.md) |
| **06** | [Coding Agent Harness](Projects/System-Design-Alex-Xu/Volume-3/06-Coding-Agent-Harness/01-Architectural-Blueprint.md) | Context Compaction, Worktree Isolation, Multi-file atomic rollbacks | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/06-Coding-Agent-Harness/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/06-Coding-Agent-Harness/03-Deep-Explainability-Guide.md) |
| **07** | [Cloud Remote Execution](Projects/System-Design-Alex-Xu/Volume-3/07-Cloud-Remote-Execution/01-Architectural-Blueprint.md) | Laptop-Lid Problem, Git Seed Bundles, Pre-Warmed MicroVMs (Firecracker) | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/07-Cloud-Remote-Execution/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/07-Cloud-Remote-Execution/03-Deep-Explainability-Guide.md) |
| **08** | [Enterprise AI Coworker](Projects/System-Design-Alex-Xu/Volume-3/08-Enterprise-AI-Coworker/01-Architectural-Blueprint.md) | Finished Deliverables Engine, Three-Tier Governance, Earned Autonomy | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/08-Enterprise-AI-Coworker/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/08-Enterprise-AI-Coworker/03-Deep-Explainability-Guide.md) |
| **09** | [Multiplayer AI Teammate](Projects/System-Design-Alex-Xu/Volume-3/09-Multiplayer-AI-Teammate/01-Architectural-Blueprint.md) | Thread Multiplayer Unit, Dedicated Identity, Shared Workspace Conflict Lock | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/09-Multiplayer-AI-Teammate/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/09-Multiplayer-AI-Teammate/03-Deep-Explainability-Guide.md) |
| **10** | [Context Engineering Platform](Projects/System-Design-Alex-Xu/Volume-3/10-Context-Engineering/01-Architectural-Blueprint.md) | Anthropic 80% Unhobbling Architecture, Dynamic Progressive Disclosure | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/10-Context-Engineering/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/10-Context-Engineering/03-Deep-Explainability-Guide.md) |
| **11** | [MCP Gateway & Federation](Projects/System-Design-Alex-Xu/Volume-3/11-MCP-Gateway-Platform/01-Architectural-Blueprint.md) | Model Context Protocol Proxy, SSE/JSON-RPC 2.0, OAuth2 Passthrough | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/11-MCP-Gateway-Platform/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/11-MCP-Gateway-Platform/03-Deep-Explainability-Guide.md) |
| **12** | [Deep Research Agent](Projects/System-Design-Alex-Xu/Volume-3/12-Deep-Research-Agent/01-Architectural-Blueprint.md) | Recursive Search DAGs, Dense Reranking, Epistemic Gap Analysis | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/12-Deep-Research-Agent/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/12-Deep-Research-Agent/03-Deep-Explainability-Guide.md) |
| **13** | [Computer-Use Platform](Projects/System-Design-Alex-Xu/Volume-3/13-Computer-Use-Platform/01-Architectural-Blueprint.md) | Multi-Modal Vision Grounding, Coordinate Normalization, Sub-Second VNC | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/13-Computer-Use-Platform/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/13-Computer-Use-Platform/03-Deep-Explainability-Guide.md) |
| **14** | [Test-Time Compute Engine](Projects/System-Design-Alex-Xu/Volume-3/14-Test-Time-Compute/01-Architectural-Blueprint.md) | Monte Carlo Tree Search (MCTS), Process Reward Models (PRM) Value Decoding | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/14-Test-Time-Compute/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/14-Test-Time-Compute/03-Deep-Explainability-Guide.md) |
| **15** | [GraphRAG Agent Memory](Projects/System-Design-Alex-Xu/Volume-3/15-GraphRAG-Agent-Memory/01-Architectural-Blueprint.md) | Dual-Layer Memory (Episodic + Semantic KG), Leiden Community Clustering | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/15-GraphRAG-Agent-Memory/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/15-GraphRAG-Agent-Memory/03-Deep-Explainability-Guide.md) |
| **16** | [Full-Duplex Voice Agent](Projects/System-Design-Alex-Xu/Volume-3/16-Full-Duplex-Voice/01-Architectural-Blueprint.md) | WebRTC Audio Transport, Sub-300ms Turnaround, Streaming VAD & Barge-In | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/16-Full-Duplex-Voice/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/16-Full-Duplex-Voice/03-Deep-Explainability-Guide.md) |
| **17** | [Distributed KV-Cache Serving](Projects/System-Design-Alex-Xu/Volume-3/17-Distributed-KV-Cache/01-Architectural-Blueprint.md) | Prefill-Decode Disaggregation, RadixAttention Routing, 400G RoCEv2 RDMA | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/17-Distributed-KV-Cache/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/17-Distributed-KV-Cache/03-Deep-Explainability-Guide.md) |
| **18** | [AI Agent Defense Mesh](Projects/System-Design-Alex-Xu/Volume-3/18-Agent-Defense-Mesh/01-Architectural-Blueprint.md) | Dual-LLM Architecture (Executive vs Untrusted), In-Flight Canaries, DLP | [Playbook](Projects/System-Design-Alex-Xu/Volume-3/18-Agent-Defense-Mesh/02-Interactive-Interview-Playbook.md) • [Guide](Projects/System-Design-Alex-Xu/Volume-3/18-Agent-Defense-Mesh/03-Deep-Explainability-Guide.md) |

---

## 🎯 Pillar II: Low-Level Design (LLD) & Design Patterns (64 Topics)

The Low-Level Design curriculum covers the theoretical foundations of object-oriented design and clean code architectures:

### [Part 1: Head First Design Patterns (14 Patterns)](Projects/LLD/01-Design-Patterns/Design%20Patterns%20-%20Index.md)
Features formal problem definitions, class diagrams (Mermaid), SOLID evaluations, and full implementations in both **Python** and **Java**:
- [01 - Strategy Pattern](Projects/LLD/01-Design-Patterns/01-Strategy-Pattern/01%20-%20Strategy%20Pattern.md) • [02 - Observer Pattern](Projects/LLD/01-Design-Patterns/02-Observer-Pattern/02%20-%20Observer%20Pattern.md) • [03 - Decorator Pattern](Projects/LLD/01-Design-Patterns/03-Decorator-Pattern/03%20-%20Decorator%20Pattern.md) • [04 - Factory Method Pattern](Projects/LLD/01-Design-Patterns/04-Factory-Method-Pattern/04%20-%20Factory%20Method%20Pattern.md)
- [05 - Abstract Factory Pattern](Projects/LLD/01-Design-Patterns/05-Abstract-Factory-Pattern/05%20-%20Abstract%20Factory%20Pattern.md) • [06 - Singleton Pattern](Projects/LLD/01-Design-Patterns/06-Singleton-Pattern/06%20-%20Singleton%20Pattern.md) • [07 - Command Pattern](Projects/LLD/01-Design-Patterns/07-Command-Pattern/07%20-%20Command%20Pattern.md) • [08 - Adapter Pattern](Projects/LLD/01-Design-Patterns/08-Adapter-Pattern/08%20-%20Adapter%20Pattern.md)
- [09 - Facade Pattern](Projects/LLD/01-Design-Patterns/09-Facade-Pattern/09%20-%20Facade%20Pattern.md) • [10 - Template Method Pattern](Projects/LLD/01-Design-Patterns/10-Template-Method-Pattern/10%20-%20Template%20Method%20Pattern.md) • [11 - Iterator Pattern](Projects/LLD/01-Design-Patterns/11-Iterator-Pattern/11%20-%20Iterator%20Pattern.md) • [12 - Composite Pattern](Projects/LLD/01-Design-Patterns/12-Composite-Pattern/12%20-%20Composite%20Pattern.md)
- [13 - State Pattern](Projects/LLD/01-Design-Patterns/13-State-Pattern/13%20-%20State%20Pattern.md) • [14 - Proxy Pattern](Projects/LLD/01-Design-Patterns/14-Proxy-Pattern/14%20-%20Proxy%20Pattern.md)

### [Part 2: Top 50 LLD Interview Questions](Projects/LLD/02-LLD-Interview-Questions/LLD%20Interview%20Questions%20-%20Index.md)
Complete object models, thread safety, and clean code for top interview systems:
- **Games**: [Parking Lot](Projects/LLD/02-LLD-Interview-Questions/01-Parking-Lot/Design%20a%20Parking%20Lot.md), [Tic-Tac-Toe](Projects/LLD/02-LLD-Interview-Questions/05-Tic-Tac-Toe/Design%20a%20Tic-Tac-Toe%20Game.md), [Chess Game](Projects/LLD/02-LLD-Interview-Questions/06-Chess-Game/Design%20a%20Chess%20Game.md), [Snake and Ladder](Projects/LLD/02-LLD-Interview-Questions/19-Snake-Ladder/Design%20a%20Snake%20and%20Ladder%20Game.md), [Bowling Alley](Projects/LLD/02-LLD-Interview-Questions/42-Bowling-Alley/Design%20a%20Bowling%20Alley.md), [Deck of Cards](Projects/LLD/02-LLD-Interview-Questions/40-Deck-of-Cards/Design%20a%20Deck%20of%20Cards.md).
- **Core Infrastructure**: [Elevator System](Projects/LLD/02-LLD-Interview-Questions/02-Elevator-System/Design%20an%20Elevator%20System.md), [LRU/LFU Cache](Projects/LLD/02-LLD-Interview-Questions/16-Cache/Design%20a%20Cache.md), [Pub-Sub Broker](Projects/LLD/02-LLD-Interview-Questions/17-Pub-Sub/Design%20a%20Pub-Sub%20Messaging%20System.md), [Task Scheduler](Projects/LLD/02-LLD-Interview-Questions/18-Task-Scheduler/Design%20a%20Task%20Scheduler.md), [Rate Limiter](Projects/LLD/02-LLD-Interview-Questions/22-Rate-Limiter/Design%20a%20Rate%20Limiter.md), [Connection Pool](Projects/LLD/02-LLD-Interview-Questions/35-Connection-Pool/Design%20a%20Database%20Connection%20Pool.md), [Thread Pool](Projects/LLD/02-LLD-Interview-Questions/39-Thread-Pool/Design%20a%20Thread%20Pool.md), [Circuit Breaker](Projects/LLD/02-LLD-Interview-Questions/50-Circuit-Breaker/Design%20a%20Circuit%20Breaker.md).
- **E-Commerce & FinTech**: [Shopping Cart](Projects/LLD/02-LLD-Interview-Questions/13-Shopping-Cart/Design%20an%20Online%20Shopping%20Cart.md), [Splitwise](Projects/LLD/02-LLD-Interview-Questions/20-Splitwise/Design%20Splitwise.md), [Payment Processor](Projects/LLD/02-LLD-Interview-Questions/24-Payment-System/Design%20a%20Payment%20Processing%20System.md), [Stock Exchange & Matching](Projects/LLD/02-LLD-Interview-Questions/46-Stock-Exchange/Design%20a%20Stock%20Exchange%20System.md).
- **Consumer Apps**: [Hotel Booking](Projects/LLD/02-LLD-Interview-Questions/07-Hotel-Booking/Design%20a%20Hotel%20Booking%20System.md), [Movie Tickets](Projects/LLD/02-LLD-Interview-Questions/08-Movie-Ticket-Booking/Design%20a%20Movie%20Ticket%20Booking%20System.md), [Food Delivery](Projects/LLD/02-LLD-Interview-Questions/25-Food-Delivery/Design%20a%20Food%20Delivery%20System.md), [Ride Sharing](Projects/LLD/02-LLD-Interview-Questions/26-Ride-Sharing/Design%20a%20Ride-Sharing%20System.md), [Chat App](Projects/LLD/02-LLD-Interview-Questions/27-Chat-Application/Design%20a%20Chat%20Application.md), [Spreadsheet Engine](Projects/LLD/02-LLD-Interview-Questions/31-Spreadsheet/Design%20a%20Spreadsheet.md).

---

## 🧩 Pillar III: Coding & LeetCode Masterclass (357 Problems)

[![LeetCode Masterclass](https://img.shields.io/badge/Curriculum-14%20Algorithmic%20Patterns-blue.svg)](Projects/Coding/README.md)
[![Total Problems](https://img.shields.io/badge/Total%20Problems-357%20Solutions-success.svg)](Projects/Coding/Coding%20-%20LeetCode%20Masterclass%20Index.md)

A production-grade, interview-tested curriculum covering all **14 foundational algorithmic patterns** and **357 LeetCode problems** with multi-language implementations in **Python 3, C++, and Java**. Each solution note includes the core problem formulation, target company tags (Amazon, Google, Meta, Microsoft, Apple), input/output specifications, constraints, first-principles algorithmic intuition, visual algorithm walkthroughs, step-by-step execution traces with multiple inputs, complexity bounds ($O(N)$ time & space), and transferrable pattern takeaways:

| # | Algorithmic Pattern | Count | Core Techniques & Focus | Catalog |
|:---:|:---|:---:|:---|:---:|
| **01** | [Arrays & Hashing](Projects/Coding/01.%20Arrays%20%26%20Hashing/) | `16` | Hash maps, frequency counters, prefix sums, duplicate detection, and sorting tricks | [Browse](Projects/Coding/01.%20Arrays%20%26%20Hashing/) |
| **02** | [Two Pointers](Projects/Coding/02.%20Two%20Pointers/) | `15` | Inward-converging pointers, fast/slow runners, sorted array transformations | [Browse](Projects/Coding/02.%20Two%20Pointers/) |
| **03** | [Sliding Window](Projects/Coding/03.%20Sliding%20Window/) | `37` | Fixed/variable windows, monotonic deques, substring frequencies, optimization | [Browse](Projects/Coding/03.%20Sliding%20Window/) |
| **04** | [Stack](Projects/Coding/04.%20Stack/) | `21` | Monotonic stacks, expression evaluators, parenthesis matching, next greater element | [Browse](Projects/Coding/04.%20Stack/) |
| **05** | [Queue](Projects/Coding/05.%20Queue/) | `12` | FIFO scheduling, circular queues, rate limiting hit counters, monotonic queues | [Browse](Projects/Coding/05.%20Queue/) |
| **06** | [Binary Search](Projects/Coding/06.%20Binary%20Search/) | `14` | Search space reduction, monotonic predicate functions, rotated arrays, boundaries | [Browse](Projects/Coding/06.%20Binary%20Search/) |
| **07** | [Linked List](Projects/Coding/07.%20Linked%20List/) | `12` | In-place reversal, Floyd cycle detection, fast & slow pointers, dummy heads | [Browse](Projects/Coding/07.%20Linked%20List/) |
| **08** | [Trees](Projects/Coding/08.%20Trees/) | `22` | DFS/BFS traversals, LCA, path sum validation, tree transformations, BST properties | [Browse](Projects/Coding/08.%20Trees/) |
| **09** | [Heap & Priority Queue](Projects/Coding/09.%20Heap%20%26%20Priority%20Queue/) | `20` | Top-K elements, streaming medians, interval scheduling, greedy priority queues | [Browse](Projects/Coding/09.%20Heap%20%26%20Priority%20Queue/) |
| **10** | [Backtracking](Projects/Coding/10.%20Backtracking/) | `35` | State space tree search, permutations, combinations, subset pruning | [Browse](Projects/Coding/10.%20Backtracking/) |
| **11** | [Graphs](Projects/Coding/11.%20Graphs/) | `22` | BFS shortest paths, DFS connectivity, topological sort, Dijkstra, union-find | [Browse](Projects/Coding/11.%20Graphs/) |
| **12** | [Dynamic Programming](Projects/Coding/12.%20Dynamic%20Programming/) | `71` | 1D/2D memoization, knapsack, longest common subsequences, interval DP | [Browse](Projects/Coding/12.%20Dynamic%20Programming/) |
| **13** | [Greedy](Projects/Coding/13.%20Greedy/) | `46` | Locally optimal choice paradigms, interval scheduling, jump games | [Browse](Projects/Coding/13.%20Greedy/) |
| **14** | [Bit Manipulation](Projects/Coding/14.%20Bit%20Manipulation/) | `14` | Bitwise XOR/AND/OR tricks, bitmasks, 2's complement properties | [Browse](Projects/Coding/14.%20Bit%20Manipulation/) |

> Complete catalog of all 357 problems with direct links and target companies: **[Coding - LeetCode Masterclass Index](Projects/Coding/Coding%20-%20LeetCode%20Masterclass%20Index.md)**

---

### 🟢 Track 1: The 14-Day Senior SDE (L5) Fast Track
The highest frequency distributed system design and LLD questions asked at Meta, Google, and Amazon:
1. [Distributed Rate Limiter (HLD)](Projects/System-Design-Alex-Xu/Volume-1/01-Rate-Limiter/01-Architectural-Blueprint.md) & [Rate Limiter (LLD)](Projects/LLD/02-LLD-Interview-Questions/22-Rate-Limiter/Design%20a%20Rate%20Limiter.md)
2. [Consistent Hashing](Projects/System-Design-Alex-Xu/Volume-1/02-Consistent-Hashing/01-Architectural-Blueprint.md)
3. [Unique ID Generator (Snowflake)](Projects/System-Design-Alex-Xu/Volume-1/03-Unique-ID-Generator/01-Architectural-Blueprint.md)
4. [Distributed Key-Value Store](Projects/System-Design-Alex-Xu/Volume-1/06-Key-Value-Store/01-Architectural-Blueprint.md) & [Cache LLD](Projects/LLD/02-LLD-Interview-Questions/16-Cache/Design%20a%20Cache.md)
5. [Distributed Message Queue](Projects/System-Design-Alex-Xu/Volume-2/04-Distributed-Message-Queue/01-Architectural-Blueprint.md) & [Pub-Sub LLD](Projects/LLD/02-LLD-Interview-Questions/17-Pub-Sub/Design%20a%20Pub-Sub%20Messaging%20System.md)
6. [Notification System](Projects/System-Design-Alex-Xu/Volume-1/07-Notification-System/01-Architectural-Blueprint.md)
7. [News Feed Architecture](Projects/System-Design-Alex-Xu/Volume-1/08-News-Feed/01-Architectural-Blueprint.md)
8. [Chat System](Projects/System-Design-Alex-Xu/Volume-1/09-Chat-System/01-Architectural-Blueprint.md) & [Chat App LLD](Projects/LLD/02-LLD-Interview-Questions/27-Chat-Application/Design%20a%20Chat%20Application.md)
9. [Distributed Task Queue](Projects/System-Design-Alex-Xu/Volume-2/15-Distributed-Task-Queue/01-Architectural-Blueprint.md) & [Task Scheduler LLD](Projects/LLD/02-LLD-Interview-Questions/18-Task-Scheduler/Design%20a%20Task%20Scheduler.md)
10. [Payment System](Projects/System-Design-Alex-Xu/Volume-2/11-Payment-System/01-Architectural-Blueprint.md) & [Payment Processor LLD](Projects/LLD/02-LLD-Interview-Questions/24-Payment-System/Design%20a%20Payment%20Processing%20System.md)

### 🔴 Track 2: The Staff / Principal (L6/L7) Systems Architect Track
Deep focus on kernel bypass, high-throughput financial matching, lock-free ringbuffers, and consensus:
1. [Stock Exchange & Order Matching](Projects/System-Design-Alex-Xu/Volume-2/13-Stock-Exchange/01-Architectural-Blueprint.md) & [Stock Exchange LLD](Projects/LLD/02-LLD-Interview-Questions/46-Stock-Exchange/Design%20a%20Stock%20Exchange%20System.md)
2. [Digital Wallet & LMAX Disruptor](Projects/System-Design-Alex-Xu/Volume-2/12-Digital-Wallet/01-Architectural-Blueprint.md)
3. [Hotel Reservation with Two-Tier Concurrency](Projects/System-Design-Alex-Xu/Volume-2/07-Hotel-Reservation/01-Architectural-Blueprint.md)
4. [S3 Object Storage & Reed-Solomon Erasure Coding](Projects/System-Design-Alex-Xu/Volume-2/09-S3-Object-Storage/01-Architectural-Blueprint.md)
5. [Customizable Contraction Hierarchies (Google Maps)](Projects/System-Design-Alex-Xu/Volume-2/03-Google-Maps/01-Architectural-Blueprint.md)
6. [Real-Time Gaming Leaderboard with Fenwick Trees](Projects/System-Design-Alex-Xu/Volume-2/10-Gaming-Leaderboard/01-Architectural-Blueprint.md)
7. [Circuit Breaker Resilience Engine](Projects/LLD/02-LLD-Interview-Questions/50-Circuit-Breaker/Design%20a%20Circuit%20Breaker.md)

### 🤖 Track 3: Frontier AI & Autonomous Agent Systems
The industry-first comprehensive curriculum on generative AI, autonomous agent harnesses, and inference infrastructure:
1. [Enterprise LLM Gateway & Distributed KV-Cache (PD Disaggregation)](Projects/System-Design-Alex-Xu/Volume-3/17-Distributed-KV-Cache/01-Architectural-Blueprint.md)
2. [Autonomous Coding Agent Harnesses](Projects/System-Design-Alex-Xu/Volume-3/06-Coding-Agent-Harness/01-Architectural-Blueprint.md)
3. [Production Context Engineering (Dynamic Tool Retrieval)](Projects/System-Design-Alex-Xu/Volume-3/10-Context-Engineering/01-Architectural-Blueprint.md)
4. [Test-Time Compute & MCTS Thought Search](Projects/System-Design-Alex-Xu/Volume-3/14-Test-Time-Compute/01-Architectural-Blueprint.md)
5. [Production Multi-Agent Platform (Temporal Orchestration)](Projects/System-Design-Alex-Xu/Volume-3/03-Multi-Agent-Orchestration/01-Architectural-Blueprint.md)
6. [AI Agent Defense Mesh & Sandboxing (Dual-LLM Guardrails)](Projects/System-Design-Alex-Xu/Volume-3/18-Agent-Defense-Mesh/01-Architectural-Blueprint.md)
7. [Low-Latency Full-Duplex WebRTC Voice Agent](Projects/System-Design-Alex-Xu/Volume-3/16-Full-Duplex-Voice/01-Architectural-Blueprint.md)

---

## 🛠️ Testing & Simulation Engines

All 45 system design chapters include runnable Python 3 simulation engines (zero dependencies):

```bash
# Run a specific chapter's benchmark lab (e.g., Chapter 1 Rate Limiter)
python3 Projects/System-Design-Alex-Xu/Volume-1/01-Rate-Limiter/rate_limiter_lab.py

# Run Distributed Task Queue verification tests
python3 Projects/System-Design-Alex-Xu/Volume-2/15-Distributed-Task-Queue/task_queue_engine.py --test

# Run Distributed KV-Cache gateway simulation
python3 Projects/System-Design-Alex-Xu/Volume-3/17-Distributed-KV-Cache/kv_cache_gateway_engine.py
```

---

## 📜 License & Citation

Curated and engineered by Kunal Kumar. Grounded in first-principles distributed systems engineering, production RFC specifications, and real-world hyperscale architectures.
