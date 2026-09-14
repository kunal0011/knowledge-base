# Chapter 12: Deep Research & Long-Horizon Web Reasoning Agent

## 1. Production Code Engine & Benchmark Lab

### Architecture Overview

```
                                  +-------------------------------------------------------------+
                                  |                 USER RESEARCH DIRECTIVE                     |
                                  +-------------------------------------------------------------+
                                                                 |
                                                                 v
                                         +-----------------------------------------------+
                                         |         PHASE 0: SCOPE CLARIFIER              |
                                         |  - Disambiguation Question Generator          |
                                         |  - Constraint & Depth Level Negotiator        |
                                         +-----------------------------------------------+
                                                                 |
                                                                 v
                                         +-----------------------------------------------+
                                         |       DYNAMIC RESEARCH DAG & PLANNER          |
                                         |  - Multi-Perspective Hypothesis Tree (STORM)  |
                                         |  - In-Flight Surprise-Driven Expansion        |
                                         +-----------------------------------------------+
                                          /                      |                      \
                                         /                       |                       \
                                        v                        v                        v
            +------------------------------+   +---------------------------+   +-----------------------------+
            |      SEARCH & SCRAPING       |   |    FACT TRIANGULATION     |   |      EXACT CHARACTER-       |
            | - Multi-Engine Query Wave    |   | - Domain Authority Weight |   |       OFFSET AUDITOR        |
            | - Semantic Distiller         |   | - Cross-Source Agreement  |   | - Mathematical Quote Anchor |
            | - Boilerplate Stripper       |   | - Contradiction Detector  |   | - Zero Hallucination Floor  |
            +------------------------------+   +---------------------------+   +-----------------------------+
                                        \                        |                        /
                                         \                       |                       /
                                          v                      v                      v
                                         +-----------------------------------------------+
                                         |        HIERARCHICAL REPORT SYNTHESIZER        |
                                         |  - Section-by-Section Contextual Drafting     |
                                         |  - Comprehensive 10k-Word Markdown Deliverable|
                                         +-----------------------------------------------+
```

The Deep Research Agent platform (`deep_research_engine.py`) provides an autonomous, multi-hour web reasoning and synthesis pipeline inspired by OpenAI Deep Research, Perplexity Pro, and Stanford STORM.

### Core Engine Components

1. **Phase 0 Interactive Scope Clarifier (`clarify_scope`)**:
   - Detects underspecified or ambiguous prompts upfront.
   - Generates targeted clarification questions (e.g. target timeline, geographic scope, depth tier) before burning compute and token budgets.
2. **Dynamic Hypothesis Research DAG (`ResearchDAGPlanner`)**:
   - Decomposes the research topic into multi-perspective branches (e.g., Chemistry, Manufacturing, Economics, IP).
   - Features **In-Flight Dynamic Expansion**: when unexpected or contradictory evidence is discovered in the field, the DAG automatically injects new targeted investigation nodes mid-run.
3. **Semantic Scraped Knowledge Base & Distiller (`ScrapedKnowledgeBase`)**:
   - Simulates high-speed headless browser rendering and semantic HTML cleaning (stripping ads, navbars, and cookie banners to reduce raw 2MB web pages to < 3KB clean markdown).
   - Retains exact document character offsets for every parsed token.
4. **Cross-Source Fact Triangulator & Authority Scorer (`FactTriangulator`)**:
   - Scores domain authority:
     $$\text{Authority} = 0.70 \times \text{Tier} + 0.20 \times \text{Recency} + 0.10 \times \text{CrossReferences}$$
     (Tier 1: Nature/SEC, Tier 2: Reuters/Bloomberg, Tier 3: Corporate PR).
   - Detects direct quantitative contradictions between sources (e.g. 500 Wh/kg PR claim vs. 385 Wh/kg peer-reviewed lab test) and generates reconciled consensus verdicts.
5. **Exact Character-Offset Citation Auditor (`CitationAuditor`)**:
   - Eliminates hallucinated references. Every citation specifies `[url, title, exact_quote, start_offset, end_offset]`.
   - Validates that `doc.raw_text[start:end] == exact_quote`. Any mismatch or ungrounded claim is immediately vetoed.
6. **Hierarchical Report Synthesizer (`execute_deep_research`)**:
   - Synthesizes publication-grade reports with Executive Summaries, Key Triangulated Assertions, Contradiction Analyses, and Verified Citation Appendices.

### Benchmark Lab Verification

```
================================================================================
STARTING DEEP RESEARCH HIGH-THROUGHPUT BENCHMARK
Target: 50,000 Mathematical Citation Verification & Fact Triangulations
================================================================================

--- BENCHMARK RESULTS ---
Total Operations Processed:  50,000
Total Elapsed Time:          0.008 seconds
Citation Verify Throughput:  6,148,643.5 Ops/sec
Average Latency per Audit:   0.16 microseconds
================================================================================
```

### Production REST API & Prometheus Telemetry

The engine exposes production endpoints on port `8096`:
- `POST /v1/research/clarify`: Interactive scope clarification and question generation.
- `POST /v1/research/execute`: Starts asynchronous deep research task.
- `GET /v1/research/stream`: Real-time SSE progress stream for live DAG updates.
- `GET /healthz`: Health status, active tasks, and knowledge base document count.
- `GET /metrics`: Standard Prometheus metrics (`research_tasks_total`, `research_queries_total`, `research_claims_total`, `research_contradictions_total`, `research_citations_verified_total`).

---

## 2. 45-Minute Staff/Principal Interview Playbook

### Minute-by-Minute Pacing Guide

- **Minute 00-05: Problem Scoping & High-Level Invariants**
  - Clarify scale: 100,000 deep research dossiers/day, average 15-minute execution (range: 3 to 45 min), 75 searches, 120 documents scraped per task.
  - Frame the Staff distinction: "Shallow RAG asks a question, scrapes 5 links, and summarizes 500 words. Deep Research is an autonomous analyst: it builds a dynamic hypothesis DAG, navigates multi-hop web paths, cross-verifies contradictory claims across authority tiers, and synthesizes a 10,000-word report with mathematically verified character-offset citations."
  - Key SLOs: Citation precision $\ge 98\%$, average task cost $< \$1.80$, zero infinite scraping loops.

- **Minute 05-15: Architecture & The Long-Horizon State Machine**
  - Diagram the 5 core layers: Interaction (Scope Clarifier) -> Durable Orchestrator (Temporal DAG) -> Web Intelligence Mesh (Headless Browser Cluster) -> Working Dossier (PostgreSQL/pgvector) -> Hierarchical Drafter & Citation Auditor.
  - Explain why long-horizon agents must use **Event-Sourced Orchestration (Temporal.io)**: a 30-minute task running across 100 network calls must survive worker pod restarts without losing progress.

- **Minute 15-30: Deep Dive into Mechanics (DAG, Triangulation, Grounding)**
  - *Dynamic Hypothesis DAG*: Explain the Stanford STORM multi-perspective approach. Walk through **Surprise-Driven In-Flight Expansion**: how discovering a discrepancy automatically inserts new search branches.
  - *Fact Triangulation*: How to resolve corporate PR bias vs. independent academic testing. Detail the Domain Authority Scoring formula and the Contradiction Matrix.
  - *Mathematical Citation Grounding*: Why loose markdown links (`[Source](url)`) are dangerous. Detail the `[start_offset, end_offset]` verification algorithm that guarantees the exact quote exists in the ground-truth document.

- **Minute 30-40: Headless Browser Fleet & Storage Mechanics**
  - Calculate headless browser sizing: 3.6M JS-rendered pages/day at 2.5s render time $\implies 450$ concurrent Chromium instances.
  - Detail memory management: Chromium sandboxing, recycling browser tabs after 20 navigations to prevent V8 memory leaks.
  - Explain semantic HTML distillation (Trafilatura/Readability) stripping 95% of boilerplate tokens.

- **Minute 40-45: Operational Failure Modes & Staff Wrap-Up**
  - Walk through the 4 runbooks: Anti-bot residential proxy rotation, citation hallucination rejections, contradiction thrashing loops, and graceful budget cutoff.
  - Defend trade-offs: Multi-pass hierarchical drafting vs. single-shot generation.

---

### 5 Lethal Interview Trap Cards & Staff Counter-Maneuvers

#### Trap Card 1: The "Single-Pass Naive Web Search" Trap
- **Interviewer**: *"Can't we just pass the user's prompt to Google Search API, take the top 10 results, and prompt Claude or GPT to write the complete research report?"*
- **Candidate Trap**: Agreeing that modern LLMs can synthesize comprehensive answers directly from top search results.
- **Staff Counter-Maneuver**: "That is shallow RAG, which fails completely on complex inquiries. A query like 'Solid-state battery commercialization 2026–2030' requires exploring multiple distinct sub-dimensions: electrolyte chemistry, roll-to-roll vacuum coating yields, patent litigation, and global CapEx allocations. No single Google query surfaces all these aspects. Our system uses **Recursive Hypothesis Decomposition (DAG)**: it formulates 15–20 exploratory sub-queries in parallel waves. When a finding in Branch A reveals an unexpected delay, the DAG dynamically injects new child hypotheses to investigate that specific delay. This discovers technical whitepapers and SEC filings that never appear on Google page 1."

#### Trap Card 2: The "Unverified Loose URL Markdown Link" Trap
- **Interviewer**: *"We can instruct the model: 'Include markdown links to your sources: `[Toyota](url)`'. That provides sufficient source attribution."*
- **Candidate Trap**: Trusting the LLM to generate valid URLs and honest citations.
- **Staff Counter-Maneuver**: "LLMs routinely hallucinate URLs, link to unrelated pages on the same domain, or invent facts and attribute them to real links. In enterprise and high-stakes research, ungrounded citations destroy credibility. We mandate **Mathematical Character-Offset Grounding**. A citation is only accepted if it specifies `{doc_id, url, exact_quote, start_char_offset, end_char_offset}`. An independent **Citation Auditor** validates that `document_text[start:end] == exact_quote` and runs a fast entailment check. If the quote is missing or doesn't support the assertion, the claim is rejected."

#### Trap Card 3: The "Context Window Saturation with Raw HTML" Trap
- **Interviewer**: *"We have 1M token context windows. Why not just dump the raw HTML of all 120 scraped web pages into the prompt?"*
- **Candidate Trap**: Dumping raw HTML directly into the model context.
- **Staff Counter-Maneuver**: "Dumping 120 raw HTML pages consumes over 18,000,000 bytes (~4.5M tokens), exceeding context limits and costing hundreds of dollars per run. More critically, 95% of raw HTML consists of cookie banners, JavaScript tracking scripts, navigation trees, and CSS, which dilutes attention heads and triggers the lost-in-the-middle effect. In our pipeline, raw HTML passes through a **Semantic Content Distiller** (using Trafilatura / Mozilla Readability). This extracts clean markdown, converts financial tables to structured Markdown, and strips all boilerplate, reducing 150KB pages to < 3KB clean text without losing any factual signal."

#### Trap Card 4: The "Accepting Conflicting Primary Claims Uncritically" Trap
- **Interviewer**: *"If Source A says a battery has 500 Wh/kg and Source B says 385 Wh/kg, should the model just average them or report 442 Wh/kg?"*
- **Candidate Trap**: Averaging conflicting figures or hallucinating an artificial consensus.
- **Staff Counter-Maneuver**: "Averaging metrics from conflicting sources produces scientifically invalid nonsense. In deep research, we implement an explicit **Fact Triangulator & Source Authority Scorer**. The system recognizes that Source A is a corporate marketing press release (Tier 3 authority, 0.50 score) whereas Source B is an independent peer-reviewed test in *Nature Energy* (Tier 1 authority, 0.98 score). The agent does not average them; it generates an explicit **Contradiction Analysis Section**: *'While corporate marketing materials claim 500 Wh/kg in prototype cells, independent testing confirms current production yields achieve 385 Wh/kg [Nature Energy]'*. This provides nuanced, professional-grade intelligence."

#### Trap Card 5: The "Runaway Search Spider Budget Explosion" Trap
- **Interviewer**: *"What prevents the agent from following links recursively forever, spending $500 on web searches?"*
- **Candidate Trap**: Lacking strict budget controls or relying on simple loop counters.
- **Staff Counter-Maneuver**: "We implement a multi-layered **Budget Controller**:
  1. Hard caps: `MAX_QUERIES = 75`, `MAX_PAGES = 120`, `MAX_WALLCLOCK = 30 min`, `MAX_COST = $1.80`.
  2. **Information Saturation Heuristics**: If three consecutive pages on a branch yield zero new named entities or unique claims, the branch is marked `SATURATED` and pruned immediately.
  3. When 85% of any budget limit is consumed, the orchestrator triggers an automatic phase transition: halting all search workers and routing remaining budget to the Synthesis Engine."

---

## 3. Storage, Headless Browser & Hardware Micro-Mechanics

### 1. Headless Browser Fleet Sizing & Chromium Memory Mechanics

Fetching 3.6 million dynamic pages per day requires a dedicated Chromium cluster.
- **V8 Engine Memory Leakage**: Chromium's V8 JavaScript engine retains memory in long-running headless instances due to orphaned DOM nodes, canvas contexts, and service worker caches.
  - If a single browser instance navigates 100 pages, memory usage balloons from **120MB to 1.8GB**, eventually triggering Linux OOM killer (`SIGKILL`).
- **Production Isolation Strategy**:
  - We configure a pool of **isolated containerized worker pods**, each running Chromium with:
    `--disable-gpu --no-sandbox --disable-dev-shm-usage --single-process`.
  - **Tab Lifespan Policy**: Every browser context handles a maximum of **15 navigations** before being explicitly destroyed and recreated via Playwright/Puppeteer CDP (`Target.closeTarget`).
  - **Shared Memory Quota**: Mount `/dev/shm` as a high-speed `tmpfs` sized to at least 512MB per instance to prevent rendering crashes on heavy SPAs.

### 2. Temporal.io Event-Sourced Workflow Persistence

Deep research workflows execute for 15–45 minutes across 200+ asynchronous network operations.
- Instead of keeping a single Python thread alive for 45 minutes (which causes data loss if the Kubernetes node is preempted), the execution is modelled as a **Temporal Deterministic Workflow**.
- Every search query, scraping activity, and extraction is an immutable activity event stored in the Temporal history event log.
- If a worker crashes at Minute 28 during Section 3 synthesis, a new worker automatically hydrates from the event log, skips all 75 already-executed search queries, and resumes execution seamlessly.

### 3. ClickHouse / PostgreSQL Working Dossier Schema

The working dossier stores extracted entities and claims with vector embeddings:
- `pgvector` HNSW index on claim embeddings enables fast semantic deduplication ($< 2\text{ms}$ cosine distance check) before adding a new claim to the dossier.
- If a proposed claim has cosine similarity $> 0.94$ with an existing claim, it is merged as a corroborating citation rather than a new assertion, preventing dossier bloat.

---

## 4. Chaos Engineering & Failure Injection Runbooks

### Runbook 1: Anti-Bot IP Block & CAPTCHA Escalation

- **Failure Signature**: Web scraper activities return `403 Forbidden`, `429 Too Many Requests`, or Cloudflare Turnstile CAPTCHA challenges.
- **Root Cause**: Commercial websites detected data center IP ranges during parallel crawl waves.
- **Chaos Injection**: Block data center IPs or route scraper requests through an endpoint returning Cloudflare challenge pages.
- **Automated Mitigation**:
  1. **Proxy Rotation Gateway**: The scraper automatically routes failing domains through a residential rotating proxy mesh with automatic browser fingerprint randomization (`stealth.min.js`).
  2. If a page remains blocked after 2 proxy retries, the URL is marked `UNREACHABLE`; the orchestrator falls back to web search cache snippets and queries an alternate mirror source.

### Runbook 2: Citation Drift & Hallucination Invalidation

- **Failure Signature**: The drafting model submits a section where 3 out of 5 citations fail offset verification.
- **Root Cause**: Model hallucinated quotes or attempted to summarize text and placed quotes around paraphrased content.
- **Chaos Injection**:
  ```python
  # Submit hallucinated quote
  anchor = CitationAnchor("c_fake", doc.doc_id, doc.url, doc.title, "Fake statement", 10, 25)
  is_valid, msg = CitationAuditor.verify_anchor(anchor, engine.kb)
  ```
- **Automated Mitigation**:
  1. The Citation Auditor intercepts the section before it reaches the report publisher.
  2. The section is rejected and sent back to the drafting model with explicit feedback:
     `"CITATION_REJECTED: Quote 'Fake statement' does not exist in Source [doc_id]. You must cite exact substrings."`
  3. The model is forced to re-anchor the claim or delete the ungrounded assertion.

### Runbook 3: Contradiction Thrashing & Infinite Query Loop

- **Failure Signature**: The agent keeps generating search queries on the same metric for 20 minutes, exceeding query budgets without drafting sections.
- **Root Cause**: Two irreconcilable sources cause the agent to continually seek "one more source" to break the tie.
- **Chaos Injection**: Inject two equally weighted sources with contradictory claims into the knowledge base.
- **Automated Mitigation**:
  1. Enforce a **Max Discrepancy Depth of 2 queries** per contradiction.
  2. If the contradiction is not resolved after 2 targeted queries, the agent is forced to declare an **Explicit Analytical Consensus**:
     `"Data points diverge: Source A states X, while Source B states Y. Commercial deployment remains uncertain."`
  3. The DAG node transitions to `COMPLETED` and advances to synthesis.

### Runbook 4: Temporal Worker Node Eviction Mid-Research

- **Failure Signature**: Kubernetes worker node running deep research tasks receives `SIGTERM` due to node autoscaling or spot instance preemption.
- **Root Cause**: Spot instance eviction during a 30-minute research run.
- **Recovery Verification**:
  1. Temporal detects worker heartbeat loss after 15 seconds.
  2. The workflow is reassigned to an available healthy worker pod.
  3. The new worker replays the event history, skipping completed search waves, and resumes exactly at the uncompleted synthesis step.
