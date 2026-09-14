# Chapter 5: Distributed Web Crawler at Hyperscale — Staff/Principal Walkthrough Playbook

> **Companion Links**:
> - Core Architecture Blueprint: [`Volume-1/Design a Web Crawler.md`](file:///Users/kunalkumar/Desktop/knowledge-base/Projects/System-Design-Alex-Xu/Volume-1/Design%20a%20Web%20Crawler.md)
> - Production Crawler Engine: [`web_crawler_engine.py`](web_crawler_engine.py)
> - Chapter Hub: [`README.md`](README.md)
> - Deep Explainability Guide: [`03-Deep-Explainability-Guide.md`](03-Deep-Explainability-Guide.md)

---

## 1. Executive Summary & The 4 Pillars

Traversing the directed graph of the World Wide Web is one of the most operationally challenging distributed systems tasks in computer science. At a scale of **1 to 50 Billion web pages per month**, a crawler must ingest tens of gigabytes per second across millions of uncooperative external web servers.

A junior engineer sketches a Breadth-First Search (BFS) queue. A **Staff/Principal candidate** is distinguished by their mastery over **Mercator two-tier URL frontiers (balancing priority with host politeness), 64-bit SimHash near-duplicate document elimination, asynchronous non-blocking DNS resolution, spider-trap defense heuristics, and ISO 28500 WARC archival storage**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          CHAPTER 5 DEEP WALKTHROUGH PILLARS                            │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Pillar 1: Production Code│ Standalone Python engine implementing Mercator frontier,    │
│                          │ 64-bit SimHash (4-table LSH), robots.txt policy manager,    │
│                          │ HTML link extractor, and compressed WARC appender.          │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Pillar 2: Interview Flow │ 45-minute whiteboarding script, dialogue progression,       │
│                          │ and 5 lethal interviewer trap cards with counter-arguments. │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Pillar 3: Micro-Mechanics│ Linux socket exhaustion (`TIME_WAIT`, `SO_REUSEADDR`),      │
│                          │ non-blocking epoll, and SimHash 4-table Hamming math.       │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Pillar 4: Chaos Lab      │ Target server tarpits (Slowloris defense), dynamic DNS      │
│                          │ outages, and circular directory spider trap isolation.      │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 2. Pillar 1: Mathematical Foundations & Core Algorithms

### 2.1 The Mercator Two-Tier Frontier (Heydon & Najork)

A naive FIFO queue causes an accidental Distributed Denial of Service (DDoS) because a web page contains dozens of internal links pointing to the same server. Mercator decouples **Importance** from **Politeness**:

```
                       [ Incoming Discovered URLs ]
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │   PRIORITY (FRONT) QUEUES     │
                     │  Queue 0: High (PageRank > 7) │
                     │  Queue 1: Normal (Default)    │
                     │  Queue 2: Low (Deep Paths)    │
                     └───────────────┬───────────────┘
                                     │ Biased Random Selector
                                     ▼
                     ┌───────────────────────────────┐
                     │   POLITENESS (BACK) QUEUES    │
                     │  Host 1 (cnn.com)    Queue    │
                     │  Host 2 (wiki.org)   Queue    │
                     │  Host N (github.com) Queue    │
                     └───────────────┬───────────────┘
                                     │ Min-Heap (Earliest Allowed Time)
                                     ▼
                           [ Fetcher Workers ]
```

1. **Front Queues (Priority)**: Enforces crawling high-value pages first. Selected using a biased lottery scheduler (e.g., Priority 0 selected 70% of the time, Priority 1 25%, Priority 2 5%).
2. **Back Queues (Politeness)**: Exactly one queue per host.
3. **Politeness Min-Heap**: Maintains tuples of `(next_available_fetch_time, host_id)`. When worker threads request a task, the heap pops the earliest ready host, pops a URL from that host's queue, and pushes `now + host_crawl_delay` back into the heap.

---

### 2.2 64-Bit SimHash Near-Duplicate Detection (Charikar / Google)

Approximately **$30\% - 40\%$ of all web pages are near-duplicates** (e.g. syndicated news, page mirrors, or identical content with different footer timestamps). Cryptographic hashes (MD5/SHA-256) change by $> 50\%$ when a single byte differs (the avalanche effect).

#### SimHash Algorithm:
1. Tokenize document into $k$-shingles (word 3-grams).
2. Hash each shingle to a 64-bit integer $h$.
3. Maintain an accumulator vector $V = [0, \dots, 0]$ of length 64.
4. For each shingle hash:
   - If bit $i == 1$: $V[i] += 1$
   - If bit $i == 0$: $V[i] -= 1$
5. Final 64-bit fingerprint:
   $$\text{Fingerprint}[i] = 1 \text{ if } V[i] > 0 \text{ else } 0$$

#### The 4-Table Indexing Trick (Sub-Millisecond Query):
To find if an existing fingerprint has **Hamming Distance $\le 3$**:
- Split the 64-bit hash into $4$ blocks of 16 bits ($64 / 4 = 16$).
- By the Pigeonhole Principle, if two 64-bit numbers differ by at most 3 bits, **at least one of the 4 16-bit blocks MUST be 100% identical!**
- Maintain 4 hash tables, each keyed by a 16-bit block.
- Query takes $O(1)$ table lookups to examine a handful of candidate matches, avoiding a linear scan over billions of documents.

---

## 3. Pillar 2: The 45-Minute Staff/Principal Interview Playbook

### Timeline & Stage Breakdown
```
00:00 ────── 05:00 ────── 10:00 ────────────────── 25:00 ───────────── 38:00 ───── 45:00
  │            │            │                         │                   │          │
Scoping      Sizing     Mercator Architecture    Low-Level Mechanics    Trap Cards  Wrap-up
& Traps      & Pipes    & Frontier Design        & SimHash 4-Table      & Faults
```

---

### Phase 1: Requirements Scoping & Scope Traps (Minutes 0:00 – 0:05)

**Candidate Opening Move**:
> *"Web crawling at scale is fundamentally about resource governance and polite distributed graph traversal. Let's align on 4 core architectural boundaries:
> 1. Scale Target: Are we crawling 1 Billion pages/month (baseline) or 50 Billion/month (Google/Bing tier)?
> 2. Content Scope: Are we parsing raw static HTML only, or must we execute JavaScript via headless browsers (Puppeteer/Playwright) for Single Page Apps (SPAs)?
> 3. Politeness & Governance: What are our per-host rate limits, and must we adhere strictly to RFC 9309 `robots.txt`?
> 4. Storage Standard: Are we storing raw compressed web pages in standard ISO 28500 WARC format for downstream search indexing and LLM tokenization?"*

---

### Phase 2: Sizing, Storage & Network Math (Minutes 0:05 – 0:10)

Write these calculations directly on the board:

#### 1. Crawl Throughput Sizing:
$$\text{Target} = 1,000,000,000\text{ pages / month}$$
$$\text{Sustained Throughput} = \frac{10^9\text{ pages}}{30 \times 86,400\text{ sec}} \approx 386\text{ pages / sec}$$
$$\text{Peak Multiplier } (2.5\times) \approx 1,000\text{ pages / sec}$$

*(For Hyperscale 50B pages/month tier: $\text{Sustained} \approx 20,000\text{ pages / sec}$)*

#### 2. Storage Capacity Math:
- Average HTML document size: $100\text{ KB}$ raw $\implies \approx 30\text{ KB}$ gzip-compressed in WARC.
- Metadata (URL, headers, SimHash, outlinks): $\approx 2\text{ KB}$.
- Total per page: $\approx 32\text{ KB}$.
- **Monthly Storage**:
  $$\text{Storage} = 10^9 \times 32\text{ KB} \approx 32\text{ Terabytes / month} \quad (384\text{ TB / year})$$

#### 3. Network Ingress Bandwidth:
$$\text{Bandwidth} = 1,000\text{ pages/sec} \times 100\text{ KB/page} = 100\text{ MB/sec} = 800\text{ Mbps}$$
*(Easily handled by a pair of 10 GbE network interfaces).*

---

### Phase 3: Mercator Frontier & End-to-End Topology (Minutes 0:10 – 0:25)

Draw the full distributed crawling architecture:

```
[ Seed URLs ] ──► [ URL Normalizer & Filter ] ──► [ Seen URLs Bloom Filter ]
                                                              │ (If not seen)
                                                              ▼
                                                   [ Mercator Frontier ]
                                                     ├── Front Queues (Priority 0, 1, 2)
                                                     ├── Back Queues (Host-Affinity)
                                                     └── Politeness Min-Heap
                                                              │
                                                              ▼
                                                   [ Asynchronous Fetcher Pool ]
                                                     ├── Asynchronous DNS Cache (Unbound)
                                                     ├── robots.txt Cache
                                                     └── Non-Blocking Socket Workers
                                                              │
                                                              ▼
                                                   [ Downloaded HTML Payload ]
                                                              │
                    ┌─────────────────────────────────────────┴────────────────────────────┐
                    ▼                                                                      ▼
        [ SimHash Near-Dup Engine ]                                            [ HTML Link Extractor ]
        (4-Table Inverted Index)                                                ├── Resolves Relative Links
          ├── If Dup: Discard                                                   ├── Normalizes & Strips Traps
          └── If Unique: Write to ISO WARC Archive ──► [ S3 / HDFS Object Store ] └── Loops back to Seen Filter
```

---

### Phase 4: The 5 Interviewer "Trap Cards" & Staff-Level Defenses (Minutes 0:38 – 0:45)

#### 🪤 Trap Card 1: The Accidental DDoS / Host Flooding Trap
- **Interviewer**: *"You crawl Wikipedia. A single article has 400 internal Wikipedia links. If your crawler puts all 400 links into a standard queue, 50 workers fetch all 400 URLs in 200ms. Wikipedia's firewall blacklists your entire IP range. How do you prevent this?"*
- **Staff-Level Response**:
  > *"This is the fundamental failure of naive BFS. We prevent this using **Mercator Politeness Back Queues with a Min-Heap**:
  > 1. All links for `wikipedia.org` are routed exclusively to `BackQueue['wikipedia.org']`.
  > 2. The host entry in the Politeness Min-Heap is stamped with `next_fetch_time = now + host_crawl_delay` (e.g. 1.0 second).
  > 3. No worker thread can touch `wikipedia.org` until that 1.0-second delay has fully elapsed.
  > 4. Across thousands of distinct domains, workers are fully utilized while ensuring that no single external host ever receives more than 1 request per second."*

#### 🪤 Trap Card 2: The Near-Duplicate Content Explosion
- **Interviewer**: *"Syndicated Reuters and AP news articles are published across 5,000 regional newspapers with identical text but different local advertisements, comments, and headers. MD5 and SHA-256 hashes are 100% different. Your crawler downloads and indexes all 5,000 copies, wasting 99.9% of your disk space. How do you solve this?"*
- **Staff-Level Response**:
  > *"Cryptographic hashes fail on near-duplicates due to the avalanche effect. We use **64-bit SimHash Locality-Sensitive Hashing (LSH)**:
  > 1. We tokenize the body text into 3-word shingles and compute a 64-bit SimHash fingerprint.
  > 2. If two pages share 95%+ identical content, their fingerprints differ by at most $\le 3$ bits (Hamming Distance).
  > 3. Using a **4-Table Inverted Index**, we partition the 64-bit hash into 4 16-bit blocks. By the Pigeonhole Principle, at least one 16-bit block must match identically. We look up candidates in $O(1)$ time and discard near-duplicates before writing to WARC storage."*

#### 🪤 Trap Card 3: The Synchronous DNS Bottleneck
- **Interviewer**: *"At 5,000 pages per second, each fetcher worker calls standard Linux `getaddrinfo()`. Each call takes 40ms over UDP. Your 500 worker threads spend 90% of their time blocked on DNS lookups. How do you eliminate this bottleneck?"*
- **Staff-Level Response**:
  > *"Standard libc `getaddrinfo()` is synchronous and thread-blocking. We eliminate this with three layers:
  > 1. **In-Memory Non-Blocking DNS Cache**: Fetchers query an in-process LRU cache (e.g. `c-ares` or `dnspython` async resolver).
  > 2. **Local Anycast DNS Daemon**: Each crawler host runs a local caching DNS proxy (**Unbound**) listening on `127.0.0.1:53` with pre-fetching.
  > 3. **Pinned Host Affinity**: The Mercator frontier assigns all URLs for a given host to the same worker node, ensuring a near-100% local DNS cache hit rate."*

#### 🪤 Trap Card 4: The Spider Trap & Infinite Path Loop
- **Interviewer**: *"A malicious website generates infinite dynamic calendars (`/events?date=2099-01-01`) or endless directory loops (`/archive/archive/archive/...`). Your crawler queue balloons to 500 million bogus URLs. How do you detect and kill spider traps?"*
- **Staff-Level Response**:
  > *"We enforce multi-tiered **Heuristic Trap Defense**:
  > 1. **Path Depth Ceiling**: Any URL with path depth $> 8$ segments or length $> 256$ characters is discarded.
  > 2. **Cyclic Path Segment Detection**: We tokenize paths and detect repetitive n-gram patterns (`/a/b/a/b/a/b`). If a segment repeats $\ge 3$ times, the URL is dropped.
  > 3. **Per-Host Crawl Quota**: Each host is allocated a maximum crawl budget (e.g. 5,000 pages/month). Once reached, subsequent URLs from that host are demoted to a dead-letter queue until an operator review.
  > 4. **Query Parameter Stripping**: We automatically strip session tokens, timestamps, and tracking params (`utm_*`, `session_id`)."*

#### 🪤 Trap Card 5: Headless Browser (Puppeteer/Playwright) Resource Exhaustion
- **Interviewer**: *"Many modern websites require JavaScript rendering. Why not just run headless Chromium for every fetch?"*
- **Staff-Level Response**:
  > *"Running Chromium for every page at 1,000 pages/sec is mathematically impossible:
  > - Headless Chrome consumes $\approx 100 - 300\text{ MB}$ of RAM per tab and takes $1.5 - 4.0\text{ seconds}$ per page.
  > - 1,000 pages/sec would require **over 300 GB of RAM and thousands of CPU cores** purely to run browser rendering engines!
  > **Staff Two-Tier Strategy**:
  > 1. **Tier 1 (95% of traffic)**: Lightweight, high-throughput asynchronous HTTP socket fetcher ($< 1\text{ MB}$ RAM, $< 50\text{ ms}$).
  > 2. **Tier 2 (5% selective trigger)**: If the Tier 1 response body contains `<div id="root"></div>` with zero substantive text, the domain is flagged for deferred rendering in a dedicated, rate-limited headless browser worker pool."*

---

## 4. Pillar 3: Kernel, Network & Micro-Mechanics

### 4.1 Linux Socket Exhaustion (`TIME_WAIT`) & `epoll`
Crawling thousands of external hosts creates and closes thousands of TCP connections:
- When a client closes a TCP socket, it enters the `TIME_WAIT` state for $2 \times \text{MSL} = 60\text{ seconds}$ to ensure late-arriving packets do not corrupt future connections.
- At 1,000 req/sec, the system accumulates $60,000$ sockets in `TIME_WAIT`, exhausting the ephemeral port range (`32768 - 60999`).
- **Kernel Tuning**:
  ```bash
  # Enable fast recycling of TIME_WAIT sockets for outgoing connections
  sysctl -w net.ipv4.tcp_tw_reuse=1
  # Expand local port range
  sysctl -w net.ipv4.ip_local_port_range="1024 65535"
  # Fast FIN timeout
  sysctl -w net.ipv4.tcp_fin_timeout=15
  ```

---

## 5. Production Engine Verification Results

From running [`web_crawler_engine.py`](web_crawler_engine.py):

```
==================================================================
  VERIFYING 64-BIT SIMHASH NEAR-DUPLICATE ENGINE
==================================================================
- Hamming Distance (Original vs Near-Duplicate): 3 bits (Threshold <= 3)
- Detected as Near-Duplicate: True (Correctly identified near-duplicate!)
- Hamming Distance (Original vs Unrelated):      35 bits
- Detected as Near-Duplicate: False (Correctly rejected unrelated doc)
- [✓] SimHash Engine Verified!

==================================================================
  LIVE MULTI-THREADED CRAWL VERIFICATION
==================================================================
- Target Pages Crawled:     5 / 5
- Seed Hosts:               example.com, httpbin.org
- Host Politeness Enforced: Strictly respected robots.txt & crawl-delay
- WARC Archive Created:     crawl_archive.warc.gz (ISO 28500 compliant)
```

---

## 6. Next Steps

- **Completed**:
  - Chapter 1: Rate Limiter (Walkthrough & Code Lab) ✅
  - Chapter 2: Consistent Hashing (Walkthrough & Code Lab) ✅
  - Chapter 3: Unique ID Generator (Walkthrough & Production Service) ✅
  - Chapter 4: URL Shortener (Walkthrough & Production Service) ✅
  - Chapter 5: Web Crawler (Walkthrough & Production Engine) ✅
- **Up Next in Volume 1**: **Chapter 6 — Design a Distributed Key-Value Store (Dynamo / Cassandra)** (Mini-LSM Tree with WAL, MemTable, SSTables, Bloom filters, compaction stalls, and Quorum Linearizability).
