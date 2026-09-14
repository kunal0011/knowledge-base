---
date: "2026-04-04"
type: system-design
status: active
volume: 2
chapter: 8
difficulty: hard
tags:
  - system-design
  - email-service
  - distributed-systems
  - alex-xu
  - interview-prep
  - scylladb
  - cassandra
  - jmap
  - elasticsearch
  - kafka
---

# Design a Distributed Email Service

> [!tip] Staff/Principal Deep Walkthrough & Production Engine
> - **Interview Playbook**: [`Volume 2 Chapter 8 Staff/Principal Walkthrough Playbook`](02-Interactive-Interview-Playbook.md)
> - **Production Email & JMAP Engine**: [`email_platform_engine.py`](email_platform_engine.py) (JMAP Stateless Protocol, RFC 5322 MIME Parser, JWZ Conversation Threading, SHA-256 CAS Attachment Deduplication, and Full-Text Inverted Index)

## Executive Architectural Blueprint

Designing a modern, hyperscale distributed email service (comparable to **Gmail** or **Microsoft Outlook**) represents one of the most demanding challenges in distributed systems engineering. Email is simultaneously an immutable append-only archival system, a real-time collaborative workspace, a complex cryptographic protocol gateway, and a multi-tenant full-text search engine.

```mermaid
graph TB
    subgraph Clients["Global Multi-Platform Clients"]
        Web["Web Client (Next.js/React)<br/>JMAP over HTTP/3"]
        Mobile["Mobile Apps (iOS/Android)<br/>JMAP Push Notifications"]
        Legacy["Desktop Mail Clients<br/>IMAP / POP3 / SMTP"]
    end

    subgraph EdgeGateways["Edge Ingress & Protocol Gateways"]
        Anycast["Global Anycast L4 Load Balancers"]
        JMAPGate["JMAP / REST API Gateway<br/>Stateless, TLS 1.3 Termination"]
        IMAPGate["IMAP / POP3 Protocol Proxy<br/>C10M epoll Connection Pool"]
        SMTP_IN["MTA Inbound SMTP Receiver<br/>Port 25 / 587, Rate Limiting"]
    end

    subgraph KafkaIngestion["Distributed Message Queues (Apache Kafka)"]
        K_In["Topic: email-inbound-raw<br/>512 Partitions"]
        K_Out["Topic: email-outbound-queue<br/>Domain-Sharded Delivery Queue"]
    end

    subgraph ProcessingPipeline["Asynchronous Processing Pipeline"]
        SecFilter["Security & Auth Pipeline<br/>DNSBL, SPF, DKIM, DMARC, ARC"]
        SpamFilter["ML Spam & Phishing Engine<br/>TensorFlow / ONNX Inference"]
        MIMEParser["MIME Body & Attachment Parser<br/>Text Stripping, SHA-256 Dedup"]
        ThreadDetector["Conversation Threading Engine<br/>Message-ID & References Matching"]
    end

    subgraph StorageTier["Distributed Storage Fabric"]
        ScyllaDB[("ScyllaDB / Cassandra Cluster<br/>Metadata, Folders, TWCS SSTables")]
        S3Blobs[("Cloud Object Storage (S3 / GCS)<br/>Attachments & Raw MIME Bodies")]
        ES[("Elasticsearch / OpenSearch Cluster<br/>Full-Text Inverted Index")]
        RedisCache[("Redis Cluster<br/>Inbox Caching & Ephemeral Sync Tokens")]
    end

    subgraph OutboundMTA["Outbound Delivery Subsystem"]
        MTA_Sender["SMTP Outbound Dispatcher<br/>MX Lookup, TLS, Backoff Retry"]
        ExternalMX["External Mail Exchangers<br/>(Google, Outlook, Yahoo)"]
    end

    Web --> Anycast
    Mobile --> Anycast
    Legacy --> Anycast

    Anycast --> JMAPGate
    Anycast --> IMAPGate
    Anycast --> SMTP_IN

    SMTP_IN --> K_In
    JMAPGate --> K_Out
    IMAPGate --> K_Out

    K_In --> SecFilter
    SecFilter --> SpamFilter
    SpamFilter --> MIMEParser
    MIMEParser --> ThreadDetector

    ThreadDetector --> ScyllaDB
    ThreadDetector --> S3Blobs
    ThreadDetector --> ES
    ThreadDetector --> RedisCache

    K_Out --> MTA_Sender
    MTA_Sender --> ExternalMX
```

### The Core Engineering Dilemma

Traditional mail servers (such as Postfix, Dovecot, and Qmail) store messages as individual files on a POSIX filesystem (the `Maildir` or `mbox` format). At consumer internet scale, this paradigm completely collapses due to:
1. **Inode Exhaustion and Metadata Latency**: Hundreds of billions of small files choke traditional ext4/XFS filesystems, generating catastrophic IOPS amplification during directory traversals.
2. **Protocol Impedance Mismatch**: Legacy protocols (IMAP4rev1, POP3) assume long-lived stateful TCP connections, client-driven folder polling, and in-band lock acquisition. Modern mobile clients on high-latency, intermittently connected cellular networks fail under these assumptions.
3. **Cryptographic & Deliverability Asymmetry**: Inbound mail requires rigorous cryptographic validation (SPF, DKIM, DMARC, ARC) and multi-tier ML abuse classification under strict $< 2\text{s}$ timeout constraints; outbound mail demands IP pool reputation warming, MX resolver resiliency, and exponential backoff retry scheduling lasting up to 96 hours.

### System Design Tenets & Service Level Objectives (SLOs)

| Metric | Target | Description & Enforcement |
| :--- | :--- | :--- |
| **Durability (Zero-Loss)** | **99.999999999% (11 9s)** | Dual-write to Kafka with `acks=all` before acknowledging SMTP `250 OK`. Raw MIME payload mirrored to S3/GCS multi-region blob storage. |
| **Inbound Processing Latency** | **p95 < 1.5s, p99 < 3.0s** | Time elapsed from SMTP `DATA` stream completion to real-time client push notification. |
| **Inbox View Fetch Latency** | **p99 < 150ms** | Single-partition range query in wide-column store using cached state tokens. |
| **Search Query Latency** | **p95 < 250ms** | Routing search queries to user-sharded Lucene/Elasticsearch indices. |
| **Service Availability** | **99.99%** | Multi-region active-active deployment across edge proxies, stateless gateways, and quorate storage tiers. |

---

## Back-of-the-Envelope Calculations & Hyperscale Baseline

### Traffic Baseline (1 Billion Users)

- **Total Registered Accounts**: $1,000,000,000$ (1 Billion).
- **Daily Active Users (DAU)**: $250,000,000$ (25% active ratio).
- **Outbound Volume (Sent)**:
  - Average per user: $10 \text{ emails/day}$.
  - Daily outbound volume: $250\text{M} \times 10 = 2.5\text{ Billion emails/day}$.
  - Average outbound QPS: $\frac{2.5 \times 10^9}{86,400} \approx 28,935\text{ msgs/sec}$.
  - Peak outbound QPS ($3\times$ multiplier): $\approx 86,800\text{ msgs/sec}$.
- **Inbound Volume (Received)**:
  - Average per user: $40 \text{ emails/day}$ (including automated notifications, newsletters, marketing, spam).
  - Daily inbound volume: $250\text{M} \times 40 = 10\text{ Billion emails/day}$.
  - Average inbound QPS: $\frac{10 \times 10^9}{86,400} \approx 115,740\text{ msgs/sec}$.
  - Peak inbound QPS ($3\times$ multiplier): $\approx 347,200\text{ msgs/sec}$.

### Storage Footprint & Ingress Bandwidth

- **Payload Breakdown**:
  - Raw metadata + HTML/Text body: $50\text{ KB}$ average.
  - Attachment probability: $20\%$ of incoming messages contain attachments.
  - Average attachment size: $500\text{ KB}$.
  - Weighted average email size: $50\text{ KB} + (0.20 \times 500\text{ KB}) = 150\text{ KB}$.
- **Daily Raw Data Ingestion**:
  - Inbound raw data volume: $10\text{B} \times 150\text{ KB} = 1.5\text{ PB/day}$.
  - Annual raw storage growth: $1.5\text{ PB/day} \times 365 \approx 547.5\text{ PB/year}$ (unreplicated).
  - Multi-datacenter replication ($3\times$ metadata, $3\times$ blob): $\approx 1.64\text{ Exabytes/year}$.
- **Network Bandwidth Demand**:
  - Inbound steady-state bandwidth: $115,740\text{ msgs/sec} \times 150\text{ KB} \approx 17.36\text{ GB/sec} \approx 138.9\text{ Gbps}$.
  - Peak ingress bandwidth: $138.9\text{ Gbps} \times 3 \approx 416.7\text{ Gbps}$.

---

## Deep-Dive Module 1: Inbound Cryptographic Verification & Spam Pipeline

When an external Mail Transfer Agent (MTA) delivers a message over port 25, the receiving infrastructure must subject the TCP stream to rigorous validation before persisting it to user mailboxes.

```mermaid
flowchart LR
    subgraph EdgeConnection["1. Connection & IP Hygiene"]
        Raw["Inbound TCP Connection"] --> RBL{"DNSBL Check"}
        RBL -- Listed on Blacklist --> Drop421["Reject Connection<br/>SMTP 550 or 421"]
        RBL -- Clean IP --> TLS["Enforce STARTTLS<br/>Inspect Client rDNS"]
    end

    subgraph CryptographicAuth["2. Cryptographic Authentication"]
        TLS --> SPF{"SPF Validation"}
        SPF -- Sender IP Authorized --> DKIM["Verify DKIM Signatures"]
        SPF -- Unauthorized IP --> DKIM
        DKIM --> DMARC{"DMARC Policy Engine"}
        DMARC -- Align Pass --> ARC["Evaluate ARC Chain<br/>(Mailing List Forwarding)"]
        DMARC -- Align Fail --> CheckPolicy{"DMARC Action"}
        CheckPolicy -- Reject Policy --> DropReject["Drop Message<br/>SMTP 550 Reject"]
        CheckPolicy -- Quarantine Policy --> TagQuarantine["Mark as Quarantine"]
    end

    subgraph MLSpamPhish["3. Machine Learning Classification"]
        ARC --> FeatExtract["Extract Features:<br/>MIME structure, URL reputation, Bayesian tokens"]
        TagQuarantine --> FeatExtract
        FeatExtract --> DLModel["Deep Learning Transformer Model<br/>Spam / Phish Probability Score"]
        DLModel --> ScoreCheck{"Score Check"}
        ScoreCheck -- High Risk --> SpamFolder["Deliver to User Spam Folder"]
        ScoreCheck -- Safe --> CleanInbox["Deliver to User Inbox"]
    end
```

### Cryptographic Authentication Mechanics

1. **SPF (Sender Policy Framework - RFC 7208)**:
   - Evaluates the envelope sender (`MAIL FROM` / Return-Path) against DNS `TXT` records of the sending domain.
   - Prevents IP spoofing, but breaks upon legitimate message forwarding (since the forwarder becomes the new connecting IP).
2. **DKIM (DomainKeys Identified Mail - RFC 6376)**:
   - Signs specified headers (e.g., `From`, `To`, `Subject`, `Date`) and the message body hash with the sender domain's private RSA/Ed25519 key.
   - The public key is retrieved from DNS (`selector._domainkey.domain.com`).
   - Remains cryptographically valid across hops provided intermediary servers do not modify signed headers or reformat body whitespace.
3. **DMARC (Domain-based Message Authentication - RFC 7489)**:
   - Solves the display spoofing vulnerability by enforcing **Identifier Alignment**: the domain in the visible `From:` header must match the validated SPF domain or the DKIM `d=` signing domain.
   - Enforces domain owner policy: `p=none` (monitor), `p=quarantine` (deliver to Spam folder), or `p=reject` (refuse at SMTP handshake with `550 5.7.1`).
4. **ARC (Authenticated Received Chain - RFC 8617)**:
   - Preserves authentication status across intermediate forwarders (e.g., mailing lists, alumni forwarding).
   - Each forwarder signs an ARC seal (`AS`), ARC message signature (`AMS`), and ARC authentication results (`AAR`), creating a trusted cryptographic chain of custody.

---

## Deep-Dive Module 2: Storage Micro-Architecture (ScyllaDB LSM & TWCS)

At $10\text{ Billion emails/day}$, traditional B-tree relational databases collapse under lock contention, random I/O fragmentation, and vacuum bloat. The system employs a wide-column NoSQL engine (**ScyllaDB / Apache Cassandra**) optimized with an LSM-tree storage layout and **Time-Window Compaction Strategy (TWCS)**.

```mermaid
graph TD
    subgraph CassandraKeyspace["ScyllaDB / Cassandra Keyspace: mail_store"]
        subgraph PartitionModel["Partition Key: ((user_id, folder_id), received_year_month)"]
            EmailRow1["Clustering Key: (internal_date DESC, email_id DESC)<br/>Subject, From, To, BodyPreview, Flags"]
            EmailRow2["Clustering Key: (internal_date DESC, email_id DESC)<br/>Subject, From, To, BodyPreview, Flags"]
            BlobPtr["Attachment Pointer Array:<br/>[sha256_hash, s3_bucket, byte_size]"]
        end
    end

    subgraph TWCSCompaction["Time-Window Compaction Strategy (TWCS)"]
        W1["Window 1: Current Month SSTables<br/>Active Concurrent Writes"]
        W2["Window 2: Last Month SSTables<br/>Immutable, Merged Once"]
        W3["Window 3: Trash / Spam Expired SSTable<br/>Entire File Dropped via OS unlink"]
    end

    subgraph S3ObjectStorage["S3 Content-Addressed Blob Storage"]
        DedupEngine["Attachment Dedup Engine:<br/>Hash payload using SHA-256"]
        S3Object["s3://mail-blobs/attachments/{sha256}<br/>Single Instance Storage Across Users"]
    end

    EmailRow1 --> BlobPtr
    BlobPtr --> DedupEngine
    DedupEngine --> S3Object
    PartitionModel --> TWCSCompaction
```

### Partitioning & Clustering Strategy

If `user_id` alone were used as the partition key, active accounts would accumulate millions of rows over decades, causing wide partitions exceeding $100\text{ MB}$, severe heap fragmentation, and degraded read performance.

We formulate a **composite partition key with temporal bucketing**:
$$\text{Partition Key} = \big((\text{user\_id}, \text{folder\_id}), \text{received\_year\_month}\big)$$
$$\text{Clustering Key} = (\text{internal\_date DESC}, \text{email\_id DESC})$$

- **Partition Sizing**: A heavy user receiving $300\text{ emails/day}$ generates $\approx 9,000\text{ rows/month}$. At $\approx 1.5\text{ KB/row}$ of metadata, the partition size is capped at $\approx 13.5\text{ MB}$, fitting comfortably within the optimal $10\text{ MB} - 50\text{ MB}$ Cassandra partition threshold.
- **TWCS (Time-Window Compaction Strategy)**: Emails are written sequentially with timestamps strictly matching arrival time. TWCS merges SSTables within the current time window (e.g., 1 day or 1 month). Once a time window closes, the SSTables become immutable. When messages in Spam or Trash expire via TTL (30 days), ScyllaDB avoids tombstone GC churn; the storage engine drops entire historical SSTable files via instant OS-level `unlink()`.

### Content-Addressed Blob Storage & Deduplication

Large email bodies ($> 64\text{ KB}$) and all attachments are stripped by the MIME parser and stored in content-addressed S3/GCS buckets:
1. The parser computes the SHA-256 cryptographic hash of the attachment payload.
2. The blob key is derived as:
   $$\text{s3://mail-attachments/}\text{hex}(H[0..1])\text{/}\text{hex}(H[2..3])\text{/}\text{sha256\_digest}$$
3. When a viral company memo or marketing attachment is sent to $50,000$ internal employees, the payload is written to S3 **exactly once**. $50,000$ individual Cassandra metadata rows store the same SHA-256 reference pointer, saving $> 99.99\%$ of attachment storage footprint.

---

## Deep-Dive Module 3: Protocol Evolution: Modern JMAP vs Legacy IMAP

For over three decades, IMAP4rev1 (RFC 3501) served as the standard client synchronization protocol. However, IMAP's stateful chatty command-response model is fundamentally incompatible with modern mobile environments.

```mermaid
sequenceDiagram
    autonumber
    participant Client as Modern Mobile Client
    participant JMAP as JMAP Gateway Server
    participant DB as Mailbox Database
    participant Push as WebPush / APNs
    participant Cache as Redis State Cache

    Client->>JMAP: HTTP POST /jmap/api<br/>('Email/get', {ids: ['e1', 'e2'], properties: ['id', 'subject', 'from']})
    JMAP->>DB: Single Batch Range Query
    DB-->>JMAP: Return JSON Data Payloads
    JMAP-->>Client: HTTP 200 OK (Single Request-Response Roundtrip)

    Note over Client,JMAP: Background Push Event Occurs
    DB->>Cache: Update state_token to s402
    DB->>Push: Invalidate State Token (new_state = s402)
    Push-->>Client: Silent Background Push: new_state=s402

    Client->>JMAP: HTTP POST /jmap/api<br/>('Email/changes', {sinceState: 's401'})
    Note over JMAP: Delta Sync Calculation
    JMAP->>DB: Fetch IDs created/destroyed since s401
    DB-->>JMAP: Delta IDs: added=[e105], updated=[e2]
    JMAP-->>Client: HTTP 200 OK with Delta Records (Minimal Bandwidth)
```

### Architectural Comparison: IMAP vs JMAP

| Architectural Dimension | Legacy IMAP4rev1 (RFC 3501) | Modern JMAP (RFC 8620 / 8621) |
| :--- | :--- | :--- |
| **Transport Layer** | Long-lived, stateful TCP connection (port 993). | Stateless HTTP/3 (QUIC) or HTTP/2 over TLS 1.3. |
| **Data Encoding** | Custom modified UTF-7, complex S-expression tokens. | Pure, standardized JSON-RPC with typed schemas. |
| **Request Roundtrips** | Requires separate serial roundtrips (`SELECT`, `SEARCH`, `FETCH FLAGS`, `FETCH BODY`). | Pipelined batch invocations: fetch metadata, thread, and unread counts in **1 single HTTP POST**. |
| **Synchronization Model** | Stateful `UIDVALIDITY` + `UIDNEXT` + `IDLE` command loop. | Monotonically incrementing `state` string tokens (`Email/changes`). |
| **Battery & Mobile Radio** | Radio must stay energized to maintain TCP keep-alives. Drops on cellular IP handover. | Zero keep-alive overhead. Wakes up strictly on standard OS push notification (APNs / FCM). |
| **Server Memory Footprint** | $\approx 25\text{ KB} - 50\text{ KB}$ kernel/process memory per idle TCP socket (C10M scale problem). | Completely stateless web tier fronted by standard Envoy/Nginx proxies. |

---

## Deep-Dive Module 4: Multi-Tenant Inverted Index & Search Sharding

Users expect sub-second full-text search across millions of personal emails spanning decades. Naively distributing an inverted index across a generic Elasticsearch cluster results in disastrous **scatter-gather fanout**, where a single user's query must contact hundreds of cluster shards.

```mermaid
flowchart TD
    subgraph UserQuery["User Search Submission"]
        Q["GET /search?q='quarterly tax' AND has:attachment<br/>user_id: 8812a"]
    end

    subgraph GatewayRouter["Query Gateway & Routing Hash"]
        Route["Route Query using user_id as Custom Routing Key:<br/>shard_id = hash(user_id) % num_shards"]
    end

    subgraph ElasticsearchCluster["Elasticsearch Multi-Tenant Shared Shards"]
        subgraph TargetShard["Assigned Single Physical Shard (Node 4)"]
            DocIndex["Target Inverted Index Segment"]
            UserTerm["Filter: term(user_id, '8812a')"]
            TextMatch["Match: match_phrase(body, 'quarterly tax')"]
            AttachFilter["Filter: term(has_attachment, true)"]
        end
        OtherShards["Remaining Shards (Node 1, 2, 3)<br/>Zero Load, Bypassed Completely"]
    end

    subgraph ResultPipeline["Result Hydration"]
        DocIDs["Matched Email IDs: [email_992, email_104]"]
        Hydrate["Hydrate Summary Snippets from ScyllaDB"]
        ClientOut["Return Ranked Results in < 80ms"]
    end

    Q --> Route
    Route --> TargetShard
    Route -.-> OtherShards
    TargetShard --> DocIndex
    DocIndex --> UserTerm
    UserTerm --> TextMatch
    TextMatch --> AttachFilter
    AttachFilter --> DocIDs
    DocIDs --> Hydrate
    Hydrate --> ClientOut
```

### Custom Routing Key Architecture

To eliminate scatter-gather overhead, every indexed email document enforces custom routing:
$$\text{Elasticsearch Index Request: } \texttt{POST /email\_index/\_doc/\{email\_id\}?routing=\{user\_id\}}$$

1. **Deterministic Single-Shard Colocation**: All emails belonging to `user_id = 8812a` reside entirely within **one physical Lucene shard**.
2. **Scatter-Gather Bypass**: When executing a search query, the gateway includes the `routing=8812a` parameter. Elasticsearch routes the query directly to the designated node. The remaining $99.9\%$ of cluster nodes experience zero CPU, network, or memory load.
3. **Two-Phase Query Execution**:
   - **Phase 1 (Index Match)**: Lucene evaluates the inverted index on the single shard, applying bitset filters on `user_id` and boolean matches on stemmed terms, returning the top 20 `email_id`s and relevance scores.
   - **Phase 2 (Hydration)**: Instead of bloating the Elasticsearch document store with massive MIME bodies, Elasticsearch stores only the search vectors and `email_id`. The gateway hydrators fetch message previews directly from ScyllaDB or Redis in $< 10\text{ms}$.

---

## Deep-Dive Module 5: Outbound Delivery Engine, MX Reputation & Backoff

Outbound email delivery is asynchronous and constrained by external recipient server policies, greylisting, and strict IP reputation thresholds.

```mermaid
graph TB
    subgraph OutboundQueue["Kafka Outbound Message Queue"]
        Msg["Outbound Email Payload<br/>Recipient: target@external.com"]
    end

    subgraph MXRouter["MX Discovery & Reputation Router"]
        DNS["DNS Resolver (DNSSEC enabled)<br/>Resolve MX records for external.com"]
        IPPool["Sender IP Reputation Pool<br/>Select High-Reputation IP based on Tenant"]
    end

    subgraph MTAEngine["MTA Sender Delivery Engine"]
        Connect["Establish TCP Port 25 / STARTTLS"]
        Handshake["SMTP Protocol Handshake:<br/>EHLO, MAIL FROM, RCPT TO, DATA"]
    end

    subgraph DeliveryResult["Delivery Outcome Evaluation"]
        Success{"Status 250 OK?"}
        Delivered["Delivery Completed<br/>Update Status: 'Sent'"]
        EvalError{"Error Classification"}
        HardBounce["Hard Bounce (5xx):<br/>User Unknown, Domain Missing<br/>Drop & Send NDR to Sender"]
        SoftBounce["Soft Bounce (4xx):<br/>Mailbox Full, Greylisting, Rate Limit"]
        Backoff["Exponential Backoff Queue:<br/>1m -> 5m -> 30m -> 2h -> 8h -> 24h"]
    end

    Msg --> DNS
    DNS --> IPPool
    IPPool --> Connect
    Connect --> Handshake
    Handshake --> Success
    Success -- Yes --> Delivered
    Success -- No --> EvalError
    EvalError -- 5xx Permanent --> HardBounce
    EvalError -- 4xx Transient --> SoftBounce
    SoftBounce --> Backoff
    Backoff -.->|Re-enqueue after delay| Connect
```

### IP Pool Partitioning & Reputation Management

Major receiving providers (Google, Microsoft, Yahoo) monitor inbound connection rates by source IP. Outbound MTAs are partitioned into isolated IP pools:

1. **Transactional Pool (Tier 1)**: Dedicated IPs reserved exclusively for password resets, 2FA tokens, and critical system alerts. Must maintain $> 99\%$ sender reputation score.
2. **Corporate & Standard User Pool (Tier 2)**: Daily peer-to-peer user communications.
3. **Bulk / Marketing Pool (Tier 3)**: High-volume newsletter dispatchers. If a tenant triggers spam complaints, their traffic is automatically isolated to this pool, preventing blast damage to Tier 1 transactional delivery.
4. **Quarantine / Warmup Pool (Tier 4)**: Newly provisioned IPs undergoing algorithmic warming (starting at $1,000\text{ msgs/day}$ and doubling daily over 30 days while monitoring Spamhaus and Return Path metrics).

### Error Classification & Backoff Schedule

- **Permanent Failures (5xx Codes - Hard Bounce)**:
  - `550 5.1.1`: User unknown / mailbox does not exist.
  - `550 5.7.1`: Relaying denied or DMARC reject policy failed.
  - *Action*: Abort retries immediately. Write to sender's mailbox with a Non-Delivery Report (NDR) bounce notification. Add recipient to suppression list.
- **Transient Failures (4xx Codes - Soft Bounce & Greylisting)**:
  - `421 4.7.0`: Service temporarily unavailable or connection limit exceeded.
  - `450 4.2.0`: Recipient mailbox full.
  - `451 4.7.1`: Greylisting enforced ("Please retry after 5 minutes").
  - *Action*: Re-enqueue into delayed Kafka retry topics with progressive, jittered exponential backoff:
    $$T_{\text{wait}} = \min(T_{\max}, T_{\text{base}} \times 2^{\text{attempt}}) \pm \text{jitter}$$
    Schedule: $1\text{m} \to 5\text{m} \to 30\text{m} \to 2\text{h} \to 8\text{h} \to 24\text{h} \to 48\text{h}$. Messages expire after $96\text{ hours}$.

---

## Data Models & Schema Definitions

### ScyllaDB CQL Schema

```sql
-- Keyspace with NetworkTopologyStrategy across two primary regions
CREATE KEYSPACE mail_store WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'us-east-1': 3,
    'us-west-2': 3
};

-- Core Email Metadata Table partitioned by user, folder, and temporal month bucket
CREATE TABLE mail_store.emails_by_folder (
    user_id          uuid,
    folder_id        text,           -- e.g., 'inbox', 'sent', 'archive', 'trash', 'spam'
    year_month       text,           -- Partition bucket: '2026-04'
    internal_date    timestamp,      -- Received or sent timestamp
    email_id         timeuuid,
    thread_id        uuid,
    message_id       text,           -- RFC 5322 globally unique identifier
    from_address     text,
    to_addresses     list<text>,
    cc_addresses     list<text>,
    bcc_addresses    list<text>,
    subject          text,
    snippet          text,           -- Sanitized 250-character plain text preview
    body_blob_id     text,           -- S3 pointer for raw body (if > 64 KB)
    body_inline_html text,           -- Inline compressed HTML if < 64 KB
    is_read          boolean,
    is_starred       boolean,
    labels           set<text>,
    attachments      list<frozen<tuple<text, text, bigint, text>>>, -- <filename, mime_type, bytes, sha256_hash>
    PRIMARY KEY (((user_id, folder_id), year_month), internal_date, email_id)
) WITH CLUSTERING ORDER BY (internal_date DESC, email_id DESC)
AND compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_unit': 'DAYS',
    'compaction_window_size': '1',
    'timestamp_resolution': 'MILLISECONDS'
};

-- Conversation Thread Table for Thread-Aggregated Views
CREATE TABLE mail_store.threads_by_user (
    user_id          uuid,
    folder_id        text,
    last_updated_at  timestamp,
    thread_id        uuid,
    subject          text,
    participants     set<text>,
    message_count    int,
    has_attachments  boolean,
    snippet          text,
    is_read          boolean,
    PRIMARY KEY ((user_id, folder_id), last_updated_at, thread_id)
) WITH CLUSTERING ORDER BY (last_updated_at DESC, thread_id DESC);

-- Counter Table for Instant Folder/Label Badge Updates
CREATE TABLE mail_store.folder_counters (
    user_id          uuid,
    folder_id        text,
    total_count      counter,
    unread_count     counter,
    PRIMARY KEY (user_id, folder_id)
);
```

### JMAP API Contract (RFC 8620 / RFC 8621)

Modern clients interact with the mail service via lightweight, batched JSON-RPC over HTTP/3:

```json
// POST /jmap/api
// Batched Client Request: Fetch changes since state token 's3109', then fetch metadata for new messages
{
  "using": [
    "urn:ietf:params:jmap:core",
    "urn:ietf:params:jmap:mail"
  ],
  "methodCalls": [
    [
      "Email/changes",
      {
        "accountId": "acc_8812a",
        "sinceState": "s3109",
        "maxChanges": 50
      },
      "call-0"
    ],
    [
      "Email/get",
      {
        "accountId": "acc_8812a",
        "#ids": {
          "resultOf": "call-0",
          "name": "Email/changes",
          "path": "/created"
        },
        "properties": ["id", "threadId", "from", "subject", "receivedAt", "keywords", "hasAttachment", "preview"]
      },
      "call-1"
    ]
  ]
}
```

```json
// Server Response (HTTP 200 OK)
{
  "methodResponses": [
    [
      "Email/changes",
      {
        "accountId": "acc_8812a",
        "oldState": "s3109",
        "newState": "s3110",
        "hasMoreChanges": false,
        "created": ["e_991823"],
        "updated": [],
        "destroyed": []
      },
      "call-0"
    ],
    [
      "Email/get",
      {
        "accountId": "acc_8812a",
        "state": "s3110",
        "list": [
          {
            "id": "e_991823",
            "threadId": "t_409112",
            "from": [{"name": "Jane Doe", "email": "jane@enterprise.org"}],
            "subject": "Q3 Cloud Migration Plan",
            "receivedAt": "2026-04-04T14:20:00Z",
            "keywords": {"$seen": true},
            "hasAttachment": true,
            "preview": "Attached is the final Terraform architectural review..."
          }
        ]
      },
      "call-1"
    ]
  ],
  "sessionState": "session_9921"
}
```

---

## Failure Modes, Resilience & Anti-Patterns

| Failure Mode | Root Cause | Catastrophic Impact | Staff-Level Engineering Mitigation |
| :--- | :--- | :--- | :--- |
| **Cassandra Tombstone Storm** | Bulk deletes of emails (e.g., user empties Spam folder containing 20,000 emails). | Queries iterating over tombstones trigger JVM GC pauses, Cassandra `TombstoneOverwhelmingException`, and node crashes. | **1.** Never execute raw CQL `DELETE` on thousands of rows. Instead, write a folder redirect marker or drop entire monthly partitions.<br/>**2.** Enforce strict TWCS TTL drop policies.<br/>**3.** Cap maximum scanned tombstones at $1,000$ per query before throwing a guarded error. |
| **SPF DNS Lookup DOS (10-Lookup Limit)** | Malicious email headers chaining nested SPF `include:` statements. | Receiver MTA blocks or throws timeout errors trying to resolve recursive DNS records. | RFC 7208 mandates a strict hard ceiling of **at most 10 DNS lookups** per SPF validation. Any check exceeding 10 lookups immediately fails with `PermError`. |
| **Attachment Bomb (Zip Decompression Bomb)** | Malicious sender sends a $1\text{ MB}$ compressed archive that inflates to $1\text{ TB}$. | Processing worker runs out of disk/memory (`OOMKill`), choking the MIME ingestion queue. | Stream decompression with a hard `CountingInputStream` limit. Abort and quarantine any attachment whose uncompressed stream exceeds $100\text{ MB}$ or has an inflation ratio $> 100:1$. |
| **Outbound IP Blacklisting via Rogue Tenant** | A compromised tenant account blasts phishing or marketing spam from a shared MTA IP. | Spamhaus lists the entire `/24` subnet. Transactional password reset emails for millions of users bounce globally. | **1.** Strict outbound rate limits per user ($500\text{ msgs/day}$ standard).<br/>**2.** Real-time pre-send outbound spam classification.<br/>**3.** Strict tenant IP pool isolation (Tier 1 transactional completely separated from Tier 2 user mail). |
| **Search Scatter-Gather Collapse** | Elastic search cluster configured without routing keys. | Every single search query fans out to 500 shards across 100 data nodes. Cluster CPU hits $100\%$, dropping all ingestion. | Enforce `routing=user_id` on all Elasticsearch read and write paths. Queries are physically bound to a single shard. |

---

## Operational SRE War Stories

### War Story 1: The Cascading Spamhaus Blacklist Catastrophe

**Context**: During a holiday weekend, an automated credential stuffing attack compromised 4,500 enterprise accounts on the email platform. The compromised accounts were instantly weaponized by a botnet to blast $12\text{ Million}$ cryptocurrency phishing emails within a 2-hour window.

**Incident**: The platform's legacy outbound MTA architecture utilized a single shared pool of 64 public IPv4 addresses. Within 45 minutes, Spamhaus placed all 64 IPs on the **SBL (Spamhaus Block List)**. Immediately, Google, Microsoft 365, and Yahoo rejected $100\%$ of outbound emails originating from the service with `550 5.7.1 Service unavailable`. Over $40\text{ Million}$ legitimate user messages, including mission-critical medical alerts and enterprise password reset tokens, were dead-lettered or queued in retry loops.

**Root Cause**: Lack of outbound tenant tiering and absence of real-time outbound anomaly detection.

**Mitigation & Permanent Fix**:
1. *Emergency Response*: Provisioned a secondary pre-warmed subnet (/26) reserved strictly for emergency Tier 1 transactional delivery, updating DNS SPF records immediately.
2. *Real-Time Outbound Tripwires*: Deployed an Apache Flink sliding window job monitoring per-account outbound rate spikes. Accounts exceeding $50\text{ emails/minute}$ or generating $> 5\%$ bounce rates are instantly routed to an isolated sandbox queue for human/ML security review.
3. *Cryptographic Tenant Sandboxing*: Enforced strict physical IP pool segregation: Tier 1 (Transactional), Tier 2 (Standard User), and Tier 3 (High-Risk/Bulk).

### War Story 2: The Midnight Cassandra Tombstone Meltdown

**Context**: A daily cron job was scheduled at 00:00 UTC to purge expired emails from user "Spam" and "Trash" folders older than 30 days.

**Incident**: At 00:05 UTC, read latency for inbox loading spiked from $80\text{ms}$ to $> 15\text{ seconds}$. Thousands of ScyllaDB/Cassandra nodes experienced severe CPU saturation. Applications threw `ReadTimeoutException` and `TombstoneOverwhelmingException` (exceeding the $100,000$ tombstone scanning threshold). The web interface became completely unresponsive for $30\text{ Million}$ active users.

**Root Cause**: The purge script executed individual CQL `DELETE FROM emails_by_folder WHERE user_id = ? AND email_id = ?`. In LSM-tree storage engines, a `DELETE` does not free disk space; it writes a **tombstone marker**. When users navigated their mailboxes, the clustering key range scan had to read tens of thousands of tombstones per folder to find surviving active emails, saturating memory and crashing read paths.

**Mitigation & Architectural Redesign**:
1. *Banned Bulk Deletes*: Purged the batch deletion script entirely.
2. *Adoption of TWCS & Temporal Partitions*: Partitioned the tables with `year_month`. Deletion of expired Spam and Trash was converted to dropping entire monthly tables or using TWCS with TTL. When an SSTable's data exceeds TTL under TWCS, ScyllaDB drops the entire SSTable file at the OS filesystem level without generating a single tombstone.
3. *Soft-Delete Folder Markers*: When a user clicks "Empty Trash", the system updates a single pointer `trash_cleared_timestamp` in the user's folder metadata table. Any email with `internal_date < trash_cleared_timestamp` is filtered out in memory during result hydration.

---

## Staff-Level Interview Follow-Up Questions

### 1. How would you support End-to-End Encryption (E2EE) like PGP or S/MIME without breaking full-text search?

In a true E2EE model, the client encrypts the message body and attachments using the recipient's public key (e.g., OpenPGP or S/MIME). The server operates purely as a blind bit-pipe and cannot read plaintext headers or content. This fundamentally breaks server-side spam classification and server-side inverted index search.

**Architectural Solutions**:
1. **Client-Side Lucene / SQLite Indexing**: Modern mobile and desktop clients maintain a local encrypted SQLite database or embedded Lucene engine (e.g., SQLite FTS5) indexing decrypted messages locally on the user's device. Search runs $100\%$ on-device.
2. **Searchable Symmetric Encryption (SSE) / Blind Indexing**: The client tokenizes the message into search terms, hashes each term with an HMAC using a secret client-side key ($\text{token} = \text{HMAC}(k_{\text{search}}, \text{word})$), and sends the hashed tokens to the server. The server indexes the hashes into Elasticsearch. When the user searches for `"tax"`, the client sends $\text{HMAC}(k_{\text{search}}, \text{"tax"})$, allowing the server to locate matching document IDs without ever learning the underlying plaintext word.

### 2. How do you implement an "Undo Send" feature in a distributed email system?

Once an email is dispatched via SMTP to an external mail exchanger (e.g., Google or Yahoo), it is physically impossible to "recall" or cancel it under RFC 5321 rules.

**The Implementation**:
- "Undo Send" is an engineered **optical illusion implemented as delayed queue dispatch**:
  1. When the user clicks "Send", the client issues a `POST /emails/send?hold_delay=15s`.
  2. The gateway assigns the message an `email_id` and writes it to Kafka with a scheduled delivery timestamp:
     $$\text{dispatch\_time} = \text{now}() + 15\text{ seconds}$$
  3. The message state in the user's outbox is marked as `PENDING_DISPATCH`.
  4. If the user clicks "Undo" within 15 seconds, the client calls `DELETE /emails/pending/{email_id}`. The gateway marks the `email_id` as revoked in Redis.
  5. When the Kafka delay consumer processes the message at $t + 15\text{s}$, it checks Redis. If revoked, it drops the message. If clean, it releases the email to the outbound SMTP dispatch queue.

### 3. How do you ensure exactly-once email delivery across distributed microservices?

SMTP and network communications are inherently **at-least-once**. True "exactly-once" delivery across independent third-party mail servers is impossible under the Two Generals' Problem. However, within our internal boundary, we enforce end-to-end idempotency:

1. **Idempotency Keys on Submission**: Clients generate a deterministic UUIDv5 based on client device ID, local draft ID, and sent timestamp. The API gateway checks Redis via a `SET NX PX 86400000` distributed lock. Duplicate submissions return the existing `202 Accepted` response.
2. **Message-ID Deduplication on Ingress**: When receiving inbound emails over SMTP, the MTA extracts the RFC 5322 `Message-ID` header. If the sender MTA retries delivery of the same message, the processing pipeline checks a Bloom Filter backed by ScyllaDB. If the `Message-ID` was successfully committed within the last 7 days, the server immediately returns `250 OK` without triggering duplicate downstream indexing, push notifications, or storage allocation.

### 4. How would you architect real-time conversation threading across multi-party replies?

Conversation threading (grouping related emails into a single chronological view) relies on RFC 5322 header graph construction:
1. **Header Parsing**:
   - `Message-ID`: Globally unique identifier for the current email (e.g., `<abc.123@domain.com>`).
   - `In-Reply-To`: The `Message-ID` of the immediate parent message being replied to.
   - `References`: An ordered list of all ancestor `Message-ID`s in the conversation chain.
2. **Graph Matching Algorithm**:
   - The threading worker queries ScyllaDB for existing thread mappings using the IDs in `References` and `In-Reply-To`.
   - If an ancestor matches an existing `thread_id`, the new email is assigned to that thread.
   - If no ancestor matches, a new `thread_id` is generated.
3. **Subject-Line Normalization Fallback**:
   - When poorly implemented email clients strip `References` headers, the system applies a fallback heuristic: normalize the subject by stripping recursive prefixes (`Re:`, `Fwd:`, `[EXTERNAL]`, `[Ticket #1234]`) and match against active threads involving the identical participant set within the past 14 days.

---

## Architectural Verification Dashboard

```
[System Design Standard: Alex Xu Vol 2 - Level 4 Staff Blueprint]
├── Scale Verification: 1B Accounts, 250M DAU, 12.5B msgs/day, 1.5 PB/day
├── Edge & Gateways: L4 Anycast, Stateless JMAP over HTTP/3, C10M IMAP Proxy
├── Cryptographic Pipeline: RFC 7208 SPF, RFC 6376 DKIM, RFC 7489 DMARC, RFC 8617 ARC
├── Storage Micro-Architecture: ScyllaDB Composite Bucketed Keyspace, TWCS SSTable Dropping
├── Deduplication Engine: Content-Addressed SHA-256 S3 Attachment Single-Instance Storage
├── Full-Text Search: Elasticsearch Multi-Tenant Shared Shards with routing=user_id
├── Outbound MTA Subsystem: Tiered IP Pools, Reputation Warmup, Jittered Exponential Backoff
└── Operational Verification: 6 / 6 Mermaid Diagrams Validated (HTTP 200 via mermaid.ink)
```
