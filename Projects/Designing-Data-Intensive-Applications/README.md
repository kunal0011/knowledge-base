# Designing Data-Intensive Applications (DDIA) Master Portal

> "Data is at the center of many challenges in system design today. Difficult issues need to be figured out, such as scalability, consistency, reliability, efficiency, and maintainability. In addition, we have an overwhelming variety of tools to choose from... We need to figure out what each tool is good for, and how to combine them to form a cohesive system."  
> — *Martin Kleppmann, Designing Data-Intensive Applications (O'Reilly)*

---

## 🏛️ Curriculum Architecture & Systems Map

This section is an authoritative, publication-grade deep-dive into the foundational principles, data models, distributed algorithms, and consensus protocols expounded in Martin Kleppmann's seminal work: **"Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems"**.

```
Projects/Designing-Data-Intensive-Applications/
├── README.md                                                 # Master Portal & Cross-Paradigm Architectural Matrix
│
├── Part-1-Foundations-of-Data-Systems/                       # Single-Node Storage, Formats & Foundational Bounds
│   ├── 01. Reliability, Scalability & Maintainability.md     # Faults vs Failures, Percentiles, SLOs, Operability
│   ├── 02. Data Models & Query Languages.md                  # Relational vs Document vs Graph (Cypher, SPARQL)
│   ├── 03. Storage & Retrieval - Engines, Indexes & Compaction.md # LSM-Trees, B-Trees, OLTP vs OLAP, Columnar
│   └── 04. Encoding & Evolution - Formats & Protocols.md     # Avro, Protobuf, Thrift, Schema Evolution, RPC
│
├── Part-2-Distributed-Data/                                  # Distributed Systems, Replication, Transactions & Consensus
│   ├── 05. Replication - Single-Leader, Multi-Leader & Quorums.md # Replication Lag, Read-Your-Writes, Dynamo Quorums
│   ├── 06. Partitioning - Strategies, Secondary Indexes & Skew.md # Hash vs Range, Document vs Term Indexes, Rebalancing
│   ├── 07. Transactions - ACID, Isolation Levels & Anomalies.md   # Dirty Reads, Read Skew, MVCC, Write Skew, 2PL, SSI
│   ├── 08. The Trouble with Distributed Systems.md           # Unreliable Networks, Clock Drift, Byzantine Faults, System Models
│   └── 09. Consistency & Consensus - Linearizability to Raft.md   # Linearizability, Total Order, 2PC, Paxos/Raft/Zab
│
└── Part-3-Derived-Data/                                      # Batch Computing, Stream Processing & Modern Unbundling
    ├── 10. Batch Processing - MapReduce, Dataflow & Joins.md # Unix Philosophy, MapReduce, Spark/Flink DAGs, Distributed Joins
    ├── 11. Stream Processing - Event Sourcing, CDC & State.md # Partitioned Logs, CDC vs Dual Writes, Stream Joins, Watermarks
    └── 12. The Future of Data Systems - Unbundled Databases.md# Unbundling, Lambda vs Kappa, Correctness, End-to-End Integrity
```

---

## 📊 Comprehensive Comparative Technology Matrix

| Dimension | Relational (OLTP) | Document / Key-Value | Column-Oriented (OLAP) | Distributed Stream / Log |
| :--- | :--- | :--- | :--- | :--- |
| **Exemplars** | PostgreSQL, MySQL, Oracle | MongoDB, Cassandra, DynamoDB | ClickHouse, Snowflake, DuckDB | Apache Kafka, Apache Pulsar |
| **Primary Data Model** | Relations, normalized tuples | JSON/BSON, Wide-column key-val | Parquet columnar vectors, arrow | Append-only byte offset stream |
| **Storage Structure** | B+Tree (4KB–8KB pages, in-place) | LSM-Tree (MemTable + SSTables) | Column chunks, RLE, Bitmaps | Sequential log segments on disk |
| **Write Path Latency** | $O(\log N)$ disk page overwrite + WAL | $O(1)$ memory write + sequential WAL | Column block bulk flush / merge | $O(1)$ sequential OS page cache hit |
| **Read Access Pattern** | Point lookups, small range scans | Key lookups, document traversals | Aggregations over subset of columns | Sequential offset tailing via DMA |
| **Concurrency Control** | MVCC + 2PL / SSI Locks | Row-level atomic LWW / Paxos | Read-only snapshot views | Partition-isolated single thread |
| **Primary Bottleneck** | Random disk I/O, lock contention | Compaction amplification, stale reads | CPU memory bandwidth, decompression | Network interface card (NIC) bandwidth |

---

## 🗺️ The DDIA Systems Spectrum

```
 [ Part I: Single-Node Foundations ]
        │
        ├── Storage Engine Anatomy: Memory (Memtable) vs Disk (SSTable / B-Tree Page)
        └── Wire Protocol Serialization: Binary tag schemas (Protobuf/Avro) vs Text JSON
        │
        ▼
 [ Part II: Distributed Realities & Fault Tolerance ]
        │
        ├── Replication: Trade-offs between Consistency (Linearizability) & Latency
        ├── Partitioning: Distributing load across shards without hotspot skew
        ├── Concurrency: Preventing Race Conditions (Write Skew, Phantoms, Lost Updates)
        └── Consensus: Reaching mathematical agreement across unreliable networks
        │
        ▼
 [ Part III: Derived Data & Unbundled Architecture ]
        │
        ├── Batch Processing: Transforming bounded historical records deterministically
        ├── Stream Processing: Transforming unbounded real-time event logs continuously
        └── The Unbundled Database: Composing heterogeneous storage engines into a cohesive whole
```

---

## 📚 12-Chapter Curriculum Index

### [Part I: Foundations of Data Systems](Part-1-Foundations-of-Data-Systems/)
1. **[01. Reliability, Scalability & Maintainability](Part-1-Foundations-of-Data-Systems/01.%20Reliability,%20Scalability%20&%20Maintainability.md)**: Faults vs Failures, SLAs, tail latencies, percentiles ($p95, p99, p99.9$), Head-of-Line blocking, operability and simplicity.
2. **[02. Data Models & Query Languages](Part-1-Foundations-of-Data-Systems/02.%20Data%20Models%20&%20Query%20Languages.md)**: Relational model vs Document model vs Graph models (Property Graphs, Neo4j, Cypher, RDF, SPARQL, Datalog).
3. **[03. Storage & Retrieval - Engines, Indexes & Compaction](Part-1-Foundations-of-Data-Systems/03.%20Storage%20&%20Retrieval%20-%20Engines,%20Indexes%20&%20Compaction.md)**: LSM-trees, SSTables, B-Trees, Write-Ahead Logs, OLTP vs OLAP, Columnar storage, Bitmaps, Run-Length Encoding.
4. **[04. Encoding & Evolution - Formats & Protocols](Part-1-Foundations-of-Data-Systems/04.%20Encoding%20&%20Evolution%20-%20Formats%20&%20Protocols.md)**: JSON/XML vs Protocol Buffers, Thrift, Avro; backward/forward compatibility, RPC vs REST vs asynchronous messaging.

### [Part II: Distributed Data](Part-2-Distributed-Data/)
5. **[05. Replication - Single-Leader, Multi-Leader & Quorums](Part-2-Distributed-Data/05.%20Replication%20-%20Single-Leader,%20Multi-Leader%20&%20Quorums.md)**: Single-leader, multi-leader, leaderless (Dynamo), replication lag anomalies (read-your-writes, monotonic reads), quorums ($W + R > N$).
6. **[06. Partitioning - Strategies, Secondary Indexes & Skew](Part-2-Distributed-Data/06.%20Partitioning%20-%20Strategies,%20Secondary%20Indexes%20&%20Skew.md)**: Hash vs range partitioning, hotspot skew, secondary indexes (document-partitioned local vs term-partitioned global), rebalancing.
7. **[07. Transactions - ACID, Isolation Levels & Anomalies](Part-2-Distributed-Data/07.%20Transactions%20-%20ACID,%20Isolation%20Levels%20&%20Anomalies.md)**: ACID definition, dirty reads, dirty writes, read skew (Snapshot Isolation / MVCC), lost updates, write skew, phantoms, 2PL, SSI.
8. **[08. The Trouble with Distributed Systems](Part-2-Distributed-Data/08.%20The%20Trouble%20with%20Distributed%20Systems.md)**: Unreliable networks, clock drift, NTP leap seconds, TrueTime API, process pauses (GC), fencing tokens, Byzantine fault tolerance.
9. **[09. Consistency & Consensus - Linearizability to Raft](Part-2-Distributed-Data/09.%20Consistency%20&%20Consensus%20-%20Linearizability%20to%20Raft.md)**: Linearizability vs Serializability, Total Order Broadcast, Two-Phase Commit (2PC), Paxos, Raft, Zab, distributed leases.

### [Part III: Derived Data](Part-3-Derived-Data/)
10. **[10. Batch Processing - MapReduce, Dataflow & Joins](Part-3-Derived-Data/10.%20Batch%20Processing%20-%20MapReduce,%20Dataflow%20&%20Joins.md)**: Unix philosophy, MapReduce execution model, distributed joins (sort-merge, broadcast hash, partitioned hash), Spark/Flink DAG dataflows.
11. **[11. Stream Processing - Event Sourcing, CDC & State](Part-3-Derived-Data/11.%20Stream%20Processing%20-%20Event%20Sourcing,%20CDC%20&%20State.md)**: Message brokers vs partitioned logs, dual writes vs log-based CDC, event sourcing, stream joins, watermarks, fault tolerance.
12. **[12. The Future of Data Systems - Unbundled Databases](Part-3-Derived-Data/12.%20The%20Future%20of%20Data%20Systems%20-%20Unbundled%20Databases.md)**: Unbundling databases, composing heterogeneous engines, end-to-end correctness, verifiable computation, audit logs, ethics and privacy.
