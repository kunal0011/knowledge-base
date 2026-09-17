# Big Data Engineering & Distributed Systems: Master Portal

> "Big data is not merely about volume; it is about building reliable, scalable, and maintainable systems from unreliable, commodity machines. It is the science of decomposing massive computations across clusters while preserving consistency, fault tolerance, and low latency."  
> — *Martin Kleppmann, Designing Data-Intensive Applications (DDIA)*

---

## 🐘 1. The Big Data Paradigm: The 5 Vs & Architectural Evolution

Traditional Relational Database Management Systems (RDBMS) scale **vertically** (buying bigger servers with more CPU and RAM). When datasets reached petabytes and streaming velocities reached millions of events per second, vertical scaling hit a physical and economic wall. 

**Big Data Engineering** scales **horizontally**: clustering hundreds or thousands of commodity servers over high-speed networks.

```text
===================================================================================================
                                       THE 5 Vs OF BIG DATA
===================================================================================================

  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. VOLUME:      Terabytes to Exabytes of data exceeding single-node disk capacity.          │
  │ 2. VELOCITY:    Real-time streaming ingestion at millions of events per second.             │
  │ 3. VARIETY:     Structured (tabular), Semi-Structured (JSON/Avro), Unstructured (video/logs)│
  │ 4. VERACITY:    Data quality, anomalies, skew, out-of-order event arrivals, and noise.       │
  │ 5. VALUE:       Transforming raw distributed data into actionable business intelligence.    │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ 2. Architectural Evolution: Lambda, Kappa & The Modern Lakehouse

Over two decades, big data architectures evolved through three major paradigms:

```text
===================================================================================================
                           ARCHITECTURAL EVOLUTION: LAMBDA TO LAKEHOUSE
===================================================================================================

 [ 1. The Lambda Architecture (Batch + Speed Layers) ]
 Real-Time Ingress ──┬──► Batch Layer (HDFS / MapReduce: Immutable Raw Log, Slow, Complete) ──┐
                     │                                                                          ▼
                     └──► Speed Layer (Storm / Spark Streaming: Low-latency Delta, Incomplete) ─► Serving
 ⚠️ Flaw: Dual-codebase nightmare! Business logic must be written and maintained twice!

 [ 2. The Kappa Architecture (Pure Stream Processing) ]
 Real-Time Ingress ──► Kafka / Redpanda ──► Apache Flink (Single Stream Engine) ──► Serving Layer
 🌟 Innovation: Everything is a stream! Batch processing is just streaming over bounded historical data.

 [ 3. The Modern Lakehouse Architecture (Decoupled Compute & Open Table Formats) ]
 Unified Ingestion (Batch & Streaming)
                  │
                  ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │                           OPEN TABLE FORMATS (ACID Transaction Layer)                       │
  │                       Apache Iceberg  │  Delta Lake  │  Apache Hudi                         │
  └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                 │
  ┌──────────────────────────────────────────────┴──────────────────────────────────────────────┐
  │                        COLUMNAR STORAGE (High-Efficiency File Formats)                      │
  │                          Apache Parquet  │  Apache ORC  │  Apache Avro                      │
  └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                 │
  ┌──────────────────────────────────────────────┴──────────────────────────────────────────────┐
  │                     OBJECT STORAGE & DISTRIBUTED FILESYSTEMS (Scale-Out)                    │
  │                         AWS S3  │  Google Cloud Storage  │  Azure ADLS  │  HDFS             │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Master Curriculum Index: Big Data Technologies

This 10-chapter curriculum synthesizes canonical knowledge from *Hadoop: The Definitive Guide* (Tom White), *Designing Data-Intensive Applications* (Martin Kleppmann), and *Learning Spark* (Jules Damji et al.):

| Module | Chapter Title | Core Theoretical & Engineering Foundations |
| :--- | :--- | :--- |
| **01** | [Foundations of Big Data - 5 Vs & Compute-Storage Decoupling](./01.%20Foundations%20of%20Big%20Data%20-%20The%205%20Vs%2C%20Distributed%20Systems%20%26%20Compute-Storage%20Decoupling.md) | SMP vs MPP scale-out, Amdahl's vs Gustafson's Law, network bisection bandwidth, and the historic shift from co-located Hadoop clusters to disaggregated cloud object stores. |
| **02** | [Distributed Storage Systems - HDFS Internals & Architecture](./02.%20Distributed%20Storage%20Systems%20-%20HDFS%20Internals%20%26%20Architecture.md) | NameNode (EditLog, FsImage, QJM High Availability), DataNode block reports, $3\times$ rack-aware replica placement, Short-Circuit reads, Erasure Coding ($6+3$), and the Small Files problem. |
| **03** | [Serialization & Row-Oriented Storage - Apache Avro & Protobuf](./03.%20Serialization%20%26%20Row-Oriented%20Storage%20-%20Apache%20Avro%2C%20Thrift%20%26%20Protocol%20Buffers.md) | Row vs columnar formats, Avro binary encoding (zigzag varints), container files with embedded schemas, Schema Evolution (Backward, Forward, Full), and Schema Registry. |
| **04** | [Columnar Storage Architecture - Apache Parquet & ORC Deep Dive](./04.%20Columnar%20Storage%20Architecture%20-%20Apache%20Parquet%20%26%20ORC%20Deep%20Dive.md) | OLTP vs OLAP physical access patterns, Parquet file layout (Row Groups, Column Chunks, Pages), Dictionary/RLE/Bit-packing encodings, and Predicate Pushdown (Min/Max row group skipping). |
| **05** | [Distributed Resource Management - Apache YARN & Kubernetes](./05.%20Distributed%20Resource%20Management%20%26%20Scheduling%20-%20Apache%20YARN%20%26%20Kubernetes.md) | YARN architecture (ResourceManager, NodeManager, ApplicationMaster), Fair vs Capacity schedulers, Dominant Resource Fairness (DRF), and running Spark natively on Kubernetes. |
| **06** | [Distributed Batch Processing - MapReduce & Apache Spark Internals](./06.%20Distributed%20Batch%20Processing%20-%20MapReduce%20Foundations%20%26%20Apache%20Spark%20Internals.md) | MapReduce shuffle mechanics, Spark RDD DAG execution, Narrow vs Wide dependencies, Shuffle spills, Catalyst query optimizer (Logical/Physical plans), and Project Tungsten whole-stage code generation. |
| **07** | [Real-Time Stream Processing - Concepts, Semantics & Apache Flink](./07.%20Real-Time%20Stream%20Processing%20-%20Concepts%2C%20Semantics%20%26%20Apache%20Flink.md) | The Streaming Framework (*What, Where, When, How*), Tumbling/Sliding/Session windows, Event Time vs Processing Time, Watermarks, Exactly-Once semantics via Chandy-Lamport snapshots, and RocksDB state backends. |
| **08** | [Distributed Message Brokers - Apache Kafka & Log-Centric Systems](./08.%20Distributed%20Message%20Brokers%20%26%20Log-Centric%20Architecture%20-%20Apache%20Kafka.md) | Distributed commit log abstraction, partition replication (Leader, Follower, ISR), zero-copy kernel transfers (`sendfile`), idempotent & transactional producers, and consumer group rebalancing. |
| **09** | [Big Data Query Engines & Distributed SQL - Trino, Presto & Hive](./09.%20Big%20Data%20Query%20Engines%20%26%20Distributed%20SQL%20-%20Trino%2C%20Presto%20%26%20Hive.md) | Evolution from Hive batch queries to MPP in-memory execution, Trino/Presto Coordinator-Worker architecture, pipelined data exchange, Dynamic Filtering, and multi-source federated queries. |
| **10** | [Modern Data Lakehouse & Table Formats - Iceberg, Delta & Hudi](./10.%20Modern%20Data%20Lakehouse%20%26%20Table%20Formats%20-%20Iceberg%2C%20Delta%20Lake%20%26%20Hudi.md) | Why raw object storage fails, ACID transactions on object storage, Apache Iceberg snapshot metadata tree, Delta Lake transaction log, Copy-on-Write vs Merge-on-Read, and Time Travel queries. |

---

## ⚖️ Storage Format Comparison: Row-Oriented vs. Columnar

```text
===================================================================================================
                       ROW-ORIENTED (AVRO) VS. COLUMNAR (PARQUET) STORAGE
===================================================================================================

  [ Logical Table Data ]
  Row 1: [ ID: 101, Name: "Alice", Age: 30, Salary: 150000 ]
  Row 2: [ ID: 102, Name: "Bob",   Age: 25, Salary: 95000  ]
  Row 3: [ ID: 103, Name: "Carol", Age: 42, Salary: 210000 ]

  [ Physical On-Disk Layout: Row-Oriented (Apache Avro) ]
  [ 101, "Alice", 30, 150000 ] [ 102, "Bob", 25, 95000 ] [ 103, "Carol", 42, 210000 ]
  🌟 Best for: OLTP, streaming message ingestion, Kafka event serialization, full-record writes.

  [ Physical On-Disk Layout: Columnar (Apache Parquet / ORC) ]
  IDs:      [ 101, 102, 103 ]
  Names:    [ "Alice", "Bob", "Carol" ]
  Ages:     [ 30, 25, 42 ]
  Salaries: [ 150000, 95000, 210000 ]
  🌟 Best for: OLAP, analytics queries ("SELECT AVG(Salary) WHERE Age > 28").
     Reads ONLY the Age and Salary columns, skipping 90% of disk I/O and achieving 10x compression!
```
