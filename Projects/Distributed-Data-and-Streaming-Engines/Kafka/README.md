# Domain I: Apache Kafka — Distributed Commit Log & Event Streaming

[![Kafka Version](https://img.shields.io/badge/Apache%20Kafka-3.x%20KRaft%20Ready-black.svg)](https://kafka.apache.org/)
[![Canonical Book](https://img.shields.io/badge/Canonical%20Book-Kafka%3A%20The%20Definitive%20Guide%20(Shapira)-darkblue.svg)](https://www.oreilly.com/library/view/kafka-the-definitive/9781492043072/)
[![Companion Book](https://img.shields.io/badge/Companion%20Book-Designing%20Event--Driven%20Systems%20(Stopford)-orange.svg)](https://www.confluent.io/designing-event-driven-systems/)

> *"At its core, Kafka is designed as a distributed, partitioned, replicated commit log service. By treating data as an immutable stream of ordered events and exploiting the operating system's page cache and zero-copy network transfer, Kafka achieves millions of messages per second with constant time $O(1)$ disk performance."*  
> — **Gwen Shapira, Todd Palino et al.**, authors of *Kafka: The Definitive Guide*

---

## 🏛️ Kafka Architectural Philosophy & Storage Blueprint

Unlike traditional enterprise message brokers (RabbitMQ, ActiveMQ) that track per-message acknowledgments and delete messages upon consumption, Kafka is an **immutable, append-only commit log**:
* Messages are written sequentially to the end of a partition log on physical disk.
* Consumers are passive and pull-based, maintaining their own read cursor (**Offset**).
* Reading messages does not mutate or lock the log; multiple independent consumer groups can read the exact same topic at different speeds without interfering with one another.

```
                    The Distributed Commit Log Architecture

Topic: 'orders' (Partition 0 Log File on Disk)
+--------+--------+--------+--------+--------+--------+--------+--------+
| Off 0  | Off 1  | Off 2  | Off 3  | Off 4  | Off 5  | Off 6  | Off 7  |  (Append-Only!)
+--------+--------+--------+--------+--------+--------+--------+--------+
    ^                          ^                                   ^
    |                          |                                   |
Consumer Group A           Consumer Group B                    Producer
(Fraud Detection)          (Data Lake Ingestion)               (Appends new events)
Offset: 0                  Offset: 3                           Log End Offset (LEO): 8
```

### The Zero-Copy Kernel Bypass (`sendfile`)
In traditional brokers, reading data from disk to network involves 4 context switches and 4 data copies across user-space buffers.  
Kafka uses the Linux **`sendfile()` system call** (Zero-Copy):
1. Data is transferred directly from the OS Page Cache to the Network Interface Card (NIC) buffer via DMA (Direct Memory Access).
2. The data **never enters JVM user-space memory**, completely eliminating garbage collection overhead!

```
                  Traditional I/O vs Linux Zero-Copy (sendfile)

Traditional Read/Send (4 Copies, 4 Context Switches):
Disk -> OS Page Cache -> JVM Heap Buffer -> OS Socket Buffer -> NIC Buffer

Linux Zero-Copy sendfile() (2 DMA Copies, 2 Context Switches, ZERO JVM Overhead!):
Disk -> OS Page Cache =================(Direct DMA)=================> NIC Buffer
```

---

## 📚 Master Chapter Index

| Chapter | Title | Core Canonical Concepts & Focus | Canonical Literature Focus |
| :---: | :--- | :--- | :--- |
| **01** | [Commit Log Architecture, Zero-Copy IO & Storage Layout](01.%20Commit%20Log%20Architecture%2C%20Zero-Copy%20IO%20%26%20Storage%20Layout.md) | Append-only write-ahead log, Page Cache, Linux `sendfile()`, avoiding GC | *Kafka: The Definitive Guide* Ch. 1 & 7 |
| **02** | [Topics, Partitions, Segment Indexing & Log Compaction](02.%20Topics%2C%20Partitions%2C%20Segment%20Indexing%20%26%20Log%20Compaction.md) | Log segments (`.log`), sparse memory-mapped `.index` files, log compaction | *Kafka: The Definitive Guide* Ch. 7 |
| **03** | [Producer Architecture, Buffering, Partitioner & Acks](03.%20Producer%20Architecture%2C%20Buffering%2C%20Partitioner%20%26%20Acks.md) | `RecordAccumulator`, memory pools, batch compression, `acks=all`, idempotence | *Kafka: The Definitive Guide* Ch. 3 |
| **04** | [Consumer Groups, Offset Commit Mechanics & Rebalancing](04.%20Consumer%20Groups%2C%20Offset%20Commit%20Mechanics%20%26%20Rebalancing.md) | Group coordinator, `__consumer_offsets`, Cooperative Sticky Rebalance | *Kafka: The Definitive Guide* Ch. 4 |
| **05** | [Replication, ISR, High Watermark & KRaft Consensus](05.%20Replication%2C%20ISR%2C%20High%20Watermark%20%26%20KRaft%20Consensus.md) | Leader/Follower replicas, In-Sync Replicas (ISR), High Watermark, KRaft quorum | *Kafka: The Definitive Guide* Ch. 6 |
| **06** | [Exactly-Once Semantics (EOS) & Transactional Messaging](06.%20Exactly-Once%20Semantics%20%28EOS%29%20%26%20Transactional%20Messaging.md) | Two-Phase Commit over Transaction Coordinator, `ProducerId`, transactional markers | *Designing Event-Driven Systems* Ch. 11 |
| **07** | [Kafka Streams, KTables & State Store Architecture](07.%20Kafka%20Streams%2C%20KTables%20%26%20State%20Store%20Architecture.md) | Stream-table duality, `KStream` vs `KTable`, local RocksDB state, changelog topics | *Kafka: The Definitive Guide* Ch. 11 |
| **08** | [Schema Registry, Avro-Protobuf & Schema Evolution](08.%20Schema%20Registry%2C%20Avro-Protobuf%20%26%20Schema%20Evolution.md) | Confluent Schema Registry, magic byte framing, Backward/Forward compatibility | *Kafka: The Definitive Guide* Ch. 3.4 |
| **09** | [Kafka Connect, Source-Sink Architecture & SMTs](09.%20Kafka%20Connect%2C%20Source-Sink%20Architecture%20%26%20SMTs.md) | Connect distributed workers, Tasks, Converters, Single Message Transforms (SMTs) | *Kafka: The Definitive Guide* Ch. 8 |
| **10** | [Kafka Operations, Monitoring, Lag & Troubleshooting](10.%20Kafka%20Operations%2C%20Monitoring%2C%20Lag%20%26%20Troubleshooting.md) | Broker thread pools, consumer lag monitoring, Under-Replicated Partitions (URP) | *Kafka: The Definitive Guide* Ch. 10 |

---

## ⚡ Quick Reference: Production Kafka Configurations

### 1. Zero-Data-Loss Producer Configuration
```properties
acks=all
enable.idempotence=true
max.in.flight.requests.per.connection=5
retries=2147483647
compression.type=zstd
linger.ms=20
batch.size=65536
```

### 2. High-Availability Topic Configuration
```properties
min.insync.replicas=2
replication.factor=3
unclean.leader.election.enable=false
```
