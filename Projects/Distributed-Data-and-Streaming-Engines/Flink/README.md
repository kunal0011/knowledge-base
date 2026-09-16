# Domain II: Apache Flink — Stateful Stream Processing & Event-Driven Engine

[![Flink Version](https://img.shields.io/badge/Apache%20Flink-1.18%2B%20Ready-orange.svg)](https://flink.apache.org/)
[![Canonical Book](https://img.shields.io/badge/Canonical%20Book-Stream%20Processing%20with%20Flink%20(Hueske)-darkred.svg)](https://www.oreilly.com/library/view/stream-processing-with/9781491974285/)
[![Companion Paper](https://img.shields.io/badge/VLDB%20Paper-Lightweight%20Asynchronous%20Snapshots-blue.svg)](https://arxiv.org/abs/1506.08603)

> *"Apache Flink is a framework and distributed processing engine for stateful computations over unbounded and bounded data streams. Flink has been designed to run in all common cluster environments, perform computations at in-memory speed and at any scale."*  
> — **Fabian Hueske & Vasiliki Kalavri**, authors of *Stream Processing with Apache Flink*

---

## 🏛️ Flink Architectural Philosophy & Streamflow Blueprint

Unlike Apache Spark—which historically simulated streaming by slicing unbounded data into discretized micro-batches of RDDs—**Apache Flink is a native, true event-driven stream processor**:
* Every single event is ingested, processed, and emitted **instantaneously (event-at-a-time)** with sub-millisecond latencies.
* Batch processing in Flink is simply treated as a special degenerate case of streaming (a bounded stream with a finite end).
* Stateful operators maintain local state (e.g., in off-heap **Embedded RocksDB**), while continuous distributed consistency is maintained via **Asynchronous Barrier Snapshotting (Chandy-Lamport)**.

```
                    Flink Distributed Cluster Architecture

                       +---------------------------------------+
                       |              JobManager               |
                       |  - Resource Manager (Slot allocation) |
                       |  - Dispatcher (REST interface)        |
                       |  - JobMaster (ExecutionGraph & ABS)   |
                       +-------------------+-------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v (RPC / Heartbeats)                          v
       +--------------------------+                  +--------------------------+
       | TaskManager 1 (Worker)   |                  | TaskManager 2 (Worker)   |
       |  +--------------------+  |                  |  +--------------------+  |
       |  | TaskSlot 1 (Core 1)|  |                  |  | TaskSlot 3 (Core 1)|  |
       |  | - Subtask A [FlatM]|  |                  |  | - Subtask A [FlatM]|  |
       |  | - RocksDB State    |  |                  |  | - RocksDB State    |  |
       |  +--------------------+  |                  |  +--------------------+  |
       |  +--------------------+  |                  |  +--------------------+  |
       |  | TaskSlot 2 (Core 2)|  |                  |  | TaskSlot 4 (Core 2)|  |
       |  | - Subtask B [Window|  |                  |  | - Subtask B [Window|  |
       |  +--------------------+  |                  |  +--------------------+  |
       +--------------------------+                  +--------------------------+
```

### The Dataflow Graph Hierarchy
A user's Flink program is compiled through four progressive representations:
1. **`StreamGraph`**: Client-side graph capturing user transformations.
2. **`JobGraph`**: Client-side optimized graph with adjacent operators fused into **Operator Chains**.
3. **`ExecutionGraph`**: JobManager-side parallel graph with parallel execution tasks.
4. **`Physical Execution Tasks`**: Threads executing inside TaskSlots on TaskManagers.

---

## 📚 Master Chapter Index

| Chapter | Title | Core Canonical Concepts & Focus | Canonical Literature Focus |
| :---: | :--- | :--- | :--- |
| **01** | [Flink Runtime Architecture, TaskManagers & TaskSlots](01.%20Flink%20Runtime%20Architecture%2C%20TaskManagers%20%26%20TaskSlots.md) | JobManager, TaskManagers, TaskSlots, Operator subtasks, Task Chaining | *Stream Processing with Flink* Ch. 3 |
| **02** | [Event Time Semantics, Watermarks & Out-of-Order Data](02.%20Event%20Time%20Semantics%2C%20Watermarks%20%26%20Out-of-Order%20Data.md) | Event Time vs Processing Time, Watermark generators, handling idle sources | *Stream Processing with Flink* Ch. 6 |
| **03** | [Windowing Mechanics, Triggers, Evictors & Side Outputs](03.%20Windowing%20Mechanics%2C%20Triggers%2C%20Evictors%20%26%20Side%20Outputs.md) | Tumbling, Sliding, Session windows, custom Triggers, Evictors, Side Outputs | *Stream Processing with Flink* Ch. 6 |
| **04** | [State Management, State Primitives & RocksDB Backend](04.%20State%20Management%2C%20State%20Primitives%20%26%20RocksDB%20Backend.md) | Keyed State (`ValueState`, `MapState`), Heap backend vs RocksDB off-heap | *Stream Processing with Flink* Ch. 7 |
| **05** | [Checkpoints, Savepoints & Asynchronous Snapshots](05.%20Checkpoints%2C%20Savepoints%20%26%20Asynchronous%20Snapshots.md) | Asynchronous Barrier Snapshotting (Chandy-Lamport), Savepoint upgrades | *Stream Processing with Flink* Ch. 8 |
| **06** | [ProcessFunctions, Timers & Low-Level Stream Processing](06.%20ProcessFunctions%2C%20Timers%20%26%20Low-Level%20Stream%20Processing.md) | `KeyedProcessFunction`, event/processing timers, `onTimer()`, state machines | *Stream Processing with Flink* Ch. 7.4 |
| **07** | [Streaming Joins, Interval Joins & Temporal Tables](07.%20Streaming%20Joins%2C%20Interval%20Joins%20%26%20Temporal%20Tables.md) | Windowed joins, Interval joins within time bounds, Temporal Table lookups | *Stream Processing with Flink* Ch. 6.5 |
| **08** | [Flink SQL, Dynamic Tables & Retraction Changelogs](08.%20Flink%20SQL%2C%20Dynamic%20Tables%20%26%20Retraction%20Changelogs.md) | Dynamic Tables, Continuous Queries, Changelog streams (`+I`, `-U`, `+U`, `-D`) | *Stream Processing with Flink* Ch. 9 |
| **09** | [End-to-End Exactly-Once Processing & Two-Phase Commits](09.%20End-to-End%20Exactly-Once%20Processing%20%26%20Two-Phase%20Commits.md) | `TwoPhaseCommitSinkFunction`, coordinating Sink commits with checkpoint barriers | *Stream Processing with Flink* Ch. 8.4 |
| **10** | [Flink Operations, Backpressure Tuning & Memory Triage](10.%20Flink%20Operations%2C%20Backpressure%20Tuning%20%26%20Memory%20Triage.md) | Credit-Based Flow Control, Web Dashboard backpressure analysis, RocksDB tuning | *Stream Processing with Flink* Ch. 10 |

---

## ⚡ Quick Reference: High-Performance Flink Idioms

### 1. Robust Event-Time Watermarking
```java
WatermarkStrategy<TransactionEvent> watermarkStrategy = WatermarkStrategy
    .<TransactionEvent>forBoundedOutOfOrderness(Duration.ofSeconds(10))
    .withTimestampAssigner((event, timestamp) -> event.getEventTimestamp())
    .withIdleness(Duration.ofMinutes(1)); // Critical: Prevents idle partitions from stalling watermarks!
```

### 2. State Retention Time-To-Live (State TTL)
Prevent unbounded RocksDB state leaks by attaching a TTL to state primitives:
```java
StateTtlConfig ttlConfig = StateTtlConfig
    .newBuilder(Time.days(7))
    .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
    .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
    .cleanupInRocksdbCompactFilter(1000) // Compact expired state during RocksDB background merges
    .build();

ValueStateDescriptor<UserSession> descriptor = new ValueStateDescriptor<>("session", UserSession.class);
descriptor.enableTimeToLive(ttlConfig);
```
