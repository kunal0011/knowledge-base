# Domain III: Apache Spark (PySpark) — Distributed Big Data Processing & Cluster Computing

[![Spark Version](https://img.shields.io/badge/Apache%20Spark-3.5%2B%20Ready-orange.svg)](https://spark.apache.org/)
[![Foundational Book](https://img.shields.io/badge/Canonical%20Book-Spark%3A%20The%20Definitive%20Guide%20(Zaharia)-darkorange.svg)](https://www.oreilly.com/library/view/spark-the-definitive/9781491912201/)
[![Companion Book](https://img.shields.io/badge/Companion%20Book-Learning%20Spark%202nd%20Ed%20(Damji)-red.svg)](https://www.oreilly.com/library/view/learning-spark-2nd/9781492050032/)

> *"Spark is a unified computing engine and a set of libraries for wide-scale data processing on computer clusters. By providing in-memory computing primitives, a state-of-the-art Catalyst query optimizer, and whole-stage code generation via Project Tungsten, Spark makes distributed big data processing up to 100 times faster than Hadoop MapReduce."*  
> — **Matei Zaharia**, Creator of Apache Spark & Chief Technologist at Databricks

---

## 🏛️ Spark Architectural Philosophy & Cluster Blueprint

Spark separates application orchestration from physical task execution using a **Driver-Executor architecture**:

```
                       +---------------------------------------+
                       |             Driver Node               |
                       |  - SparkSession / SparkContext        |
                       |  - Catalyst Optimizer (Logical Plan)  |
                       |  - DAGScheduler (Stages)              |
                       |  - TaskScheduler (Tasks)              |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |            Cluster Manager            |
                       |     (YARN / Kubernetes / Standalone)  |
                       +-------------------+-------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
       +--------------------------+                  +--------------------------+
       |   Worker Node 1          |                  |   Worker Node 2          |
       |  +--------------------+  |                  |  +--------------------+  |
       |  |  Executor (JVM)    |  |                  |  |  Executor (JVM)    |  |
       |  |  - Task 1 (Core 1) |  |                  |  |  - Task 3 (Core 1) |  |
       |  |  - Task 2 (Core 2) |  |                  |  |  - Task 4 (Core 2) |  |
       |  |  - BlockManager    |  |                  |  |  - BlockManager    |  |
       |  +--------------------+  |                  |  +--------------------+  |
       +--------------------------+                  +--------------------------+
```

### The Catalyst Query Optimizer Pipeline
When you execute a DataFrame transformation in Spark SQL / PySpark, your code is not immediately executed. Instead, it passes through the Catalyst Optimizer:

```
Unresolved Logical Plan
         |  (Catalog lookup, schema resolution)
         v
Analyzed Logical Plan
         |  (Predicate pushdown, projection pruning, constant folding)
         v
Optimized Logical Plan
         |  (Cost-Based Optimizer, Join reordering, Broadcast hints)
         v
Physical Plans (Multiple Candidate Strategies)
         |  (Cost Model selects best Physical Plan)
         v
Selected Physical Plan
         |  (Project Tungsten: Whole-Stage Code Generation to Java bytecode)
         v
Compiled Java Bytecode Executed on JVM Tasks
```

---

## 📚 Master Chapter Index

| Chapter | Title | Core Canonical Concepts & Focus | Canonical Literature Focus |
| :---: | :--- | :--- | :--- |
| **01** | [Spark Architecture, Cluster Execution & Catalyst-Tungsten](01.%20Spark%20Architecture%2C%20Cluster%20Execution%20%26%20Catalyst-Tungsten.md) | Driver, Executors, Tasks, Catalyst query pipeline, Tungsten CodeGen | *Spark: The Definitive Guide* Ch. 1–2 |
| **02** | [RDD Fundamentals, DAG Lineage & Fault Tolerance](02.%20RDD%20Fundamentals%2C%20DAG%20Lineage%20%26%20Fault%20Tolerance.md) | Lineage graph, lazy evaluation, narrow vs wide dependencies, recomputation | *Spark: The Definitive Guide* Ch. 12 |
| **03** | [DataFrames, Datasets & Spark SQL Engine](03.%20DataFrames%2C%20Datasets%20%26%20Spark%20SQL%20Engine.md) | StructType schemas, Column expressions, Catalyst optimization phases | *Learning Spark (2nd Ed.)* Ch. 3 |
| **04** | [Partitioning, Shuffling & Data Skew Optimization](04.%20Partitioning%2C%20Shuffling%20%26%20Data%20Skew%20Optimization.md) | Hash/Range partitioners, shuffle spill, Adaptive Query Execution, salting | *Spark: The Definitive Guide* Ch. 19 |
| **05** | [Distributed Join Strategies & Performance Tuning](05.%20Distributed%20Join%20Strategies%20%26%20Performance%20Tuning.md) | Broadcast Hash Join, Shuffle Hash Join, Sort-Merge Join, broadcast limits | *Spark: The Definitive Guide* Ch. 8 |
| **06** | [User-Defined Functions (UDFs) & Vectorized Pandas UDFs](06.%20User-Defined%20Functions%20%28UDFs%29%20%26%20Vectorized%20Pandas%20UDFs.md) | Python worker serialization bottleneck, PyArrow Vectorized Pandas UDFs | *Learning Spark (2nd Ed.)* Ch. 3 |
| **07** | [Memory Management, Storage Levels & Caching](07.%20Memory%20Management%2C%20Storage%20Levels%20%26%20Caching.md) | Unified memory model, on-heap vs off-heap, `persist()` levels, GC tuning | *High Performance Spark* Ch. 7 |
| **08** | [Distributed File Formats, Parquet, ORC & Delta Lake](08.%20Distributed%20File%20Formats%2C%20Parquet%2C%20ORC%20%26%20Delta%20Lake.md) | Parquet columnar encoding, predicate pushdown, Delta Lake ACID logs | *Spark: The Definitive Guide* Ch. 9 & Delta Lake |
| **09** | [Structured Streaming, Watermarks & Stateful Processing](09.%20Structured%20Streaming%2C%20Watermarks%20%26%20Stateful%20Processing.md) | Micro-batching, event-time semantics, watermarking, exactly-once sinks | *Spark: The Definitive Guide* Ch. 20–21 |
| **10** | [Spark Operations, Monitoring, UI Profiling & Troubleshooting](10.%20Spark%20Operations%2C%20Monitoring%2C%20UI%20Profiling%20%26%20Troubleshooting.md) | Spark Web UI analysis, OOM diagnosis, FetchFailedException, dynamic scaling | *Spark: The Definitive Guide* Ch. 18 |

---

## ⚡ Quick Reference: High-Performance PySpark Idioms

### 1. Broadcast Hash Join for Small-to-Large Joins
Eliminate costly cluster-wide shuffles when joining a large dataset with a lookup table:
```python
from pyspark.sql import functions as F

# Hints Catalyst to broadcast the small dimension table to all executors
joined_df = large_fact_df.join(
    F.broadcast(small_dim_df),
    on="user_id",
    how="inner"
)
```

### 2. High-Performance Vectorized Pandas UDF (PyArrow)
Avoid standard Python UDF row-by-row serialization overhead:
```python
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

@pandas_udf(DoubleType())
def calculate_z_score_udf(series: pd.Series) -> pd.Series:
    # Executes vector-at-a-time in compiled C/Arrow inside worker
    return (series - series.mean()) / (series.std() + 1e-8)

df = df.withColumn("z_score", calculate_z_score_udf(F.col("metric")))
```

### 3. Salting Skewed Keys to Prevent Straggler Tasks
```python
import random
from pyspark.sql import functions as F

# Add salt 0..19 to distribute hot key across 20 partitions
salted_df = skewed_df.withColumn("salt", (F.rand() * 20).cast("int"))
salted_df = salted_df.withColumn("salted_key", F.concat(F.col("key"), F.lit("_"), F.col("salt")))
```
