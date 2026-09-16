# Domain III: Apache Airflow — Distributed Workflow Orchestration & Scheduling

[![Airflow Version](https://img.shields.io/badge/Apache%20Airflow-2.8%2B%20Ready-teal.svg)](https://airflow.apache.org/)
[![Canonical Book](https://img.shields.io/badge/Canonical%20Book-Data%20Pipelines%20with%20Airflow%20(Harenslak)-darkblue.svg)](https://www.manning.com/books/data-pipelines-with-apache-airflow)
[![Architecture Paradigm](https://img.shields.io/badge/Architecture-TaskFlow%20%26%20Dynamic%20Task%20Mapping-green.svg)](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/taskflow.html)

> *"Airflow is a platform created by the community to programmatically author, schedule, and monitor workflows. By representing pipelines as Directed Acyclic Graphs (DAGs) written in dynamic Python code, Airflow provides unlimited extensibility, robust backfilling, and rich operational observability across enterprise data platforms."*  
> — **Bas P. Harenslak & Julian Rutger de Ruiter**, authors of *Data Pipelines with Apache Airflow*

---

## 🏛️ Airflow Architectural Philosophy & Topology Blueprint

Apache Airflow is not a distributed data processing framework (it does not compute big data in memory like Spark or Flink); it is an **Orchestration and Scheduling Engine**. Airflow manages the dependencies, schedules, retries, and execution states of tasks that coordinate work across external compute engines (Snowflake, Spark, BigQuery, Kubernetes, Docker).

```
                      Airflow Distributed Architecture (Celery / K8s)

+-------------------------------------------------------------------------------+
| User Code: Python DAG Files (/dags directory)                                 |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| Airflow Scheduler Process                                                     |
|  - DagFileProcessor: Parses Python files into Serialized DAGs                 |
|  - SchedulerLoop: Evaluates task dependencies, states, and data intervals     |
|  - Transitions TaskInstances: Scheduled -> Queued                            |
+---------------------------------------+---------------------------------------+
                                        |
           +----------------------------+----------------------------+
           | Writes State / Heartbeats                               | Reads State
           v                                                         v
+---------------------------------------+         +-----------------------------+
| Metadata Database (PostgreSQL / MySQL)|         | Airflow Webserver (UI)      |
+-------------------+-------------------+         +-----------------------------+
                    |
                    v Reads Queued Tasks
+-------------------------------------------------------------------------------+
| Distributed Executor (Celery / KubernetesExecutor)                            |
|                                                                               |
|  +--------------------------+                  +--------------------------+  |
|  | Celery Worker Node 1     |                  | Kubernetes Pod Worker    |  |
|  |  - TaskInstance 101      |                  |  - Ephemeral Task Pod    |  |
|  |  - Custom Operators      |                  |  - Isolated dependencies |  |
|  +--------------------------+                  +--------------------------+  |
|                                                                               |
|  +-------------------------------------------------------------------------+  |
|  | Airflow Triggerer Process (Asyncio Event Loop for Deferrable Sensors)   |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
```

---

## 📚 Master Chapter Index

| Chapter | Title | Core Canonical Concepts & Focus | Canonical Literature Focus |
| :---: | :--- | :--- | :--- |
| **01** | [Airflow Architecture, Schedulers & Execution Topologies](01.%20Airflow%20Architecture%2C%20Schedulers%20%26%20Execution%20Topologies.md) | Webserver, Scheduler, Metadata DB, Triggerer; Celery vs KubernetesExecutor | *Data Pipelines with Airflow* Ch. 2 |
| **02** | [DAG Internals, Parsing Loops & Task State Machines](02.%20DAG%20Internals%2C%20Parsing%20Loops%20%26%20Task%20State%20Machines.md) | DagFileProcessor loop, DagBag, TaskInstance lifecycle states | *Data Pipelines with Airflow* Ch. 3 |
| **03** | [Operators, Sensors, Hooks & Deferrable Operators](03.%20Operators%2C%20Sensors%2C%20Hooks%20%26%20Deferrable%20Operators.md) | BaseOperator vs Hooks, Sensors (poke vs reschedule), Deferrable Triggerer | *Data Pipelines with Airflow* Ch. 4 & 9 |
| **04** | [Scheduling Mechanics, Data Intervals, Logical Date & Backfills](04.%20Scheduling%20Mechanics%2C%20Data%20Intervals%2C%20Logical%20Date%20%26%20Backfills.md) | Timetables, `logical_date` vs intervals, catchup, CLI backfill commands | *Data Pipelines with Airflow* Ch. 5 |
| **05** | [Data Sharing, XCom Architecture & Custom XCom Backends](05.%20Data%20Sharing%2C%20XCom%20Architecture%20%26%20Custom%20XCom%20Backends.md) | XCom metadata DB table, size limits, custom S3/GCS blob backends | *Data Pipelines with Airflow* Ch. 6 |
| **06** | [TaskFlow API & Dynamic Task Mapping (expand-partial)](06.%20TaskFlow%20API%20%26%20Dynamic%20Task%20Mapping%20%28expand-partial%29.md) | Pythonic `@dag` and `@task`, Dynamic Task Mapping (`.expand()` / `.partial()`) | *Data Pipelines with Airflow* & AIP-42 |
| **07** | [Concurrency Management, Worker Pools & SLAs](07.%20Concurrency%20Management%2C%20Worker%20Pools%20%26%20SLAs.md) | `max_active_tasks`, Worker Pools (protecting DB bottlenecks), SLA Miss callbacks | *Data Pipelines with Airflow* Ch. 11 |
| **08** | [Secrets Management, Connections & Variable Masking](08.%20Secrets%20Management%2C%20Connections%20%26%20Variable%20Masking.md) | Airflow Connections, Fernet encryption, AWS Secrets Manager / Vault backends | *Data Pipelines with Airflow* Ch. 12 |
| **09** | [DAG Testing, CI-CD Validation & Mock Testing](09.%20DAG%20Testing%2C%20CI-CD%20Validation%20%26%20Mock%20Testing.md) | DAG integrity tests, unit testing custom operators, `dag.test()` CLI | *Data Pipelines with Airflow* Ch. 10 |
| **10** | [Airflow Operations, Database Pool Tuning & Zombie Triage](10.%20Airflow%20Operations%2C%20Database%20Pool%20Tuning%20%26%20Zombie%20Triage.md) | Diagnosing Zombie Tasks, PgBouncer connection pool tuning, scheduler latency | *Data Pipelines with Airflow* Ch. 13 |

---

## ⚡ Quick Reference: Modern Airflow 2.x Idioms

### 1. Modern Pythonic TaskFlow API (`@dag`, `@task`)
```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule="@daily", start_date=datetime(2026, 1, 1), catchup=False, tags=["etl"])
def enterprise_etl_pipeline():
    
    @task
    def extract_order_ids() -> list[int]:
        return [101, 102, 103, 104]

    @task
    def process_order(order_id: int) -> dict:
        return {"id": order_id, "status": "VERIFIED"}

    # Dynamic Task Mapping (AIP-42): Spawns 4 parallel TaskInstances dynamically at runtime!
    orders = extract_order_ids()
    results = process_order.expand(order_id=orders)

pipeline = enterprise_etl_pipeline()
```

### 2. Async Deferrable Operator (Releases Worker Slot)
```python
from airflow.sensors.base import BaseSensorOperator

# Uses async Triggerer event-loop; zero worker slot consumed while sleeping!
wait_for_s3_file = S3KeySensorAsync(
    task_id="wait_for_s3_file",
    bucket_key="s3://bucket/landing/data.csv",
    timeout=3600
)
```
