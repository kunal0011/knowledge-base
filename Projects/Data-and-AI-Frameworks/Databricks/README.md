# Domain V: Databricks (Unified Data Analytics & AI Lakehouse Platform)

> "The data lakehouse solves the fundamental friction between the data warehouse and the data lake. Databricks builds upon Apache Spark, Delta Lake, MLflow, and Photon to create a unified platform where data engineering, BI, and AI coexist on an open, governed lakehouse."  
> — *Matei Zaharia & Ali Ghodsi, Founders of Databricks*

---

## 🏛️ Executive Architecture: The Databricks Lakehouse Platform

The Databricks Lakehouse Platform combines the reliability, ACID governance, and performance of enterprise data warehouses with the openness, flexibility, and machine learning scalability of cloud data lakes:

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   APPLICATIONS & WORKLOADS                              │
 │                                                                                         │
 │   [ Data Engineering (DLT) ]   [ BI & Analytics (DBSQL) ]   [ Generative AI & MLflow ] │
 └───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
 ┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
 │                       UNITY CATALOG (UNIFIED GOVERNANCE LAYER)                          │
 │                                                                                         │
 │   • 3-Level Namespace: `catalog.schema.table | volume | model`                          │
 │   • Fine-Grained Access Control: Row-Filters, Column-Masks, Attribute Tags              │
 │   • Open Sharing via Delta Sharing | Lineage Tracking & System Tables                   │
 └───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
 ┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
 │                         DATABRICKS RUNTIME & EXECUTION ENGINES                          │
 │                                                                                         │
 │             ┌─────────────────────────────────┬───────────────────────────────┐         │
 │             │          PHOTON ENGINE          │      DATABRICKS RUNTIME       │         │
 │             │   Native C++ Vectorized Core    │   Optimized Apache Spark 3.5+ │         │
 │             └─────────────────────────────────┴───────────────────────────────┘         │
 │   • Serverless Compute (Instant warm pools, automated scaling, zero cold-starts)        │
 │   • Predictive I/O (AI-driven indexing, data skipping, and caching)                     │
 └───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
 ┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
 │                         STORAGE LAYER: DELTA LAKE PROTOCOL                              │
 │                                                                                         │
 │   • ACID Transactions via Serialized Log (`_delta_log/`)                                │
 │   • Liquid Clustering (Dynamic Hilbert Space-Filling Curves replacing Z-Order)          │
 │   • Deletion Vectors (Roaring Bitmaps for O(1) row-level updates & deletes)             │
 │   • UniForm (Universal Format: Auto-generates Iceberg & Hudi metadata from Delta)       │
 └───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
 ┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
 │                        CLOUD OBJECT STORAGE (CUSTOMER DATA PLANE)                       │
 │                                                                                         │
 │                 [ AWS S3 ]            [ Azure ADLS Gen2 ]            [ GCP GCS ]        │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚖️ Architectural Matrix: Open-Source Spark vs. Databricks Runtime

| Dimension | Open-Source Apache Spark | Databricks Lakehouse Platform (DBR) |
| :--- | :--- | :--- |
| **Execution Engine** | JVM-based Whole-Stage Java CodeGen (Tungsten) | **Photon Engine: 100% C++ Vectorized Execution Engine** |
| **Table Optimization**| Manual Z-Ordering, manual bin-pack compaction | **Liquid Clustering, Predictive I/O, Auto-Compaction** |
| **Governance** | Apache Hive Metastore / External AWS Glue | **Unity Catalog: Centralized 3-tier governance for data & AI** |
| **ETL Orchestration** | Raw PySpark scripts orchestrated via Airflow | **Delta Live Tables (DLT): Declarative DAGs with quality SLAs** |
| **Row Deletions** | Full-file Copy-on-Write (COW) | **Deletion Vectors (Roaring Bitmaps) with near-instant deletes** |
| **Compute Provisioning**| Manual Kubernetes/YARN cluster provisioning (5–15 min)| **Serverless Compute (< 5 second instantaneous startup)** |
| **Interoperability** | Format-locked | **UniForm: Auto-publishes Delta tables as Apache Iceberg** |

---

## 📚 Curriculum Syllabus: 10 Comprehensive Chapters

| Chapter | Title & Domain Focus | Core Mechanics & Code Deliverables |
| :---: | :--- | :--- |
| **[01](01.%20Databricks%20Architecture,%20Control%20Plane%20&%20Data%20Plane.md)** | **Databricks Architecture, Control Plane & Data Plane** | Control Plane vs Data Plane (Egress/Ingress), Classic vs Serverless Compute, Cluster architectures (All-Purpose, Job, SQL Warehouses), Photon C++ execution engine. |
| **[02](02.%20Unity%20Catalog%20&%20Unified%20Lakehouse%20Governance.md)** | **Unity Catalog & Unified Lakehouse Governance** | 3-Level Namespace (`catalog.schema.table/volume/model`), Managed vs External Tables, Volumes, Dynamic Row Filters & Column Masks, Delta Sharing open protocol. |
| **[03](03.%20Delta%20Lake%20on%20Databricks%20&%20Storage%20Engine%20Mechanics.md)** | **Delta Lake on Databricks & Storage Engine Mechanics** | Liquid Clustering vs Z-Order, Deletion Vectors, Change Data Feed (CDF), UniForm (Delta $\rightarrow$ Iceberg sync), `OPTIMIZE` and `VACUUM` safety invariants. |
| **[04](04.%20Delta%20Live%20Tables%20(DLT)%20&%20Declarative%20Pipeline%20Orchestration.md)** | **Delta Live Tables (DLT) & Declarative Pipeline Orchestration** | Declarative pipeline syntax (`@dlt.table`), Streaming Tables vs Materialized Views, Auto Loader (`cloudFiles`), Expectations data quality circuit breakers. |
| **[05](05.%20Databricks%20Workflows,%20Task%20Orchestration%20&%20Serverless%20Jobs.md)** | **Databricks Workflows, Task Orchestration & Serverless Jobs** | Multi-task DAG workflows, Serverless Jobs execution, Parameter passing via Task Values, Conditional task routing (If/Else, For-Each), Failure alerts. |
| **[06](06.%20Databricks%20SQL%20&%20Serverless%20Photon%20Warehousing.md)** | **Databricks SQL & Serverless Photon Warehousing** | Serverless SQL Warehouses, Photon SIMD memory layout, Query Profile waterfall diagnostics, Disk Cache vs Result Cache, Materialized Views in DBSQL. |
| **[07](07.%20MLflow%20&%20End-to-End%20Machine%20Learning%20Operations%20(MLOps).md)** | **MLflow & End-to-End Machine Learning Operations (MLOps)** | MLflow Tracking, Autologging, Model Registry in Unity Catalog, Feature Store / Feature Engineering in UC, Canary deployments and model governance. |
| **[08](08.%20Generative%20AI,%20Mosaic%20AI%20&%20Model%20Serving.md)** | **Generative AI, Mosaic AI & Model Serving** | Mosaic AI Model Serving, Vector Search with Delta sync, RAG architectures on Lakehouse, AI SQL Functions (`ai_query()`, `ai_summarize()`, `ai_classify()`). |
| **[09](09.%20Lakehouse%20Security,%20Networking%20&%20Identity%20Federation.md)** | **Lakehouse Security, Networking & Identity Federation** | Secure Cluster Connectivity (No Public IPs), Storage Credentials & External Locations, SCIM provisioning, System Tables (`system.access.audit`, `system.billing.usage`). |
| **[10](10.%20Enterprise%20Production%20Blueprint%20&%20CICD%20with%20Databricks%20Asset%20Bundles%20(DABs).md)** | **Enterprise Production Blueprint & CI/CD with Databricks Asset Bundles (DABs)** | Databricks Asset Bundles (`databricks.yml`), Multi-environment CI/CD (Dev/Staging/Prod), End-to-end production pipeline: Auto Loader $\rightarrow$ DLT $\rightarrow$ MLflow $\rightarrow$ Serverless Job. |
