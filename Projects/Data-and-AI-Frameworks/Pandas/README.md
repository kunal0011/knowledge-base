# Domain II: Pandas — High-Performance Tabular Analytics & Relational Algebra

[![Pandas Version](https://img.shields.io/badge/Pandas-2.x%20Ready-green.svg)](https://pandas.pydata.org/)
[![Foundational Book](https://img.shields.io/badge/Canonical%20Book-Python%20for%20Data%20Analysis%20(McKinney)-darkgreen.svg)](https://wesmckinney.com/book/)
[![Companion Book](https://img.shields.io/badge/Companion%20Book-Effective%20Pandas%20(Harrison)-orange.svg)](https://store.metasnake.com/effective-pandas-book)

> *"pandas was designed from the beginning to make data analysis and modeling work fast, productive, and enjoyable in Python. It provides high-performance, easy-to-use data structures and data analysis tools, unifying tabular data manipulation across relational SQL algebra, multidimensional tensor indexing, and high-frequency time series."*  
> — **Wes McKinney**, Creator of pandas & author of *Python for Data Analysis*

---

## 🏛️ Pandas Architectural Philosophy & Memory Blueprint

A pandas `DataFrame` is fundamentally an ordered collection of 1D `Series` sharing a common index. Historically, pandas stored homogeneous columns together in 2D contiguous arrays managed by the internal **`BlockManager`**. In modern pandas (v2.0+), the storage backend can leverage **Apache Arrow**, eliminating the historical boxing overhead and fragmentation of NumPy object dtypes.

```
                  Traditional Pandas Architecture (BlockManager)
DataFrame:
  Index: ['2026-01-01', '2026-01-02', '2026-01-03']
  Columns: ['price', 'volume', 'symbol', 'spread']
        |
        v
  +-----------------------------------------------------------+
  | FloatBlock (2D NumPy array): ['price', 'spread'] (float64) |
  +-----------------------------------------------------------+
  | IntBlock (2D NumPy array):   ['volume']          (int64)  |
  +-----------------------------------------------------------+
  | ObjectBlock (PyObject ptrs): ['symbol']          (string) |
  +-----------------------------------------------------------+

               Modern Pandas 2.0+ Architecture (Apache Arrow)
  +-----------------------------------------------------------+
  | Arrow String Array: UTF-8 contiguous binary buffer + offsets|
  +-----------------------------------------------------------+
  | Arrow Int64 Array:  Contiguous 64-bit int + null bitmap    |
  +-----------------------------------------------------------+
  | Arrow Float64 Array: Contiguous 64-bit float + null bitmap |
  +-----------------------------------------------------------+
  (Zero-copy data sharing, SIMD-accelerated, native null handling)
```

---

## 📚 Master Chapter Index

| Chapter | Title | Core Canonical Concepts & Focus | Canonical Literature Focus |
| :---: | :--- | :--- | :--- |
| **01** | [Series & DataFrame Internals, BlockManager & Arrow Backend](01.%20Series%20%26%20DataFrame%20Internals%2C%20BlockManager%20%26%20Arrow%20Backend.md) | Series/DataFrame architecture, BlockManager, Arrow table integration | *Python for Data Analysis* Ch. 5 |
| **02** | [Data Ingestion, Serialization & High-Performance IO](02.%20Data%20Ingestion%2C%20Serialization%20%26%20High-Performance%20IO.md) | Parquet, Feather, CSV chunking, Arrow engines, memory-efficient dtypes | *Python for Data Analysis* Ch. 6 |
| **03** | [Indexing, Selection, MultiIndex & SettingWithCopyWarning](03.%20Indexing%2C%20Selection%2C%20MultiIndex%20%26%20SettingWithCopyWarning.md) | `.loc` vs `.iloc`, MultiIndex slicing, chained indexing, `SettingWithCopy` | *Effective Pandas* & McKinney Ch. 5 |
| **04** | [Missing Data Mechanics, Nullable Types & Imputation](04.%20Missing%20Data%20Mechanics%2C%20Nullable%20Types%20%26%20Imputation.md) | `NaN` vs `None` vs `pd.NA`, nullable integer/bool types, 3-valued logic | *Python for Data Analysis* Ch. 7 |
| **05** | [Transformations, Vectorization & Method Chaining](05.%20Transformations%2C%20Vectorization%20%26%20Method%20Chaining.md) | `.apply()` traps, vectorization, `pd.eval()`, `.query()`, method chaining | *Effective Pandas* (Matt Harrison) |
| **06** | [GroupBy & Split-Apply-Combine Mechanics](06.%20GroupBy%20%26%20Split-Apply-Combine%20Mechanics.md) | Hash partitioning of groups, `.agg()`, `.transform()`, `.filter()`, windows | *Python for Data Analysis* Ch. 10 |
| **07** | [Reshaping, Pivoting, Melting & Crosstabs](07.%20Reshaping%2C%20Pivoting%2C%20Melting%20%26%20Crosstabs.md) | `pivot`, `pivot_table`, wide-to-long `melt`, `stack`/`unstack`, `crosstab` | *Python for Data Analysis* Ch. 8 |
| **08** | [Relational Algebra, Merges, Joins & Concatenation](08.%20Relational%20Algebra%2C%20Merges%2C%20Joins%20%26%20Concatenation.md) | Relational merge mechanics, hash joins, `validate`, `indicator`, Cartesian | *Python for Data Analysis* Ch. 8 |
| **09** | [Time Series Analysis, Resampling & Rolling Windows](09.%20Time%20Series%20Analysis%2C%20Resampling%20%26%20Rolling%20Windows.md) | `DatetimeIndex`, frequency offsets, timezones, resampling, rolling stats | *Python for Data Analysis* Ch. 11 |
| **10** | [Categorical Data, Memory Optimization & PyArrow Dtypes](10.%20Categorical%20Data%2C%20Memory%20Optimization%20%26%20PyArrow%20Dtypes.md) | `CategoricalDtype` dictionary encoding, 90%+ RAM reduction, PyArrow dtypes | *Python for Data Analysis* Ch. 12 |

---

## ⚡ Quick Reference: High-Performance Pandas Idioms

### 1. Eliminating `SettingWithCopyWarning` Forever
Never chain indexing expressions (`df[cond]['col'] = val`). Always use explicit label-based `.loc`:
```python
# ANTI-PATTERN (Triggers SettingWithCopyWarning, may fail silently)
# df[df['score'] > 90]['grade'] = 'A'

# PRODUCTION IDIOM
df.loc[df['score'] > 90, 'grade'] = 'A'
```

### 2. Method Chaining for Clean, Reproducible Pipelines
Avoid mutating DataFrames in-place across disparate notebook cells:
```python
clean_df = (
    raw_df
    .rename(columns=str.lower)
    .drop(columns=['unused_col'])
    .dropna(subset=['critical_id'])
    .assign(
        log_revenue=lambda d: np.log1p(d['revenue']),
        is_active=lambda d: d['status'] == 'ACTIVE'
    )
    .query('log_revenue > 0')
)
```

### 3. PyArrow Engine Ingestion (10x Faster CSV & Native Nulls)
```python
import pandas as pd

# Ingest using Apache Arrow backend for instant loading and zero-copy string memory
df = pd.read_csv("massive_dataset.csv", engine="pyarrow", dtype_backend="pyarrow")
```
