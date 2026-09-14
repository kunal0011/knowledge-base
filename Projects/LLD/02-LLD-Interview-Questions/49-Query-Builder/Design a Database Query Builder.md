---
date: "2026-04-06"
type: lld-question
difficulty: hard
tags: [lld, interview-prep, database, composite-pattern]
---

# Design a Database Query Builder

## 1. Problem Statement
Design a fluent query builder supporting SELECT, WHERE, JOIN, GROUP BY, ORDER BY, and LIMIT with type-safe chaining.

## 2. Key Implementation (Python)

```python
from typing import List, Optional, Tuple

class QueryBuilder:
    def __init__(self):
        self._select_cols: List[str] = ["*"]
        self._from_table: str = ""
        self._joins: List[str] = []
        self._where_clauses: List[str] = []
        self._group_by: List[str] = []
        self._having: List[str] = []
        self._order_by: List[Tuple[str, str]] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._params: List = []

    def select(self, *columns: str) -> 'QueryBuilder':
        self._select_cols = list(columns)
        return self

    def from_table(self, table: str) -> 'QueryBuilder':
        self._from_table = table
        return self

    def where(self, condition: str, *params) -> 'QueryBuilder':
        self._where_clauses.append(condition)
        self._params.extend(params)
        return self

    def and_where(self, condition: str, *params) -> 'QueryBuilder':
        return self.where(condition, *params)

    def or_where(self, condition: str, *params) -> 'QueryBuilder':
        if self._where_clauses:
            last = self._where_clauses.pop()
            self._where_clauses.append(f"({last} OR {condition})")
        else:
            self._where_clauses.append(condition)
        self._params.extend(params)
        return self

    def join(self, table: str, on: str, join_type: str = "INNER") -> 'QueryBuilder':
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self

    def left_join(self, table: str, on: str) -> 'QueryBuilder':
        return self.join(table, on, "LEFT")

    def group_by(self, *columns: str) -> 'QueryBuilder':
        self._group_by.extend(columns)
        return self

    def having(self, condition: str) -> 'QueryBuilder':
        self._having.append(condition)
        return self

    def order_by(self, column: str, direction: str = "ASC") -> 'QueryBuilder':
        self._order_by.append((column, direction.upper()))
        return self

    def limit(self, count: int) -> 'QueryBuilder':
        self._limit = count
        return self

    def offset(self, count: int) -> 'QueryBuilder':
        self._offset = count
        return self

    def build(self) -> Tuple[str, List]:
        parts = [f"SELECT {', '.join(self._select_cols)}"]
        parts.append(f"FROM {self._from_table}")

        for j in self._joins:
            parts.append(j)

        if self._where_clauses:
            parts.append(f"WHERE {' AND '.join(self._where_clauses)}")

        if self._group_by:
            parts.append(f"GROUP BY {', '.join(self._group_by)}")

        if self._having:
            parts.append(f"HAVING {' AND '.join(self._having)}")

        if self._order_by:
            order = ', '.join(f"{col} {d}" for col, d in self._order_by)
            parts.append(f"ORDER BY {order}")

        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")

        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")

        return '\n'.join(parts), self._params

    def __str__(self):
        sql, _ = self.build()
        return sql

if __name__ == "__main__":
    query = (QueryBuilder()
        .select("u.name", "COUNT(o.id) as order_count", "SUM(o.total) as total_spent")
        .from_table("users u")
        .left_join("orders o", "u.id = o.user_id")
        .where("u.active = ?", True)
        .and_where("u.created_at > ?", "2024-01-01")
        .group_by("u.id", "u.name")
        .having("COUNT(o.id) > 5")
        .order_by("total_spent", "DESC")
        .limit(10))

    sql, params = query.build()
    print(sql)
    print(f"Params: {params}")
```

## 3. Patterns: **Builder** (fluent API) | **Composite** (nested conditions) | **Strategy** (dialect-specific SQL generation)
## 4. Follow-ups: **Multi-dialect?** Strategy for MySQL/PostgreSQL/SQLite differences | **Subqueries?** Nested QueryBuilder | **SQL injection?** Parameterized queries only.

---
**Related:** [[03 - Decorator Pattern]] | [[01 - Strategy Pattern]]
