---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 51: N-Queens"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 51: N-Queens

## LeetCode 51 — N-Queens

---

## Problem Statement

The **n-queens puzzle** is the problem of placing `n` queens on an `n × n` chessboard such that **no two queens attack each other**.

A queen can attack:

* Horizontally (same row)
* Vertically (same column)
* Diagonally

Return **all distinct solutions**, where each solution is represented as a list of strings:

* `'Q'` denotes a queen
* `'.'` denotes an empty space

**Constraints**

* `1 ≤ n ≤ 9`

**Example**

```text
Input: n = 4
Output:
[
 [".Q..",
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",
  "Q...",
  "...Q",
  ".Q.."]
]
```

---

## Key Observations (Critical)

1. **Exactly one queen per row**

   * This allows us to decide **row by row**.
2. Columns must be unique → use a `cols` set.
3. Diagonals must be unique:

   * Main diagonal (`row - col`)
   * Anti-diagonal (`row + col`)
4. If a position violates **any** of the above, it is immediately invalid → prune.
5. The problem is a **constraint satisfaction backtracking** problem.

---

## Backtracking Strategy

* Recursively place a queen **row by row**
* At each row, try all columns
* Only continue recursion if the placement is valid
* When `row == n`, a complete solution is formed

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        result: List[List[str]] = []
        board = [["."] * n for _ in range(n)]

        cols = set()
        diag1 = set()  # row - col
        diag2 = set()  # row + col

        def backtrack(row: int) -> None:
            if row == n:
                result.append(["".join(r) for r in board])
                return

            for col in range(n):
                if col in cols or (row - col) in diag1 or (row + col) in diag2:
                    continue

                board[row][col] = "Q"
                cols.add(col)
                diag1.add(row - col)
                diag2.add(row + col)

                backtrack(row + 1)

                board[row][col] = "."
                cols.remove(col)
                diag1.remove(row - col)
                diag2.remove(row + col)

        backtrack(0)
        return result
```

---

## Example Explanation (`n = 4`)

We place queens **row by row**.

### One valid placement path

```
Row 0 → Col 1
Row 1 → Col 3
Row 2 → Col 0
Row 3 → Col 2
```

Board:

```
.Q..
...Q
Q...
..Q.
```

This satisfies:

* All columns unique
* No diagonal conflicts

---

## Backtracking Tree Structure

### (Complete Conceptual Tree for `n = 4`)

Each level = one row  
Each node = column chosen for that row  
❌ = pruned due to conflict  
✅ = complete solution

```
Row 0
├── Col 0
│   ├── Col 0 ❌ (same column)
│   ├── Col 1 ❌ (diagonal)
│   ├── Col 2
│   │   ├── Col 0 ❌
│   │   ├── Col 1 ❌
│   │   ├── Col 2 ❌
│   │   └── Col 3 ❌
│   └── Col 3
│       ├── Col 0 ❌
│       ├── Col 1 ❌
│       ├── Col 2 ❌
│       └── Col 3 ❌
│
├── Col 1
│   ├── Col 0 ❌
│   ├── Col 1 ❌
│   ├── Col 2 ❌
│   └── Col 3
│       ├── Col 0
│       │   ├── Col 0 ❌
│       │   ├── Col 1 ❌
│       │   ├── Col 2 ✅  ← Solution 1
│       │   └── Col 3 ❌
│       ├── Col 1 ❌
│       ├── Col 2 ❌
│       └── Col 3 ❌
│
├── Col 2
│   ├── Col 0
│   │   ├── Col 0 ❌
│   │   ├── Col 1 ❌
│   │   ├── Col 2 ❌
│   │   └── Col 3
│   │       ├── Col 0 ❌
│   │       ├── Col 1 ✅  ← Solution 2
│   │       ├── Col 2 ❌
│   │       └── Col 3 ❌
│   ├── Col 1 ❌
│   ├── Col 2 ❌
│   └── Col 3 ❌
│
└── Col 3
    ├── Col 0
    │   ├── Col 0 ❌
    │   ├── Col 1 ❌
    │   ├── Col 2 ❌
    │   └── Col 3 ❌
    ├── Col 1
    │   ├── Col 0 ❌
    │   ├── Col 1 ❌
    │   ├── Col 2 ❌
    │   └── Col 3 ❌
    ├── Col 2 ❌
    └── Col 3 ❌
```

---

## Why This Tree Is Powerful

* **Row-wise placement** reduces branching
* **Early pruning** avoids exploring invalid boards
* Only **2 valid leaves** exist for `n = 4`
* Demonstrates exponential pruning from `4⁴ = 256` → only a handful explored

---

## Complexity Analysis

* **Time Complexity:** `O(n!)` (pruned heavily in practice)
* **Space Complexity:** `O(n)` recursion depth + board state

---

## Pattern Recognition

LeetCode 51 is the **canonical constraint backtracking problem**:

* Fixed depth
* Strong pruning
* State represented by sets

This pattern generalizes to:

* Sudoku
* Graph coloring
* Crossword / CSP solvers

---

### One-Line Interview Explanation

> “We place one queen per row and use sets to track columns and diagonals, pruning invalid placements early using backtracking.”

If you want next:

* Side-by-side **LC 51 vs LC 52**
* Bitmask optimization for large `n`
* Visual animation logic for this tree