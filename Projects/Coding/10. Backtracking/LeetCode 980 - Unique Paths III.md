---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 980: Unique Paths III"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 980: Unique Paths III

## LeetCode 980 — Unique Paths III

---

## Problem Statement

You are given an `m x n` grid representing a map where:

* `1` → starting square (exactly one)
* `2` → ending square (exactly one)
* `0` → empty square (must be walked over)
* `-1` → obstacle (cannot be walked over)

You must find the **number of unique paths** from the start to the end such that:

* You **visit every non-obstacle square exactly once**
* You can move **up, down, left, or right**

---

## Key Observations

1. This is **not a shortest path** problem; it is a **Hamiltonian path–style DFS**.
2. Every valid path must:

   * Start at `1`
   * End at `2`
   * Visit **all `0` cells exactly once**
3. The traversal state depends on:

   * Current `(row, col)`
   * Number of remaining squares to visit
4. This is a **backtracking with visited marking** problem:

   * Choose a direction
   * Mark cell as visited
   * Recurse
   * Unmark (backtrack)
5. A path is valid **only if**:

   * We reach `2`
   * AND there are **no remaining unvisited empty cells**

---

## Core Backtracking State

* `(r, c)` → current position
* `remain` → number of squares still to be visited (including `0` and `2`)
* Grid itself is used to mark visited cells temporarily

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        remain = 0
        start_r = start_c = 0

        # Count non-obstacle cells and find start
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != -1:
                    remain += 1
                if grid[r][c] == 1:
                    start_r, start_c = r, c

        def backtrack(r: int, c: int, remain: int) -> int:
            # Out of bounds or obstacle
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == -1:
                return 0

            # Reached end
            if grid[r][c] == 2:
                return 1 if remain == 1 else 0

            # Mark as visited
            temp = grid[r][c]
            grid[r][c] = -1

            paths = (
                backtrack(r + 1, c, remain - 1) +
                backtrack(r - 1, c, remain - 1) +
                backtrack(r, c + 1, remain - 1) +
                backtrack(r, c - 1, remain - 1)
            )

            # Backtrack (unvisit)
            grid[r][c] = temp
            return paths

        return backtrack(start_r, start_c, remain)
```

---

## Example Explanation

### Input Grid

```
[
  [1, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 2, -1]
]
```

* Total non-obstacle cells = **11**
* All 11 must be visited exactly once
* DFS explores all possible paths from `1`
* Only paths that reach `2` **after visiting all 0s** are counted

---

## Complete Conceptual Backtracking Tree (Navigation View)

This is a **conceptual tree**, not a full enumeration (actual tree is exponential).

Legend:

* `(r,c | remain)`
* ❌ = dead end (hit obstacle / revisited / early end)
* ✓ = valid complete path

```
Start
(0,0 | 11)
│
├── (0,1 | 10)
│   ├── (0,2 | 9)
│   │   ├── (0,3 | 8)
│   │   │   ├── (1,3 | 7)
│   │   │   │   ├── (1,2 | 6)
│   │   │   │   │   ├── (2,2 | 5) ❌ (end too early)
│   │   │   │   │   └── (1,1 | 5)
│   │   │   │   │       ├── ...
│   │   │   │   │       └── ✓ (valid path)
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
├── (1,0 | 10)
│   ├── (2,0 | 9)
│   │   ├── (2,1 | 8)
│   │   │   ├── (2,2 | 7)
│   │   │   │   └── ❌ (still unvisited cells)
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
└── ❌ (invalid moves)
```

### Key Tree Properties

* **Depth = number of non-obstacle cells**
* Each node branches into **up to 4 directions**
* Many branches terminate early:

  * Hit obstacle
  * Revisit a cell
  * Reach `2` before visiting all cells
* Only leaf nodes that:

  * End at `2`
  * Have `remain == 1`  
    are counted

---

## Why This Is Pure Backtracking (Not DP)

* State space is exponential
* Each cell’s visited/unvisited status matters
* No overlapping subproblems with identical states
* Memoization is infeasible due to grid mutation

---

## Complexity Analysis

* **Time Complexity:** `O(4^(m*n))` (worst case, heavily pruned)
* **Space Complexity:** `O(m*n)` recursion depth

---

## One-Line Interview Explanation

> “We perform DFS from the start, marking cells as visited and counting paths that reach the end only after all non-obstacle cells have been visited exactly once.”