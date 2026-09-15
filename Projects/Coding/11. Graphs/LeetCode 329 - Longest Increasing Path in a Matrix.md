---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 329: Longest Increasing Path in a Matrix"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 329: Longest Increasing Path in a Matrix

**Target Companies:** Google (Signature Hard), Amazon

---

### Problem Statement

Given an `m x n` integers matrix, return the length of the longest increasing path in matrix.

---

### Key Observation

* Since paths must be strictly increasing, the grid forms a Directed Acyclic Graph (DAG) with no cycles.
* We can find the longest path from each cell using **DFS + Memoization** (or Kahn's Topological Sort).
* `dp[r][c] = 1 + max(dfs(nr, nc))` for all strictly greater neighbors.

---

### Core Technique: Memoized Depth-First Search on Matrix DAG

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        memo = {}
        
        def dfs(r, c):
            if (r, c) in memo:
                return memo[(r, c)]
                
            res = 1
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and matrix[nr][nc] > matrix[r][c]:
                    res = max(res, 1 + dfs(nr, nc))
                    
            memo[(r, c)] = res
            return res
            
        longest = 0
        for r in range(m):
            for c in range(n):
                longest = max(longest, dfs(r, c))
                
        return longest
```

---

### Worked-Out Example

```python
matrix = [
  [9, 9, 4],
  [6, 6, 8],
  [2, 1, 1]
]
Longest path: [1 -> 2 -> 6 -> 9]
Result length = 4
```

---

### Complexity Analysis

* **Time Complexity:** `O(m * n) (each cell is computed once and cached)`
* **Space Complexity:** `O(m * n)`

---

### Takeaway Pattern

Strictly increasing paths cannot contain cycles, allowing direct DFS + Memoization without visited set resets.