---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 200: Number of Islands"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 200: Number of Islands

---

### Problem Statement

Given an `m x n` 2D binary grid which represents a map of '1's (land) and '0's (water), return the number of islands.

---

### Key Observation

* Each connected cluster of '1's forms an island.
* When encountering a '1', increment island count and trigger DFS/BFS to sink/visit all adjacent connected '1's.

---

### Core Technique: Grid Connected Components via DFS / BFS

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        m, n = len(grid), len(grid[0])
        islands = 0
        
        def dfs(r, c):
            if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != '1':
                return
            grid[r][c] = '0'  # mark visited by sinking
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
            
        for r in range(m):
            for c in range(n):
                if grid[r][c] == '1':
                    islands += 1
                    dfs(r, c)
                    
        return islands
```

---

### Worked-Out Example

```python
grid = [
  ["1","1","0","0"],
  ["1","1","0","0"],
  ["0","0","1","0"],
  ["0","0","0","1"]
]
(0,0) starts island 1 -> sinks connected 4 cells
(2,2) starts island 2 -> sinks cell
(3,3) starts island 3 -> sinks cell
Result = 3
```

---

### Complexity Analysis

* **Time Complexity:** `O(m * n)`
* **Space Complexity:** `O(m * n) recursion call stack worst case`

---

### Takeaway Pattern

In-place cell modification `grid[r][c] = '0'` serves as an O(1) auxiliary visited set.