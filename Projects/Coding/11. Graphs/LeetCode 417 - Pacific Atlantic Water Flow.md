---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 417: Pacific Atlantic Water Flow"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 417: Pacific Atlantic Water Flow

---

### Problem Statement

Given an `m x n` grid of heights, return coordinates from which water can flow to BOTH the Pacific and Atlantic oceans.

---

### Key Observation

* Instead of simulating flow downhill from every single cell (expensive), reverse the flow and search uphill from the ocean borders.
* DFS/BFS from Pacific borders and Atlantic borders separately.
* The intersection of both reachable sets is the answer.

---

### Core Technique: Multi-Source Reverse Ocean Traversal

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        m, n = len(heights), len(heights[0])
        pac, atl = set(), set()
        
        def dfs(r, c, visit, prev_height):
            if ((r, c) in visit or r < 0 or c < 0 or 
                r >= m or c >= n or heights[r][c] < prev_height):
                return
            visit.add((r, c))
            dfs(r + 1, c, visit, heights[r][c])
            dfs(r - 1, c, visit, heights[r][c])
            dfs(r, c + 1, visit, heights[r][c])
            dfs(r, c - 1, visit, heights[r][c])
            
        for c in range(n):
            dfs(0, c, pac, heights[0][c])
            dfs(m - 1, c, atl, heights[m - 1][c])
        for r in range(m):
            dfs(r, 0, pac, heights[r][0])
            dfs(r, n - 1, atl, heights[r][n - 1])
            
        return list(pac & atl)
```

---

### Worked-Out Example

```
Top & Left edges start Pacific flow
Bottom & Right edges start Atlantic flow
Return all (r, c) present in both sets.
```

---

### Complexity Analysis

* **Time Complexity:** `O(m * n)`
* **Space Complexity:** `O(m * n)`

---

### Takeaway Pattern

When searching for paths to multiple boundaries, reverse the search direction from the boundaries inward.