---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 2290: Minimum Obstacle Removal to Reach Corner"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 2290: Minimum Obstacle Removal to Reach Corner

**Target Companies:** Google (Hard), Amazon | **Algorithm:** 0-1 BFS with Double-Ended Queue (Deque)

---

### Problem Statement

Given an `m x n` grid where 0 is empty and 1 is obstacle. Return the minimum number of obstacles to remove to move from (0, 0) to (m - 1, n - 1).

---

### Key Observation

* Edge weights are either `0` (moving to empty cell) or `1` (removing obstacle).
* On graphs with only weights 0 and 1, Dijkstra's algorithm $O((V+E)\log V)$ can be optimized to **0-1 BFS in $O(V + E)$**!
* When weight is 0: `deque.appendleft()` (explore immediately at same distance).
* When weight is 1: `deque.append()` (explore at next distance level).

---

### Core Algorithm & Technique: 0-1 BFS via Double-Ended Queue (Linear Time Shortest Path)

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dist = [[float('inf')] * n for _ in range(m)]
        dist[0][0] = 0
        
        q = deque([(0, 0, 0)])  # (cost, r, c)
        
        while q:
            cost, r, c = q.popleft()
            
            if r == m - 1 and c == n - 1:
                return cost
                
            if cost > dist[r][c]:
                continue
                
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    new_cost = cost + grid[nr][nc]
                    if new_cost < dist[nr][nc]:
                        dist[nr][nc] = new_cost
                        if grid[nr][nc] == 0:
                            q.appendleft((new_cost, nr, nc))  # 0-weight edge to front
                        else:
                            q.append((new_cost, nr, nc))      # 1-weight edge to back
                            
        return dist[m - 1][n - 1]
```

---

### Worked-Out Example

```python
grid = [[0,1,1],[1,1,0],[1,1,0]]
Path (0,0) [cost 0] -> (0,1) [cost 1] -> (1,2) [cost 2] -> (2,2) [cost 2]
Minimum obstacles removed = 2
```

---

### Complexity Analysis

* **Time Complexity:** `O(m * n) (strictly linear, faster than Dijkstra)`
* **Space Complexity:** `O(m * n)`

---

### Takeaway Pattern

When edge weights are strictly {0, 1}, use 0-1 BFS (`appendleft` for 0, `append` for 1) to achieve O(V + E) shortest paths.