---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 317: Shortest Distance from All Buildings"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 317: Shortest Distance from All Buildings

**Target Companies:** Google (Top #1 Hard Classic), Amazon | **Algorithm:** Multi-Source BFS Distance & Reachability Accumulation

---

### Problem Statement

You are given an `m x n` grid with 0 (empty land), 1 (building), and 2 (obstacle). Find an empty land that minimizes the total travel distance to all buildings (-1 if impossible).

---

### Key Observation

* Start BFS from each **building (1)** rather than from each empty land (fewer buildings than lands).
* Maintain `total_dist[r][c]` and `reach_count[r][c]`.
* A land is valid only if `reach_count[r][c] == total_buildings`.
* Pruning optimization: only expand to land reachable by all previously processed buildings.

---

### Core Algorithm & Technique: Multi-Source BFS Distance Accumulation with Reachability Pruning

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        if not grid:
            return -1
        m, n = len(grid), len(grid[0])
        total_dist = [[0] * n for _ in range(m)]
        reach_count = [[0] * n for _ in range(m)]
        buildings = 0
        
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 1:
                    buildings += 1
                    queue = deque([(r, c, 0)])
                    visited = set()
                    
                    while queue:
                        cr, cc, d = queue.popleft()
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < m and 0 <= nc < n and 
                                grid[nr][nc] == 0 and (nr, nc) not in visited):
                                visited.add((nr, nc))
                                total_dist[nr][nc] += (d + 1)
                                reach_count[nr][nc] += 1
                                queue.append((nr, nc, d + 1))
                                
        min_dist = float('inf')
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 0 and reach_count[r][c] == buildings:
                    min_dist = min(min_dist, total_dist[r][c])
                    
        return min_dist if min_dist != float('inf') else -1
```

---

### Worked-Out Example

```
grid = [
  [1, 0, 2, 0, 1],
  [0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0]
]
BFS from 3 buildings accumulates distances at each empty cell (0).
Cell (1, 2) reaches all 3 buildings with distances (3 + 3 + 1 = 7).
Result = 7
```

---

### Complexity Analysis

* **Time Complexity:** `O(buildings * m * n)`
* **Space Complexity:** `O(m * n)`

---

### Takeaway Pattern

Run BFS from sources (buildings) to targets (lands) to accumulate distances and verify 100% reachability.