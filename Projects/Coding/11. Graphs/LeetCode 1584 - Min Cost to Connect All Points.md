---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 1584: Min Cost to Connect All Points"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 1584: Min Cost to Connect All Points

**Target Companies:** Amazon, Google, Microsoft | **Algorithm:** Prim's Algorithm (Minimum Spanning Tree with Min-Heap)

---

### Problem Statement

You are given an array `points` representing integer coordinates of `n` points. Return the minimum cost to make all points connected (Manhattan distance: `|x_i - x_j| + |y_i - y_j|`).

---

### Key Observation

* This problem asks for the **Minimum Spanning Tree (MST)** of a complete graph with `n` nodes.
* **Prim's Algorithm**: Start from arbitrary node (node 0). Maintain a Min-Heap of `(cost, next_node)` edges connecting the visited tree to unvisited nodes.
* Greedily extract the smallest edge to expand the MST until all `n` nodes are included.

---

### Core Algorithm & Technique: Prim's Algorithm with Min-Heap for Minimum Spanning Tree

---

### Python 3 Solution (with typing)

```python
import heapq
from typing import List

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        visited = set()
        min_heap = [(0, 0)]  # (cost, node)
        total_cost = 0
        
        while len(visited) < n:
            cost, u = heapq.heappop(min_heap)
            if u in visited:
                continue
                
            visited.add(u)
            total_cost += cost
            
            # Add all edges from newly added node u to unvisited nodes
            x1, y1 = points[u]
            for v in range(n):
                if v not in visited:
                    x2, y2 = points[v]
                    dist = abs(x1 - x2) + abs(y1 - y2)
                    heapq.heappush(min_heap, (dist, v))
                    
        return total_cost
```

---

### Worked-Out Example

```python
points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
Start at point 0 [0,0]:
  heap: (4, 1), (13, 2), (7, 3), (7, 4) -> pick point 1 (cost 4)
From point 1 [2,2]:
  heap: (3, 3) -> pick point 3 (cost 3)
From point 3 [5,2]:
  heap: (4, 4) -> pick point 4 (cost 4)
From point 4 [7,0]:
  heap: (9, 2) -> pick point 2 (cost 9)
Total MST cost = 4 + 3 + 4 + 9 = 20
```

---

### Complexity Analysis

* **Time Complexity:** `O(N^2 log N) with Min-Heap (or O(N^2) using dense Prim's array)`
* **Space Complexity:** `O(N^2) for heap edges`

---

### Takeaway Pattern

Use Prim's Algorithm when building minimum connected spanning trees from coordinates or weighted graphs.