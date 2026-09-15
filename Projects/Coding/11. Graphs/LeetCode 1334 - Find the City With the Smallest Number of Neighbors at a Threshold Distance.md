---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 1334: Find the City With the Smallest Number of Neighbors at a Threshold Distance"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 1334: Find the City With the Smallest Number of Neighbors at a Threshold Distance

**Target Companies:** Amazon, Google, Microsoft | **Algorithm:** Floyd-Warshall Algorithm (All-Pairs Shortest Path)

---

### Problem Statement

There are `n` cities and weighted `edges`. Find the city with the smallest number of reachable cities within distance `distanceThreshold` (if tie, return greatest city index).

---

### Key Observation

* We need shortest distances between **ALL pairs** of cities `(u, v)`.
* **Floyd-Warshall Algorithm**: Dynamic programming on intermediate nodes `k`:
* `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])` for all `k = 0..n-1`.
* Runs in $O(V^3)$ time and $O(V^2)$ space on a matrix.

---

### Core Algorithm & Technique: Floyd-Warshall All-Pairs Shortest Path Dynamic Programming

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            
        for u, v, w in edges:
            dist[u][v] = w
            dist[v][u] = w
            
        # Floyd-Warshall 3-nested loops: k is intermediate vertex
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        
        min_reachable = float('inf')
        best_city = -1
        
        for i in range(n):
            reachable = sum(1 for j in range(n) if i != j and dist[i][j] <= distanceThreshold)
            if reachable <= min_reachable:
                min_reachable = reachable
                best_city = i  # automatically picks greater index on tie
                
        return best_city
```

---

### Worked-Out Example

```python
n = 4, edges = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]], distanceThreshold = 4
Distance Matrix after Floyd-Warshall:
0: [0, 3, 4, 5] -> reaches 1, 2 (count: 2)
1: [3, 0, 1, 2] -> reaches 0, 2, 3 (count: 3)
2: [4, 1, 0, 1] -> reaches 0, 1, 3 (count: 3)
3: [5, 2, 1, 0] -> reaches 1, 2 (count: 2)
Tie between city 0 and city 3 -> return greatest index 3.
```

---

### Complexity Analysis

* **Time Complexity:** `O(n^3)`
* **Space Complexity:** `O(n^2)`

---

### Takeaway Pattern

When all-pairs shortest paths are required on dense networks with n <= 100, Floyd-Warshall's 3-loop structure is unmatched in simplicity.