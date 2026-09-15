---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 743: Network Delay Time"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 743: Network Delay Time

---

### Problem Statement

You are given a network of `n` nodes with directed weighted edges `[u, v, w]`. Return the minimum time for all nodes to receive a signal from source `k` (-1 if impossible).

---

### Key Observation

* All edge weights are positive (`w >= 0`).
* This is the Single-Source Shortest Path problem on weighted graphs.
* Use Dijkstra's Algorithm with a Min-Heap (Priority Queue).

---

### Core Technique: Dijkstra's Shortest Path with Min-Heap

---

### Python 3 Solution (with typing)

```python
import heapq
from typing import List
from collections import defaultdict

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        adj = defaultdict(list)
        for u, v, w in times:
            adj[u].append((v, w))
            
        min_heap = [(0, k)]  # (distance, node)
        distances = {}
        
        while min_heap:
            dist, node = heapq.heappop(min_heap)
            if node in distances:
                continue
            distances[node] = dist
            
            for neighbor, weight in adj[node]:
                if neighbor not in distances:
                    heapq.heappush(min_heap, (dist + weight, neighbor))
                    
        return max(distances.values()) if len(distances) == n else -1
```

---

### Worked-Out Example

```
times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
distances: {2: 0, 1: 1, 3: 1, 4: 2}
all 4 nodes reached! Max time = 2
```

---

### Complexity Analysis

* **Time Complexity:** `O((V + E) log V) with binary heap`
* **Space Complexity:** `O(V + E)`

---

### Takeaway Pattern

Use Dijkstra's algorithm for shortest paths on non-negative weighted graphs.