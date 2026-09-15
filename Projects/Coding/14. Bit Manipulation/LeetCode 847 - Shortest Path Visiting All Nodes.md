---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 847: Shortest Path Visiting All Nodes"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 847: Shortest Path Visiting All Nodes

**Target Companies:** Google (Signature Hard), Amazon

---

### Problem Statement

You have an undirected connected graph of `n` nodes. Return the length of the shortest path that visits every node (you may revisit nodes and reuse edges).

---

### Key Observation

* Since `n <= 12`, use a **Bitmask** to represent the subset of visited nodes (e.g. `1101_2` = nodes 0, 2, 3 visited).
* Target state: all `n` bits set: `(1 << n) - 1`.
* State in BFS: `(current_node, visited_mask)`.
* Multi-source BFS starting from every node at step 0 guarantees finding the global shortest path first.

---

### Core Technique: Bitmask State Space Multi-Source BFS

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        if n == 1:
            return 0
            
        target_mask = (1 << n) - 1
        queue = deque()  # (node, mask, dist)
        visited = set()  # (node, mask)
        
        # Multi-source init: path can start at any node
        for i in range(n):
            mask = 1 << i
            queue.append((i, mask, 0))
            visited.add((i, mask))
            
        while queue:
            node, mask, dist = queue.popleft()
            
            if mask == target_mask:
                return dist
                
            for neighbor in graph[node]:
                next_mask = mask | (1 << neighbor)
                if (neighbor, next_mask) not in visited:
                    visited.add((neighbor, next_mask))
                    queue.append((neighbor, next_mask, dist + 1))
                    
        return 0
```

---

### Worked-Out Example

```python
graph = [[1,2,3],[0],[0],[0]], n = 4, target_mask = 1111_2 (15)
Start BFS with all nodes.
Path: 1 -> 0 -> 2 -> 0 -> 3 (visiting all nodes in 4 steps)
Shortest distance = 4
```

---

### Complexity Analysis

* **Time Complexity:** `O(n * 2^n)`
* **Space Complexity:** `O(n * 2^n)`

---

### Takeaway Pattern

When n <= 15 and graph paths revisit nodes, encode visited subsets as bitmasks in BFS state `(node, mask)`.