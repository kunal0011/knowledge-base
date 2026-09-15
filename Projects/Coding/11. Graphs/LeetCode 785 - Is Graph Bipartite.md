---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 785: Is Graph Bipartite?"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 785: Is Graph Bipartite?

**Target Companies:** Amazon, Google, Meta | **Algorithm:** Bipartite Graph Verification (2-Coloring BFS / DFS)

---

### Problem Statement

There is an undirected graph with `n` nodes. Return `true` if and only if it is bipartite (nodes can be partitioned into two independent sets such that every edge connects between the two sets).

---

### Key Observation

* A graph is bipartite if and only if it contains **NO odd-length cycles**.
* Color nodes using two colors (e.g. `1` and `-1`).
* For every edge `(u, v)`, `v` must have the opposite color `-color[u]`. If a neighbor already has the same color, the graph is NOT bipartite.
* Handle disconnected components by checking every unvisited node.

---

### Core Algorithm & Technique: Graph 2-Coloring via Breadth-First Search

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        color = [0] * n  # 0: unvisited, 1: red, -1: blue
        
        for start in range(n):
            if color[start] != 0:
                continue
                
            color[start] = 1
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in graph[u]:
                    if color[v] == 0:
                        color[v] = -color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False  # same color conflict!
                        
        return True
```

---

### Worked-Out Example

```python
graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
Node 0 (color 1) -> neighbors 1, 2, 3 colored -1
Node 1 (color -1) connects to Node 2 (already colored -1!) -> conflict!
Graph is NOT bipartite -> return False
```

---

### Complexity Analysis

* **Time Complexity:** `O(V + E)`
* **Space Complexity:** `O(V) for color array and BFS queue`

---

### Takeaway Pattern

Verify bipartite graphs by alternating colors (+1 / -1); any neighbor with matching color indicates an odd cycle.