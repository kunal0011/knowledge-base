---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 684: Redundant Connection"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 684: Redundant Connection

---

### Problem Statement

Given an undirected graph that started as a tree with `n` nodes, with one additional edge added resulting in a cycle, return an edge that can be removed so that the resulting graph is a tree.

---

### Key Observation

* A tree of `n` nodes has no cycles and exactly `n - 1` edges.
* Adding an edge between two nodes already in the same connected component creates a cycle.
* Use Union-Find (Disjoint Set Union) with path compression and rank.

---

### Core Technique: Disjoint Set Union (DSU / Union-Find)

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        parent = list(range(len(edges) + 1))
        rank = [1] * (len(edges) + 1)
        
        def find(n):
            p = parent[n]
            while p != parent[p]:
                parent[p] = parent[parent[p]]  # path compression
                p = parent[p]
            return p
            
        def union(n1, n2):
            p1, p2 = find(n1), find(n2)
            if p1 == p2:
                return False  # cycle detected
            if rank[p1] > rank[p2]:
                parent[p2] = p1
                rank[p1] += rank[p2]
            else:
                parent[p1] = p2
                rank[p2] += rank[p1]
            return True
            
        for u, v in edges:
            if not union(u, v):
                return [u, v]
        return []
```

---

### Worked-Out Example

```python
edges = [[1,2],[1,3],[2,3]]
union(1, 2) -> OK
union(1, 3) -> OK
union(2, 3) -> find(2)==find(3)==1 -> cycle! Return [2, 3]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n * alpha(n)) (nearly linear)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Use Union-Find for instant cycle detection and dynamic connectivity in undirected graphs.